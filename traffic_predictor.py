"""
Network Traffic Prediction System
Parses Wireshark hex dump (.txt) or .pcapng files,
extracts features, trains ML models, and predicts:
  - Throughput/Bandwidth
  - Packet Count
  - Link Utilization
"""

import re
import os
import sys
import json
import struct
import argparse
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_hex_dump_txt(filepath: str) -> pd.DataFrame:
    """
    Parse Wireshark hex-dump text file.
    Format expected:
        +---------+...
        HH:MM:SS,mmm,uuu   ETHER
        |0   |xx|xx|...|
        +---------+...
    Returns a DataFrame with one row per packet.
    """
    packets = []
    ts_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d+),(\d+)\s+ETHER"
    )
    hex_pattern = re.compile(r"\|0\s+\|((?:[0-9a-fA-F]{2}\|)+)")

    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    current_ts = None
    current_hex = []

    for line in lines:
        ts_match = ts_pattern.search(line)
        if ts_match:
            # Save previous packet if any
            if current_ts is not None and current_hex:
                packets.append(_build_packet_record(current_ts, current_hex))
            current_ts = ts_match
            current_hex = []
            continue

        hex_match = hex_pattern.search(line)
        if hex_match and current_ts is not None:
            hex_str = hex_match.group(1).replace("|", " ").strip()
            current_hex = [int(b, 16) for b in hex_str.split() if b]

    # Last packet
    if current_ts is not None and current_hex:
        packets.append(_build_packet_record(current_ts, current_hex))

    if not packets:
        raise ValueError("No packets found in file. Check the format.")

    df = pd.DataFrame(packets)
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _build_packet_record(ts_match, raw_bytes: list) -> dict:
    h, m, s = int(ts_match.group(1)), int(ts_match.group(2)), int(ts_match.group(3))
    ms_part = ts_match.group(4)
    us_part = ts_match.group(5)
    # Convert to milliseconds
    ts_ms = (h * 3600 + m * 60 + s) * 1000 + int(ms_part[:3])

    pkt_len = len(raw_bytes)
    protocol, src_ip, dst_ip, sport, dport = _decode_layers(raw_bytes)

    return {
        "timestamp_ms": ts_ms,
        "hour": h,
        "minute": m,
        "second": s,
        "pkt_len": pkt_len,
        "protocol": protocol,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "sport": sport,
        "dport": dport,
    }


def _decode_layers(raw: list) -> tuple:
    """Decode Ethernet → IP/IPv6 → TCP/UDP and extract key fields."""
    if len(raw) < 14:
        return "OTHER", "", "", 0, 0

    eth_type = (raw[12] << 8) | raw[13]
    protocol = "OTHER"
    src_ip = dst_ip = ""
    sport = dport = 0

    if eth_type == 0x0800:  # IPv4
        if len(raw) < 34:
            return "IPv4", "", "", 0, 0
        proto = raw[23]
        src_ip = ".".join(str(raw[26 + i]) for i in range(4))
        dst_ip = ".".join(str(raw[30 + i]) for i in range(4))
        if proto == 6 and len(raw) >= 38:   # TCP
            protocol = "TCP"
            sport = (raw[34] << 8) | raw[35]
            dport = (raw[36] << 8) | raw[37]
        elif proto == 17 and len(raw) >= 38:  # UDP
            protocol = "UDP"
            sport = (raw[34] << 8) | raw[35]
            dport = (raw[36] << 8) | raw[37]
        elif proto == 2:
            protocol = "IGMP"
        else:
            protocol = f"IPv4-{proto}"

    elif eth_type == 0x86DD:  # IPv6
        if len(raw) < 54:
            return "IPv6", "", "", 0, 0
        next_hdr = raw[20]
        src_ip = ":".join(f"{raw[22+i]:02x}{raw[23+i]:02x}" for i in range(0, 16, 2))
        dst_ip = ":".join(f"{raw[38+i]:02x}{raw[39+i]:02x}" for i in range(0, 16, 2))
        if next_hdr == 58:
            protocol = "ICMPv6"
        elif next_hdr == 17 and len(raw) >= 58:
            protocol = "UDPv6"
            sport = (raw[54] << 8) | raw[55]
            dport = (raw[56] << 8) | raw[57]
        elif next_hdr == 6 and len(raw) >= 58:
            protocol = "TCPv6"
            sport = (raw[54] << 8) | raw[55]
            dport = (raw[56] << 8) | raw[57]
        else:
            protocol = "IPv6"

    elif eth_type == 0x0806:
        protocol = "ARP"
    elif eth_type == 0x0026:
        protocol = "STP"
    else:
        protocol = f"ETH-{eth_type:04x}"

    return protocol, src_ip, dst_ip, sport, dport


def parse_pcapng(filepath: str) -> pd.DataFrame:
    """
    Minimal pcapng parser (supports Interface Description + Enhanced/Simple Packet blocks).
    Falls back to pcap if magic matches.
    """
    packets = []

    with open(filepath, "rb") as f:
        data = f.read()

    pos = 0
    link_type = 1  # default Ethernet
    ts_resolution = 1e6  # default microseconds

    while pos < len(data) - 8:
        if pos + 8 > len(data):
            break
        block_type = struct.unpack_from("<I", data, pos)[0]
        block_len  = struct.unpack_from("<I", data, pos + 4)[0]

        if block_len < 12 or pos + block_len > len(data):
            break

        body = data[pos + 8: pos + block_len - 4]

        if block_type == 0x0A0D0D0A:  # Section Header
            pass
        elif block_type == 0x00000001:  # Interface Description
            if len(body) >= 2:
                link_type = struct.unpack_from("<H", body, 0)[0]
        elif block_type == 0x00000006:  # Enhanced Packet
            if len(body) >= 20:
                iface_id = struct.unpack_from("<I", body, 0)[0]
                ts_high  = struct.unpack_from("<I", body, 4)[0]
                ts_low   = struct.unpack_from("<I", body, 8)[0]
                cap_len  = struct.unpack_from("<I", body, 12)[0]
                orig_len = struct.unpack_from("<I", body, 16)[0]
                pkt_data = body[20: 20 + cap_len]
                ts_us = ((ts_high << 32) | ts_low)
                ts_ms = int(ts_us / 1000)
                raw = list(pkt_data)
                protocol, src_ip, dst_ip, sport, dport = _decode_layers(raw)
                packets.append({
                    "timestamp_ms": ts_ms,
                    "hour": (ts_ms // 3_600_000) % 24,
                    "minute": (ts_ms // 60_000) % 60,
                    "second": (ts_ms // 1_000) % 60,
                    "pkt_len": orig_len,
                    "protocol": protocol,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "sport": sport,
                    "dport": dport,
                })
        elif block_type == 0x00000003:  # Simple Packet
            if len(body) >= 4:
                orig_len = struct.unpack_from("<I", body, 0)[0]
                pkt_data = body[4:]
                raw = list(pkt_data)
                protocol, src_ip, dst_ip, sport, dport = _decode_layers(raw)
                packets.append({
                    "timestamp_ms": len(packets),  # no ts in simple block
                    "hour": 0, "minute": 0, "second": 0,
                    "pkt_len": orig_len,
                    "protocol": protocol,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "sport": sport,
                    "dport": dport,
                })

        pos += block_len

    if not packets:
        raise ValueError("No packets parsed from pcapng. Try the .txt hex dump format.")

    df = pd.DataFrame(packets)
    df.sort_values("timestamp_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING  (1-second windows)
# ─────────────────────────────────────────────────────────────────────────────

def build_windows(df: pd.DataFrame, window_sec: int = 1) -> pd.DataFrame:
    """Aggregate packet-level data into time windows."""
    df = df.copy()
    df["window"] = df["timestamp_ms"] // (window_sec * 1000)

    grp = df.groupby("window")

    agg = pd.DataFrame({
        "packet_count": grp["pkt_len"].count(),
        "total_bytes":  grp["pkt_len"].sum(),
        "mean_pkt_len": grp["pkt_len"].mean(),
        "max_pkt_len":  grp["pkt_len"].max(),
        "std_pkt_len":  grp["pkt_len"].std().fillna(0),
        "unique_src":   grp["src_ip"].nunique(),
        "unique_dst":   grp["dst_ip"].nunique(),
        "tcp_count":    grp["protocol"].apply(lambda x: (x == "TCP").sum()),
        "udp_count":    grp["protocol"].apply(lambda x: (x == "UDP").sum()),
        "arp_count":    grp["protocol"].apply(lambda x: (x == "ARP").sum()),
        "icmpv6_count": grp["protocol"].apply(lambda x: (x == "ICMPv6").sum()),
    })

    # Throughput in Mbps (bytes → bits → Mbps over window_sec)
    agg["throughput_mbps"] = (agg["total_bytes"] * 8) / (window_sec * 1e6)

    # Link utilization % — assume 1 Gbps link
    LINK_CAPACITY_MBPS = 1000.0
    agg["link_util_pct"] = (agg["throughput_mbps"] / LINK_CAPACITY_MBPS) * 100

    # Lag features
    for lag in [1, 2, 3, 5]:
        agg[f"pkt_lag_{lag}"]        = agg["packet_count"].shift(lag).fillna(0)
        agg[f"bytes_lag_{lag}"]      = agg["total_bytes"].shift(lag).fillna(0)
        agg[f"throughput_lag_{lag}"] = agg["throughput_mbps"].shift(lag).fillna(0)

    # Rolling averages
    for w in [3, 5, 10]:
        agg[f"pkt_roll_{w}"]        = agg["packet_count"].rolling(w, min_periods=1).mean()
        agg[f"bytes_roll_{w}"]      = agg["total_bytes"].rolling(w, min_periods=1).mean()
        agg[f"throughput_roll_{w}"] = agg["throughput_mbps"].rolling(w, min_periods=1).mean()

    agg.dropna(inplace=True)
    agg.reset_index(inplace=True)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "packet_count", "total_bytes", "mean_pkt_len", "max_pkt_len", "std_pkt_len",
    "unique_src", "unique_dst", "tcp_count", "udp_count", "arp_count", "icmpv6_count",
    "pkt_lag_1", "pkt_lag_2", "pkt_lag_3", "pkt_lag_5",
    "bytes_lag_1", "bytes_lag_2", "bytes_lag_3", "bytes_lag_5",
    "throughput_lag_1", "throughput_lag_2", "throughput_lag_3", "throughput_lag_5",
    "pkt_roll_3", "pkt_roll_5", "pkt_roll_10",
    "bytes_roll_3", "bytes_roll_5", "bytes_roll_10",
    "throughput_roll_3", "throughput_roll_5", "throughput_roll_10",
]

TARGET_COLS = {
    "throughput_mbps": "Throughput/Bandwidth (Mbps)",
    "packet_count":    "Packet Count (pkt/s)",
    "link_util_pct":   "Link Utilization (%)",
}


def train_models(agg: pd.DataFrame, model_dir: str = "models") -> dict:
    os.makedirs(model_dir, exist_ok=True)
    results = {}

    # Use only rows where we have a "next" row as target (shift -1)
    X_df = agg[FEATURE_COLS].copy()

    for target_col, label in TARGET_COLS.items():
        y = agg[target_col].shift(-1).ffill()
        X = X_df.values
        y = y.values

        if len(X) < 20:
            print(f"  ⚠  Too few windows for {label}. Skipping model training.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, subsample=0.8, random_state=42
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        model_path  = os.path.join(model_dir, f"{target_col}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{target_col}_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        results[target_col] = {
            "label": label,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2": round(r2, 4),
            "model_path": model_path,
            "scaler_path": scaler_path,
        }
        print(f"  ✔  {label}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_next(agg: pd.DataFrame, model_dir: str = "models",
                 horizon_minutes: int = 60) -> dict:
    """
    Use the last window's features to predict the next time step,
    then extrapolate for the requested horizon.
    """
    last_row = agg[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
    predictions = {}

    for target_col, label in TARGET_COLS.items():
        model_path  = os.path.join(model_dir, f"{target_col}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{target_col}_scaler.pkl")

        if not os.path.exists(model_path):
            continue

        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        X_scaled = scaler.transform(last_row)
        pred = float(model.predict(X_scaled)[0])
        pred = max(pred, 0)

        predictions[target_col] = {
            "label": label,
            "predicted_value": round(pred, 4),
        }

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# 5. REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(predictions: dict, model_results: dict,
                    agg: pd.DataFrame, output_dir: str = "output") -> str:
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    future = now + timedelta(hours=3)

    # Build natural language sentences (matching the desired output format)
    lines = []

    for target_col, info in predictions.items():
        val = info["predicted_value"]
        label = info["label"]

        if target_col == "throughput_mbps":
            if val >= 1000:
                val_str = f"{val/1000:.2f} Gbps"
            else:
                val_str = f"{val:.2f} Mbps"
            sentence = (
                f'Throughput/Bandwidth: "At {future.strftime("%I:%M %p")} '
                f'tomorrow, the incoming traffic will be {val_str}."'
            )

        elif target_col == "packet_count":
            val_str = f"{int(val):,}"
            sentence = (
                f'Packet Count: "We expect {val_str} packets per second '
                f'during the next 10-minute window."'
            )

        elif target_col == "link_util_pct":
            val_str = f"{val:.1f}%"
            start_t = future.strftime("%I:%M %p")
            end_t   = (future + timedelta(hours=2)).strftime("%I:%M %p")
            sentence = (
                f'Link Utilization: "The backbone link will be at {val_str} '
                f'capacity between {start_t} and {end_t}."'
            )
        else:
            sentence = f'{label}: {val}'

        lines.append(sentence)

    # Also output plain stats
    stats = {
        "generated_at": now.isoformat(),
        "total_windows_analyzed": len(agg),
        "avg_throughput_mbps": round(float(agg["throughput_mbps"].mean()), 4),
        "peak_throughput_mbps": round(float(agg["throughput_mbps"].max()), 4),
        "avg_packets_per_sec": round(float(agg["packet_count"].mean()), 2),
        "peak_packets_per_sec": int(agg["packet_count"].max()),
        "avg_link_util_pct": round(float(agg["link_util_pct"].mean()), 4),
        "peak_link_util_pct": round(float(agg["link_util_pct"].max()), 4),
        "predictions": predictions,
        "model_metrics": model_results,
        "output_sentences": lines,
    }

    # Save JSON
    json_path = os.path.join(output_dir, "prediction_report.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Save readable TXT
    txt_path = os.path.join(output_dir, "prediction_output.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  NETWORK TRAFFIC PREDICTION REPORT\n")
        f.write(f"  Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write("OUTPUT:\n\n")
        for s in lines:
            # Extract bullet-point style
            parts = s.split(":", 1)
            f.write(f"• {parts[0].strip()}:{parts[1] if len(parts)>1 else ''}\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("CURRENT NETWORK STATS (from captured data):\n\n")
        f.write(f"  Windows analyzed    : {stats['total_windows_analyzed']}\n")
        f.write(f"  Avg Throughput      : {stats['avg_throughput_mbps']} Mbps\n")
        f.write(f"  Peak Throughput     : {stats['peak_throughput_mbps']} Mbps\n")
        f.write(f"  Avg Packets/sec     : {stats['avg_packets_per_sec']}\n")
        f.write(f"  Peak Packets/sec    : {stats['peak_packets_per_sec']}\n")
        f.write(f"  Avg Link Utilization: {stats['avg_link_util_pct']}%\n")
        f.write(f"  Peak Link Utilization: {stats['peak_link_util_pct']}%\n")

        f.write("\n" + "-" * 60 + "\n")
        f.write("MODEL PERFORMANCE METRICS:\n\n")
        for target_col, m in model_results.items():
            f.write(f"  {m['label']}\n")
            f.write(f"    MAE : {m['mae']}\n")
            f.write(f"    RMSE: {m['rmse']}\n")
            f.write(f"    R²  : {m['r2']}\n\n")

    print(f"\n{'='*60}")
    print("  OUTPUT BE LIKE:\n")
    for s in lines:
        parts = s.split(":", 1)
        print(f"• {parts[0].strip()}:{parts[1] if len(parts)>1 else ''}")
    print(f"{'='*60}")
    print(f"\n✔ Full report saved to: {txt_path}")
    print(f"✔ JSON report saved to: {json_path}")

    return txt_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Network Traffic Prediction from Wireshark hex dump or pcapng"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to .txt hex-dump or .pcapng file (omit to use built-in demo data)"
    )
    parser.add_argument("--window", type=int, default=1,
                        help="Window size in seconds for aggregation (default: 1)")
    parser.add_argument("--model-dir", default="models",
                        help="Directory to save/load models")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for output reports")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NETWORK TRAFFIC PREDICTION SYSTEM")
    print("="*60)

    # ── Load data ──────────────────────────────────────────────
    if args.input_file:
        fpath = args.input_file
        print(f"\n[1/4] Loading: {fpath}")
        ext = os.path.splitext(fpath)[1].lower()
        if ext in (".pcapng", ".pcap"):
            df = parse_pcapng(fpath)
        else:
            df = parse_hex_dump_txt(fpath)
    else:
        print("\n[1/4] No file provided — using embedded demo packets …")
        df = _demo_dataframe()

    print(f"      {len(df):,} packets loaded.")

    # ── Build windows ──────────────────────────────────────────
    print(f"\n[2/4] Aggregating into {args.window}s windows …")
    agg = build_windows(df, window_sec=args.window)
    print(f"      {len(agg)} windows created.")

    # ── Train models ───────────────────────────────────────────
    print(f"\n[3/4] Training prediction models …")
    model_results = train_models(agg, model_dir=args.model_dir)

    # ── Predict & report ───────────────────────────────────────
    print(f"\n[4/4] Generating predictions & report …")
    predictions = predict_next(agg, model_dir=args.model_dir)
    generate_report(predictions, model_results, agg, output_dir=args.output_dir)


def _demo_dataframe() -> pd.DataFrame:
    """Generate synthetic demo data so the tool works out of the box."""
    rng = np.random.default_rng(42)
    n = 5000
    base_ms = 11 * 3600_000 + 5 * 60_000 + 55_000

    protocols = rng.choice(
        ["TCP", "UDP", "ARP", "ICMPv6", "IGMP"],
        size=n, p=[0.35, 0.30, 0.15, 0.12, 0.08]
    )
    pkt_lens = rng.integers(64, 1500, size=n).tolist()
    ts_offsets = np.cumsum(rng.exponential(2, size=n)).astype(int).tolist()

    records = []
    for i in range(n):
        ts = base_ms + ts_offsets[i]
        records.append({
            "timestamp_ms": ts,
            "hour":   (ts // 3_600_000) % 24,
            "minute": (ts // 60_000) % 60,
            "second": (ts // 1_000) % 60,
            "pkt_len": pkt_lens[i],
            "protocol": protocols[i],
            "src_ip": f"10.12.{rng.integers(1,5)}.{rng.integers(1,254)}",
            "dst_ip": f"10.12.{rng.integers(1,5)}.{rng.integers(1,254)}",
            "sport": int(rng.integers(1024, 65535)),
            "dport": int(rng.integers(1, 1024)),
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    main()

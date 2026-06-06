"""
kaggle_predict.py — Predict future traffic using models trained by kaggle_train.py

USAGE:
─────────────────────────────────────────────────────────────────────────────
    # Basic prediction (interactive input)
    python kaggle_predict.py --model-dir kaggle_models

    # Predict from a new CSV row (pass a CSV file)
    python kaggle_predict.py --model-dir kaggle_models --file new_data.csv

    # Custom steps and window
    python kaggle_predict.py --model-dir kaggle_models --file data.csv --steps 20 --window 60

    # No graph (text only)
    python kaggle_predict.py --model-dir kaggle_models --no-plot

NOTE:
─────────────────────────────────────────────────────────────────────────────
    This script uses models saved by kaggle_train.py.
    Run kaggle_train.py first if you haven't already:
        python kaggle_train.py --file your_dataset.csv
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

LINK_CAPACITY_MBPS  = 1000.0    # Assumed 1 Gbps link — change if needed
MAX_PACKETS_PER_SEC = 100_000   # For packet count % calculation

TARGETS = {
    "throughput_mbps": "Throughput/Bandwidth (Mbps)",
    "packet_count":    "Packet Count (pkt/s)",
    "link_util_pct":   "Link Utilization (%)",
}

PALETTE = {
    "bg":      "#1a1d2e",
    "card":    "#252840",
    "accent1": "#00d4ff",
    "accent2": "#ff6b6b",
    "accent3": "#ffd93d",
    "accent4": "#6bcb77",
    "text":    "#e0e0e0",
    "grid":    "#2e3150",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["card"],
    "axes.edgecolor":   PALETTE["grid"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["text"],
    "ytick.color":      PALETTE["text"],
    "text.color":       PALETTE["text"],
    "grid.color":       PALETTE["grid"],
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def load_models(model_dir: str) -> dict:
    loaded = {}
    missing = []

    for target_col in TARGETS:
        model_path   = os.path.join(model_dir, f"kaggle_{target_col}_model.pkl")
        scaler_path  = os.path.join(model_dir, f"kaggle_{target_col}_scaler.pkl")
        feature_path = os.path.join(model_dir, f"kaggle_{target_col}_features.pkl")

        if not os.path.exists(model_path):
            missing.append(target_col)
            continue

        loaded[target_col] = {
            "model":    joblib.load(model_path),
            "scaler":   joblib.load(scaler_path),
            "features": joblib.load(feature_path) if os.path.exists(feature_path) else [],
        }

    if missing:
        print(f"\n❌ Missing models for: {missing}")
        print(f"   Run training first:")
        print(f"   python kaggle_train.py --file your_dataset.csv --model-dir {model_dir}\n")
        sys.exit(1)

    print(f"   ✔ Loaded {len(loaded)} models from '{model_dir}/'")
    return loaded


# ─────────────────────────────────────────────────────────────────────────────
# GET INPUT FEATURES
# From CSV file OR interactive prompt
# ─────────────────────────────────────────────────────────────────────────────
def get_input_features(models: dict, csv_file: str = None) -> np.ndarray:
    # Use feature list from first available model
    feature_cols = next(iter(models.values()))["features"]

    if not feature_cols:
        print("⚠  No feature list saved with model. Using default feature set.")
        feature_cols = [
            "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
            "fwd_pkt_len_mean", "bwd_pkt_len_mean",
        ]

    if csv_file:
        # Load last row of CSV as input
        df = pd.read_csv(csv_file, low_memory=False, nrows=10000)
        df.columns = [c.strip() for c in df.columns]

        # Build reverse alias map: alias → standard name
        COLUMN_ALIASES = {
            "throughput_mbps":  ["Flow Bytes/s","flow bytes/s","Flow.Bytes.s","bytes_per_sec","bps"],
            "packet_count":     ["Flow Packets/s","flow packets/s","Flow.Packets.s","packets_per_sec","pps"],
            "link_util_pct":    ["Flow Duration","flow duration","Flow.Duration","dur","duration"],
            "fwd_packets":      ["Total Fwd Packets","total fwd packets","Total.Fwd.Packets","spkts"],
            "bwd_packets":      ["Total Backward Packets","total backward packets","Total.Backward.Packets","dpkts"],
            "fwd_bytes":        ["Total Length of Fwd Packets","Total.Length.of.Fwd.Packets","sbytes"],
            "bwd_bytes":        ["Total Length of Bwd Packets","Total.Length.of.Bwd.Packets","dbytes"],
            "fwd_pkt_len_mean": ["Fwd Packet Length Mean","Fwd.Packet.Length.Mean","smean"],
            "bwd_pkt_len_mean": ["Bwd Packet Length Mean","Bwd.Packet.Length.Mean","dmean"],
            "fwd_pkt_len_max":  ["Fwd Packet Length Max","Fwd.Packet.Length.Max"],
            "bwd_pkt_len_max":  ["Bwd Packet Length Max","Bwd.Packet.Length.Max"],
            "iat_mean":         ["Flow IAT Mean","Flow.IAT.Mean","sinpkt"],
            "iat_std":          ["Flow IAT Std","Flow.IAT.Std"],
            "active_mean":      ["Active Mean","Active.Mean"],
            "idle_mean":        ["Idle Mean","Idle.Mean"],
        }
        # Build lookup: raw col → standard name
        alias_to_std = {}
        for std, aliases in COLUMN_ALIASES.items():
            for a in aliases:
                alias_to_std[a.lower()] = std
        # Rename df columns to standard names where possible
        rename = {}
        for col in df.columns:
            std = alias_to_std.get(col.lower())
            if std and col != std:
                rename[col] = std
        if rename:
            df = df.rename(columns=rename)

        col_lower = {c.lower(): c for c in df.columns}
        row_values = []
        used_cols  = []

        for feat in feature_cols:
            if feat in df.columns:
                val = pd.to_numeric(df[feat], errors="coerce").dropna()
                val = float(val.iloc[-1]) if len(val) else 0.0
                row_values.append(val)
                used_cols.append(feat)
            elif feat.lower() in col_lower:
                actual = col_lower[feat.lower()]
                val = pd.to_numeric(df[actual], errors="coerce").dropna()
                val = float(val.iloc[-1]) if len(val) else 0.0
                row_values.append(val)
                used_cols.append(feat)
            else:
                row_values.append(0.0)
                used_cols.append(f"{feat}(→0)")

        matched = sum(1 for c in used_cols if "→0" not in c)
        print(f"   Matched {matched}/{len(feature_cols)} features from CSV.")
        return np.array(row_values).reshape(1, -1), feature_cols

    else:
        # Interactive: ask user to enter values
        print("\n  Enter current network snapshot values")
        print("  (Press Enter to use default/example values)\n")

        DEFAULTS = {
            "fwd_packets":       500,
            "bwd_packets":       450,
            "fwd_bytes":         750000,
            "bwd_bytes":         680000,
            "fwd_pkt_len_mean":  800,
            "bwd_pkt_len_mean":  750,
            "fwd_pkt_len_max":   1500,
            "bwd_pkt_len_max":   1500,
            "iat_mean":          2000,
            "iat_std":           500,
            "active_mean":       50000,
            "idle_mean":         100000,
            "throughput_mbps":   6.0,
            "packet_count":      950,
        }

        row_values = []
        for feat in feature_cols:
            default_val = DEFAULTS.get(feat, 0.0)
            try:
                raw = input(f"  {feat:<25} [default={default_val}]: ").strip()
                val = float(raw) if raw else float(default_val)
            except ValueError:
                val = float(default_val)
            row_values.append(val)

        return np.array(row_values).reshape(1, -1), feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT FUTURE STEPS
# ─────────────────────────────────────────────────────────────────────────────
def predict_future(models: dict, input_row: np.ndarray,
                   feature_cols: list, steps: int, window_sec: int) -> dict:
    now = datetime.now()
    all_forecasts = {}

    for target_col, label in TARGETS.items():
        m       = models[target_col]["model"]
        scaler  = models[target_col]["scaler"]
        current = input_row.copy()

        forecasts = []
        for step in range(steps):
            X_scaled = scaler.transform(current)
            pred = max(0.0, float(m.predict(X_scaled)[0]))

            # Compute traffic %
            if target_col == "throughput_mbps":
                traffic_pct = min((pred / LINK_CAPACITY_MBPS) * 100, 100)
            elif target_col == "packet_count":
                traffic_pct = min((pred / MAX_PACKETS_PER_SEC) * 100, 100)
            elif target_col == "link_util_pct":
                traffic_pct = min(pred, 100)
            else:
                traffic_pct = 0.0

            future_time = now + timedelta(seconds=(step + 1) * window_sec)
            forecasts.append({
                "step":        step + 1,
                "time":        future_time.strftime("%H:%M:%S"),
                "value":       round(pred, 4),
                "traffic_pct": round(traffic_pct, 2),
            })

            # Update input with latest prediction for rolling forecast
            if "throughput_mbps" in feature_cols:
                idx = feature_cols.index("throughput_mbps")
                current[0][idx] = pred
            if "packet_count" in feature_cols and target_col == "packet_count":
                idx = feature_cols.index("packet_count")
                current[0][idx] = pred

        all_forecasts[target_col] = forecasts

    return all_forecasts


# ─────────────────────────────────────────────────────────────────────────────
# PRINT TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_table(all_forecasts: dict, steps: int, window_sec: int):
    now = datetime.now()
    print(f"\n{'='*65}")
    print(f"  FUTURE TRAFFIC FORECAST — Next {steps} × {window_sec}s windows")
    print(f"  From: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    for target_col, label in TARGETS.items():
        forecasts = all_forecasts[target_col]
        print(f"  📊 {label}")
        print(f"  {'Step':<6} {'Time':<12} {'Predicted Value':<24} {'Traffic %'}")
        print(f"  {'-'*60}")

        for f in forecasts:
            val = f["value"]
            pct = f["traffic_pct"]

            if target_col == "throughput_mbps":
                val_str = f"{val/1000:.3f} Gbps" if val >= 1000 else f"{val:.4f} Mbps"
            elif target_col == "packet_count":
                val_str = f"{int(val):,} pkt/s"
            elif target_col == "link_util_pct":
                val_str = f"{val:.3f} %"
            else:
                val_str = str(val)

            bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            pct_str = f"{pct:5.1f}%  [{bar}]"

            print(f"  {f['step']:<6} {f['time']:<12} {val_str:<24} {pct_str}")
        print()

    # Natural language summary (matching slide output format)
    print(f"\n{'─'*65}")
    print("  OUTPUT BE LIKE:\n")

    for target_col, forecasts in all_forecasts.items():
        first = forecasts[0]
        val   = first["value"]
        t1    = forecasts[0]["time"]
        t2    = forecasts[-1]["time"] if len(forecasts) > 1 else t1

        if target_col == "throughput_mbps":
            val_str = f"{val/1000:.2f} Gbps" if val >= 1000 else f"{val:.2f} Mbps"
            print(f'  • Throughput/Bandwidth: "At {t1} tomorrow, '
                  f'the incoming traffic will be {val_str}."')

        elif target_col == "packet_count":
            print(f'  • Packet Count: "We expect {int(val):,} packets per second '
                  f'during the next 10-minute window."')

        elif target_col == "link_util_pct":
            print(f'  • Link Utilization: "The backbone link will be at {val:.1f}% '
                  f'capacity between {t1} and {t2}."')

    print(f"{'─'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT (does NOT auto-save — use matplotlib window to save manually)
# ─────────────────────────────────────────────────────────────────────────────
def plot_forecast(all_forecasts: dict, steps: int, window_sec: int):
    now   = datetime.now()
    times = [f["time"] for f in list(all_forecasts.values())[0]]
    tick_every = max(1, steps // 8)

    fig = plt.figure(figsize=(16, 11), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Kaggle-Trained Traffic Forecast — Next {steps} × {window_sec}s windows\n"
        f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14, color=PALETTE["accent1"], fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    metric_configs = [
        ("throughput_mbps", "Throughput (Mbps)",    PALETTE["accent1"], gs[0, 0]),
        ("packet_count",    "Packet Count (pkt/s)", PALETTE["accent4"], gs[0, 1]),
        ("link_util_pct",   "Link Utilization (%)", PALETTE["accent3"], gs[1, 0]),
    ]

    for target_col, label, color, pos in metric_configs:
        ax     = fig.add_subplot(pos)
        values = [f["value"] for f in all_forecasts[target_col]]
        x      = range(len(values))

        ax.plot(x, values, color=color, linewidth=2, label=label, zorder=3)
        ax.fill_between(x, values, alpha=0.18, color=color)

        peak_idx = int(np.argmax(values))
        ax.scatter([peak_idx], [values[peak_idx]], color="white", s=60, zorder=5)
        ax.annotate(
            f"Peak\n{values[peak_idx]:.2f}",
            xy=(peak_idx, values[peak_idx]),
            xytext=(min(peak_idx + max(1, steps // 10), len(values) - 1),
                    values[peak_idx]),
            fontsize=7, color="white",
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8),
        )

        ax.set_title(label, fontsize=11, color=color, pad=8)
        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel("Time", fontsize=9)
        ax.set_xticks(range(0, len(times), tick_every))
        ax.set_xticklabels(
            [times[i] for i in range(0, len(times), tick_every)],
            rotation=35, fontsize=7
        )
        ax.grid(True)
        ax.legend(fontsize=8)

    # Traffic % panel (bottom right)
    ax_pct = fig.add_subplot(gs[1, 1])
    colors_map = {
        "throughput_mbps": PALETTE["accent1"],
        "packet_count":    PALETTE["accent4"],
        "link_util_pct":   PALETTE["accent3"],
    }

    for target_col in TARGETS:
        pcts  = [f["traffic_pct"] for f in all_forecasts[target_col]]
        label = TARGETS[target_col].split("(")[0].strip()
        ax_pct.plot(range(len(pcts)), pcts,
                    color=colors_map[target_col],
                    linewidth=1.8, label=label)

    ax_pct.axhline(80, color=PALETTE["accent2"], linestyle="--",
                   linewidth=1.2, alpha=0.8, label="80% danger zone")
    ax_pct.fill_between(range(steps), 80, 105,
                        color=PALETTE["accent2"], alpha=0.07)
    ax_pct.set_ylim(0, 105)
    ax_pct.set_title("Traffic Load % (All Metrics)",
                     fontsize=11, color=PALETTE["accent2"], pad=8)
    ax_pct.set_ylabel("Traffic %", fontsize=9)
    ax_pct.set_xlabel("Time", fontsize=9)
    ax_pct.set_xticks(range(0, len(times), tick_every))
    ax_pct.set_xticklabels(
        [times[i] for i in range(0, len(times), tick_every)],
        rotation=35, fontsize=7
    )
    ax_pct.grid(True)
    ax_pct.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()   # ← displays window, does NOT auto-save


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Predict future traffic using Kaggle-trained models"
    )
    parser.add_argument("--model-dir", default="kaggle_models",
                        help="Folder with trained .pkl files (default: kaggle_models/)")
    parser.add_argument("--file",    default=None,
                        help="CSV file with current network data (optional)")
    parser.add_argument("--steps",   type=int, default=10,
                        help="How many future windows to predict (default: 10)")
    parser.add_argument("--window",  type=int, default=1,
                        help="Window size in seconds (default: 1)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib graph")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KAGGLE MODEL — TRAFFIC PREDICTOR")
    print("="*60)

    print(f"\n[1/3] Loading models from '{args.model_dir}/' …")
    models = load_models(args.model_dir)

    print(f"\n[2/3] Getting input features …")
    input_row, feature_cols = get_input_features(models, args.file)

    print(f"\n[3/3] Predicting next {args.steps} × {args.window}s windows …")
    all_forecasts = predict_future(
        models, input_row, feature_cols, args.steps, args.window
    )

    print_table(all_forecasts, args.steps, args.window)

    if not args.no_plot:
        plot_forecast(all_forecasts, args.steps, args.window)


if __name__ == "__main__":
    main()

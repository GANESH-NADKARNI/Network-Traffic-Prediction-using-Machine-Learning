"""
kaggle_train.py — Train traffic prediction models from any Kaggle CSV dataset.

RECOMMENDED KAGGLE DATASETS (download any one):
─────────────────────────────────────────────────────────────────────────────
1. CIC-IDS-2017 (BEST - most complete)
   https://www.kaggle.com/datasets/cicdataset/cicids2017

2. UNSW-NB15
   https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

3. Network Traffic Flows (87 apps)
   https://www.kaggle.com/datasets/jsrojas/ip-network-traffic-flows-labeled-with-87-apps

4. KDD Cup 1999
   https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data

5. Network Intrusion Detection
   https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection

HOW TO DOWNLOAD FROM KAGGLE:
─────────────────────────────────────────────────────────────────────────────
Option A — Manual (easiest):
    1. Open any link above
    2. Click Download button
    3. Extract ZIP → place CSV file in same folder as this script

Option B — Kaggle API:
    1. pip install kaggle
    2. Go to https://www.kaggle.com/settings → Create New API Token
    3. Place kaggle.json in ~/.kaggle/kaggle.json
    4. kaggle datasets download -d cicdataset/cicids2017 --unzip

USAGE:
─────────────────────────────────────────────────────────────────────────────
    python kaggle_train.py --file your_dataset.csv
    python kaggle_train.py --file Friday-WorkingHours.csv
    python kaggle_train.py --file UNSW_NB15_training-set.csv
    python kaggle_train.py --file your_dataset.csv --model-dir my_models
    python kaggle_train.py --file your_dataset.csv --rows 500000
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN NAME MAPPINGS FOR POPULAR DATASETS
# Maps raw Kaggle column names → our standard internal names
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_ALIASES = {
    # ── Throughput / bytes ───────────────────────────────────────────────────
    "throughput_mbps":    ["Flow Bytes/s", "flow bytes/s", "flow_bytes_s",
                           "Flow.Bytes.s", "TotBytes", "total_bytes",
                           "bytes_per_sec", "bps"],

    # ── Packet count / rate ──────────────────────────────────────────────────
    "packet_count":       ["Flow Packets/s", "flow packets/s", "flow_packets_s",
                           "Flow.Packets.s", "TotPkts", "total_packets",
                           "packets_per_sec", "pps", "num_pkts_sent"],

    # ── Link utilization proxy (flow duration in microseconds) ───────────────
    "link_util_pct":      ["Flow Duration", "flow duration", "flow_duration",
                           "Flow.Duration", "dur", "duration", "Dur"],

    # ── Supporting features ──────────────────────────────────────────────────
    "fwd_packets":        ["Total Fwd Packets", "total fwd packets",
                           "Total.Fwd.Packets", "spkts", "src_pkts"],
    "bwd_packets":        ["Total Backward Packets", "total backward packets",
                           "Total.Backward.Packets", "dpkts", "dst_pkts"],
    "fwd_bytes":          ["Total Length of Fwd Packets", "totlen fwd pkts",
                           "Total.Length.of.Fwd.Packets", "sbytes", "src_bytes"],
    "bwd_bytes":          ["Total Length of Bwd Packets", "totlen bwd pkts",
                           "Total.Length.of.Bwd.Packets", "dbytes", "dst_bytes"],
    "fwd_pkt_len_mean":   ["Fwd Packet Length Mean", "fwd packet length mean",
                           "Fwd.Packet.Length.Mean", "smean"],
    "bwd_pkt_len_mean":   ["Bwd Packet Length Mean", "bwd packet length mean",
                           "Bwd.Packet.Length.Mean", "dmean"],
    "fwd_pkt_len_max":    ["Fwd Packet Length Max", "fwd packet length max",
                           "Fwd.Packet.Length.Max"],
    "bwd_pkt_len_max":    ["Bwd Packet Length Max", "bwd packet length max",
                           "Bwd.Packet.Length.Max"],
    "iat_mean":           ["Flow IAT Mean", "flow iat mean",
                           "Flow.IAT.Mean", "sinpkt", "dinpkt"],
    "iat_std":            ["Flow IAT Std", "flow iat std", "Flow.IAT.Std"],
    "active_mean":        ["Active Mean", "active mean", "Active.Mean"],
    "idle_mean":          ["Idle Mean", "idle mean", "Idle.Mean"],
}

LINK_CAPACITY_MBPS = 1000.0   # Assumed 1 Gbps link — change if needed
MAX_PACKETS_PER_SEC = 100000  # For normalising packet count to %


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CSV
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(filepath: str, max_rows: int) -> pd.DataFrame:
    size_mb = os.path.getsize(filepath) / 1e6
    print(f"   File : {filepath}  ({size_mb:.1f} MB)")
    df = pd.read_csv(filepath, low_memory=False, nrows=max_rows)
    df.columns = [c.strip() for c in df.columns]   # strip whitespace
    print(f"   Rows : {len(df):,}   Columns: {len(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — AUTO-MAP COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
def auto_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to find and rename columns to our standard names.
    Works across CIC-IDS, UNSW-NB15, KDD, NetFlow, and similar datasets.
    """
    rename = {}
    df_cols_lower = {c.lower(): c for c in df.columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        if standard_name in df.columns:
            continue   # already has the right name
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = standard_name
                break
            if alias.lower() in df_cols_lower:
                rename[df_cols_lower[alias.lower()]] = standard_name
                break

    if rename:
        df = df.rename(columns=rename)
        print(f"   Mapped columns: {list(rename.values())}")
    else:
        print("   ⚠  No automatic column mapping matched.")
        print(f"   Available columns: {list(df.columns[:15])} ...")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DERIVE TARGETS IF MISSING
# ─────────────────────────────────────────────────────────────────────────────
def derive_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a target column is missing, try to derive it from other columns.
    """
    # throughput_mbps — from bytes columns
    if "throughput_mbps" not in df.columns:
        if "fwd_bytes" in df.columns and "bwd_bytes" in df.columns:
            total_bytes = pd.to_numeric(df["fwd_bytes"], errors="coerce").fillna(0) + \
                          pd.to_numeric(df["bwd_bytes"], errors="coerce").fillna(0)
            dur_sec = 1.0
            if "link_util_pct" in df.columns:
                dur_sec = pd.to_numeric(df["link_util_pct"], errors="coerce") \
                            .fillna(1e6).clip(lower=1) / 1e6
            df["throughput_mbps"] = (total_bytes * 8) / (dur_sec * 1e6)
            print("   ✔ Derived throughput_mbps from fwd_bytes + bwd_bytes")

    # packet_count — from fwd + bwd packets
    if "packet_count" not in df.columns:
        if "fwd_packets" in df.columns and "bwd_packets" in df.columns:
            df["packet_count"] = \
                pd.to_numeric(df["fwd_packets"], errors="coerce").fillna(0) + \
                pd.to_numeric(df["bwd_packets"], errors="coerce").fillna(0)
            print("   ✔ Derived packet_count from fwd_packets + bwd_packets")

    # link_util_pct — normalised to 0-100 always
    if "link_util_pct" not in df.columns:
        if "throughput_mbps" in df.columns:
            df["link_util_pct"] = (
                pd.to_numeric(df["throughput_mbps"], errors="coerce")
                  .fillna(0) / LINK_CAPACITY_MBPS * 100
            ).clip(0, 100)
            print("   ✔ Derived link_util_pct from throughput_mbps")
    else:
        # If column came from flow duration (microseconds), normalise to 0-100
        raw = pd.to_numeric(df["link_util_pct"], errors="coerce").fillna(0)
        if raw.max() > 100:
            df["link_util_pct"] = (
                (raw - raw.min()) / (raw.max() - raw.min() + 1e-9) * 100
            ).clip(0, 100)
            print("   ✔ Normalised link_util_pct to 0-100 range")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_CANDIDATES = [
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean", "fwd_pkt_len_max", "bwd_pkt_len_max",
    "iat_mean", "iat_std", "active_mean", "idle_mean",
    "throughput_mbps", "packet_count",
]

TARGETS = {
    "throughput_mbps": "Throughput/Bandwidth (Mbps)",
    "packet_count":    "Packet Count (pkt/s)",
    "link_util_pct":   "Link Utilization (%)",
}


def build_features(df: pd.DataFrame):
    # Only use columns that exist
    feature_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]

    if len(feature_cols) < 2:
        print("\n❌ Not enough feature columns found.")
        print(f"   Found: {feature_cols}")
        print("   Make sure your CSV has columns like:")
        print("   Flow Bytes/s, Flow Packets/s, Flow Duration, etc.")
        sys.exit(1)

    print(f"   Features used ({len(feature_cols)}): {feature_cols}")

    # Convert all to numeric, drop bad rows
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for target in TARGETS:
        if target in df.columns:
            df[target] = pd.to_numeric(df[target], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols)
    df = df.dropna(subset=[t for t in TARGETS if t in df.columns])
    df = df.reset_index(drop=True)

    print(f"   Clean rows: {len(df):,}")
    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
def train_models(df: pd.DataFrame, feature_cols: list, model_dir: str) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    results = {}

    X = df[feature_cols].values

    for target_col, label in TARGETS.items():
        if target_col not in df.columns:
            print(f"   ⚠  Skipping {label} — column not found.")
            continue

        y = df[target_col].values

        if len(X) < 50:
            print(f"   ⚠  Too few rows for {label}. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_split=10,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        model_path  = os.path.join(model_dir, f"kaggle_{target_col}_model.pkl")
        scaler_path = os.path.join(model_dir, f"kaggle_{target_col}_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Save feature list so predictor knows what columns to use
        joblib.dump(feature_cols,
                    os.path.join(model_dir, f"kaggle_{target_col}_features.pkl"))

        results[target_col] = {
            "label":       label,
            "mae":         round(mae, 4),
            "rmse":        round(rmse, 4),
            "r2":          round(r2, 4),
            "model_path":  model_path,
            "scaler_path": scaler_path,
            "n_train":     len(X_train),
            "n_test":      len(X_test),
        }
        print(f"   ✔  {label}")
        print(f"      MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
        print(f"      Saved → {model_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — SAVE METRICS + PLOT FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics_and_plots(results: dict, df: pd.DataFrame,
                           feature_cols: list, model_dir: str):
    import json

    metrics_path = os.path.join(model_dir, "kaggle_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   ✔ Metrics saved → {metrics_path}")

    # Feature importance plot for each model
    PALETTE = {"bg": "#1a1d2e", "card": "#252840",
               "accent1": "#00d4ff", "text": "#e0e0e0", "grid": "#2e3150"}

    for target_col, info in results.items():
        model = joblib.load(info["model_path"])
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
        ax.set_facecolor(PALETTE["card"])
        bars = ax.barh(
            [feature_cols[i] for i in indices],
            [importances[i] for i in indices],
            color=PALETTE["accent1"], alpha=0.85
        )
        ax.set_xlabel("Importance", color=PALETTE["text"])
        ax.set_title(f"Feature Importance — {info['label']}\nR²={info['r2']}",
                     color=PALETTE["accent1"], fontsize=12, fontweight="bold")
        ax.tick_params(colors=PALETTE["text"])
        ax.spines["bottom"].set_color(PALETTE["grid"])
        ax.spines["left"].set_color(PALETTE["grid"])
        plt.tight_layout()

        plot_path = os.path.join(model_dir, f"kaggle_{target_col}_importance.png")
        plt.savefig(plot_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        plt.close()
        print(f"   ✔ Importance plot → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train traffic prediction models from a Kaggle CSV dataset"
    )
    parser.add_argument("--file",      required=True,
                        help="Path to downloaded Kaggle CSV file")
    parser.add_argument("--model-dir", default="kaggle_models",
                        help="Where to save trained models (default: kaggle_models/)")
    parser.add_argument("--rows",      type=int, default=300_000,
                        help="Max rows to load (default: 300000)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  KAGGLE DATASET TRAINER")
    print("="*60)

    print("\n[1/5] Loading CSV …")
    df = load_csv(args.file, args.rows)

    print("\n[2/5] Mapping columns …")
    df = auto_map_columns(df)

    print("\n[3/5] Deriving target columns …")
    df = derive_targets(df)

    print("\n[4/5] Building feature matrix …")
    df, feature_cols = build_features(df)

    print(f"\n[5/5] Training models → saving to '{args.model_dir}/' …\n")
    results = train_models(df, feature_cols, args.model_dir)

    save_metrics_and_plots(results, df, feature_cols, args.model_dir)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    print(f"\n  Models saved in: {args.model_dir}/")
    print("  Now run prediction:")
    print(f"  python kaggle_predict.py --model-dir {args.model_dir}\n")


if __name__ == "__main__":
    main()

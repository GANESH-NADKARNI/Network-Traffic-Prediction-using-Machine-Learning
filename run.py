#!/usr/bin/env python3
"""
run.py — Entry point for the Network Traffic Prediction System.

Usage:
    python run.py                          # demo mode (built-in synthetic data)
    python run.py capture.txt              # Wireshark hex-dump .txt
    python run.py capture.pcapng           # pcapng file
    python run.py capture.txt --window 5   # 5-second aggregation windows
"""

import sys
import os

# Ensure local imports work regardless of cwd
sys.path.insert(0, os.path.dirname(__file__))

from traffic_predictor import (
    parse_hex_dump_txt, parse_pcapng,
    build_windows, train_models, predict_next,
    generate_report, _demo_dataframe
)
from visualize import plot_all
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Network Traffic Prediction — Wireshark hex dump / pcapng"
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Path to .txt or .pcapng file"
    )
    parser.add_argument("--window", type=int, default=1,
                        help="Window size in seconds (default: 1)")
    parser.add_argument("--model-dir",  default="models")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  NETWORK TRAFFIC PREDICTION SYSTEM")
    print("="*60)

    # 1. Load ─────────────────────────────────────────────────────
    if args.input_file:
        fpath = args.input_file
        print(f"\n[1/5] Loading: {fpath}")
        ext = os.path.splitext(fpath)[1].lower()
        if ext in (".pcapng", ".pcap"):
            df = parse_pcapng(fpath)
        else:
            df = parse_hex_dump_txt(fpath)
    else:
        print("\n[1/5] No file provided — using built-in demo dataset …")
        df = _demo_dataframe()

    print(f"      {len(df):,} packets loaded.")

    # 2. Feature engineering ──────────────────────────────────────
    print(f"\n[2/5] Aggregating into {args.window}s windows …")
    agg = build_windows(df, window_sec=args.window)
    print(f"      {len(agg)} windows created.")

    # 3. Train ────────────────────────────────────────────────────
    print(f"\n[3/5] Training Gradient Boosting models …")
    model_results = train_models(agg, model_dir=args.model_dir)

    # 4. Predict & report ─────────────────────────────────────────
    print(f"\n[4/5] Predicting next window …")
    predictions = predict_next(agg, model_dir=args.model_dir)
    generate_report(predictions, model_results, agg, output_dir=args.output_dir)

    # 5. Visualize ────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\n[5/5] Generating visualizations …")
        plot_all(agg, predictions, model_results, output_dir=args.output_dir)
    else:
        print("\n[5/5] Skipping plots (--no-plots).")

    print("\n" + "="*60)
    print("  Done! Check the 'output/' folder for results.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

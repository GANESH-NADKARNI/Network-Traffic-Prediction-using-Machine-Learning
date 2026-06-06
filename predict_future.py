import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
from traffic_predictor import parse_hex_dump_txt, build_windows, FEATURE_COLS, TARGET_COLS

LINK_CAPACITY_MBPS = 1000.0  # Assume 1 Gbps link — change if needed

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
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
})


def predict_future(capture_file, steps=60, window_sec=1, model_dir="models"):
    """
    steps      = how many windows ahead to predict
    window_sec = size of each window in seconds
    """

    # 1. Load & process capture file
    df  = parse_hex_dump_txt(capture_file)
    agg = build_windows(df, window_sec=window_sec)

    # 2. Start from last known window
    last_row = agg[FEATURE_COLS].iloc[-1].values.copy()
    now = datetime.now()

    print(f"\n{'='*60}")
    print(f"  FUTURE TRAFFIC FORECAST — Next {steps} windows")
    print(f"  Each window = {window_sec} second(s)")
    print(f"  From: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    all_forecasts = {}

    for target_col, label in TARGET_COLS.items():
        model  = joblib.load(f"{model_dir}/{target_col}_model.pkl")
        scaler = joblib.load(f"{model_dir}/{target_col}_scaler.pkl")

        forecasts = []
        current = last_row.copy().reshape(1, -1)

        for step in range(steps):
            X_scaled = scaler.transform(current)
            pred = float(model.predict(X_scaled)[0])
            pred = max(0, pred)

            future_time = now + timedelta(seconds=(step + 1) * window_sec)

            # Compute traffic % relative to link capacity
            if target_col == "throughput_mbps":
                traffic_pct = min((pred / LINK_CAPACITY_MBPS) * 100, 100)
            elif target_col == "link_util_pct":
                traffic_pct = min(pred, 100)
            elif target_col == "packet_count":
                # % relative to assumed max 10,000 pkt/s
                traffic_pct = min((pred / 10000) * 100, 100)
            else:
                traffic_pct = 0.0

            forecasts.append({
                "step":        step + 1,
                "time":        future_time.strftime("%H:%M:%S"),
                "value":       round(pred, 4),
                "traffic_pct": round(traffic_pct, 2),
            })

            # Roll lag features forward
            current[0][13] = current[0][12]
            current[0][12] = current[0][11]
            current[0][11] = current[0][10]
            current[0][10] = pred
            current[0][0]  = pred

        all_forecasts[target_col] = forecasts

    # 3. Print table
    _print_forecast_table(all_forecasts, TARGET_COLS, window_sec)

    # 4. Show graph (does NOT save)
    _plot_forecast(all_forecasts, TARGET_COLS, steps, window_sec, now)

    return all_forecasts


def _print_forecast_table(all_forecasts, TARGET_COLS, window_sec):
    for target_col, label in TARGET_COLS.items():
        forecasts = all_forecasts[target_col]
        print(f"  {label}")
        print(f"  {'Step':<6} {'Time':<12} {'Predicted Value':<22} {'Traffic %'}")
        print(f"  {'-'*52}")
        for f in forecasts:
            val = f['value']
            pct = f['traffic_pct']

            if target_col == "throughput_mbps":
                val_str = f"{val:.3f} Mbps"
            elif target_col == "packet_count":
                val_str = f"{int(val):,} pkt/s"
            elif target_col == "link_util_pct":
                val_str = f"{val:.3f} %"
            else:
                val_str = str(val)

            # Traffic % bar indicator
            bar_len  = int(pct / 5)   # max 20 chars
            bar      = "█" * bar_len + "░" * (20 - bar_len)
            pct_str  = f"{pct:5.1f}%  [{bar}]"

            print(f"  {f['step']:<6} {f['time']:<12} {val_str:<22} {pct_str}")
        print()


def _plot_forecast(all_forecasts, TARGET_COLS, steps, window_sec, now):
    fig = plt.figure(figsize=(16, 11), facecolor=PALETTE["bg"])
    fig.suptitle(
        f"Network Traffic Forecast — Next {steps} × {window_sec}s windows\n"
        f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14, color=PALETTE["accent1"], fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    metric_configs = [
        ("throughput_mbps", "Throughput (Mbps)",    PALETTE["accent1"], gs[0, 0]),
        ("packet_count",    "Packet Count (pkt/s)", PALETTE["accent4"], gs[0, 1]),
        ("link_util_pct",   "Link Utilization (%)", PALETTE["accent3"], gs[1, 0]),
    ]

    times = [f["time"] for f in list(all_forecasts.values())[0]]
    tick_every = max(1, steps // 8)   # show ~8 x-axis labels max

    for target_col, label, color, subplot_pos in metric_configs:
        ax = fig.add_subplot(subplot_pos)
        forecasts = all_forecasts[target_col]
        values = [f["value"] for f in forecasts]
        pcts   = [f["traffic_pct"] for f in forecasts]
        x      = range(len(values))

        # Main value line
        ax.plot(x, values, color=color, linewidth=2, label=label, zorder=3)
        ax.fill_between(x, values, alpha=0.18, color=color)

        # Mark peak
        peak_idx = int(np.argmax(values))
        ax.scatter([peak_idx], [values[peak_idx]],
                   color="white", s=60, zorder=5)
        ax.annotate(
            f"Peak\n{values[peak_idx]:.2f}",
            xy=(peak_idx, values[peak_idx]),
            xytext=(peak_idx + max(1, steps // 10), values[peak_idx]),
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

    # ── Traffic % comparison panel (bottom right) ──────────────────────────
    ax_pct = fig.add_subplot(gs[1, 1])
    colors_pct = [PALETTE["accent1"], PALETTE["accent4"], PALETTE["accent3"]]
    labels_pct = []

    for (target_col, label, color, _), cpct in zip(metric_configs, colors_pct):
        pcts = [f["traffic_pct"] for f in all_forecasts[target_col]]
        ax_pct.plot(range(len(pcts)), pcts, color=color,
                    linewidth=1.8, label=label.split("(")[0].strip())
        labels_pct.append(label)

    # Danger zone line at 80%
    ax_pct.axhline(80, color=PALETTE["accent2"], linestyle="--",
                   linewidth=1.2, alpha=0.8, label="80% danger threshold")
    ax_pct.fill_between(range(steps), 80, 100,
                        color=PALETTE["accent2"], alpha=0.08)

    ax_pct.set_ylim(0, 105)
    ax_pct.set_title("Traffic Load % (All Metrics)", fontsize=11,
                     color=PALETTE["accent2"], pad=8)
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
    plt.show()   # ← displays, does NOT save


if __name__ == "__main__":
    import sys
    file   = sys.argv[1] if len(sys.argv) > 1 else "sample_capture.txt"
    steps  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    window = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    predict_future(file, steps=steps, window_sec=window)
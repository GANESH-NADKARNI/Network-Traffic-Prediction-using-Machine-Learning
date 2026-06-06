"""
visualize.py — Plot traffic statistics and prediction results.
All charts saved to output/plots/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from collections import Counter


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
    "font.family":       "DejaVu Sans",
})


def plot_all(agg: pd.DataFrame, predictions: dict,
             model_results: dict, output_dir: str = "output"):
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    _plot_dashboard(agg, predictions, model_results, plot_dir)
    _plot_protocol_pie(agg, plot_dir)
    _plot_throughput_ts(agg, plot_dir)
    print(f"  ✔  Charts saved to {plot_dir}/")


def _plot_dashboard(agg, predictions, model_results, plot_dir):
    fig = plt.figure(figsize=(18, 12), facecolor=PALETTE["bg"])
    fig.suptitle("Network Traffic Prediction Dashboard",
                 fontsize=20, color=PALETTE["accent1"],
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Throughput time series ──
    ax1 = fig.add_subplot(gs[0, :2])
    x = range(len(agg))
    ax1.plot(x, agg["throughput_mbps"], color=PALETTE["accent1"],
             linewidth=1.5, label="Throughput (Mbps)")
    ax1.fill_between(x, agg["throughput_mbps"],
                     alpha=0.2, color=PALETTE["accent1"])
    ax1.set_title("Throughput Over Time", fontsize=12, color=PALETTE["accent1"])
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Mbps")
    ax1.grid(True)
    ax1.legend(fontsize=9)

    # ── Packet count ──
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.bar(x, agg["packet_count"], color=PALETTE["accent4"],
            alpha=0.75, width=0.8, label="Packets/s")
    ax2.set_title("Packet Count Per Window", fontsize=12, color=PALETTE["accent4"])
    ax2.set_xlabel("Window")
    ax2.set_ylabel("Packets")
    ax2.grid(True, axis="y")
    ax2.legend(fontsize=9)

    # ── Link utilization ──
    ax3 = fig.add_subplot(gs[2, :2])
    colors = [PALETTE["accent2"] if v > 50 else PALETTE["accent3"]
              for v in agg["link_util_pct"]]
    ax3.bar(x, agg["link_util_pct"], color=colors, alpha=0.8, width=0.8)
    ax3.axhline(50, color=PALETTE["accent2"], linestyle="--", alpha=0.7, label="50% threshold")
    ax3.set_title("Link Utilization %", fontsize=12, color=PALETTE["accent3"])
    ax3.set_xlabel("Window")
    ax3.set_ylabel("%")
    ax3.grid(True, axis="y")
    ax3.legend(fontsize=9)

    # ── Prediction cards ──
    card_ax = fig.add_subplot(gs[:, 2])
    card_ax.set_xlim(0, 1)
    card_ax.set_ylim(0, 1)
    card_ax.axis("off")
    card_ax.set_title("Predictions", fontsize=13,
                       color=PALETTE["accent1"], pad=12)

    card_colors = [PALETTE["accent1"], PALETTE["accent4"], PALETTE["accent3"]]
    card_labels = []
    card_values = []

    for target_col, info in predictions.items():
        val = info["predicted_value"]
        lbl = info["label"].split("(")[0].strip()
        if target_col == "throughput_mbps":
            if val >= 1000:
                card_values.append(f"{val/1000:.2f} Gbps")
            else:
                card_values.append(f"{val:.2f} Mbps")
        elif target_col == "packet_count":
            card_values.append(f"{int(val):,} pkt/s")
        elif target_col == "link_util_pct":
            card_values.append(f"{val:.1f} %")
        else:
            card_values.append(str(val))
        card_labels.append(lbl)

    y_positions = [0.82, 0.52, 0.22]
    for i, (lbl, val, col, ypos) in enumerate(
            zip(card_labels, card_values, card_colors, y_positions)):
        rect = FancyBboxPatch((0.05, ypos - 0.08), 0.9, 0.22,
                               boxstyle="round,pad=0.02",
                               facecolor=PALETTE["bg"],
                               edgecolor=col, linewidth=2)
        card_ax.add_patch(rect)
        card_ax.text(0.5, ypos + 0.08, lbl, ha="center", va="center",
                     fontsize=10, color=PALETTE["text"])
        card_ax.text(0.5, ypos - 0.02, val, ha="center", va="center",
                     fontsize=18, fontweight="bold", color=col)

    # Model metrics mini table
    if model_results:
        y_start = 0.10
        card_ax.text(0.5, y_start - 0.01, "Model R² Scores",
                     ha="center", fontsize=9,
                     color=PALETTE["text"], alpha=0.7)
        for j, (tc, mr) in enumerate(model_results.items()):
            short = mr["label"].split("/")[0][:16]
            card_ax.text(0.08, y_start - 0.06 - j * 0.05,
                         f"{short}: R²={mr['r2']}",
                         fontsize=8, color=PALETTE["accent4"], alpha=0.85)

    path = os.path.join(plot_dir, "dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"      → dashboard.png")


def _plot_protocol_pie(agg, plot_dir):
    # Reconstruct protocol counts from tcp/udp/arp/icmpv6 columns
    total = len(agg)
    if total == 0:
        return

    proto_data = {
        "TCP":    agg["tcp_count"].sum(),
        "UDP":    agg["udp_count"].sum(),
        "ARP":    agg["arp_count"].sum(),
        "ICMPv6": agg["icmpv6_count"].sum(),
    }
    # OTHER
    known = sum(proto_data.values())
    other = agg["packet_count"].sum() - known
    if other > 0:
        proto_data["OTHER"] = other

    labels = list(proto_data.keys())
    sizes  = list(proto_data.values())
    colors = [PALETTE["accent1"], PALETTE["accent4"],
              PALETTE["accent3"], PALETTE["accent2"], "#9b59b6"]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors[:len(labels)],
        autopct="%1.1f%%", startangle=140,
        textprops={"color": PALETTE["text"], "fontsize": 11},
        wedgeprops={"linewidth": 1.5, "edgecolor": PALETTE["bg"]},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_color(PALETTE["bg"])
    ax.set_title("Protocol Distribution", fontsize=14,
                 color=PALETTE["accent1"], fontweight="bold")

    path = os.path.join(plot_dir, "protocol_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"      → protocol_distribution.png")


def _plot_throughput_ts(agg, plot_dir):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             facecolor=PALETTE["bg"], sharex=True)
    fig.suptitle("Traffic Metrics Time Series", fontsize=15,
                 color=PALETTE["accent1"], fontweight="bold")

    metrics = [
        ("throughput_mbps", "Throughput (Mbps)",   PALETTE["accent1"]),
        ("packet_count",    "Packet Count (pkt/s)", PALETTE["accent4"]),
        ("link_util_pct",   "Link Utilization (%)", PALETTE["accent3"]),
    ]

    x = range(len(agg))
    for ax, (col, label, color) in zip(axes, metrics):
        ax.set_facecolor(PALETTE["card"])
        ax.plot(x, agg[col], color=color, linewidth=1.5)
        ax.fill_between(x, agg[col], alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True)
        # Moving avg
        roll = agg[col].rolling(5, min_periods=1).mean()
        ax.plot(x, roll, color="white", linewidth=1,
                linestyle="--", alpha=0.6, label="5-window avg")
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Window Index", fontsize=10)
    plt.tight_layout()

    path = os.path.join(plot_dir, "traffic_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"      → traffic_timeseries.png")

#!/usr/bin/env python3
"""Plot benchmark results for ampholyte zwitterion fraction predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_CSV = (
    SCRIPT_DIR / "results" / "f_zwit_benchmark" / "benchmark_results.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "f_zwit_benchmark"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize ampholyte zwitterion benchmark results."
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=DEFAULT_BENCHMARK_CSV,
        help="Aggregate benchmark CSV produced by benchmark_fzwit.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saved figures.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    benchmark_csv = args.benchmark_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(benchmark_csv)

    fig, ax = plt.subplots(figsize=(13, 6))
    scatter = ax.scatter(
        df["experimental_f_zwit"],
        df["predicted_f_zwit"],
        c=pd.Categorical(df["source"]).codes,
        cmap="tab10",
        label=df["source"],
    )

    df = df[df["dtype"] != "COSMO-RS"]

    xmin = -0.1
    xmax = 1.1
    ax.plot([xmin, xmax], [xmin, xmax], "k--", lw=1)
    ax.set_xlabel("Experimental $f_{zwit}$")
    ax.set_ylabel("Predicted $f_{zwit}$")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    handles, _ = scatter.legend_elements(prop="colors")
    labels = pd.Categorical(df["source"]).categories
    ax.legend(handles, labels, title="Source", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(output_dir / "f_zwit_scatter.png", dpi=300, bbox_inches="tight")

    fz_exp = df["experimental_f_zwit"].to_numpy()
    fz_pred = df["predicted_f_zwit"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        kz_exp = np.log10(fz_exp / (1 - fz_exp))
        kz_pred = np.log10(fz_pred / (1 - fz_pred))
    valid = np.isfinite(kz_exp) & np.isfinite(kz_pred)
    kz_exp_valid = kz_exp[valid]
    kz_pred_valid = kz_pred[valid]
    df_valid = df.loc[valid].reset_index(drop=True)

    log10_kz_equal = 0.0
    uncharged_region = (kz_exp_valid < log10_kz_equal) & (kz_pred_valid < log10_kz_equal)
    zwitterion_region = (kz_exp_valid > log10_kz_equal) & (kz_pred_valid > log10_kz_equal)
    outside_region = ~(uncharged_region | zwitterion_region)

    print(f"Points outside highlighted quadrants: {outside_region.sum()} / {len(kz_exp_valid)}")
    for idx in np.where(outside_region)[0]:
        molecule = df_valid.iloc[idx]["molecule"] if "molecule" in df_valid.columns else idx
        print(
            f"  {molecule}: "
            f"exp log10(K_zwit)={kz_exp_valid[idx]:.4f}, "
            f"pred log10(K_zwit)={kz_pred_valid[idx]:.4f}"
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    xmin_k = np.min((np.min(kz_exp_valid - 1), np.min(kz_pred_valid - 1)))
    xmax_k = np.max((np.max(kz_exp_valid + 1), np.max(kz_pred_valid + 1)))
    ax.set_xlim(xmin_k, xmax_k)
    ax.set_ylim(xmin_k, xmax_k)

    ax.add_patch(
        plt.Rectangle(
            (xmin_k, xmin_k),
            log10_kz_equal - xmin_k,
            log10_kz_equal - xmin_k,
            facecolor="#fd8d3c",
            alpha=0.18,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (log10_kz_equal, log10_kz_equal),
            xmax_k - log10_kz_equal,
            xmax_k - log10_kz_equal,
            facecolor="#fd8d3c",
            alpha=0.18,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(log10_kz_equal, color="gray", lw=0.8, alpha=0.5, zorder=1)
    ax.axhline(log10_kz_equal, color="gray", lw=0.8, alpha=0.5, zorder=1)

    ax.scatter(kz_exp_valid, kz_pred_valid, c="red", label="Kz", zorder=3)
    ax.set_xlabel(r"Experimental $\log_{10}$ $K_{\mathrm{zwit}}$", fontsize=20)
    ax.set_ylabel(r"Predicted $\log_{10}$ $K_{\mathrm{zwit}}$", fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.plot([xmin_k, xmax_k], [xmin_k, xmax_k], "k--", lw=1, zorder=2)
    ax.plot([xmin_k, xmax_k], [xmin_k + 1, xmax_k + 1], "k--", lw=1, alpha=0.5, zorder=2)
    ax.plot([xmin_k, xmax_k], [xmin_k - 1, xmax_k - 1], "k--", lw=1, alpha=0.5, zorder=2)
    ax.plot([xmin_k, xmax_k], [xmin_k + 2, xmax_k + 2], "k--", lw=1, alpha=0.2, zorder=2)
    ax.plot([xmin_k, xmax_k], [xmin_k - 2, xmax_k - 2], "k--", lw=1, alpha=0.2, zorder=2)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    n = len(kz_exp_valid)
    mae = np.mean(np.abs(kz_pred_valid - kz_exp_valid))
    rmse = np.sqrt(np.mean((kz_pred_valid - kz_exp_valid) ** 2))
    stats_text = f"N = {n}\nMAE = {mae:.2f}\nRMSE = {rmse:.2f}"
    ax.text(
        0.97,
        0.03,
        stats_text,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "kz_scatter_log10.svg", bbox_inches="tight", transparent=True)
    plt.savefig(output_dir / "kz_scatter_log10.png", dpi=300, bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()

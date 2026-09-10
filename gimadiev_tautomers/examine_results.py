#!/usr/bin/env python3
"""Plot Gimadiev tautomer-equilibrium benchmark results as log10(K_T)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_tautomers import DATASETS
from parse_rdf import canonical_dataset_name


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Gimadiev tautomer benchmark results as log10(K_T), "
            "restricted to equilibria whose reactant and product both appear "
            "in PEACE tautomer enumeration coverage."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="extracted",
        help=(
            "Named tautomer set: main/extracted (MOESM4), test_set_1 (MOESM2), "
            "or test_set_2 (MOESM3)."
        ),
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Aggregate benchmark CSV produced by benchmark_tautomers.py.",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=None,
        help=(
            "Enumeration coverage table; analysis is restricted to rows where "
            "both reactant and product were found in the enumerated pool."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--no-coverage-filter",
        action="store_true",
        help="Plot every scored row, even if enumeration did not find the product.",
    )
    return parser


def _gaussian_kde_density(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate a 2-D Gaussian KDE (Scott bandwidth) at the sample points."""
    pts = np.column_stack([x, y])
    n_pts, n_dim = pts.shape
    cov = np.cov(pts, rowvar=False)
    bandwidth = n_pts ** (-1.0 / (n_dim + 4))
    cov_bw = cov * (bandwidth**2)
    inv_cov = np.linalg.inv(cov_bw)
    norm = n_pts * (2.0 * np.pi) * np.sqrt(np.linalg.det(cov_bw))
    delta = pts[:, None, :] - pts[None, :, :]
    mahalanobis = np.einsum("...i,ij,...j->...", delta, inv_cov, delta)
    return np.exp(-0.5 * mahalanobis).sum(axis=1) / norm


def main() -> None:
    args = _build_parser().parse_args()
    spec = DATASETS[args.dataset]
    benchmark_csv = (args.benchmark_csv or (Path(spec["results_root"]) / "benchmark_tautomers_results.csv")).resolve()
    coverage_csv = (args.coverage_csv or spec["coverage_csv"]).resolve()
    output_dir = (args.output_dir or spec["results_root"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_stem = spec.get("plot_stem", f"{args.dataset}_kt_scatter_log10")

    df = pd.read_csv(benchmark_csv)
    if not args.no_coverage_filter:
        if not coverage_csv.exists():
            raise FileNotFoundError(
                f"Coverage CSV not found at {coverage_csv}. "
                f"Run check_tautomer_enumeration.py --dataset {args.dataset} "
                "or pass --no-coverage-filter."
            )
        coverage = pd.read_csv(coverage_csv)
        required = {"reaction_index", "reactant_in_pool", "product_in_pool"}
        missing = required - set(coverage.columns)
        if missing:
            raise ValueError(f"{coverage_csv} is missing columns: {sorted(missing)}")
        covered = coverage.loc[
            coverage["reactant_in_pool"].astype(bool)
            & coverage["product_in_pool"].astype(bool),
            "reaction_index",
        ].astype(int)
        n_before = len(df)
        df = df[df["reaction_index"].astype(int).isin(set(covered))].copy()
        print(
            f"Restricted to enumerated reactant+product pairs: "
            f"{len(df)} / {n_before} benchmark rows "
            f"({covered.nunique()} coverage indices)."
        )
        if canonical_dataset_name(args.dataset) == "main" and len(df) != 191:
            print(f"Warning: expected 191 main/extracted entries after coverage filter, found {len(df)}.")

    if "tabulated_constant" not in df.columns:
        raise ValueError(f"{benchmark_csv} is missing column 'tabulated_constant'.")
    kt_exp = pd.to_numeric(df["tabulated_constant"], errors="coerce").to_numpy()
    kt_pred = pd.to_numeric(df["predicted_log10Kz"], errors="coerce").to_numpy()
    valid = np.isfinite(kt_exp) & np.isfinite(kt_pred)
    kt_exp_valid = kt_exp[valid]
    kt_pred_valid = kt_pred[valid]
    df_valid = df.loc[valid].reset_index(drop=True)
    if len(kt_exp_valid) == 0:
        n_exp = int(np.isfinite(kt_exp).sum())
        n_pred = int(np.isfinite(kt_pred).sum())
        raise ValueError(
            f"No rows with both experimental tabulated_constant and predicted_log10Kz "
            f"({n_exp} experimental, {n_pred} predicted). "
            "test_set_1 RDF records do not include tabulated constants."
        )

    log10_kt_equal = 0.0
    reactant_region = (kt_exp_valid < log10_kt_equal) & (kt_pred_valid < log10_kt_equal)
    product_region = (kt_exp_valid > log10_kt_equal) & (kt_pred_valid > log10_kt_equal)
    outside_region = ~(reactant_region | product_region)

    print(f"Points outside highlighted quadrants: {outside_region.sum()} / {len(kt_exp_valid)}")
    for idx in np.where(outside_region)[0]:
        row = df_valid.iloc[idx]
        print(
            f"  rxn {int(row['reaction_index'])} ({row['solvent']}): "
            f"exp log10(K_T)={kt_exp_valid[idx]:.4f}, "
            f"pred log10(K_T)={kt_pred_valid[idx]:.4f}"
        )

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    xmin_k = np.min((np.min(kt_exp_valid - 1), np.min(kt_pred_valid - 1)))
    xmax_k = np.max((np.max(kt_exp_valid + 1), np.max(kt_pred_valid + 1)))
    ax.set_xlim(xmin_k, xmax_k)
    ax.set_ylim(xmin_k, xmax_k)

    ax.add_patch(
        plt.Rectangle(
            (xmin_k, xmin_k),
            log10_kt_equal - xmin_k,
            log10_kt_equal - xmin_k,
            facecolor="#fd8d3c",
            alpha=0.18,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (log10_kt_equal, log10_kt_equal),
            xmax_k - log10_kt_equal,
            xmax_k - log10_kt_equal,
            facecolor="#fd8d3c",
            alpha=0.18,
            edgecolor="none",
            zorder=0,
        )
    )
    ax.axvline(log10_kt_equal, color="gray", lw=0.8, alpha=0.5, zorder=1)
    ax.axhline(log10_kt_equal, color="gray", lw=0.8, alpha=0.5, zorder=1)

    density = _gaussian_kde_density(kt_exp_valid, kt_pred_valid)
    order = np.argsort(density)
    ax.scatter(
        kt_exp_valid[order],
        kt_pred_valid[order],
        c=density[order],
        cmap="winter",
        zorder=3,
    )
    ax.set_xlabel(r"Experimental $\log_{10}$ $K_{\mathrm{T}}$", fontsize=20)
    ax.set_ylabel(r"Predicted $\log_{10}$ $K_{\mathrm{T}}$", fontsize=20)
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

    n = len(kt_exp_valid)
    mae = np.mean(np.abs(kt_pred_valid - kt_exp_valid))
    rmse = np.sqrt(np.mean((kt_pred_valid - kt_exp_valid) ** 2))
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
    plt.savefig(output_dir / f"{plot_stem}.svg", bbox_inches="tight", transparent=True)
    plt.savefig(output_dir / f"{plot_stem}.png", dpi=300, bbox_inches="tight", transparent=True)
    print(f"Wrote {output_dir / plot_stem}.png")


if __name__ == "__main__":
    main()

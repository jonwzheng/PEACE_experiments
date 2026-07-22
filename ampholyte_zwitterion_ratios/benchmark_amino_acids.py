#!/usr/bin/env python3
"""Benchmark zwitterion fractions for amino-acid test sets in aqueous and non-aqueous solvents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.benchmark_common import (
    add_benchmark_runtime_args,
    build_peace_command,
    resolve_peace_root,
    run_peace_job,
    slugify,
)

DEFAULT_AQ_CSV = SCRIPT_DIR / "data" / "amino_acids_aq_test.csv"
DEFAULT_NONAQ_CSV = SCRIPT_DIR / "data" / "amino_acids_nonaq_test.csv"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "amino_acids"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark predicted zwitterion fractions for amino-acid test sets "
            "in ampholyte_zwitterion_ratios/data/."
        )
    )
    parser.add_argument(
        "--aq-csv",
        type=Path,
        default=DEFAULT_AQ_CSV,
        help="Aqueous amino-acid input CSV (SMILES, solvent).",
    )
    parser.add_argument(
        "--nonaq-csv",
        type=Path,
        default=DEFAULT_NONAQ_CSV,
        help="Non-aqueous amino-acid input CSV (SMILES, solvent, optional f_zwit).",
    )
    add_benchmark_runtime_args(parser, default_results_root=DEFAULT_RESULTS_ROOT)
    return parser


def _benchmark_dataset(
    *,
    dataset_name: str,
    input_csv: Path,
    results_root: Path,
    peace_root: Path,
    main_extra_args: list[str],
) -> pd.DataFrame:
    data = pd.read_csv(input_csv)
    required = {"molecule", "SMILES", "solvent"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    dataset_root = results_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx, row in data.iterrows():
        name = str(row["molecule"])
        smiles = str(row["SMILES"])
        solvent = str(row["solvent"])
        experimental = row["f_zwit"] if "f_zwit" in data.columns else pd.NA
        if pd.notna(experimental):
            experimental = float(experimental)

        mol_dir = dataset_root / f"{idx:03d}_{slugify(name)}"
        output_csv = mol_dir / "results.csv"

        cmd = build_peace_command(
            smiles=smiles,
            solvent=solvent,
            scratch_root=mol_dir / "solvation",
            output_csv=output_csv,
            extra_args=main_extra_args,
        )

        print(f"[{dataset_name} {idx + 1}/{len(data)}] Running: {name} ({solvent})")
        outcome = run_peace_job(
            peace_root=peace_root,
            cmd=cmd,
            mol_dir=mol_dir,
            output_csv=output_csv,
        )

        rows.append(
            {
                "molecule": name,
                "SMILES": smiles,
                "solvent": solvent,
                "f_zwit_exp": experimental,
                "f_zwit_pred": outcome.predicted_f_zwit,
                "job_messages": outcome.job_messages,
                "run_ok": outcome.returncode == 0,
                "run_returncode": outcome.returncode,
                "result_dir": str(mol_dir),
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = results_root / f"amino_acids_{dataset_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved {dataset_name} summary to: {summary_path}")
    return summary


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()

    results_root = args.results_root.resolve()
    peace_root = resolve_peace_root(args.peace_root)
    results_root.mkdir(parents=True, exist_ok=True)

    _benchmark_dataset(
        dataset_name="aq",
        input_csv=args.aq_csv.resolve(),
        results_root=results_root,
        peace_root=peace_root,
        main_extra_args=main_extra_args,
    )
    _benchmark_dataset(
        dataset_name="nonaq",
        input_csv=args.nonaq_csv.resolve(),
        results_root=results_root,
        peace_root=peace_root,
        main_extra_args=main_extra_args,
    )


if __name__ == "__main__":
    main()

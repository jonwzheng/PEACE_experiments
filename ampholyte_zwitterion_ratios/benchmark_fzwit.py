#!/usr/bin/env python3
"""Benchmark predicted zwitterion fractions against experimental tautomer-ratio data."""

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

DEFAULT_INPUT_CSV = SCRIPT_DIR / "data" / "realistic_mol_aq_test.csv"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "f_zwit_benchmark"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark predicted zwitterion fractions against tautomer-ratio test data "
            "(default: ampholyte_zwitterion_ratios/data/realistic_mol_aq_test.csv)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Input CSV with SMILES and experimental f_zwit.",
    )
    add_benchmark_runtime_args(parser, default_results_root=DEFAULT_RESULTS_ROOT)
    return parser


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()

    input_csv = args.input_csv.resolve()
    results_root = args.results_root.resolve()
    peace_root = resolve_peace_root(args.peace_root)
    results_root.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    if "SMILES" not in data.columns or "f_zwit" not in data.columns:
        raise ValueError("Input CSV must contain 'SMILES' and 'f_zwit' columns.")

    rows: list[dict] = []
    for idx, row in data.iterrows():
        name = str(row.get("molecule", f"row_{idx}"))
        smiles = str(row["SMILES"])
        experimental = float(row["f_zwit"])
        source = str(row.get("source", ""))
        dtype = str(row.get("type", ""))
        solvent = str(row.get("solvent", "water"))
        mol_dir = results_root / f"{idx:03d}_{slugify(name)}"
        output_csv = mol_dir / "results.csv"

        cmd = build_peace_command(
            smiles=smiles,
            solvent=solvent,
            scratch_root=mol_dir / "solvation",
            output_csv=output_csv,
            extra_args=main_extra_args,
        )

        print(f"[{idx + 1}/{len(data)}] Running: {name}")
        outcome = run_peace_job(
            peace_root=peace_root,
            cmd=cmd,
            mol_dir=mol_dir,
            output_csv=output_csv,
        )

        predicted = outcome.predicted_f_zwit
        abs_error = abs(predicted - experimental) if predicted is not None else None

        rows.append(
            {
                "molecule": name,
                "SMILES": smiles,
                "solvent": solvent,
                "source": source,
                "dtype": dtype,
                "experimental_f_zwit": experimental,
                "predicted_f_zwit": predicted,
                "abs_error": abs_error,
                "run_ok": outcome.returncode == 0,
                "run_returncode": outcome.returncode,
                "job_messages": outcome.job_messages,
                "result_dir": str(mol_dir),
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = results_root / "benchmark_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved benchmark summary to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark PEACE dominant tautomers against SAMPL6 observed microstates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.benchmark_common import (
    add_benchmark_runtime_args,
    build_peace_command,
    canon_smiles,
    extract_dominant_by_charge,
    resolve_peace_root,
    run_peace_job,
    slugify,
)

DEFAULT_INPUT_CSV = SCRIPT_DIR / "observed_microstates.csv"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "microstate_benchmark"

MOL_ID_COL = "SAMPL6 Molecule ID"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PEACE dominant tautomers against SAMPL6 observed microstates "
            "(default: observed_microstates.csv). Runs one PEACE job per molecule using "
            "the charge-0 canonical SMILES only as the enumeration seed, with "
            "--charge-min/--charge-max spanning the observed charge states. Every observed "
            "charge — including +0 — is scored by comparing PEACE's dominant tautomer to "
            "rdkit_canon_smi (the seed is not assumed to remain dominant after tautomer search)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Observed microstates CSV (molecule ID, rdkit_canon_smi, charge).",
    )
    parser.add_argument(
        "--molecule",
        action="append",
        dest="molecules",
        default=None,
        help="Optional molecule ID filter (repeatable), e.g. --molecule SM07.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help=(
            "Do not re-run PEACE; rebuild the summary from existing per-molecule "
            "results.csv files under --results-root."
        ),
    )
    add_benchmark_runtime_args(parser, default_results_root=DEFAULT_RESULTS_ROOT)
    return parser


def _apply_predictions(
    data: pd.DataFrame,
    *,
    mol_rows: pd.DataFrame,
    input_smiles: str,
    charge_min: int,
    charge_max: int,
    dominant_by_charge: dict,
    run_ok: bool,
    run_returncode: int | None,
    job_messages: str | None,
    mol_dir: Path,
) -> None:
    for idx in mol_rows.index:
        charge = int(data.at[idx, "charge"])
        observed = canon_smiles(data.at[idx, "rdkit_canon_smi"])
        dominant = dominant_by_charge.get(charge)

        data.at[idx, "input_smiles"] = input_smiles
        data.at[idx, "charge_min"] = charge_min
        data.at[idx, "charge_max"] = charge_max
        data.at[idx, "run_ok"] = run_ok
        data.at[idx, "run_returncode"] = run_returncode
        data.at[idx, "job_messages"] = job_messages
        data.at[idx, "result_dir"] = str(mol_dir)

        if dominant is None:
            data.at[idx, "peace_match"] = False
            print(f"  charge {charge:+d}: no dominant tautomer found [MISMATCH]")
            continue

        data.at[idx, "peace_n_tautomers"] = dominant.n_tautomers

        if dominant.smiles is None or dominant.fraction is None:
            data.at[idx, "peace_match"] = False
            print(
                f"  charge {charge:+d}: n_tautomers={dominant.n_tautomers} "
                f"no dominant tautomer found [MISMATCH]"
            )
            continue

        data.at[idx, "peace_dominant_smiles"] = dominant.smiles
        data.at[idx, "peace_dominant_fraction"] = dominant.fraction
        data.at[idx, "peace_dominant_fraction_pct"] = 100.0 * dominant.fraction
        data.at[idx, "peace_match"] = (
            observed is not None and dominant.smiles == observed
        )

        match_label = "MATCH" if data.at[idx, "peace_match"] else "MISMATCH"
        print(
            f"  charge {charge:+d}: n_tautomers={dominant.n_tautomers} "
            f"pred={dominant.smiles} "
            f"f={dominant.fraction:.4f} ({100.0 * dominant.fraction:.2f}%) "
            f"obs={observed} [{match_label}]"
        )


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()

    input_csv = args.input_csv.resolve()
    results_root = args.results_root.resolve()
    peace_root = None if args.rescore_only else resolve_peace_root(args.peace_root)
    results_root.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    required = {MOL_ID_COL, "rdkit_canon_smi", "charge"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    data = data.copy()
    data["charge"] = data["charge"].astype(int)

    if args.molecules:
        wanted = {m.strip().upper() for m in args.molecules}
        data = data[data[MOL_ID_COL].astype(str).str.upper().isin(wanted)].copy()
        if data.empty:
            raise ValueError(f"No rows matched --molecule filters: {sorted(wanted)}")

    # Preserve original row order for an output of the same size/order as the filtered input.
    data["_row_order"] = range(len(data))

    prediction_cols = {
        "input_smiles": pd.NA,
        "charge_min": pd.NA,
        "charge_max": pd.NA,
        "peace_n_tautomers": pd.NA,
        "peace_dominant_smiles": pd.NA,
        "peace_dominant_fraction": pd.NA,
        "peace_dominant_fraction_pct": pd.NA,
        "peace_match": pd.NA,
        "run_ok": False,
        "run_returncode": pd.NA,
        "job_messages": pd.NA,
        "result_dir": pd.NA,
    }
    for col, default in prediction_cols.items():
        data[col] = default

    molecule_ids = list(dict.fromkeys(data[MOL_ID_COL].astype(str)))
    for mol_i, mol_id in enumerate(molecule_ids, start=1):
        mol_rows = data[data[MOL_ID_COL].astype(str) == mol_id]
        charges = sorted(int(c) for c in mol_rows["charge"].unique())
        charge_min = min(charges)
        charge_max = max(charges)

        # Charge-0 observed SMILES seeds PEACE only; the +0 dominant tautomer is still
        # extracted from results and matched below (seed need not remain major).
        zero_rows = mol_rows[mol_rows["charge"] == 0]
        if zero_rows.empty:
            raise ValueError(f"{mol_id}: no charge-0 row found to use as PEACE input SMILES.")
        input_smiles = str(zero_rows.iloc[0]["rdkit_canon_smi"])

        mol_dir = results_root / f"{mol_i:03d}_{slugify(mol_id)}"
        output_csv = mol_dir / "results.csv"

        print(
            f"[{mol_i}/{len(molecule_ids)}] {mol_id}: "
            f"seed={input_smiles} charge=[{charge_min}, {charge_max}]"
        )

        if args.rescore_only:
            if not output_csv.exists():
                raise FileNotFoundError(
                    f"--rescore-only requires existing results at {output_csv}"
                )
            dominant_by_charge = extract_dominant_by_charge(output_csv)
            _apply_predictions(
                data,
                mol_rows=mol_rows,
                input_smiles=input_smiles,
                charge_min=charge_min,
                charge_max=charge_max,
                dominant_by_charge=dominant_by_charge,
                run_ok=True,
                run_returncode=0,
                job_messages=None,
                mol_dir=mol_dir,
            )
            continue

        cmd = build_peace_command(
            smiles=input_smiles,
            scratch_root=mol_dir / "solvation",
            output_csv=output_csv,
            charge_min=charge_min,
            charge_max=charge_max,
            extra_args=main_extra_args,
        )
        outcome = run_peace_job(
            peace_root=peace_root,
            cmd=cmd,
            mol_dir=mol_dir,
            output_csv=output_csv,
            compute_f_zwit=False,
        )

        dominant_by_charge = {}
        if outcome.returncode == 0 and output_csv.exists():
            dominant_by_charge = extract_dominant_by_charge(output_csv)

        _apply_predictions(
            data,
            mol_rows=mol_rows,
            input_smiles=input_smiles,
            charge_min=charge_min,
            charge_max=charge_max,
            dominant_by_charge=dominant_by_charge,
            run_ok=outcome.returncode == 0,
            run_returncode=outcome.returncode,
            job_messages=outcome.job_messages,
            mol_dir=mol_dir,
        )

    out_df = data.sort_values("_row_order").drop(columns=["_row_order"])
    out_path = results_root / "benchmark_microstates_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved benchmark summary to: {out_path}")

    n_match = int(out_df["peace_match"].fillna(False).sum())
    n_filled = int(out_df["peace_dominant_smiles"].notna().sum())
    print(f"Filled predictions: {n_filled}/{len(out_df)}; matches: {n_match}/{len(out_df)}")


if __name__ == "__main__":
    main()

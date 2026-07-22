#!/usr/bin/env python3
"""Benchmark PEACE protomer rankings against SAMPL7 microstate free energies."""

from __future__ import annotations

import argparse
import ast
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
    resolve_peace_root,
    run_peace_job,
    slugify,
)

DEFAULT_INPUT_CSV = SCRIPT_DIR / "SAMPL7_molecule_ID_and_SMILES_downsampled.csv"
DEFAULT_NUMBERS_CSV = SCRIPT_DIR / "numbers.csv"
DEFAULT_MICROSTATES_DIR = SCRIPT_DIR / "microstates"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "microstate_benchmark"

MOL_ID_COL = "SAMPL7 Molecule ID"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PEACE protomer rankings / relative free energies against "
            "SAMPL7 labeled microstates (default: "
            "SAMPL7_molecule_ID_and_SMILES_downsampled.csv). Runs one PEACE job "
            "per molecule using charge_range for --charge-min/--charge-max, then "
            "compares ranked protomers at comparison_charges to SAMPL7 average FE "
            "predictions. Charge-0 ΔGs are kept relative to micro000 (=0); other "
            "charges are re-zeroed within the charge state."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Molecule ID / SMILES / comparison_charges / charge_range CSV.",
    )
    parser.add_argument(
        "--numbers-csv",
        type=Path,
        default=DEFAULT_NUMBERS_CSV,
        help="SAMPL7 numbers.csv with 'ID tag' and 'average FE prediction'.",
    )
    parser.add_argument(
        "--microstates-dir",
        type=Path,
        default=DEFAULT_MICROSTATES_DIR,
        help="Directory containing {ID}_microstates_labeled.csv files.",
    )
    parser.add_argument(
        "--molecule",
        action="append",
        dest="molecules",
        default=None,
        help="Optional molecule ID filter (repeatable), e.g. --molecule SM25.",
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


def _parse_int_list(value: object) -> list[int]:
    """Parse a CSV cell like '[0,1]', '[-1,0]', or '[0]' into a list of ints."""
    if isinstance(value, list):
        return [int(v) for v in value]
    text = str(value).strip()
    parsed = ast.literal_eval(text)
    if isinstance(parsed, int):
        return [parsed]
    if not isinstance(parsed, (list, tuple)) or not parsed:
        raise ValueError(f"Expected a non-empty int list, got {value!r}")
    return [int(v) for v in parsed]


def _charge_bounds(charge_range: object) -> tuple[int, int]:
    charges = _parse_int_list(charge_range)
    return min(charges), max(charges)


def _load_sampl7_fe(numbers_csv: Path) -> pd.Series:
    """Map microstate ID tag -> average FE prediction (kcal/mol)."""
    numbers = pd.read_csv(numbers_csv)
    required = {"ID tag", "average FE prediction"}
    missing = required - set(numbers.columns)
    if missing:
        raise ValueError(f"{numbers_csv} is missing required columns: {sorted(missing)}")
    fe = pd.to_numeric(numbers["average FE prediction"], errors="coerce")
    return pd.Series(fe.to_numpy(), index=numbers["ID tag"].astype(str), name="sampl7_fe")


def _is_canonical_micro(micro_id: str, mol_id: str) -> bool:
    return micro_id == f"{mol_id}_micro000" or micro_id.endswith("_micro000")


def _rank_by_relative_energy(
    df: pd.DataFrame,
    energy_col: str,
    *,
    shift_to_min: bool = True,
) -> pd.DataFrame:
    """Add relative energy and 1-based rank within the frame.

    When ``shift_to_min`` is True (default), energies are re-zeroed so the
    lowest value is 0. When False, ``energy_col`` is treated as already
    relative (e.g. SAMPL7 charge-0 ΔG vs micro000) and used as-is.
    """
    out = df.copy()
    energies = pd.to_numeric(out[energy_col], errors="coerce")
    out["_energy"] = energies
    valid = out["_energy"].notna()
    out["relative_energy_kcal_mol"] = pd.NA
    out["rank"] = pd.NA
    if not valid.any():
        return out.drop(columns=["_energy"])

    subset = out.loc[valid].sort_values(
        ["_energy", "canon_smiles"], ascending=[True, True], kind="mergesort"
    )
    if shift_to_min:
        e_ref = float(subset["_energy"].min())
        out.loc[subset.index, "relative_energy_kcal_mol"] = subset["_energy"] - e_ref
    else:
        out.loc[subset.index, "relative_energy_kcal_mol"] = subset["_energy"]
    out.loc[subset.index, "rank"] = range(1, len(subset) + 1)
    return out.drop(columns=["_energy"])


def _load_sampl7_microstates_for_molecule(
    *,
    mol_id: str,
    microstates_dir: Path,
    comparison_charges: set[int],
    fe_by_id: pd.Series,
) -> pd.DataFrame:
    path = microstates_dir / f"{mol_id}_microstates_labeled.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing labeled microstates file: {path}")

    labeled = pd.read_csv(path)
    required = {"microstate ID", "canonical isomeric SMILES", "charge"}
    missing = required - set(labeled.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in labeled.iterrows():
        charge = int(row["charge"])
        if charge not in comparison_charges:
            continue
        micro_id = str(row["microstate ID"])
        smiles = canon_smiles(row["canonical isomeric SMILES"])
        if smiles is None:
            raise ValueError(f"{micro_id}: could not canonicalize SMILES")
        fe = fe_by_id.get(micro_id)
        # Charge-0 ΔGs are relative to the canonical micro000 tautomer (always 0).
        if charge == 0 and _is_canonical_micro(micro_id, mol_id):
            fe_value: float | object = 0.0
        elif fe is not None and pd.notna(fe):
            fe_value = float(fe)
        else:
            fe_value = pd.NA
        rows.append(
            {
                "molecule_id": mol_id,
                "charge": charge,
                "canon_smiles": smiles,
                "sampl7_microstate_id": micro_id,
                "sampl7_average_fe_prediction": fe_value,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "molecule_id",
                "charge",
                "canon_smiles",
                "sampl7_microstate_id",
                "sampl7_rank",
                "sampl7_delta_g_kcal_mol",
                "sampl7_average_fe_prediction",
            ]
        )

    df = pd.DataFrame(rows)
    ranked_parts: list[pd.DataFrame] = []
    for charge, group in df.groupby("charge", sort=True):
        # Charge 0: keep ΔG vs micro000 (do not re-zero to the lowest tautomer).
        # Other charges: re-zero within the charge state so the most stable is 0.
        ranked = _rank_by_relative_energy(
            group,
            "sampl7_average_fe_prediction",
            shift_to_min=(int(charge) != 0),
        )
        ranked = ranked.rename(
            columns={
                "rank": "sampl7_rank",
                "relative_energy_kcal_mol": "sampl7_delta_g_kcal_mol",
            }
        )
        ranked_parts.append(ranked)
    return pd.concat(ranked_parts, ignore_index=True)


def _extract_peace_protomers(
    results_csv: Path,
    *,
    mol_id: str,
    comparison_charges: set[int],
) -> pd.DataFrame:
    """Rank PEACE protomers within each comparison charge (relative ΔG, min=0)."""
    df = pd.read_csv(results_csv)
    required = {"formal_charge", "protomer_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {results_csv}; missing {sorted(missing)}"
        )

    energy_col = None
    for candidate in (
        "solution_phase_free_energy_kcal_mol",
        "delta_g_kcal_mol",
    ):
        if candidate in df.columns:
            energy_col = candidate
            break
    if energy_col is None:
        raise ValueError(
            f"{results_csv} has neither solution_phase_free_energy_kcal_mol nor "
            "delta_g_kcal_mol"
        )

    rows: list[dict] = []
    for _, row in df.iterrows():
        charge = int(row["formal_charge"])
        if charge not in comparison_charges:
            continue
        smiles = canon_smiles(row["protomer_smiles"])
        if smiles is None:
            smiles = str(row["protomer_smiles"])
        energy = pd.to_numeric(row[energy_col], errors="coerce")
        rows.append(
            {
                "molecule_id": mol_id,
                "charge": charge,
                "canon_smiles": smiles,
                "peace_energy_kcal_mol": float(energy) if pd.notna(energy) else pd.NA,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "molecule_id",
                "charge",
                "canon_smiles",
                "peace_rank",
                "peace_delta_g_kcal_mol",
            ]
        )

    peace = pd.DataFrame(rows)
    # Deduplicate identical canonical SMILES within a charge (keep lowest energy).
    peace = (
        peace.sort_values(
            ["charge", "peace_energy_kcal_mol", "canon_smiles"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        .groupby(["molecule_id", "charge", "canon_smiles"], as_index=False, sort=False)
        .first()
    )

    ranked_parts: list[pd.DataFrame] = []
    for charge, group in peace.groupby("charge", sort=True):
        ranked = _rank_by_relative_energy(group, "peace_energy_kcal_mol")
        ranked = ranked.rename(
            columns={
                "rank": "peace_rank",
                "relative_energy_kcal_mol": "peace_delta_g_kcal_mol",
            }
        )
        ranked_parts.append(ranked.drop(columns=["peace_energy_kcal_mol"]))
    return pd.concat(ranked_parts, ignore_index=True)


def _merge_peace_and_sampl7(
    *,
    peace: pd.DataFrame,
    sampl7: pd.DataFrame,
    mol_id: str,
    input_smiles: str,
    charge_min: int,
    charge_max: int,
    comparison_charges: list[int],
    run_ok: bool,
    run_returncode: int | None,
    job_messages: str | None,
    mol_dir: Path,
) -> pd.DataFrame:
    keys = ["molecule_id", "charge", "canon_smiles"]
    if peace.empty and sampl7.empty:
        merged = pd.DataFrame(
            columns=[
                *keys,
                "peace_rank",
                "peace_delta_g_kcal_mol",
                "sampl7_microstate_id",
                "sampl7_rank",
                "sampl7_delta_g_kcal_mol",
                "sampl7_average_fe_prediction",
            ]
        )
    elif peace.empty:
        merged = sampl7.copy()
        merged["peace_rank"] = pd.NA
        merged["peace_delta_g_kcal_mol"] = pd.NA
    elif sampl7.empty:
        merged = peace.copy()
        merged["sampl7_microstate_id"] = pd.NA
        merged["sampl7_rank"] = pd.NA
        merged["sampl7_delta_g_kcal_mol"] = pd.NA
        merged["sampl7_average_fe_prediction"] = pd.NA
    else:
        merged = peace.merge(sampl7, on=keys, how="outer")

    merged["molecule_id"] = merged.get("molecule_id", mol_id).fillna(mol_id)
    merged["input_smiles"] = input_smiles
    merged["charge_min"] = charge_min
    merged["charge_max"] = charge_max
    merged["comparison_charges"] = str(comparison_charges)
    merged["in_peace"] = merged["peace_rank"].notna()
    merged["in_sampl7"] = merged["sampl7_microstate_id"].notna()
    merged["run_ok"] = run_ok
    merged["run_returncode"] = run_returncode
    merged["job_messages"] = job_messages
    merged["result_dir"] = str(mol_dir)

    column_order = [
        "molecule_id",
        "charge",
        "canon_smiles",
        "peace_rank",
        "peace_delta_g_kcal_mol",
        "sampl7_microstate_id",
        "sampl7_rank",
        "sampl7_delta_g_kcal_mol",
        "sampl7_average_fe_prediction",
        "in_peace",
        "in_sampl7",
        "input_smiles",
        "charge_min",
        "charge_max",
        "comparison_charges",
        "run_ok",
        "run_returncode",
        "job_messages",
        "result_dir",
    ]
    return merged.reindex(columns=column_order).sort_values(
        ["molecule_id", "charge", "peace_rank", "sampl7_rank", "canon_smiles"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    )


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()

    input_csv = args.input_csv.resolve()
    numbers_csv = args.numbers_csv.resolve()
    microstates_dir = args.microstates_dir.resolve()
    results_root = args.results_root.resolve()
    peace_root = None if args.rescore_only else resolve_peace_root(args.peace_root)
    results_root.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(input_csv)
    required = {MOL_ID_COL, "SMILES", "comparison_charges", "charge_range"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

    if args.molecules:
        wanted = {m.strip().upper() for m in args.molecules}
        data = data[data[MOL_ID_COL].astype(str).str.upper().isin(wanted)].copy()
        if data.empty:
            raise ValueError(f"No rows matched --molecule filters: {sorted(wanted)}")

    fe_by_id = _load_sampl7_fe(numbers_csv)
    summary_parts: list[pd.DataFrame] = []

    for mol_i, (_, mol_row) in enumerate(data.iterrows(), start=1):
        mol_id = str(mol_row[MOL_ID_COL])
        input_smiles = canon_smiles(mol_row["SMILES"])
        if input_smiles is None:
            raise ValueError(f"{mol_id}: could not canonicalize input SMILES")

        comparison_charges_list = _parse_int_list(mol_row["comparison_charges"])
        comparison_charges = set(comparison_charges_list)
        charge_min, charge_max = _charge_bounds(mol_row["charge_range"])

        mol_dir = results_root / f"{mol_i:03d}_{slugify(mol_id)}"
        output_csv = mol_dir / "results.csv"

        print(
            f"[{mol_i}/{len(data)}] {mol_id}: seed={input_smiles} "
            f"charge=[{charge_min}, {charge_max}] "
            f"compare={comparison_charges_list}"
        )

        sampl7 = _load_sampl7_microstates_for_molecule(
            mol_id=mol_id,
            microstates_dir=microstates_dir,
            comparison_charges=comparison_charges,
            fe_by_id=fe_by_id,
        )
        for _, srow in sampl7.iterrows():
            fe = srow["sampl7_average_fe_prediction"]
            rel = srow["sampl7_delta_g_kcal_mol"]
            fe_txt = f"{float(fe):.2f}" if pd.notna(fe) else "NA"
            rel_txt = f"{float(rel):.2f}" if pd.notna(rel) else "NA"
            print(
                f"  SAMPL7 q={int(srow['charge']):+d} "
                f"{srow['sampl7_microstate_id']}: "
                f"rank={srow['sampl7_rank']} "
                f"ΔG={rel_txt} (raw FE={fe_txt}) "
                f"smi={srow['canon_smiles']}"
            )

        peace = pd.DataFrame()
        run_ok = False
        run_returncode: int | None = None
        job_messages: str | None = None

        if args.rescore_only:
            if not output_csv.exists():
                raise FileNotFoundError(
                    f"--rescore-only requires existing results at {output_csv}"
                )
            peace = _extract_peace_protomers(
                output_csv, mol_id=mol_id, comparison_charges=comparison_charges
            )
            run_ok = True
            run_returncode = 0
        else:
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
            run_ok = outcome.returncode == 0
            run_returncode = outcome.returncode
            job_messages = outcome.job_messages
            if run_ok and output_csv.exists():
                peace = _extract_peace_protomers(
                    output_csv, mol_id=mol_id, comparison_charges=comparison_charges
                )

        for _, prow in peace.iterrows():
            print(
                f"  PEACE  q={int(prow['charge']):+d} "
                f"rank={prow['peace_rank']} "
                f"ΔG={float(prow['peace_delta_g_kcal_mol']):.2f} "
                f"smi={prow['canon_smiles']}"
            )

        merged = _merge_peace_and_sampl7(
            peace=peace,
            sampl7=sampl7,
            mol_id=mol_id,
            input_smiles=input_smiles,
            charge_min=charge_min,
            charge_max=charge_max,
            comparison_charges=comparison_charges_list,
            run_ok=run_ok,
            run_returncode=run_returncode,
            job_messages=job_messages,
            mol_dir=mol_dir,
        )
        summary_parts.append(merged)

        n_both = int((merged["in_peace"] & merged["in_sampl7"]).sum())
        n_peace_only = int((merged["in_peace"] & ~merged["in_sampl7"]).sum())
        n_sampl_only = int((~merged["in_peace"] & merged["in_sampl7"]).sum())
        print(
            f"  joined rows: {len(merged)} "
            f"(both={n_both}, peace_only={n_peace_only}, sampl7_only={n_sampl_only})"
        )

    out_df = pd.concat(summary_parts, ignore_index=True)
    out_path = results_root / "benchmark_microstates_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved benchmark summary to: {out_path}")

    n_both = int((out_df["in_peace"] & out_df["in_sampl7"]).sum())
    n_peace_only = int((out_df["in_peace"] & ~out_df["in_sampl7"]).sum())
    n_sampl_only = int((~out_df["in_peace"] & out_df["in_sampl7"]).sum())
    print(
        f"Totals: rows={len(out_df)}; both={n_both}; "
        f"peace_only={n_peace_only}; sampl7_only={n_sampl_only}"
    )


if __name__ == "__main__":
    main()

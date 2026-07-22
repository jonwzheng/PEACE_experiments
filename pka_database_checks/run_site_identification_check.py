#!/usr/bin/env python3
"""
Run PEACE site-identification checks against preprocessed pKa database species.

For each species, PEACE is run over merged contiguous charge ranges (to minimize
jobs), then each required charge state from the database is checked. Missing charge
states are flagged.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from peace_enumeration import (
    ChargeEngine,
    PROGRESSIVE_EXPANSION_STAGES,
    enumerate_species_charge_states,
    enumerate_species_charge_states_progressive,
    protomer_count,
    redirect_peace_logs,
    seed_round_cap_label,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREPROCESSED = SCRIPT_DIR / "preprocessed" / "species_charge_requirements.csv"
RESULTS_DIR = SCRIPT_DIR / "results"


def slice_output_path(stem: str, extension: str, slice_start: int, slice_end: int) -> Path:
    """Build a results path like ``site_identification_results_0_99.csv``."""
    return RESULTS_DIR / f"{stem}_{slice_start}_{slice_end}{extension}"


def _log(message: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def parse_charge_ranges(charge_ranges_text: str) -> list[tuple[int, int]]:
    if not charge_ranges_text or pd.isna(charge_ranges_text):
        return []
    ranges: list[tuple[int, int]] = []
    for part in str(charge_ranges_text).split(";"):
        lo_s, hi_s = part.split(":")
        ranges.append((int(lo_s), int(hi_s)))
    return ranges


def parse_required_charges(required_charges_text: str) -> list[int]:
    if not required_charges_text or pd.isna(required_charges_text):
        return []
    return [int(x) for x in str(required_charges_text).split(";") if x]


def check_species(
    row: pd.Series,
    *,
    engine: ChargeEngine,
    site_search_mode: str,
    peace_log_path: Path | None,
    max_seed_rounds: int | None,
    seed_round_cap_factor: float,
    progressive_expansion: bool,
    charge_seed_first_only: bool,
) -> list[dict]:
    unique_id = row["unique_ID"]
    smiles = row["SMILES"]
    transitions = json.loads(row["pka_transitions_json"])
    required_charges = parse_required_charges(row["required_charges"])
    charge_ranges = parse_charge_ranges(row["charge_ranges"])

    if not required_charges:
        return []

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [
            {
                "unique_ID": unique_id,
                "SMILES": smiles,
                "pka_type": pka_type,
                "required_charge": charge,
                "found": False,
                "protomer_count": 0,
                "reason": "invalid_smiles",
                "expansion_stage": "",
            }
            for pka_type, charge in transitions.items()
        ]

    # Only transitions in pka_transitions_json are validated; intermediate charges
    # (e.g. 0 from a merged -1:1 job) are enumerated but not checked.
    species_by_charge: dict[int, object] = {}
    enumeration_error = ""
    expansion_stage = ""
    species_header = (
        f"{unique_id} {smiles} charges={row['required_charges']} ranges={row['charge_ranges']}"
    )
    log_ctx = (
        redirect_peace_logs(peace_log_path, species_header=species_header)
        if peace_log_path is not None
        else contextlib.nullcontext()
    )
    with log_ctx:
        for charge_min, charge_max in charge_ranges:
            range_required = {
                charge
                for charge in required_charges
                if charge_min <= charge <= charge_max
            }
            try:
                if progressive_expansion and max_seed_rounds is None:
                    if peace_log_path is not None:
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(peace_log_path, "a", encoding="utf-8") as log_file:
                            log_file.write(
                                f"[{ts}] Progressive expansion for charge range "
                                f"{charge_min}:{charge_max} "
                                f"(stages: {', '.join(seed_round_cap_label(s) for s in PROGRESSIVE_EXPANSION_STAGES)})\n"
                            )
                    found, range_stage = enumerate_species_charge_states_progressive(
                        smiles,
                        charge_min=charge_min,
                        charge_max=charge_max,
                        required_charges=range_required,
                        site_search_mode=site_search_mode,
                        engine=engine,
                        charge_seed_first_only=charge_seed_first_only,
                    )
                    expansion_stage = range_stage
                else:
                    found = enumerate_species_charge_states(
                        smiles,
                        charge_min=charge_min,
                        charge_max=charge_max,
                        site_search_mode=site_search_mode,
                        engine=engine,
                        max_seed_rounds=max_seed_rounds,
                        seed_round_cap_factor=int(seed_round_cap_factor),
                        required_charges=range_required,
                        charge_seed_first_only=charge_seed_first_only,
                    )
                    expansion_stage = seed_round_cap_label(max_seed_rounds)
            except Exception as exc:
                enumeration_error = f"{type(exc).__name__}: {exc}"
                fail_msg = (
                    f"PEACE enumeration failed for {unique_id} "
                    f"({charge_min}:{charge_max}): {enumeration_error}"
                )
                if peace_log_path is None:
                    _log(f"  {fail_msg}")
                else:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(peace_log_path, "a", encoding="utf-8") as log_file:
                        log_file.write(f"[{ts}] {fail_msg}\n")
                break
            species_by_charge.update(found)

    results: list[dict] = []
    for pka_type, required_charge in sorted(transitions.items(), key=lambda x: x[1]):
        spec = species_by_charge.get(required_charge)
        if enumeration_error:
            found = False
            count = 0
            reason = f"peace_error: {enumeration_error}"
        elif spec is None:
            found = False
            count = 0
            reason = "charge_state_unreachable"
        else:
            count = protomer_count(spec)
            found = count > 0
            reason = "" if found else "no_protomers_at_charge"

        results.append(
            {
                "unique_ID": unique_id,
                "SMILES": smiles,
                "original_IUPAC_names": row.get("original_IUPAC_names", ""),
                "seed_formal_charge": row.get("seed_formal_charge"),
                "pka_type": pka_type,
                "required_charge": required_charge,
                "found": found,
                "protomer_count": count,
                "reason": reason,
                "expansion_stage": expansion_stage,
            }
        )

    return results


def resolve_species_slice(
    total: int,
    *,
    start: int | None,
    end: int | None,
    limit: int | None,
) -> tuple[int, int]:
    """
    Return inclusive (start, end) indices into the filtered species dataframe.

    ``--limit N`` is shorthand for ``--start 0 --end N-1``.
    """
    if limit is not None:
        if start is not None or end is not None:
            raise ValueError("--limit cannot be combined with --start or --end")
        if limit <= 0:
            raise ValueError("--limit must be a positive integer")
        return 0, limit - 1

    slice_start = 0 if start is None else start
    slice_end = (total - 1) if end is None else end

    if slice_start < 0:
        raise ValueError(f"--start must be >= 0, got {slice_start}")
    if slice_end < slice_start:
        raise ValueError(f"--end ({slice_end}) must be >= --start ({slice_start})")
    if slice_start >= total:
        raise ValueError(f"--start ({slice_start}) is out of range for {total} species")
    if slice_end >= total:
        raise ValueError(f"--end ({slice_end}) is out of range for {total} species (max {total - 1})")

    return slice_start, slice_end


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check PEACE ionization-site identification against pKa database species."
    )
    parser.add_argument(
        "--preprocessed",
        type=Path,
        default=DEFAULT_PREPROCESSED,
        help="Preprocessed species CSV from preprocess_pka_species.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Full per-transition results CSV (default: results/site_identification_results_START_END.csv)",
    )
    parser.add_argument(
        "--flagged-output",
        type=Path,
        default=None,
        help="CSV containing only failed transitions (default: results/site_identification_flagged_START_END.csv)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="First species index to process, 0-based inclusive (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last species index to process, 0-based inclusive (default: last row)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Shorthand for --start 0 --end N-1 (process the first N species)",
    )
    parser.add_argument(
        "--site-search-mode",
        type=str,
        default=None,
        choices=["default", "strong", "all", "very_weak", "none"],
        help="PEACE ionizable-site search strategy (omit for context defaults)",
    )
    parser.add_argument(
        "--max-seed-rounds",
        type=int,
        default=None,
        help=(
            "Fixed cap on iterative protomer seed rounds per tautomer (disables progressive "
            "expansion). Default with --progressive-expansion: try 0, 10, 100, then unlimited. "
            "Otherwise: --seed-round-cap-factor × ionizable groups. Set to -1 for no cap."
        ),
    )
    parser.add_argument(
        "--seed-round-cap-factor",
        type=float,
        default=5.0,
        help=(
            "Multiplier for automatic seed-round cap when --max-seed-rounds is unset and "
            "--no-progressive-expansion is set"
        ),
    )
    parser.add_argument(
        "--progressive-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For existence checks, try protomer expansion in stages: base protomer only (0 "
            "rounds), then 10, 100, and unlimited seed rounds until required charges are found"
        ),
    )
    parser.add_argument(
        "--charge-seed-first-only",
        action="store_true",
        help=(
            "When seeding adjacent charge states, use only the first unique "
            "protonation/deprotonation product per structural tautomer instead of "
            "keeping the full per-tautomer shift pool."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Write PEACE enumeration logs to --peace-log and show a tqdm progress bar",
    )
    parser.add_argument(
        "--peace-log",
        type=Path,
        default=None,
        help="Log file for PEACE enumeration output when --quiet is set (default: results/peace_START_END.log)",
    )
    args = parser.parse_args()

    if not args.preprocessed.is_file():
        parser.error(
            f"Preprocessed file not found: {args.preprocessed}. "
            "Run preprocess_pka_species.py first."
        )

    full_species_df = pd.read_csv(args.preprocessed)
    full_species_df = full_species_df[full_species_df["n_transitions"] > 0].reset_index(drop=True)
    total_species = len(full_species_df)

    try:
        slice_start, slice_end = resolve_species_slice(
            total_species,
            start=args.start,
            end=args.end,
            limit=args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    species_df = full_species_df.iloc[slice_start : slice_end + 1].reset_index(drop=True)
    n_in_run = len(species_df)

    args.output = args.output or slice_output_path(
        "site_identification_results", ".csv", slice_start, slice_end
    )
    args.flagged_output = args.flagged_output or slice_output_path(
        "site_identification_flagged", ".csv", slice_start, slice_end
    )
    args.peace_log = args.peace_log or slice_output_path(
        "peace", ".log", slice_start, slice_end
    )

    peace_log_path = args.peace_log if args.quiet else None
    if peace_log_path is not None:
        peace_log_path.parent.mkdir(parents=True, exist_ok=True)
        peace_log_path.write_text("", encoding="utf-8")

    _log(f"Total species in dataset: {total_species}")
    _log(f"Processing indices: {slice_start} to {slice_end} inclusive ({n_in_run} species this run)")
    if args.progressive_expansion and args.max_seed_rounds is None:
        stage_labels = ", ".join(seed_round_cap_label(s) for s in PROGRESSIVE_EXPANSION_STAGES)
        expansion_msg = f"progressive ({stage_labels})"
    elif args.max_seed_rounds is not None and args.max_seed_rounds < 0:
        expansion_msg = "unlimited rounds per tautomer"
    elif args.max_seed_rounds is not None:
        expansion_msg = f"{args.max_seed_rounds} rounds per tautomer"
    else:
        expansion_msg = f"{args.seed_round_cap_factor:g} × ionizable groups per tautomer"

    charge_seed_msg = (
        ", charge_seed_first_only"
        if args.charge_seed_first_only
        else ""
    )
    _log(
        f"Checking {n_in_run} species "
        f"(site_search_mode={args.site_search_mode}, expansion: {expansion_msg}"
        f"{charge_seed_msg})"
    )
    if peace_log_path is not None:
        _log(f"PEACE enumeration logs -> {peace_log_path}")

    engine = ChargeEngine()
    all_results: list[dict] = []

    species_iter = species_df.iterrows()
    if args.quiet:
        species_iter = tqdm(
            species_iter,
            total=len(species_df),
            desc="Species",
            unit="species",
        )

    for run_idx, row in species_iter:
        global_idx = slice_start + run_idx
        if not args.quiet:
            _log(
                f"Species {global_idx + 1}/{total_species} "
                f"(run {run_idx + 1}/{n_in_run}, index {global_idx}): {row['unique_ID']} "
                f"charges={row['required_charges']} ranges={row['charge_ranges']}"
            )
        all_results.extend(
            check_species(
                row,
                engine=engine,
                site_search_mode=args.site_search_mode,
                peace_log_path=peace_log_path,
                max_seed_rounds=args.max_seed_rounds,
                seed_round_cap_factor=args.seed_round_cap_factor,
                progressive_expansion=args.progressive_expansion,
                charge_seed_first_only=args.charge_seed_first_only,
            )
        )

    results_df = pd.DataFrame(all_results)
    flagged_df = results_df[~results_df["found"]].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    flagged_df.to_csv(args.flagged_output, index=False)

    n_checked = len(results_df)
    n_flagged = len(flagged_df)
    n_species_flagged = flagged_df["unique_ID"].nunique() if n_flagged else 0

    _log(f"Wrote {n_checked} transition checks to {args.output}")
    _log(f"Flagged {n_flagged} transitions across {n_species_flagged} species -> {args.flagged_output}")


if __name__ == "__main__":
    main()

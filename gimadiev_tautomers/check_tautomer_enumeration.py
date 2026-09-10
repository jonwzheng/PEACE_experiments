#!/usr/bin/env python3
"""Check whether PEACE enumeration finds Gimadiev reactant/product forms.

No geometry optimization or solvation is performed. Each reactant is expanded
with PEACE tautomer enumeration and same-charge site search (protomers /
zwitterions), matching `python -m peace.main --site-search-mode default`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.benchmark_common import PEACE_ROOT, resolve_peace_root

from benchmark_tautomers import (
    DATASETS,
    _ensure_peace_on_path,
    _smiles_match_keys,
    ensure_processed_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate PEACE tautomers and same-charge protomers for each Gimadiev "
            "reactant and report whether the reactant and product SMILES appear in "
            "the pool (no optimization)."
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
        "--input-csv",
        type=Path,
        default=None,
        help="Processed tautomer-equilibrium CSV.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Corrected extraction CSV used if --input-csv is missing.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write the enumeration coverage table.",
    )
    parser.add_argument(
        "--peace-root",
        type=Path,
        default=PEACE_ROOT,
        help="Path to the PEACE repository (ChargeEngine tautomer/protomer enumeration).",
    )
    parser.add_argument(
        "--site-search-mode",
        type=str,
        default="default",
        choices=["default", "strong", "all", "very_weak", "none"],
        help=(
            "Ionizable-site search used for same-charge protomer/zwitterion expansion "
            "(default: default)."
        ),
    )
    return parser


def _match_in_pool(target_smiles: str, pool_keys: list[tuple[str | None, str | None]]) -> str | None:
    iso_target, noniso_target = _smiles_match_keys(target_smiles)
    if iso_target is None and noniso_target is None:
        return None
    for iso, noniso in pool_keys:
        if iso_target is not None and iso == iso_target:
            return "isomeric"
        if noniso_target is not None and noniso == noniso_target:
            return "nonisomeric"
    return None


def _enumerate_pool(seed_smiles: str, *, site_search_mode: str) -> tuple[list[str], int]:
    from peace.engine import ChargeEngine
    from peace.main import _enumerate_species_protomers, _make_species

    engine = ChargeEngine()
    spec = _make_species(seed_smiles, engine=engine, only_protomer_search=False)
    _enumerate_species_protomers(
        spec,
        engine=engine,
        site_search_mode=site_search_mode,
    )
    return spec.get_all_smiles(), len(spec.tautomers)


def enumerate_reactions(
    data: pd.DataFrame,
    *,
    site_search_mode: str,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Enumerate tautomers/protomers for each row and record pool membership."""
    required = {"reaction_index", "reactant_smiles", "product_smiles"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Enumeration input is missing required columns: {sorted(missing)}")

    pool_cache: dict[str, tuple[list[str], int]] = {}
    rows: list[dict] = []
    n_product_found = 0
    n_reactant_found = 0

    for idx, row in data.iterrows():
        reaction_index = int(row["reaction_index"])
        reactant = str(row["reactant_smiles"])
        product = str(row["product_smiles"])
        if reactant not in pool_cache:
            pool_cache[reactant] = _enumerate_pool(
                reactant,
                site_search_mode=site_search_mode,
            )
        pool, n_tautomers = pool_cache[reactant]
        pool_keys = [_smiles_match_keys(smi) for smi in pool]
        reactant_mode = _match_in_pool(reactant, pool_keys)
        product_mode = _match_in_pool(product, pool_keys)
        reactant_present = reactant_mode is not None
        product_present = product_mode is not None
        n_reactant_found += int(reactant_present)
        n_product_found += int(product_present)

        print(
            f"[{idx + 1}/{len(data)}] rxn {reaction_index}: "
            f"n_tautomers={n_tautomers} n_protomers={len(pool)} "
            f"reactant={reactant_present} product={product_present} "
            f"({product_mode or 'missing'})"
        )
        rows.append(
            {
                "reaction_index": reaction_index,
                "reactant_smiles": reactant,
                "product_smiles": product,
                "site_search_mode": site_search_mode,
                "n_enumerated_tautomers": n_tautomers,
                "n_enumerated_protomers": len(pool),
                "reactant_in_pool": reactant_present,
                "product_in_pool": product_present,
                "reactant_match_mode": reactant_mode,
                "product_match_mode": product_mode,
                "enumerated_smiles": "|".join(pool),
            }
        )

    coverage = pd.DataFrame(rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(output_csv, index=False)
        print(f"Saved enumeration coverage to: {output_csv}")
    print(
        f"Reactant in pool: {n_reactant_found}/{len(data)}; "
        f"product in pool: {n_product_found}/{len(data)}"
    )
    return coverage


def main() -> None:
    args = _build_parser().parse_args()
    spec = DATASETS[args.dataset]
    peace_root = resolve_peace_root(args.peace_root)
    _ensure_peace_on_path(peace_root)

    input_csv = (args.input_csv or spec["processed_csv"]).resolve()
    raw_csv = (args.raw_csv or spec["raw_csv"]).resolve()
    output_csv = (args.output_csv or spec["coverage_csv"]).resolve()
    ensure_processed_csv(
        spec,
        raw_csv=raw_csv,
        processed_csv=input_csv,
        peace_root=peace_root,
    )

    data = pd.read_csv(input_csv)
    enumerate_reactions(
        data,
        site_search_mode=args.site_search_mode,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()

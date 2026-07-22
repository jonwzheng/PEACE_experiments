#!/usr/bin/env python3
"""
Preprocess the IUPAC pKa database into unique species with required charge states.

Reads iupac_high-confidence_v2_3.csv and writes one row per unique species with the
charge transitions implied by standard pKa1–pKa6, pKaH1–pKaH6, and pKb1–pKb6 entries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data" / "iupac_high-confidence_v2_3.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "preprocessed" / "species_charge_requirements.csv"

PKA_ACID = re.compile(r"^pKa(\d+)$")
PKA_BASE = re.compile(r"^pKaH(\d+)$")
PKB_BASE = re.compile(r"^pKb(\d+)$")


def contiguous_charge_ranges(charges: set[int]) -> list[tuple[int, int]]:
    """Group integer charges into minimal contiguous [min, max] ranges."""
    if not charges:
        return []

    sorted_charges = sorted(charges)
    ranges: list[tuple[int, int]] = []
    start = sorted_charges[0]
    end = start

    for charge in sorted_charges[1:]:
        if charge == end + 1:
            end = charge
        else:
            ranges.append((start, end))
            start = end = charge

    ranges.append((start, end))
    return ranges


def peace_charge_ranges(required_charges: set[int]) -> list[tuple[int, int]]:
    """
    Charge ranges for PEACE jobs, minimizing the number of runs.

    Contiguous required charges are grouped normally. When requirements span both
    negative and positive values (a gap across 0), a single [min, max] job is used
    instead of separate negative and positive runs; charge 0 is enumerated only as
    an intermediate and is not a required transition.
    """
    if not required_charges:
        return []

    min_charge = min(required_charges)
    max_charge = max(required_charges)
    if min_charge < 0 < max_charge:
        return [(min_charge, max_charge)]

    return contiguous_charge_ranges(required_charges)


def charge_from_pka_type(pka_type: str) -> int | None:
    """Map pKa1 -> -1, pKaH1/pKb1 -> +1, etc. Nonstandard types return None."""
    text = str(pka_type).strip()
    acid_match = PKA_ACID.match(text)
    if acid_match:
        return -int(acid_match.group(1))
    for base_pattern in (PKA_BASE, PKB_BASE):
        base_match = base_pattern.match(text)
        if base_match:
            return int(base_match.group(1))
    return None


def seed_formal_charge(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return int(Chem.GetFormalCharge(mol))


def build_species_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    grouped = df.groupby(["unique_ID", "SMILES"], sort=False)
    for (unique_id, smiles), group in grouped:
        inchi = group["InChI"].iloc[0]
        name = group["original_IUPAC_names"].iloc[0] if "original_IUPAC_names" in group else ""
        seed_charge = seed_formal_charge(smiles)

        transitions: dict[str, int] = {}
        for pka_type in group["pka_type"].unique():
            charge = charge_from_pka_type(pka_type)
            if charge is None:
                continue
            transitions[str(pka_type)] = charge

        required_charges = sorted(set(transitions.values()))
        charge_ranges = peace_charge_ranges(set(required_charges))

        rows.append(
            {
                "unique_ID": unique_id,
                "SMILES": smiles,
                "InChI": inchi,
                "original_IUPAC_names": name,
                "seed_formal_charge": seed_charge,
                "required_charges": ";".join(str(c) for c in required_charges),
                "charge_ranges": ";".join(f"{lo}:{hi}" for lo, hi in charge_ranges),
                "pka_transitions_json": json.dumps(transitions, sort_keys=True),
                "n_transitions": len(transitions),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("unique_ID").reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess pKa database rows into unique species charge requirements."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to iupac_high-confidence_v2_3.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for preprocessed species CSV",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    species_table = build_species_table(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    species_table.to_csv(args.output, index=False)

    n_with_transitions = int((species_table["n_transitions"] > 0).sum())
    print(f"Wrote {len(species_table)} unique species to {args.output}")
    print(f"  Species with at least one standard pKa/pKb transition: {n_with_transitions}")
    print(f"  Species with invalid SMILES (seed charge unknown): {species_table['seed_formal_charge'].isna().sum()}")


if __name__ == "__main__":
    main()

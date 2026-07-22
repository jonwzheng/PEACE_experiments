"""Thin wrapper around PEACE protomer enumeration for experiment scripts."""

from __future__ import annotations

import contextlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, TextIO

from rdkit.Chem import AllChem

_PEACE_ROOT = Path(__file__).resolve().parents[2] / "PEACE"
if str(_PEACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PEACE_ROOT))

from peace.engine import ChargeEngine  # noqa: E402
from peace.main import (  # noqa: E402
    _enumerate_species_protomers,
    _make_species,
    _seed_adjacent_charge_species,
)
from peace import main as peace_main  # noqa: E402
from peace.protomer import Species  # noqa: E402

# Existence-check expansion stages: base protomer only, then 10 / 100 rounds, then unlimited.
PROGRESSIVE_EXPANSION_STAGES: tuple[int, ...] = (0, 10, 100, -1)


def seed_round_cap_label(max_seed_rounds: int | None) -> str:
    if max_seed_rounds is None:
        return "auto"
    if max_seed_rounds < 0:
        return "unlimited"
    return str(max_seed_rounds)


def charges_satisfied(
    species_by_charge: dict[int, Species],
    required_charges: set[int],
) -> bool:
    if not required_charges:
        return True
    return all(
        charge in species_by_charge and protomer_count(species_by_charge[charge]) > 0
        for charge in required_charges
    )


@contextlib.contextmanager
def redirect_peace_logs(
    log_path: Path,
    *,
    species_header: str | None = None,
) -> Iterator[None]:
    """Send PEACE ``_log`` output to *log_path* instead of stdout."""
    original_log = peace_main._log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as log_file:
        if species_header:
            log_file.write(f"\n=== {species_header} ===\n")
            log_file.flush()

        peace_main._log = _make_file_logger(log_file)
        try:
            yield
        finally:
            peace_main._log = original_log


def _make_file_logger(log_file: TextIO):
    def _file_log(message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{ts}] {message}\n")
        log_file.flush()

    return _file_log


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
    """Like contiguous_charge_ranges, but merge across 0 when span is negative to positive."""
    if not required_charges:
        return []

    min_charge = min(required_charges)
    max_charge = max(required_charges)
    if min_charge < 0 < max_charge:
        return [(min_charge, max_charge)]

    return contiguous_charge_ranges(required_charges)


def edge_stop_charges_for_range(
    required_charges: set[int],
    charge_min: int,
    charge_max: int,
    seed_charge: int,
) -> set[int]:
    """
    Required charges where edge early-stopping is safe for a charge search window.

    When the interval includes *seed_charge* (e.g. -1 to +1 with seed 0), both
    range boundaries may early-stop. When the interval lies entirely on one side
    of the seed (e.g. -3:-1 or +1:+3 with seed 0), only the furthest boundary
    from the seed may early-stop; the boundary adjacent to the seed must still
    receive full protomer expansion so downstream charge states seed correctly.
    """
    boundary_required = {
        charge
        for charge in required_charges
        if charge == charge_min or charge == charge_max
    }
    if not boundary_required:
        return set()

    if charge_min <= seed_charge <= charge_max:
        return boundary_required

    if charge_max < seed_charge:
        return {charge for charge in boundary_required if charge == charge_min}

    if charge_min > seed_charge:
        return {charge for charge in boundary_required if charge == charge_max}

    return boundary_required


def _store_charge_state(
    species_by_charge: dict[int, Species],
    charge: int,
    spec: Species,
    *,
    edge_stop_charges: set[int],
    site_search_mode: str | None,
    engine: ChargeEngine,
    max_seed_rounds: int | None,
    seed_round_cap_factor: int,
    context: str = "same_charge",
) -> bool:
    """
    Record *spec* at *charge*, optionally skipping full protomer enumeration.

    Returns True when edge early-stop conditions are met (any protomer at a
    range-boundary required charge).
    """
    if charge in edge_stop_charges and protomer_count(spec) > 0:
        species_by_charge[charge] = spec
        return True

    if max_seed_rounds == 0:
        species_by_charge[charge] = spec
        return False

    _enumerate_species_protomers(
        spec,
        engine=engine,
        site_search_mode=site_search_mode,
        max_seed_rounds=max_seed_rounds,
        seed_round_cap_factor=seed_round_cap_factor,
        context=context,
    )
    species_by_charge[charge] = spec
    return False


def enumerate_species_charge_states(
    smiles: str,
    *,
    charge_min: int,
    charge_max: int,
    site_search_mode: str | None = None,
    engine: Optional[ChargeEngine] = None,
    max_seed_rounds: int | None = None,
    seed_round_cap_factor: int = 5,
    required_charges: Optional[set[int]] = None,
    edge_stop_charges: Optional[set[int]] = None,
    charge_seed_first_only: bool = False,
) -> dict[int, Species]:
    """
    Enumerate PEACE protomers for reachable charge states in [charge_min, charge_max].

    Mirrors the charge-seeding loop in ``peace.main`` without solvation or plotting.

    When *required_charges* / *edge_stop_charges* are supplied, range-boundary
    acid/base checks may stop early once seeding reaches a safe boundary charge
    and at least one protomer is already present. Safe boundaries depend on
    whether the charge interval includes the seed charge; see
    ``edge_stop_charges_for_range``. The opposite direction is also skipped when
    no required charges lie beyond the seed in that direction.
    """
    if charge_min > charge_max:
        raise ValueError("charge_min must be <= charge_max")

    engine = engine or ChargeEngine()
    seed_spec = _make_species(smiles, engine=engine)
    seed_charge = int(AllChem.GetFormalCharge(seed_spec.tautomers[0].protomers[0].mol))

    if edge_stop_charges is None and required_charges is not None:
        edge_stop_charges = edge_stop_charges_for_range(
            required_charges, charge_min, charge_max, seed_charge
        )
    edge_stop_charges = edge_stop_charges or set()

    species_by_charge: dict[int, Species] = {}
    seed_edge_stop = _store_charge_state(
        species_by_charge,
        seed_charge,
        seed_spec,
        edge_stop_charges=edge_stop_charges,
        site_search_mode=site_search_mode,
        engine=engine,
        max_seed_rounds=max_seed_rounds,
        seed_round_cap_factor=seed_round_cap_factor,
    )

    explore_lower = required_charges is None or any(
        charge < seed_charge for charge in required_charges
    )
    explore_higher = required_charges is None or any(
        charge > seed_charge for charge in required_charges
    )

    if explore_lower and not (seed_edge_stop and seed_charge == charge_min):
        current_charge = seed_charge
        current_spec = seed_spec
        while current_charge - 1 >= charge_min:
            target_charge = current_charge - 1
            next_spec = _seed_adjacent_charge_species(
                current_spec,
                engine=engine,
                charge_step=-1,
                site_search_mode=site_search_mode,
                charge_seed_first_only=charge_seed_first_only,
            )
            if next_spec is None:
                break
            if _store_charge_state(
                species_by_charge,
                target_charge,
                next_spec,
                edge_stop_charges=edge_stop_charges,
                site_search_mode=site_search_mode,
                engine=engine,
                max_seed_rounds=max_seed_rounds,
                seed_round_cap_factor=seed_round_cap_factor,
                context="inter_charge",
            ):
                break
            current_spec = next_spec
            current_charge = target_charge

    if explore_higher and not (seed_edge_stop and seed_charge == charge_max):
        current_charge = seed_charge
        current_spec = seed_spec
        while current_charge + 1 <= charge_max:
            target_charge = current_charge + 1
            next_spec = _seed_adjacent_charge_species(
                current_spec,
                engine=engine,
                charge_step=1,
                site_search_mode=site_search_mode,
                charge_seed_first_only=charge_seed_first_only,
            )
            if next_spec is None:
                break
            if _store_charge_state(
                species_by_charge,
                target_charge,
                next_spec,
                edge_stop_charges=edge_stop_charges,
                site_search_mode=site_search_mode,
                engine=engine,
                max_seed_rounds=max_seed_rounds,
                seed_round_cap_factor=seed_round_cap_factor,
                context="inter_charge",
            ):
                break
            current_spec = next_spec
            current_charge = target_charge

    return {
        charge: spec
        for charge, spec in species_by_charge.items()
        if charge_min <= charge <= charge_max
    }


def enumerate_species_charge_states_progressive(
    smiles: str,
    *,
    charge_min: int,
    charge_max: int,
    required_charges: set[int],
    site_search_mode: str | None = None,
    engine: Optional[ChargeEngine] = None,
    stages: tuple[int, ...] = PROGRESSIVE_EXPANSION_STAGES,
    charge_seed_first_only: bool = False,
) -> tuple[dict[int, Species], str]:
    """
    Try progressively larger protomer expansions until required charges are reachable.

    Stages default to 0 (base protomer only), 10, 100, then unlimited seed rounds.
    Returns the species-by-charge map and a label for the stage that succeeded (or the
    final stage attempted if none satisfied all requirements).
    """
    range_required = {
        charge for charge in required_charges if charge_min <= charge <= charge_max
    }
    last_result: dict[int, Species] = {}
    last_stage_label = seed_round_cap_label(stages[-1])

    for stage_cap in stages:
        last_stage_label = seed_round_cap_label(stage_cap)
        last_result = enumerate_species_charge_states(
            smiles,
            charge_min=charge_min,
            charge_max=charge_max,
            site_search_mode=site_search_mode,
            engine=engine,
            max_seed_rounds=stage_cap,
            required_charges=range_required,
            charge_seed_first_only=charge_seed_first_only,
        )
        if charges_satisfied(last_result, range_required):
            return last_result, last_stage_label

    return last_result, last_stage_label


def protomer_count(spec: Species) -> int:
    return sum(len(taut.protomers) for taut in spec.tautomers.values())

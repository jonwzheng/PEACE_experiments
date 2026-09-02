"""Shared helpers for PEACE_experiments benchmark scripts."""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
PEACE_ROOT = EXPERIMENTS_ROOT.parent / "PEACE"

DEFAULT_SCREEN_THRESHOLD = "80.0"
DEFAULT_MAX_CONFORMERS = "10"
DEFAULT_CONFORMER_ENERGY_THRESHOLD = "40.0"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "molecule"


def canon_smiles(smiles: str) -> str | None:
    """RDKit-canonical SMILES for comparing predicted vs observed structures."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_predicted_f_zwit(results_csv: Path) -> float:
    df = pd.read_csv(results_csv)
    if "is_zwitterion" not in df.columns or "boltzmann_fraction" not in df.columns:
        raise ValueError(
            f"Expected columns 'is_zwitterion' and 'boltzmann_fraction' in {results_csv}."
        )

    zwit = df["is_zwitterion"].astype(str).str.lower().isin(["true", "1", "yes"])
    fractions = pd.to_numeric(df["boltzmann_fraction"], errors="coerce").fillna(0.0)
    return float(fractions[zwit].sum())


@dataclass(frozen=True)
class DominantTautomer:
    smiles: str | None
    fraction: float | None
    n_tautomers: int


# kcal/mol/K; matches PEACE Species.assign_boltzmann_microstate_populations
_GAS_CONSTANT_KCAL = 0.00198720425864083
_DEFAULT_TEMPERATURE_K = 298.15


def _boltzmann_fractions_from_energies(
    energies: pd.Series,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
) -> pd.Series:
    """Compute Boltzmann fractions from solution-phase free energies (kcal/mol)."""
    g = pd.to_numeric(energies, errors="coerce")
    valid = g.dropna()
    out = pd.Series(index=energies.index, dtype=float)
    if valid.empty or temperature_k <= 0:
        return out
    rt = _GAS_CONSTANT_KCAL * float(temperature_k)
    weights = np.exp(-(valid.to_numpy(dtype=float) - float(valid.min())) / rt)
    partition_q = float(np.sum(weights))
    if partition_q <= 0 or not math.isfinite(partition_q):
        return out
    out.loc[valid.index] = weights / partition_q
    return out


def extract_dominant_by_charge(results_csv: Path) -> dict[int, DominantTautomer]:
    """Summarize each formal charge in a PEACE results CSV.

    For every charge state, reports:
      - ``n_tautomers``: number of enumerated protomers (all tautomer forms)
      - dominant SMILES / Boltzmann fraction when ranking is possible

    Uses ``boltzmann_fraction`` when present. If a charge group has no fractions
    (e.g. older PEACE runs that only Boltzmann-weighted the last charge), falls
    back to recomputing fractions from ``solution_phase_free_energy_kcal_mol``.
    """
    df = pd.read_csv(results_csv)
    required = {"formal_charge", "protomer_smiles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Expected columns {sorted(required)} in {results_csv}; missing {sorted(missing)}"
        )

    out: dict[int, DominantTautomer] = {}
    stored_frac = (
        pd.to_numeric(df["boltzmann_fraction"], errors="coerce")
        if "boltzmann_fraction" in df.columns
        else pd.Series(index=df.index, dtype=float)
    )
    energies = (
        df["solution_phase_free_energy_kcal_mol"]
        if "solution_phase_free_energy_kcal_mol" in df.columns
        else None
    )

    for charge, group in df.groupby(df["formal_charge"].astype(int), sort=True):
        n_tautomers = int(len(group))
        frac = stored_frac.loc[group.index]
        if frac.notna().any():
            ranked = group.assign(_frac=frac).dropna(subset=["_frac"])
        elif energies is not None:
            ranked = group.assign(
                _frac=_boltzmann_fractions_from_energies(energies.loc[group.index])
            ).dropna(subset=["_frac"])
        else:
            ranked = group.iloc[0:0]

        if ranked.empty:
            out[int(charge)] = DominantTautomer(
                smiles=None, fraction=None, n_tautomers=n_tautomers
            )
            continue

        best = ranked.sort_values(
            ["_frac", "protomer_smiles"], ascending=[False, True]
        ).iloc[0]
        smiles = canon_smiles(best["protomer_smiles"]) or str(best["protomer_smiles"])
        out[int(charge)] = DominantTautomer(
            smiles=smiles,
            fraction=float(best["_frac"]),
            n_tautomers=n_tautomers,
        )
    return out


def add_benchmark_runtime_args(
    parser: argparse.ArgumentParser,
    *,
    default_results_root: Path,
) -> None:
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help="Folder to store per-molecule PEACE outputs and aggregate benchmark CSVs.",
    )
    parser.add_argument(
        "--peace-root",
        type=Path,
        default=PEACE_ROOT,
        help="Path to the PEACE repository used to run `python -m peace.main`.",
    )


def build_peace_command(
    *,
    smiles: str,
    scratch_root: Path,
    output_csv: Path,
    solvent: str = "water",
    temperature: float | None = None,
    charge_min: int | None = None,
    charge_max: int | None = None,
    site_search_mode: str | None = None,
    keep_tautomer_smiles: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "peace.main",
        "--smiles",
        smiles,
        "--solvent",
        solvent,
        "--solvation",
        "--override-solvation",
        "--scratch-root",
        str(scratch_root),
        "--output-csv",
        str(output_csv),
        "--no-plot",
        "--screen-threshold",
        DEFAULT_SCREEN_THRESHOLD,
        "--max-conformers",
        DEFAULT_MAX_CONFORMERS,
        "--conformer-energy-threshold",
        DEFAULT_CONFORMER_ENERGY_THRESHOLD,
    ]
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    if charge_min is not None:
        cmd.extend(["--charge-min", str(charge_min)])
    if charge_max is not None:
        cmd.extend(["--charge-max", str(charge_max)])
    if site_search_mode is not None:
        cmd.extend(["--site-search-mode", str(site_search_mode)])
    if keep_tautomer_smiles:
        for tautomer_smiles in keep_tautomer_smiles:
            cmd.extend(["--keep-tautomer-smiles", tautomer_smiles])
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def log10_k_from_delta_g(
    delta_g_kcal_mol: float,
    *,
    temperature_k: float = _DEFAULT_TEMPERATURE_K,
) -> float:
    """Convert ΔG (kcal/mol) to log10(K) at the given temperature."""
    rt_ln10 = _GAS_CONSTANT_KCAL * float(temperature_k) * math.log(10.0)
    return -float(delta_g_kcal_mol) / rt_ln10


@dataclass(frozen=True)
class PeaceRunOutcome:
    returncode: int
    stdout: str
    stderr: str
    predicted_f_zwit: float | None
    job_messages: str | None


def summarize_job_messages(*, returncode: int, stdout: str, stderr: str) -> str | None:
    parts: list[str] = []
    stderr = stderr.strip()
    stdout = stdout.strip()

    if returncode != 0:
        parts.append(f"exit code {returncode}")

    warning_lines = [
        line.strip()
        for line in (stdout + "\n" + stderr).splitlines()
        if re.search(r"\b(warning|error|traceback|exception)\b", line, re.IGNORECASE)
    ]
    if warning_lines:
        parts.extend(warning_lines)

    if stderr and returncode != 0:
        parts.append(stderr)

    if not parts:
        return None
    return " | ".join(dict.fromkeys(parts))


def run_peace_job(
    *,
    peace_root: Path,
    cmd: list[str],
    mol_dir: Path,
    output_csv: Path,
    compute_f_zwit: bool = True,
) -> PeaceRunOutcome:
    mol_dir.mkdir(parents=True, exist_ok=True)
    run = subprocess.run(
        cmd,
        cwd=str(peace_root),
        capture_output=True,
        text=True,
    )
    (mol_dir / "stdout.log").write_text(run.stdout)
    (mol_dir / "stderr.log").write_text(run.stderr)

    predicted = None
    if compute_f_zwit and run.returncode == 0 and output_csv.exists():
        predicted = compute_predicted_f_zwit(output_csv)

    return PeaceRunOutcome(
        returncode=run.returncode,
        stdout=run.stdout,
        stderr=run.stderr,
        predicted_f_zwit=predicted,
        job_messages=summarize_job_messages(
            returncode=run.returncode,
            stdout=run.stdout,
            stderr=run.stderr,
        ),
    )


def resolve_peace_root(peace_root: Path) -> Path:
    resolved = peace_root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"PEACE repository not found at {resolved}. Pass --peace-root explicitly."
        )
    return resolved

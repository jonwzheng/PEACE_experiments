#!/usr/bin/env python3
"""Benchmark PEACE tautomer log10(Kz) against Gimadiev et al. extracted equilibria."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_ROOT = SCRIPT_DIR.parent
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from common.benchmark_common import (
    add_benchmark_runtime_args,
    build_peace_command,
    canon_smiles,
    log10_k_from_delta_g,
    resolve_peace_root,
    run_peace_job,
    slugify,
)

from parse_rdf import DATASETS, prepare_dataset, write_extracted_csv

DEFAULT_SCREEN_THRESHOLD = "500.0"
DEFAULT_CONFORMER_ENERGY_THRESHOLD = "40.0"
DEFAULT_EMBEDDED_CONFORMERS = "500"

DEFAULT_RAW_CSV = DATASETS["main"]["raw_csv"]
DEFAULT_PROCESSED_CSV = DATASETS["main"]["processed_csv"]
DEFAULT_RESULTS_ROOT = DATASETS["main"]["results_root"]
DEFAULT_COVERAGE_CSV = DATASETS["main"]["coverage_csv"]
SUMMARY_NAME = "benchmark_tautomers_results.csv"
ENERGY_COL = "solution_phase_free_energy_kcal_mol"

# IUPAC / dataset solvent names that are the same PEACE-supported solvent under
# a name not listed in ALPB/CPCM alias files.
_DATASET_SOLVENT_ALIASES = {
    "methanesulfinylmethane": "dmso",
    "oxolane": "thf",
    "tetrahydrofuran": "thf",
    "dimethylsulfoxide": "dmso",
    "dimethyl sulfoxide": "dmso",
}

_SALT_REMOVER = SaltRemover()

SUMMARY_COLUMNS = [
    "reaction_index",
    "reactant_smiles",
    "product_smiles",
    "solvent",
    "solvent_original",
    "temperature",
    "tabulated_constant",
    "predicted_log10Kz",
    "abs_error",
    "delta_g_kcal_mol",
    "reactant_energy_kcal_mol",
    "product_energy_kcal_mol",
    "reactant_tautomer_present",
    "product_tautomer_present",
    "reactant_match_mode",
    "product_match_mode",
    "matched_reactant_smiles",
    "matched_product_smiles",
    "n_peace_protomers",
    "site_search_mode",
    "reactant_is_zwitterion",
    "product_is_zwitterion",
    "run_ok",
    "run_returncode",
    "job_messages",
    "result_dir",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PEACE tautomer free-energy ratios against Gimadiev equilibria. "
            "For test_set_1 / test_set_2, tautomer enumeration runs first and PEACE "
            "is scored only on pairs whose product is found in the enumerated pool."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="extracted",
        help=(
            "Named tautomer set: main/extracted (MOESM4), test_set_1 (MOESM2), "
            "or test_set_2 (MOESM3). Scripts use the corrected CSVs."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Processed tautomer-equilibrium CSV (created from --raw-csv if missing).",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="Corrected atom-mapped extraction CSV (original RDF extract plus data_to_correct.csv).",
    )
    parser.add_argument(
        "--coverage-csv",
        type=Path,
        default=None,
        help="Enumeration coverage table written/read when --enumerate-first is set.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help=(
            "Rebuild the processed CSV from the corrected --raw-csv "
            "(preparing original/corrected extracts if needed) and exit."
        ),
    )
    parser.add_argument(
        "--enumerate-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Run tautomer enumeration first and keep only rows whose enumerated "
            "pool includes the target product. Default: on for test_set_1/2, off "
            "for main/extracted."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many unfinished entries this run (partial run).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run PEACE even when a summary row or results.csv already exists.",
    )
    parser.add_argument(
        "--rescore-only",
        action="store_true",
        help=(
            "Do not re-run PEACE; rebuild the summary from existing per-entry "
            "results.csv files under --results-root."
        ),
    )
    add_benchmark_runtime_args(parser, default_results_root=DEFAULT_RESULTS_ROOT)
    return parser


def _ensure_peace_on_path(peace_root: Path) -> None:
    root = str(peace_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _normalize_solvent_key(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _resolve_dataset_solvent(name: str):
    from peace.solvents import resolve_solvent

    text = str(name).strip()
    candidates = [text]
    alias = _DATASET_SOLVENT_ALIASES.get(text.lower())
    if alias:
        candidates.append(alias)
    alias_norm = _DATASET_SOLVENT_ALIASES.get(_normalize_solvent_key(text))
    if alias_norm:
        candidates.append(alias_norm)
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            return resolve_solvent(candidate)
        except ValueError:
            continue
    return None


def _unmap_and_clean_smiles(smiles: str) -> str | None:
    """Return a salt-stripped, non-atom-mapped canonical SMILES, or None."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = _SALT_REMOVER.StripMol(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol)


def _formal_charge(smiles: str) -> int | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return int(Chem.GetFormalCharge(mol))


def _is_zwitterion_smiles(smiles: str) -> bool:
    """Reuse PEACE's zwitterion definition (charged heavy atoms, net charge may be 0)."""
    from peace.protomer import Protomer

    return bool(Protomer.from_smiles(str(smiles)).is_zwitterion)


def _as_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def preprocess_reactions(
    raw_csv: Path,
    *,
    peace_root: Path,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Clean mapped SMILES and keep pure solvents supported by ALPB + CPCM-X."""
    _ensure_peace_on_path(peace_root)
    raw = pd.read_csv(raw_csv)
    required = {
        "reaction_index",
        "temperature",
        "solvent",
        "solvent_part",
        "reactant_smiles",
        "product_smiles",
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{raw_csv} is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, row in raw.iterrows():
        solvent_part = pd.to_numeric(row["solvent_part"], errors="coerce")
        if pd.isna(solvent_part) or not math.isclose(float(solvent_part), 1.0, abs_tol=1e-9):
            continue

        spec = _resolve_dataset_solvent(row["solvent"])
        if spec is None:
            continue

        reactant = _unmap_and_clean_smiles(row["reactant_smiles"])
        product = _unmap_and_clean_smiles(row["product_smiles"])
        if reactant is None or product is None:
            continue

        charge = _formal_charge(reactant)
        if charge is None:
            continue

        reactant_zwit = _is_zwitterion_smiles(reactant)
        product_zwit = _is_zwitterion_smiles(product)

        rows.append(
            {
                "reaction_index": int(row["reaction_index"]),
                "tabulated_constant": pd.to_numeric(
                    row.get("tabulated_constant", pd.NA), errors="coerce"
                ),
                "temperature": pd.to_numeric(row["temperature"], errors="coerce"),
                "temperature_celsius": pd.to_numeric(
                    row.get("temperature_celsius", pd.NA), errors="coerce"
                ),
                "solvent": spec.alpb,
                "solvent_original": str(row["solvent"]),
                "solvent_part": float(solvent_part),
                "reactant_smiles": reactant,
                "product_smiles": product,
                "reactant_smiles_mapped": str(row["reactant_smiles"]),
                "product_smiles_mapped": str(row["product_smiles"]),
                "formal_charge": charge,
                "reactant_is_zwitterion": reactant_zwit,
                "product_is_zwitterion": product_zwit,
                "n_reactants": row.get("n_reactants", pd.NA),
                "n_products": row.get("n_products", pd.NA),
            }
        )

    processed = pd.DataFrame(rows)
    if processed.empty:
        raise ValueError(f"No rows remained after preprocessing {raw_csv}.")

    out_path = output_csv or DEFAULT_PROCESSED_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(out_path, index=False)
    return processed


def ensure_raw_csv(spec: dict, raw_csv: Path) -> Path:
    """Ensure original RDF extracts exist and write the corrected CSV used by benchmarks."""
    if raw_csv.resolve() == Path(spec["raw_csv"]).resolve():
        prepare_dataset(spec, include_processed=False, include_results=False)
        return Path(spec["raw_csv"])
    if raw_csv.exists():
        return raw_csv
    rdf_path = spec.get("rdf")
    if rdf_path is None:
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv}")
    print(f"Raw CSV not found at {raw_csv}; parsing {rdf_path}")
    write_extracted_csv(
        Path(rdf_path),
        raw_csv,
        index_mode=spec.get("index_mode", "mireg"),
    )
    return raw_csv


def ensure_processed_csv(spec: dict, *, raw_csv: Path, processed_csv: Path, peace_root: Path) -> Path:
    ensure_raw_csv(spec, raw_csv)
    if not processed_csv.exists():
        print(f"Processed CSV not found at {processed_csv}; building it from {raw_csv}")
        preprocess_reactions(raw_csv, peace_root=peace_root, output_csv=processed_csv)
    return processed_csv


def _smiles_match_keys(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return (
        Chem.MolToSmiles(mol, isomericSmiles=True),
        Chem.MolToSmiles(mol, isomericSmiles=False),
    )


def _strip_stereochemistry(smiles: str) -> str | None:
    """Canonical SMILES with tetrahedral and double-bond stereo removed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def _lookup_protomer_energy(
    peace: pd.DataFrame,
    *,
    target_smiles: str,
) -> tuple[float | None, str | None, str | None]:
    """Return (energy, matched SMILES, match_mode) for a tautomer in PEACE output."""
    iso_target, noniso_target = _smiles_match_keys(target_smiles)
    if iso_target is None and noniso_target is None:
        return None, None, None

    iso_hits: list[tuple[float, str]] = []
    noniso_hits: list[tuple[float, str]] = []
    for _, row in peace.iterrows():
        energy = pd.to_numeric(row.get(ENERGY_COL), errors="coerce")
        if pd.isna(energy):
            continue
        iso, noniso = _smiles_match_keys(str(row["protomer_smiles"]))
        smiles = canon_smiles(row["protomer_smiles"]) or str(row["protomer_smiles"])
        if iso_target is not None and iso == iso_target:
            iso_hits.append((float(energy), smiles))
        elif noniso_target is not None and noniso == noniso_target:
            noniso_hits.append((float(energy), smiles))

    if iso_hits:
        energy, smiles = min(iso_hits, key=lambda item: (item[0], item[1]))
        return energy, smiles, "isomeric"
    if noniso_hits:
        energy, smiles = min(noniso_hits, key=lambda item: (item[0], item[1]))
        return energy, smiles, "nonisomeric"
    return None, None, None


def _peace_results_complete(output_csv: Path) -> bool:
    if not output_csv.exists():
        return False
    try:
        df = pd.read_csv(output_csv)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return False
    return "protomer_smiles" in df.columns and len(df) > 0


def _workflow_notes(results_csv: Path) -> str | None:
    try:
        df = pd.read_csv(results_csv)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
        return None
    parts: list[str] = []
    if "workflow_error" in df.columns:
        errors = [
            text
            for text in df["workflow_error"].dropna().astype(str)
            if text.strip() and text.strip().lower() not in {"nan", "none"}
        ]
        parts.extend(dict.fromkeys(errors))
    return " | ".join(parts) if parts else None


def _combine_messages(*messages: str | None) -> str | None:
    parts = [m.strip() for m in messages if m and str(m).strip()]
    if not parts:
        return None
    return " | ".join(dict.fromkeys(parts))


def _entry_dir(
    results_root: Path,
    *,
    row_number: int,
    reaction_index: int,
    smiles: str,
    solvent: str,
    temperature: float,
) -> Path:
    return results_root / (
        f"{row_number:04d}_{int(reaction_index)}_"
        f"{slugify(smiles)[:30]}_{slugify(solvent)}_{int(round(float(temperature)))}"
    )


def _empty_score_fields() -> dict:
    return {
        "predicted_log10Kz": pd.NA,
        "abs_error": pd.NA,
        "delta_g_kcal_mol": pd.NA,
        "reactant_energy_kcal_mol": pd.NA,
        "product_energy_kcal_mol": pd.NA,
        "reactant_tautomer_present": False,
        "product_tautomer_present": False,
        "reactant_match_mode": pd.NA,
        "product_match_mode": pd.NA,
        "matched_reactant_smiles": pd.NA,
        "matched_product_smiles": pd.NA,
        "n_peace_protomers": pd.NA,
    }


def score_peace_results(
    *,
    output_csv: Path,
    reactant_smiles: str,
    product_smiles: str,
    temperature: float,
    tabulated_constant: float,
) -> dict:
    fields = _empty_score_fields()
    if not _peace_results_complete(output_csv):
        return fields

    peace = pd.read_csv(output_csv)
    fields["n_peace_protomers"] = int(len(peace))
    g_react, r_smi, r_mode = _lookup_protomer_energy(peace, target_smiles=reactant_smiles)
    g_prod, p_smi, p_mode = _lookup_protomer_energy(peace, target_smiles=product_smiles)

    fields["reactant_tautomer_present"] = g_react is not None
    fields["product_tautomer_present"] = g_prod is not None
    fields["reactant_match_mode"] = r_mode if r_mode is not None else pd.NA
    fields["product_match_mode"] = p_mode if p_mode is not None else pd.NA
    fields["matched_reactant_smiles"] = r_smi if r_smi is not None else pd.NA
    fields["matched_product_smiles"] = p_smi if p_smi is not None else pd.NA
    fields["reactant_energy_kcal_mol"] = g_react if g_react is not None else pd.NA
    fields["product_energy_kcal_mol"] = g_prod if g_prod is not None else pd.NA

    if g_react is None or g_prod is None:
        return fields

    delta_g = float(g_prod) - float(g_react)
    predicted = log10_k_from_delta_g(delta_g, temperature_k=float(temperature))
    fields["delta_g_kcal_mol"] = delta_g
    fields["predicted_log10Kz"] = predicted
    if pd.notna(tabulated_constant):
        fields["abs_error"] = abs(predicted - float(tabulated_constant))
    return fields


def _load_summary(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "reaction_index" not in df.columns:
        return {}
    out: dict[int, dict] = {}
    for _, row in df.iterrows():
        out[int(row["reaction_index"])] = row.to_dict()
    return out


def _write_summary(path: Path, rows_by_id: dict[int, dict], order: list[int]) -> None:
    rows = [rows_by_id[rid] for rid in order if rid in rows_by_id]
    df = pd.DataFrame(rows)
    df = df.reindex(columns=SUMMARY_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _base_row(row: pd.Series) -> dict:
    return {
        "reaction_index": int(row["reaction_index"]),
        "reactant_smiles": str(row["reactant_smiles"]),
        "product_smiles": str(row["product_smiles"]),
        "solvent": str(row["solvent"]),
        "solvent_original": str(row.get("solvent_original", row["solvent"])),
        "temperature": float(row["temperature"]),
        "tabulated_constant": pd.to_numeric(row["tabulated_constant"], errors="coerce"),
    }


def main() -> None:
    args, main_extra_args = _build_parser().parse_known_args()
    spec = DATASETS[args.dataset]
    peace_root = resolve_peace_root(args.peace_root)
    _ensure_peace_on_path(peace_root)
    user_set_site_search = "--site-search-mode" in main_extra_args

    raw_csv = (args.raw_csv or spec["raw_csv"]).resolve()
    input_csv = (args.input_csv or spec["processed_csv"]).resolve()
    coverage_csv = (args.coverage_csv or spec["coverage_csv"]).resolve()
    results_root = (
        args.results_root.resolve()
        if "--results-root" in sys.argv
        else Path(spec["results_root"]).resolve()
    )
    enumerate_first = (
        spec["enumerate_first"] if args.enumerate_first is None else args.enumerate_first
    )

    if args.preprocess_only:
        ensure_raw_csv(spec, raw_csv)
        processed = preprocess_reactions(
            raw_csv,
            peace_root=peace_root,
            output_csv=input_csv,
        )
        print(f"Wrote {len(processed)} processed rows to {input_csv}")
        return

    ensure_processed_csv(
        spec,
        raw_csv=raw_csv,
        processed_csv=input_csv,
        peace_root=peace_root,
    )

    data = pd.read_csv(input_csv)
    required = {
        "reaction_index",
        "reactant_smiles",
        "product_smiles",
        "solvent",
        "temperature",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")
    if "tabulated_constant" not in data.columns:
        data["tabulated_constant"] = pd.NA

    if enumerate_first:
        from check_tautomer_enumeration import enumerate_reactions

        if coverage_csv.exists() and not args.force_rerun:
            coverage = pd.read_csv(coverage_csv)
            print(f"Loaded enumeration coverage from {coverage_csv}")
        else:
            coverage = enumerate_reactions(
                data,
                site_search_mode="default",
                output_csv=coverage_csv,
            )
        if "product_in_pool" not in coverage.columns:
            raise ValueError(f"{coverage_csv} is missing column 'product_in_pool'.")
        keep = set(
            coverage.loc[coverage["product_in_pool"].astype(bool), "reaction_index"].astype(int)
        )
        n_before = len(data)
        data = data[data["reaction_index"].astype(int).isin(keep)].copy()
        data = data.reset_index(drop=True)
        print(
            f"Keeping {len(data)} / {n_before} equilibria whose enumerated pool "
            f"includes the target product."
        )
        if data.empty:
            raise ValueError(
                "No equilibria remained after requiring the product in the enumerated pool."
            )

    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / SUMMARY_NAME
    completed = {} if args.rescore_only else _load_summary(summary_path)
    order = [int(v) for v in data["reaction_index"].tolist()]

    n_new = 0
    for row_number, row in data.iterrows():
        reaction_index = int(row["reaction_index"])
        reactant = str(row["reactant_smiles"])
        product = str(row["product_smiles"])
        product_tautomer = _strip_stereochemistry(product) or product
        solvent = str(row["solvent"])
        temperature = float(row["temperature"])
        tabulated = pd.to_numeric(row["tabulated_constant"], errors="coerce")
        charge = int(row["formal_charge"]) if "formal_charge" in data.columns else 0
        if "reactant_is_zwitterion" in data.columns and "product_is_zwitterion" in data.columns:
            reactant_zwit = _as_bool(row["reactant_is_zwitterion"])
            product_zwit = _as_bool(row["product_is_zwitterion"])
        else:
            reactant_zwit = _is_zwitterion_smiles(reactant)
            product_zwit = _is_zwitterion_smiles(product)
        if user_set_site_search:
            site_search_mode = None
            site_search_label = "user"
        else:
            site_search_mode = "none"
            site_search_label = "none"
        only_protomer_search = "--only-protomer-search" not in main_extra_args
        add_product_tautomer = "--add-tautomers" not in main_extra_args

        mol_dir = _entry_dir(
            results_root,
            row_number=int(row_number),
            reaction_index=reaction_index,
            smiles=reactant,
            solvent=solvent,
            temperature=temperature,
        )
        output_csv = mol_dir / "results.csv"
        already_scored = reaction_index in completed and not args.force_rerun

        if already_scored and not args.rescore_only:
            continue
        if args.limit is not None and n_new >= args.limit:
            break

        print(
            f"[{row_number + 1}/{len(data)}] rxn {reaction_index}: "
            f"{solvent} T={temperature:g} K site-search={site_search_label} "
            f"only-protomer-search={only_protomer_search} seed={reactant}"
        )

        run_ok = False
        run_returncode: int | None = None
        job_messages: str | None = None

        if args.rescore_only:
            if not _peace_results_complete(output_csv):
                raise FileNotFoundError(
                    f"--rescore-only requires existing results at {output_csv}"
                )
            run_ok = True
            run_returncode = 0
        elif _peace_results_complete(output_csv) and not args.force_rerun:
            run_ok = True
            run_returncode = 0
            print(f"  skipping PEACE (found {output_csv})")
        else:
            cmd = build_peace_command(
                smiles=reactant,
                solvent=solvent,
                temperature=temperature,
                scratch_root=mol_dir / "solvation",
                output_csv=output_csv,
                charge_min=charge,
                charge_max=charge,
                site_search_mode=site_search_mode,
                only_protomer_search=only_protomer_search,
                add_tautomers=[product_tautomer] if add_product_tautomer else None,
                screen_threshold=DEFAULT_SCREEN_THRESHOLD,
                embedded_conformers=DEFAULT_EMBEDDED_CONFORMERS,
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

        score = score_peace_results(
            output_csv=output_csv,
            reactant_smiles=reactant,
            product_smiles=product_tautomer,
            temperature=temperature,
            tabulated_constant=float(tabulated) if pd.notna(tabulated) else float("nan"),
        )
        if not score["product_tautomer_present"]:
            job_messages = _combine_messages(job_messages, "product tautomer not present")
        if run_ok and not score["reactant_tautomer_present"]:
            job_messages = _combine_messages(job_messages, "reactant tautomer not present")
        if _peace_results_complete(output_csv):
            job_messages = _combine_messages(job_messages, _workflow_notes(output_csv))

        record = {
            **_base_row(row),
            **score,
            "site_search_mode": site_search_label,
            "reactant_is_zwitterion": reactant_zwit,
            "product_is_zwitterion": product_zwit,
            "run_ok": run_ok,
            "run_returncode": run_returncode,
            "job_messages": job_messages,
            "result_dir": str(mol_dir),
        }
        completed[reaction_index] = record
        n_new += 1
        _write_summary(summary_path, completed, order)

        pred = record["predicted_log10Kz"]
        pred_txt = f"{float(pred):.3f}" if pd.notna(pred) else "NA"
        tab_txt = f"{float(tabulated):.3f}" if pd.notna(tabulated) else "NA"
        present = "yes" if record["product_tautomer_present"] else "NO"
        print(
            f"  product_present={present} log10Kz_pred={pred_txt} "
            f"tabulated={tab_txt} run_ok={run_ok}"
        )

    _write_summary(summary_path, completed, order)
    n_remaining = len(data) - len(completed)
    print(f"Saved benchmark summary to: {summary_path}")
    print(f"Scored {len(completed)}/{len(data)} entries ({n_new} updated this run).")
    if n_remaining > 0:
        print(f"{n_remaining} unfinished entries remain; re-run the script to continue.")


if __name__ == "__main__":
    main()

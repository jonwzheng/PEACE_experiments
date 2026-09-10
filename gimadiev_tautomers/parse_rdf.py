#!/usr/bin/env python3
"""Parse Gimadiev RDF files and apply tabulated-constant corrections.

Each dataset keeps an uncorrected extract under ``original/`` and a corrected
copy that the benchmark scripts consume:

- ``main`` (MOESM4; CLI alias ``extracted``)
- ``test_set_1`` (MOESM2)
- ``test_set_2`` (MOESM3)
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
CORRECTIONS_CSV = DATA_DIR / "data_to_correct.csv"
SUMMARY_NAME = "benchmark_tautomers_results.csv"

# Values of data_to_correct.csv "set" (and CLI --dataset) mapped to a folder name.
SET_ALIASES = {
    "original": "main",
    "main": "main",
    "extracted": "main",
    "test_set_1": "test_set_1",
    "test_set_2": "test_set_2",
}

EXTRACTED_COLUMNS = [
    "reaction_index",
    "reaction_smiles",
    "tabulated_constant",
    "temperature_celsius",
    "solvent",
    "temperature",
    "solvent_part",
    "reactant_smiles",
    "product_smiles",
    "n_reactants",
    "n_products",
]


def canonical_dataset_name(name: str) -> str:
    try:
        return SET_ALIASES[str(name).strip()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset {name!r}. Expected one of {sorted(SET_ALIASES)}.") from exc


def _make_spec(
    name: str,
    *,
    rdf: Path,
    results_root: Path,
    coverage_csv: Path,
    plot_stem: str,
    enumerate_first: bool,
    index_mode: str,
) -> dict:
    data_dir = DATA_DIR / name
    original_dir = data_dir / "original"
    return {
        "name": name,
        "rdf": rdf,
        "index_mode": index_mode,
        "data_dir": data_dir,
        "original_dir": original_dir,
        "original_raw_csv": original_dir / "reactions_extracted.csv",
        "original_processed_csv": original_dir / "reactions_extracted_processed.csv",
        "original_results_csv": original_dir / SUMMARY_NAME,
        "raw_csv": data_dir / "reactions_extracted.csv",
        "processed_csv": data_dir / "reactions_extracted_processed.csv",
        "results_root": results_root,
        "results_csv": results_root / SUMMARY_NAME,
        "coverage_csv": coverage_csv,
        "plot_stem": plot_stem,
        "enumerate_first": enumerate_first,
    }


_CANONICAL_DATASETS = {
    "main": _make_spec(
        "main",
        rdf=DATA_DIR / "10822_2018_101_MOESM4_ESM.rdf",
        results_root=SCRIPT_DIR / "results" / "tautomer_benchmark",
        coverage_csv=SCRIPT_DIR / "results" / "tautomer_enumeration" / "enumeration_coverage.csv",
        plot_stem="gimadiev_kt_scatter_log10",
        enumerate_first=False,
        # Existing main extraction uses 0-based RDF file order, not $MIREG.
        index_mode="sequential0",
    ),
    "test_set_1": _make_spec(
        "test_set_1",
        rdf=DATA_DIR / "10822_2018_101_MOESM2_ESM.rdf",
        results_root=SCRIPT_DIR / "results" / "test_set_1",
        coverage_csv=SCRIPT_DIR / "results" / "test_set_1" / "enumeration_coverage.csv",
        plot_stem="test_set_1_kt_scatter_log10",
        enumerate_first=True,
        index_mode="mireg",
    ),
    "test_set_2": _make_spec(
        "test_set_2",
        rdf=DATA_DIR / "10822_2018_101_MOESM3_ESM.rdf",
        results_root=SCRIPT_DIR / "results" / "test_set_2",
        coverage_csv=SCRIPT_DIR / "results" / "test_set_2" / "enumeration_coverage.csv",
        plot_stem="test_set_2_kt_scatter_log10",
        enumerate_first=True,
        index_mode="mireg",
    ),
}

CANONICAL_DATASET_NAMES = ("main", "test_set_1", "test_set_2")
DATASETS = {
    **_CANONICAL_DATASETS,
    "extracted": _CANONICAL_DATASETS["main"],
}
TEST_SETS = {name: _CANONICAL_DATASETS[name] for name in ("test_set_1", "test_set_2")}


def _clean_numeric(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    number = float(text)
    if not math.isfinite(number):
        return None
    rounded = round(number, 6)
    hundredths = round(number, 2)
    if abs(number - hundredths) < 1e-6 * max(1.0, abs(number)):
        return float(int(hundredths)) if hundredths == int(hundredths) else hundredths
    return rounded


def _parse_properties(chunk: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for key, value in re.findall(r"^\$DTYPE\s+(.+)\n\$DATUM\s+(.+)$", chunk, flags=re.M)
    }


def parse_rdf_file(path: Path, *, index_mode: str = "mireg") -> pd.DataFrame:
    """Return a reactions_extracted-style table from a MDL RD file of $RXN records."""
    if index_mode not in {"mireg", "sequential0"}:
        raise ValueError(f"Unknown index_mode {index_mode!r}.")
    RDLogger.DisableLog("rdApp.warning")
    text = path.read_text(encoding="utf-8", errors="replace")
    rows: list[dict] = []
    for chunk in re.split(r"(?=\$RFMT\b)", text):
        if "$RXN" not in chunk:
            continue
        index_match = re.search(r"\$RFMT\s+\$MIREG\s+(\d+)", chunk)
        if index_match is None:
            raise ValueError(f"{path} contains an $RXN record without $MIREG.")
        rxn_start = chunk.find("$RXN")
        dtype_start = chunk.find("$DTYPE")
        rxn_block = chunk[rxn_start:dtype_start if dtype_start >= 0 else None]
        rxn = rdChemReactions.ReactionFromRxnBlock(rxn_block)
        if rxn is None:
            raise ValueError(
                f"{path} reaction {index_match.group(1)} could not be parsed as an RXN block."
            )
        n_reactants = int(rxn.GetNumReactantTemplates())
        n_products = int(rxn.GetNumProductTemplates())
        if n_reactants < 1 or n_products < 1:
            raise ValueError(
                f"{path} reaction {index_match.group(1)} is missing a reactant or product."
            )
        reactant_smiles = ".".join(
            Chem.MolToSmiles(rxn.GetReactantTemplate(i)) for i in range(n_reactants)
        )
        product_smiles = ".".join(
            Chem.MolToSmiles(rxn.GetProductTemplate(i)) for i in range(n_products)
        )
        props = _parse_properties(chunk)
        temperature = _clean_numeric(props.get("temperature"))
        temperature_celsius = (
            None if temperature is None else _clean_numeric(str(temperature - 273.15))
        )
        if index_mode == "mireg":
            reaction_index = int(index_match.group(1))
        else:
            reaction_index = len(rows)
        rows.append(
            {
                "reaction_index": reaction_index,
                "reaction_smiles": f"{reactant_smiles}>>{product_smiles}",
                "tabulated_constant": _clean_numeric(props.get("tabulated_constant")),
                "temperature_celsius": temperature_celsius,
                "solvent": props.get("solvent", ""),
                "temperature": temperature,
                "solvent_part": _clean_numeric(props.get("solvent_part")),
                "reactant_smiles": reactant_smiles,
                "product_smiles": product_smiles,
                "n_reactants": n_reactants,
                "n_products": n_products,
            }
        )
    if not rows:
        raise ValueError(f"No $RXN records found in {path}")
    return pd.DataFrame(rows, columns=EXTRACTED_COLUMNS)


def write_extracted_csv(
    rdf_path: Path,
    output_csv: Path,
    *,
    index_mode: str = "mireg",
) -> pd.DataFrame:
    extracted = parse_rdf_file(rdf_path, index_mode=index_mode)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    extracted.to_csv(output_csv, index=False)
    return extracted


def load_corrections(dataset_name: str) -> pd.DataFrame:
    """Rows of data_to_correct.csv that apply to this canonical dataset."""
    canonical = canonical_dataset_name(dataset_name)
    if not CORRECTIONS_CSV.exists():
        return pd.DataFrame()
    corrections = pd.read_csv(CORRECTIONS_CSV)
    if corrections.empty:
        return corrections
    if "set" not in corrections.columns:
        raise ValueError(f"{CORRECTIONS_CSV} is missing column 'set'.")
    mapped = corrections["set"].map(lambda value: SET_ALIASES.get(str(value).strip()))
    unknown = sorted({str(value) for value, mapped_value in zip(corrections["set"], mapped) if mapped_value is None})
    if unknown:
        raise ValueError(
            f"{CORRECTIONS_CSV} has unknown set values {unknown}. "
            f"Expected one of {sorted(SET_ALIASES)}."
        )
    return corrections.loc[mapped == canonical].copy()


def _numeric_close(left, right) -> bool:
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    if pd.isna(left_num) and pd.isna(right_num):
        return True
    if pd.isna(left_num) or pd.isna(right_num):
        return False
    return math.isclose(float(left_num), float(right_num), rel_tol=1e-6, abs_tol=1e-6)


def apply_tabulated_corrections(
    df: pd.DataFrame,
    dataset_name: str,
    *,
    source: Path | str | None = None,
    require_all_indices: bool = True,
) -> tuple[pd.DataFrame, int, int]:
    """Apply sign-flip / REMOVE rows from data_to_correct.csv.

    Returns (corrected_frame, n_updated, n_removed).
    """
    if "reaction_index" not in df.columns:
        raise ValueError("Table is missing column 'reaction_index'.")
    corrections = load_corrections(dataset_name)
    if corrections.empty:
        return df.copy(), 0, 0

    out = df.copy()
    label = str(source) if source is not None else "table"
    drop_indices: list[int] = []
    n_updated = 0
    n_skipped = 0

    for _, correction in corrections.iterrows():
        idx = int(correction["reaction_index"])
        mask = out["reaction_index"].astype(int) == idx
        n_match = int(mask.sum())
        if n_match == 0:
            if require_all_indices:
                raise ValueError(f"{label} has no reaction_index {idx} for dataset {dataset_name}.")
            n_skipped += 1
            continue
        if n_match != 1:
            raise ValueError(f"{label} has {n_match} rows with reaction_index {idx}.")

        original_value = correction.get("original_tabulated_constant", pd.NA)
        current_value = out.loc[mask, "tabulated_constant"].iloc[0] if "tabulated_constant" in out.columns else pd.NA
        if pd.notna(original_value) and str(original_value).strip() != "" and "tabulated_constant" in out.columns:
            if not _numeric_close(current_value, original_value):
                raise ValueError(
                    f"{label} reaction_index {idx}: expected original tabulated_constant "
                    f"{original_value}, found {current_value}."
                )

        new_value = correction["corrected_tabulated_constant"]
        if str(new_value).strip().upper() == "REMOVE":
            drop_indices.append(idx)
            continue
        if "tabulated_constant" not in out.columns:
            raise ValueError(f"{label} is missing column 'tabulated_constant'.")
        out.loc[mask, "tabulated_constant"] = pd.to_numeric(new_value, errors="coerce")
        n_updated += 1

    if drop_indices:
        out = out[~out["reaction_index"].astype(int).isin(set(drop_indices))].copy()
    out = out.reset_index(drop=True)

    if "predicted_log10Kz" in out.columns and "abs_error" in out.columns:
        pred = pd.to_numeric(out["predicted_log10Kz"], errors="coerce")
        exp = pd.to_numeric(out["tabulated_constant"], errors="coerce")
        out["abs_error"] = (pred - exp).abs()

    if n_skipped:
        print(
            f"  skipped {n_skipped} correction(s) whose reaction_index is absent from {label}"
        )
    return out, n_updated, len(drop_indices)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _maybe_archive_as_original(working: Path, original: Path) -> None:
    """If a working file exists and no original archive does, keep the working copy as original."""
    if working.exists() and not original.exists():
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(working, original)
        print(f"Archived uncorrected {working.name} -> {original}")


def prepare_dataset(
    spec: dict,
    *,
    from_rdf: bool = False,
    include_processed: bool = True,
    include_results: bool = True,
) -> None:
    """Ensure original extracts exist, then write corrected tables used by the benchmarks."""
    name = spec["name"]
    original_raw = Path(spec["original_raw_csv"])
    raw_csv = Path(spec["raw_csv"])
    original_processed = Path(spec["original_processed_csv"])
    processed_csv = Path(spec["processed_csv"])
    original_results = Path(spec["original_results_csv"])
    results_csv = Path(spec["results_csv"])

    _maybe_archive_as_original(raw_csv, original_raw)
    if include_processed:
        _maybe_archive_as_original(processed_csv, original_processed)
    if include_results:
        _maybe_archive_as_original(results_csv, original_results)

    if from_rdf or not original_raw.exists():
        rdf_path = spec.get("rdf")
        if rdf_path is None:
            raise FileNotFoundError(f"Original extract not found at {original_raw}")
        print(f"Parsing {rdf_path} -> {original_raw}")
        write_extracted_csv(
            Path(rdf_path),
            original_raw,
            index_mode=spec.get("index_mode", "mireg"),
        )

    original = pd.read_csv(original_raw)
    corrected, n_updated, n_removed = apply_tabulated_corrections(
        original,
        name,
        source=original_raw,
        require_all_indices=True,
    )
    _write_csv(corrected, raw_csv)
    print(
        f"{name}: wrote {len(corrected)} corrected reactions "
        f"({n_updated} updated, {n_removed} removed) to {raw_csv}"
    )

    if include_processed and original_processed.exists():
        processed_original = pd.read_csv(original_processed)
        processed_corrected, n_updated, n_removed = apply_tabulated_corrections(
            processed_original,
            name,
            source=original_processed,
            require_all_indices=False,
        )
        _write_csv(processed_corrected, processed_csv)
        print(
            f"{name}: wrote {len(processed_corrected)} corrected processed rows "
            f"({n_updated} updated, {n_removed} removed) to {processed_csv}"
        )

    if include_results and original_results.exists():
        results_original = pd.read_csv(original_results)
        results_corrected, n_updated, n_removed = apply_tabulated_corrections(
            results_original,
            name,
            source=original_results,
            require_all_indices=False,
        )
        _write_csv(results_corrected, results_csv)
        print(
            f"{name}: wrote {len(results_corrected)} corrected result rows "
            f"({n_updated} updated, {n_removed} removed) to {results_csv}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse Gimadiev RDF files into original extracts and apply "
            "data_to_correct.csv onto the corrected tables used by the benchmarks."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        action="append",
        help=(
            "Dataset to prepare (repeatable). Default: main, test_set_1, and test_set_2. "
            "'extracted' is an alias for main (MOESM4)."
        ),
    )
    parser.add_argument(
        "--from-rdf",
        action="store_true",
        help="Re-parse RDF files into original/ even when an original extract already exists.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    requested = args.dataset or list(CANONICAL_DATASET_NAMES)
    seen: set[str] = set()
    for name in requested:
        spec = DATASETS[name]
        canonical = spec["name"]
        if canonical in seen:
            continue
        seen.add(canonical)
        prepare_dataset(spec, from_rdf=args.from_rdf)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Grid visualization of ampholyte / zwitterion benchmark molecules.

Reads summary CSVs from ``benchmark_fzwit.py`` or ``benchmark_amino_acids.py``,
draws RDKit structures, and annotates each panel with:

- molecule name (above)
- ``f_z^PEACE`` and ``f_z^exp`` (below)

Outputs high-DPI PNG and a vector SVG (RDKit MolDraw2DSVG panels + SVG text).
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FZWIT_CSV = (
    SCRIPT_DIR / "results" / "f_zwit_benchmark" / "benchmark_results.csv"
)
DEFAULT_AA_NONAQ_CSV = (
    SCRIPT_DIR / "results" / "amino_acids" / "amino_acids_nonaq_summary.csv"
)
DEFAULT_AA_AQ_CSV = (
    SCRIPT_DIR / "results" / "amino_acids" / "amino_acids_aq_summary.csv"
)

# Panel geometry (SVG units / PNG inches share the same aspect).
MOL_W = 320
MOL_H = 260
TITLE_H = 24
FOOTER_H = 12
PAD = 6
DEFAULT_NCOLS = 6
PNG_DPI = 300


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot RDKit molecule grids for ampholyte zwitterion benchmark summaries."
        )
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        action="append",
        dest="summary_csvs",
        default=None,
        help=(
            "Summary CSV to visualize (repeatable). Defaults to f_zwit benchmark "
            "and amino-acid nonaq (plus aq if present)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside each summary CSV).",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=DEFAULT_NCOLS,
        help=f"Molecules per row (default: {DEFAULT_NCOLS}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=PNG_DPI,
        help=f"PNG DPI (default: {PNG_DPI}).",
    )
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip SVG export.",
    )
    return parser


def _default_summary_csvs() -> list[Path]:
    paths = [DEFAULT_FZWIT_CSV, DEFAULT_AA_NONAQ_CSV]
    if DEFAULT_AA_AQ_CSV.exists():
        paths.append(DEFAULT_AA_AQ_CSV)
    return paths


def normalize_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Unify column names across f_zwit and amino-acid summary CSVs."""
    out = df.copy()
    rename: dict[str, str] = {}
    if "predicted_f_zwit" in out.columns and "f_zwit_pred" not in out.columns:
        rename["predicted_f_zwit"] = "f_zwit_pred"
    if "experimental_f_zwit" in out.columns and "f_zwit_exp" not in out.columns:
        rename["experimental_f_zwit"] = "f_zwit_exp"
    if rename:
        out = out.rename(columns=rename)

    required = {"molecule", "SMILES"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Summary CSV missing required columns: {sorted(missing)}")
    if "f_zwit_pred" not in out.columns:
        out["f_zwit_pred"] = pd.NA
    if "f_zwit_exp" not in out.columns:
        out["f_zwit_exp"] = pd.NA
    return out


def _fmt_frac(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    if x == 0.0:
        return "0"
    if abs(x) >= 0.01 or x == 0.0:
        return f"{x:.3f}"
    return f"{x:.2e}"


def _panel_title(row: pd.Series) -> str:
    name = str(row["molecule"])
    solvent = str(row["solvent"]) if "solvent" in row.index and pd.notna(row["solvent"]) else ""
    title = name if solvent == "water" else f"{name} ({solvent})"
    # Soft-wrap long names. Disable hyphen-based breaks so names like
    # ``a-dimethyl...`` are not split after the leading ``a-``.
    wrapper = textwrap.TextWrapper(
        width=30,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return "\n".join(wrapper.wrap(title)) or title


def _panel_labels(row: pd.Series) -> tuple[str, str]:
    title = _panel_title(row)
    f_pred = _fmt_frac(row.get("f_zwit_pred"))
    f_exp = _fmt_frac(row.get("f_zwit_exp"))
    footer = (
        rf"$f_{{\mathrm{{z}}}}^{{\mathrm{{PEACE}}}}={f_pred}$"
        "\n"
        rf"$f_{{\mathrm{{z}}}}^{{\mathrm{{exp}}}}={f_exp}$"
    )
    return title, footer


def _footer_plain(row: pd.Series) -> list[str]:
    """Plain-text footer lines for SVG (no matplotlib mathtext)."""
    f_pred = _fmt_frac(row.get("f_zwit_pred"))
    f_exp = _fmt_frac(row.get("f_zwit_exp"))
    return [
        f"f_z^PEACE = {f_pred}",
        f"f_z^exp = {f_exp}",
    ]


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def _prepare_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in normalize_summary(df).iterrows():
        title, footer_mpl = _panel_labels(row)
        rows.append(
            {
                "title": title,
                "footer_mpl": footer_mpl,
                "footer_lines": _footer_plain(row),
                "mol": _mol_from_smiles(row["SMILES"]),
                "smiles": str(row["SMILES"]),
            }
        )
    return rows


def _output_stem(summary_csv: Path) -> str:
    stem = summary_csv.stem
    # amino_acids_nonaq_summary -> amino_acids_nonaq_molecules
    if stem.endswith("_summary"):
        stem = stem[: -len("_summary")]
    if stem == "benchmark_results":
        return "f_zwit_benchmark_molecules"
    return f"{stem}_molecules"


def plot_png(
    panels: list[dict],
    *,
    out_path: Path,
    ncols: int,
    dpi: int,
) -> None:
    n = len(panels)
    if n == 0:
        raise ValueError("No molecules to plot.")
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    # inches: keep molecule area roughly square-ish at high DPI
    cell_w = 2.6
    cell_h = 2.9
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows), dpi=dpi)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.30, hspace=0.55)

    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        title_fontsize = 8 if "\n" in panel["title"] or len(panel["title"]) > 24 else 9
        ax.set_title(panel["title"], fontsize=title_fontsize, pad=6)
        mol = panel["mol"]
        if mol is None:
            ax.text(
                0.5,
                0.55,
                "invalid SMILES",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#999999",
            )
        else:
            img = Draw.MolToImage(mol, size=(MOL_W, MOL_H))
            ax.imshow(img)
            ax.set_xlim(0, img.size[0])
            ax.set_ylim(img.size[1], 0)

        ax.text(
            0.5,
            -0.02,
            panel["footer_mpl"],
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            linespacing=1.35,
        )

    # Hide unused axes
    for j in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _strip_svg_xml_decl(svg_text: str) -> str:
    return re.sub(r"<\?xml[^>]*\?>", "", svg_text).strip()


def _mol_to_nested_svg(mol: Chem.Mol | None, width: int, height: int) -> str:
    """Return a self-contained nested ``<svg>`` fragment for embedding."""
    if mol is None:
        return (
            f'<svg width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
            f'fill="#999999" font-size="14" font-family="sans-serif">'
            f"invalid SMILES</text></svg>"
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return _strip_svg_xml_decl(drawer.GetDrawingText())


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def plot_svg(
    panels: list[dict],
    *,
    out_path: Path,
    ncols: int,
) -> None:
    """Compose a multi-panel SVG with vector RDKit drawings + text labels."""
    n = len(panels)
    if n == 0:
        raise ValueError("No molecules to plot.")
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    cell_w = MOL_W + 2 * PAD
    cell_h = TITLE_H + MOL_H + FOOTER_H + 2 * PAD
    total_w = ncols * cell_w
    total_h = nrows * cell_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for i, panel in enumerate(panels):
        r, c = divmod(i, ncols)
        x0 = c * cell_w
        y0 = r * cell_h

        # Title (may be multi-line)
        title_lines = panel["title"].split("\n")
        title_y0 = y0 + PAD + 14
        for j, line in enumerate(title_lines):
            parts.append(
                f'<text x="{x0 + cell_w/2}" y="{title_y0 + j * 14}" '
                f'text-anchor="middle" font-size="12" font-family="sans-serif" '
                f'font-weight="600">{_escape_xml(line)}</text>'
            )

        # Molecule as nested SVG (keeps RDKit vectors + namespaces intact)
        mol_x = x0 + (cell_w - MOL_W) / 2
        mol_y = y0 + PAD + TITLE_H
        nested = _mol_to_nested_svg(panel["mol"], MOL_W, MOL_H)
        parts.append(f'<g transform="translate({mol_x},{mol_y})">{nested}</g>')

        # Footer
        footer_y = mol_y + MOL_H + 16
        for j, line in enumerate(panel["footer_lines"]):
            parts.append(
                f'<text x="{x0 + cell_w/2}" y="{footer_y + j * 14}" '
                f'text-anchor="middle" font-size="11" font-family="sans-serif">'
                f"{_escape_xml(line)}</text>"
            )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def visualize_summary(
    summary_csv: Path,
    *,
    output_dir: Path | None,
    ncols: int,
    dpi: int,
    write_svg: bool,
) -> list[Path]:
    summary_csv = summary_csv.resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    panels = _prepare_rows(df)
    out_dir = (output_dir or summary_csv.parent).resolve()
    stem = _output_stem(summary_csv)

    written: list[Path] = []
    png_path = out_dir / f"{stem}.png"
    plot_png(panels, out_path=png_path, ncols=ncols, dpi=dpi)
    written.append(png_path)
    print(f"Wrote {png_path}")

    if write_svg:
        svg_path = out_dir / f"{stem}.svg"
        plot_svg(panels, out_path=svg_path, ncols=ncols)
        written.append(svg_path)
        print(f"Wrote {svg_path}")

    return written


def main() -> None:
    args = _build_parser().parse_args()
    csvs = args.summary_csvs or _default_summary_csvs()
    written_any = False
    for csv_path in csvs:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"Skipping missing summary CSV: {csv_path}")
            continue
        visualize_summary(
            csv_path,
            output_dir=args.output_dir,
            ncols=args.ncols,
            dpi=args.dpi,
            write_svg=not args.no_svg,
        )
        written_any = True
    if not written_any:
        raise SystemExit("No summary CSVs found to visualize.")


if __name__ == "__main__":
    main()

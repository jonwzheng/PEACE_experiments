#!/usr/bin/env python3
"""Grid of SAMPL6 charge-0 observed microstates.

Reads ``observed_microstates.csv``, keeps charge = 0 rows, and draws each
structure with its molecule ID (e.g. SM08). No energies, opacity, or
PEACE filtering.

Outputs high-DPI PNG and a vector SVG.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_CSV = SCRIPT_DIR / "observed_microstates.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "microstate_benchmark" / "figures"

MOL_ID_COL = "SAMPL6 Molecule ID"
MOL_W = 280
MOL_H = 220
TITLE_H = 28
FOOTER_H = 0
PAD = 10
DEFAULT_NCOLS = 4
PNG_DPI = 300


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot SAMPL6 charge-0 observed microstates in an RDKit grid."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Observed microstates CSV (default: observed_microstates.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG/SVG outputs.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="sampl6_charge0_grid",
        help="Output filename stem.",
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
    parser.add_argument(
        "--molecule",
        action="append",
        dest="molecules",
        default=None,
        help="Optional molecule ID filter (repeatable), e.g. --molecule SM08.",
    )
    return parser


def _mol_from_smiles(smiles: object) -> Chem.Mol | None:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None
    text = str(smiles).strip()
    if not text or text.lower() == "nan":
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    try:
        Chem.rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def load_charge0_panels(input_csv: Path, molecules: list[str] | None) -> list[dict]:
    df = pd.read_csv(input_csv)
    required = {MOL_ID_COL, "rdkit_canon_smi", "charge"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_csv} missing columns: {sorted(missing)}")

    df0 = df[df["charge"].astype(int) == 0].copy()
    if molecules:
        wanted = {m.strip().upper() for m in molecules}
        df0 = df0[df0[MOL_ID_COL].astype(str).str.upper().isin(wanted)].copy()
        if df0.empty:
            raise ValueError(f"No charge-0 rows matched --molecule: {sorted(wanted)}")

    df0 = df0.sort_values(MOL_ID_COL, kind="mergesort").reset_index(drop=True)

    panels: list[dict] = []
    for _, row in df0.iterrows():
        mol_id = str(row[MOL_ID_COL])
        panels.append(
            {
                "title": mol_id,
                "mol": _mol_from_smiles(row["rdkit_canon_smi"]),
            }
        )
    return panels


def plot_png(
    panels: list[dict],
    *,
    out_path: Path,
    ncols: int,
    dpi: int,
) -> None:
    n = len(panels)
    if n == 0:
        raise ValueError("No charge-0 molecules to plot.")
    ncols = max(1, min(ncols, n))
    nrows = math.ceil(n / ncols)

    cell_w = 2.6
    cell_h = 2.7
    fig = plt.figure(figsize=(cell_w * ncols, cell_h * nrows), dpi=dpi)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.25, hspace=0.40)

    for i, panel in enumerate(panels):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(panel["title"], fontsize=12, pad=6, fontweight="bold")
        mol = panel["mol"]
        if mol is None:
            ax.text(
                0.5,
                0.5,
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

    for j in range(n, nrows * ncols):
        ax = fig.add_subplot(gs[j // ncols, j % ncols])
        ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _strip_svg_xml_decl(svg_text: str) -> str:
    return re.sub(r"<\?xml[^>]*\?>", "", svg_text).strip()


def _mol_to_nested_svg(mol: Chem.Mol | None, width: int, height: int) -> str:
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
    n = len(panels)
    if n == 0:
        raise ValueError("No charge-0 molecules to plot.")
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

        parts.append(
            f'<text x="{x0 + cell_w/2}" y="{y0 + PAD + 16}" '
            f'text-anchor="middle" font-size="14" font-family="sans-serif" '
            f'font-weight="700">{_escape_xml(panel["title"])}</text>'
        )

        mol_x = x0 + (cell_w - MOL_W) / 2
        mol_y = y0 + PAD + TITLE_H
        nested = _mol_to_nested_svg(panel["mol"], MOL_W, MOL_H)
        parts.append(f'<g transform="translate({mol_x},{mol_y})">{nested}</g>')

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV not found: {input_csv}")

    panels = load_charge0_panels(input_csv, args.molecules)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = max(1, int(args.ncols))
    stem = args.stem

    png_path = out_dir / f"{stem}.png"
    plot_png(panels, out_path=png_path, ncols=ncols, dpi=args.dpi)
    print(f"Wrote {png_path} ({len(panels)} molecules)")

    if not args.no_svg:
        svg_path = out_dir / f"{stem}.svg"
        plot_svg(panels, out_path=svg_path, ncols=ncols)
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()

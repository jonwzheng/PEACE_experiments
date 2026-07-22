#!/usr/bin/env python3
"""Visualize SAMPL7 PEACE microstate rankings as a charge-column grid.

Reads ``benchmark_microstates_results.csv`` from ``benchmark_microstates.py`` and
draws PEACE-identified protomers in a 3-column layout (charges −1 / 0 / +1):

- One output figure per parent molecule (SM25, …).
- Within each charge column, protomers are placed left→right in PEACE rank
  order, wrapping after ``--per-row`` entries (default 3).
- Charge-0 rel. G values are relative to ``{mol}_micro000`` (= 0) for both PEACE
  and SAMPL7; other charges keep within-charge relative DG (min = 0).
- Full-opacity structure + dark text if the protomer is in both PEACE and SAMPL7;
  PEACE-only structures are drawn at half opacity with solid dark-blue labels.
- Labels: PEACE / SAMPL ranks, rel. G^PEACE, DG^SAMPL.

Outputs high-DPI PNG and a vector SVG per molecule.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image, ImageChops, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_CSV = (
    SCRIPT_DIR / "results" / "microstate_benchmark" / "benchmark_microstates_results.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "microstate_benchmark" / "figures"

CHARGES = (-1, 0, 1)
CHARGE_COLORS = {
    -1: ("#e8f1fa", "#c5d9ef"),  # column fill, header fill
    0: ("#f3f3f3", "#dddddd"),
    1: ("#f8ebe3", "#edd4c4"),
}
# PEACE-only label colors tuned to each charge column's theme.
PEACE_ONLY_TEXT_COLORS = {
    -1: "#4575FF",  # blue
    0: "#5A5A5A",  # gray
    1: "#C45C26",  # orange
}

LABEL_W = 56
HEADER_H = 48
# Per-protomer cell (sized so several fit comfortably in a charge column).
MOL_W = 220
MOL_H = 160
FOOTER_H = 140
CELL_PAD = 10
PANEL_INSET = 8
CELL_W = MOL_W + 2 * CELL_PAD
CELL_H = MOL_H + FOOTER_H + PANEL_INSET
FOOTER_FONT = 20
FOOTER_SUP_FONT = 14
HEADER_FONT = 22
MOL_LABEL_FONT = 22
FOOTER_LINE_H = 24
# Extra gap between rank lines (PEACE/SAMPL) and rel. G lines (superscripts need room).
FOOTER_ENERGY_GAP = 12
DEFAULT_PER_ROW = 2

BOTH_OPACITY = 1.0
PEACE_ONLY_OPACITY = 0.5
BOTH_TEXT_COLOR = "#222222"
PNG_DPI_SCALE = 2  # rasterize at 2× SVG logical pixels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize SAMPL7 PEACE microstate rankings by charge."
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Aggregate CSV from benchmark_microstates.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for per-molecule PNG/SVG outputs.",
    )
    parser.add_argument(
        "--per-row",
        type=int,
        default=DEFAULT_PER_ROW,
        help=(
            "Max protomers side-by-side within each charge column before wrapping "
            f"(default: {DEFAULT_PER_ROW})."
        ),
    )
    parser.add_argument(
        "--no-svg",
        action="store_true",
        help="Skip SVG export.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Skip PNG export.",
    )
    parser.add_argument(
        "--molecule",
        action="append",
        dest="molecules",
        default=None,
        help="Optional molecule ID filter (repeatable), e.g. --molecule SM25.",
    )
    return parser


def _panel_text_color(*, in_both: bool, charge: int) -> str:
    if in_both:
        return BOTH_TEXT_COLOR
    return PEACE_ONLY_TEXT_COLORS.get(charge, BOTH_TEXT_COLOR)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "t"}


def _fmt_rank(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    try:
        if pd.isna(value):
            return "—"
        return str(int(value))
    except (TypeError, ValueError):
        return "—"


def _fmt_dg(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    return f"{value:.2f}"


def _optional_float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        if pd.isna(value):
            return None
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def _is_micro000(micro_id: object, mol_id: str) -> bool:
    if micro_id is None or (isinstance(micro_id, float) and math.isnan(micro_id)):
        return False
    text = str(micro_id)
    if pd.isna(micro_id) or text.lower() == "nan":
        return False
    return text == f"{mol_id}_micro000" or text.endswith("_micro000")


@dataclass
class ProtomerPanel:
    molecule_id: str
    charge: int
    smiles: str
    peace_rank: int
    sampl_rank: object
    delta_g_peace: float | None
    delta_g_sampl: float | None
    in_both: bool
    mol: Chem.Mol | None


def add_display_energies(df: pd.DataFrame) -> pd.DataFrame:
    """Add charge-0 energies re-zeroed to micro000 for PEACE (and SAMPL if needed).

    Charge-0: both PEACE and SAMPL rel. G are relative to ``{mol}_micro000`` (= 0).
    Other charges: keep within-charge relative energies from the benchmark CSV
    (min = 0), since micro000 is a charge-0 microstate.
    """
    out = df.copy()
    out["display_delta_g_peace"] = pd.to_numeric(
        out["peace_delta_g_kcal_mol"], errors="coerce"
    )
    if "sampl7_delta_g_kcal_mol" in out.columns:
        out["display_delta_g_sampl"] = pd.to_numeric(
            out["sampl7_delta_g_kcal_mol"], errors="coerce"
        )
    elif "sampl7_average_fe_prediction" in out.columns:
        out["display_delta_g_sampl"] = pd.to_numeric(
            out["sampl7_average_fe_prediction"], errors="coerce"
        )
    else:
        out["display_delta_g_sampl"] = pd.NA

    for mol_id, group in out.groupby("molecule_id", sort=False):
        mask0 = group["charge"].astype(int) == 0
        if not mask0.any():
            continue

        micro0 = group.loc[
            mask0
            & group["sampl7_microstate_id"].map(lambda x: _is_micro000(x, str(mol_id)))
        ]
        idx0 = group.index[mask0]

        # PEACE: shift so the PEACE energy of micro000 is 0 (when present in PEACE).
        micro0_peace = micro0[micro0["in_peace"].map(_as_bool)]
        if not micro0_peace.empty:
            ref_peace = pd.to_numeric(
                micro0_peace.iloc[0]["peace_delta_g_kcal_mol"], errors="coerce"
            )
            if pd.notna(ref_peace):
                out.loc[idx0, "display_delta_g_peace"] = (
                    pd.to_numeric(out.loc[idx0, "peace_delta_g_kcal_mol"], errors="coerce")
                    - float(ref_peace)
                )

        # SAMPL: prefer stored sampl7_delta_g (already vs micro000 at charge 0).
        # If only raw FE is available, re-zero to micro000's FE.
        if (
            "sampl7_delta_g_kcal_mol" not in out.columns
            and "sampl7_average_fe_prediction" in out.columns
            and not micro0.empty
        ):
            ref_sampl = pd.to_numeric(
                micro0.iloc[0]["sampl7_average_fe_prediction"], errors="coerce"
            )
            if pd.notna(ref_sampl):
                out.loc[idx0, "display_delta_g_sampl"] = (
                    pd.to_numeric(
                        out.loc[idx0, "sampl7_average_fe_prediction"], errors="coerce"
                    )
                    - float(ref_sampl)
                )

    return out


def prepare_panels(df: pd.DataFrame) -> dict[str, dict[int, list[ProtomerPanel]]]:
    """molecule_id -> charge -> panels in PEACE rank order."""
    work = df[df["in_peace"].map(_as_bool)].copy()
    work = add_display_energies(work)

    by_mol: dict[str, dict[int, list[ProtomerPanel]]] = {}
    mol_order = list(dict.fromkeys(work["molecule_id"].astype(str).tolist()))

    for mol_id in mol_order:
        mol_df = work[work["molecule_id"].astype(str) == mol_id]
        charge_map: dict[int, list[ProtomerPanel]] = {c: [] for c in CHARGES}
        for charge in CHARGES:
            sub = mol_df[mol_df["charge"].astype(int) == charge].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("peace_rank", ascending=True, kind="mergesort")
            for _, row in sub.iterrows():
                in_sampl = _as_bool(row.get("in_sampl7"))
                charge_map[charge].append(
                    ProtomerPanel(
                        molecule_id=mol_id,
                        charge=int(charge),
                        smiles=str(row["canon_smiles"]),
                        peace_rank=int(row["peace_rank"]),
                        sampl_rank=row.get("sampl7_rank"),
                        delta_g_peace=_optional_float(row.get("display_delta_g_peace")),
                        delta_g_sampl=_optional_float(row.get("display_delta_g_sampl")),
                        in_both=in_sampl,
                        mol=_mol_from_smiles(row["canon_smiles"]),
                    )
                )
        by_mol[mol_id] = charge_map
    return by_mol


def _footer_rank_lines(panel: ProtomerPanel) -> list[str]:
    return [
        f"PEACE: {panel.peace_rank}",
        f"SAMPL: {_fmt_rank(panel.sampl_rank)}",
    ]


def _footer_energy_rows(panel: ProtomerPanel) -> list[tuple[str, str]]:
    """Return [(superscript_label, value_str), ...] for rel. G lines."""
    return [
        ("PEACE", _fmt_dg(panel.delta_g_peace)),
        ("SAMPL", _fmt_dg(panel.delta_g_sampl)),
    ]


def _col_w(per_row: int) -> int:
    return max(1, int(per_row)) * CELL_W


def _active_charges(charge_map: dict[int, list[ProtomerPanel]]) -> list[int]:
    """Charges that have at least one PEACE protomer (preserve CHARGES order)."""
    return [c for c in CHARGES if charge_map.get(c)]


def _n_wrap_rows(n_panels: int, per_row: int) -> int:
    per_row = max(1, int(per_row))
    if n_panels <= 0:
        return 0
    return math.ceil(n_panels / per_row)


def _molecule_rows(
    charge_map: dict[int, list[ProtomerPanel]],
    per_row: int,
    *,
    charges: list[int] | None = None,
) -> int:
    use = charges if charges is not None else _active_charges(charge_map)
    if not use:
        return 0
    return max(_n_wrap_rows(len(charge_map[c]), per_row) for c in use)


def _layout_one(
    charge_map: dict[int, list[ProtomerPanel]],
    *,
    per_row: int,
) -> tuple[int, int, int, list[int]]:
    """Return (n_rows, canvas_width, canvas_height, active_charges)."""
    charges = _active_charges(charge_map)
    col_w = _col_w(per_row)
    total_w = LABEL_W + max(len(charges), 1) * col_w
    n_rows = _molecule_rows(charge_map, per_row, charges=charges)
    total_h = HEADER_H + n_rows * CELL_H
    return n_rows, total_w, total_h, charges


def _strip_svg_xml_decl(svg_text: str) -> str:
    return re.sub(r"<\?xml[^>]*\?>", "", svg_text).strip()


def _mol_to_nested_svg(mol: Chem.Mol | None, width: int, height: int) -> str:
    if mol is None:
        return (
            f'<svg width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<text x="{width/2}" y="{height/2}" text-anchor="middle" '
            f'fill="#999999" font-size="12" font-family="sans-serif">'
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


def _column_x(charge: int, per_row: int, active_charges: list[int]) -> int:
    return LABEL_W + active_charges.index(charge) * _col_w(per_row)


def plot_svg_molecule(
    mol_id: str,
    charge_map: dict[int, list[ProtomerPanel]],
    *,
    out_path: Path,
    per_row: int,
) -> None:
    per_row = max(1, int(per_row))
    n_rows, total_w, total_h, active = _layout_one(charge_map, per_row=per_row)
    if n_rows == 0 or not active:
        raise ValueError(f"No PEACE protomers for {mol_id}")
    col_w = _col_w(per_row)
    content_y0 = HEADER_H

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{total_w}" height="{total_h}" '
        f'viewBox="0 0 {total_w} {total_h}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for charge in active:
        x = _column_x(charge, per_row, active)
        fill, header = CHARGE_COLORS[charge]
        parts.append(
            f'<rect x="{x}" y="0" width="{col_w}" height="{total_h}" fill="{fill}"/>'
        )
        parts.append(
            f'<rect x="{x + 4}" y="6" width="{col_w - 8}" height="36" rx="6" '
            f'fill="{header}" stroke="#888888" stroke-width="0.8"/>'
        )
        label = f"charge {charge:+d}" if charge != 0 else "charge 0"
        parts.append(
            f'<text x="{x + col_w/2}" y="30" text-anchor="middle" '
            f'font-size="{HEADER_FONT}" font-family="sans-serif" font-weight="700">'
            f"{_escape_xml(label)}</text>"
        )

    label_y = content_y0 + (n_rows * CELL_H) / 2
    parts.append(
        f'<text x="{LABEL_W/2}" y="{label_y}" text-anchor="middle" '
        f'dominant-baseline="middle" font-size="{MOL_LABEL_FONT}" '
        f'font-family="sans-serif" '
        f'font-weight="700" transform="rotate(-90 {LABEL_W/2} {label_y})">'
        f"{_escape_xml(mol_id)}</text>"
    )

    for charge in active:
        panels = charge_map[charge]
        col_x = _column_x(charge, per_row, active)
        for idx, panel in enumerate(panels):
            wrap_row, wrap_col = divmod(idx, per_row)
            cell_x = col_x + wrap_col * CELL_W
            cell_y = content_y0 + wrap_row * CELL_H
            opacity = BOTH_OPACITY if panel.in_both else PEACE_ONLY_OPACITY
            text_color = _panel_text_color(in_both=panel.in_both, charge=charge)
            mol_x = cell_x + CELL_PAD
            mol_y = cell_y + PANEL_INSET
            nested = _mol_to_nested_svg(panel.mol, MOL_W, MOL_H)
            parts.append(
                f'<g opacity="{opacity:.3f}" '
                f'transform="translate({mol_x},{mol_y})">{nested}</g>'
            )

            cx = cell_x + CELL_W / 2
            footer_y = mol_y + MOL_H + 18
            for j, line in enumerate(_footer_rank_lines(panel)):
                parts.append(
                    f'<text x="{cx}" y="{footer_y + j * FOOTER_LINE_H}" '
                    f'text-anchor="middle" font-size="{FOOTER_FONT}" '
                    f'font-family="sans-serif" fill="{text_color}">'
                    f"{_escape_xml(line)}</text>"
                )
            # rel. G with proper SVG superscripts: rel. G^{PEACE}, rel. G^{SAMPL}
            energy_y0 = footer_y + 2 * FOOTER_LINE_H + FOOTER_ENERGY_GAP
            for j, (sup, value) in enumerate(_footer_energy_rows(panel)):
                y = energy_y0 + j * FOOTER_LINE_H
                parts.append(
                    f'<text x="{cx}" y="{y}" text-anchor="middle" '
                    f'font-size="{FOOTER_FONT}" font-family="sans-serif" '
                    f'fill="{text_color}">'
                    f"rel. G"
                    f'<tspan baseline-shift="super" font-size="{FOOTER_SUP_FONT}">'
                    f"{_escape_xml(sup)}</tspan>"
                    f" = {_escape_xml(value)}"
                    f"</text>"
                )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _load_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_bold_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return _load_font(size)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _whitish_to_transparent(img: Image.Image, threshold: int = 245) -> Image.Image:
    rgba = img.convert("RGBA")
    bands = rgba.split()
    mask = ImageChops.lighter(
        ImageChops.lighter(
            bands[0].point(lambda v: 255 if v < threshold else 0),
            bands[1].point(lambda v: 255 if v < threshold else 0),
        ),
        bands[2].point(lambda v: 255 if v < threshold else 0),
    )
    rgba.putalpha(mask)
    return rgba


def _apply_opacity_on_bg(
    img: Image.Image,
    opacity: float,
    bg_rgb: tuple[int, int, int],
) -> Image.Image:
    rgba = _whitish_to_transparent(img)
    if opacity < 0.999:
        alpha = rgba.getchannel("A").point(lambda a: int(a * opacity))
        rgba.putalpha(alpha)
    base = Image.new("RGBA", rgba.size, (*bg_rgb, 255))
    return Image.alpha_composite(base, rgba).convert("RGB")


def plot_png_molecule(
    mol_id: str,
    charge_map: dict[int, list[ProtomerPanel]],
    *,
    out_path: Path,
    per_row: int,
    scale: int = PNG_DPI_SCALE,
) -> None:
    per_row = max(1, int(per_row))
    n_rows, total_w, total_h, active = _layout_one(charge_map, per_row=per_row)
    if n_rows == 0 or not active:
        raise ValueError(f"No PEACE protomers for {mol_id}")
    col_w = _col_w(per_row)
    content_y0 = HEADER_H
    s = max(1, int(scale))

    canvas = Image.new("RGB", (total_w * s, total_h * s), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(FOOTER_FONT * s)
    font_bold = _load_bold_font(HEADER_FONT * s)
    font_label = _load_bold_font(MOL_LABEL_FONT * s)

    def sx(v: float) -> int:
        return int(round(v * s))

    for charge in active:
        x = _column_x(charge, per_row, active)
        fill, header = CHARGE_COLORS[charge]
        draw.rectangle([sx(x), 0, sx(x + col_w) - 1, sx(total_h) - 1], fill=fill)
        draw.rounded_rectangle(
            [sx(x + 4), sx(6), sx(x + col_w - 4) - 1, sx(42) - 1],
            radius=6 * s,
            fill=header,
            outline="#888888",
            width=max(1, s),
        )
        label = f"charge {charge:+d}" if charge != 0 else "charge 0"
        tw = draw.textlength(label, font=font_bold)
        draw.text(
            (sx(x + col_w / 2) - tw / 2, sx(10)),
            label,
            font=font_bold,
            fill="#222222",
        )

    block_h = n_rows * CELL_H
    label_img = Image.new("RGBA", (sx(block_h), sx(LABEL_W)), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_img)
    tw = label_draw.textlength(mol_id, font=font_label)
    label_draw.text(
        ((sx(block_h) - tw) / 2, (sx(LABEL_W) - MOL_LABEL_FONT * s) / 2),
        mol_id,
        font=font_label,
        fill="#222222",
    )
    rotated = label_img.rotate(90, expand=True)
    canvas.paste(rotated, (0, sx(content_y0)), rotated)

    for charge in active:
        panels = charge_map[charge]
        col_x = _column_x(charge, per_row, active)
        bg_rgb = _hex_to_rgb(CHARGE_COLORS[charge][0])
        for idx, panel in enumerate(panels):
            wrap_row, wrap_col = divmod(idx, per_row)
            cell_x = col_x + wrap_col * CELL_W
            cell_y = content_y0 + wrap_row * CELL_H
            opacity = BOTH_OPACITY if panel.in_both else PEACE_ONLY_OPACITY
            text_color = _panel_text_color(in_both=panel.in_both, charge=charge)
            mol_x = cell_x + CELL_PAD
            mol_y = cell_y + PANEL_INSET

            if panel.mol is None:
                cell = Image.new("RGB", (MOL_W * s, MOL_H * s), bg_rgb)
                cd = ImageDraw.Draw(cell)
                msg = "invalid SMILES"
                mw = cd.textlength(msg, font=font)
                cd.text(
                    ((MOL_W * s - mw) / 2, MOL_H * s / 2),
                    msg,
                    font=font,
                    fill="#999999",
                )
            else:
                cell = Draw.MolToImage(panel.mol, size=(MOL_W * s, MOL_H * s))
                cell = _apply_opacity_on_bg(cell, opacity, bg_rgb)
            canvas.paste(cell, (sx(mol_x), sx(mol_y)))

            cx = cell_x + CELL_W / 2
            footer_y = mol_y + MOL_H + 18
            for j, line in enumerate(_footer_rank_lines(panel)):
                tw = draw.textlength(line, font=font)
                draw.text(
                    (sx(cx) - tw / 2, sx(footer_y + j * FOOTER_LINE_H)),
                    line,
                    font=font,
                    fill=text_color,
                )
            # Approximate superscripts in raster text (PIL has no mathtext).
            font_sup = _load_font(FOOTER_SUP_FONT * s)
            energy_y0 = footer_y + 2 * FOOTER_LINE_H + FOOTER_ENERGY_GAP
            for j, (sup, value) in enumerate(_footer_energy_rows(panel)):
                y = energy_y0 + j * FOOTER_LINE_H
                # Compose "rel. G" + raised "PEACE"/"SAMPL" + " = value"
                prefix = "rel. G"
                mid = sup
                suffix = f" = {value}"
                w_prefix = draw.textlength(prefix, font=font)
                w_mid = draw.textlength(mid, font=font_sup)
                w_suffix = draw.textlength(suffix, font=font)
                total = w_prefix + w_mid + w_suffix
                x = sx(cx) - total / 2
                draw.text((x, sx(y)), prefix, font=font, fill=text_color)
                draw.text(
                    (x + w_prefix, sx(y) - 8 * s),
                    mid,
                    font=font_sup,
                    fill=text_color,
                )
                draw.text(
                    (x + w_prefix + w_mid, sx(y)),
                    suffix,
                    font=font,
                    fill=text_color,
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG", dpi=(150 * s, 150 * s))


def main() -> None:
    args = _build_parser().parse_args()
    results_csv = args.results_csv.resolve()
    if not results_csv.exists():
        raise SystemExit(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv)
    required = {
        "molecule_id",
        "charge",
        "canon_smiles",
        "peace_rank",
        "peace_delta_g_kcal_mol",
        "in_peace",
        "in_sampl7",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Results CSV missing columns: {sorted(missing)}")

    if args.molecules:
        wanted = {m.strip().upper() for m in args.molecules}
        df = df[df["molecule_id"].astype(str).str.upper().isin(wanted)].copy()
        if df.empty:
            raise SystemExit(f"No rows matched --molecule filters: {sorted(wanted)}")

    by_mol = prepare_panels(df)
    if not by_mol:
        raise SystemExit("No PEACE protomers found to visualize.")

    per_row = max(1, int(args.per_row))
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for mol_id, charge_map in by_mol.items():
        if not args.no_svg:
            svg_path = out_dir / f"{mol_id}_microstates.svg"
            plot_svg_molecule(
                mol_id, charge_map, out_path=svg_path, per_row=per_row
            )
            print(f"Wrote {svg_path}")
        if not args.no_png:
            png_path = out_dir / f"{mol_id}_microstates.png"
            plot_png_molecule(
                mol_id, charge_map, out_path=png_path, per_row=per_row
            )
            print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()

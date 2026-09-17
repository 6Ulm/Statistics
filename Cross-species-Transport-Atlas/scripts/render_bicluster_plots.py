#!/usr/bin/env python3
"""Render the supplied reference block-diagonal category plot for every rho asset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


CATEGORY_COLORS = [
    "#0f9a9a", "#d62728", "#1f5fbf", "#2ca02c", "#7b3fb0",
    "#e67e00", "#d6339a", "#8c564b", "#9a9a00", "#00a0e0",
]
GUIDE = "0.3"
FS = 6.5
LH = FS * 1.45 / 72
PAD = 0.09
AX_W = AX_H = 9.0
BRACKET, CONN = 0.15, 0.35

mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})


def spread_intervals(centers, sizes, gap, lo, hi):
    values = np.asarray(centers, float).copy()
    extents = np.asarray(sizes, float)
    for _ in range(500):
        moved = False
        for index in range(1, len(values)):
            needed = (extents[index - 1] + extents[index]) / 2 + gap
            if values[index] - values[index - 1] < needed - 1e-9:
                delta = (needed - (values[index] - values[index - 1])) / 2
                values[index - 1] -= delta
                values[index] += delta
                moved = True
        if len(values):
            if values[0] - extents[0] / 2 < lo:
                values += lo - (values[0] - extents[0] / 2)
                moved = True
            if values[-1] + extents[-1] / 2 > hi and values[0] - extents[0] / 2 > lo + 1e-9:
                values -= min(values[-1] + extents[-1] / 2 - hi, values[0] - extents[0] / 2 - lo)
                moved = True
        if not moved:
            break
    return values


def render(source: Path, categories: list[dict], destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    original_blocks = sorted(payload["biclusters"], key=lambda row: row["transportMass"], reverse=True)

    category_order = list(dict.fromkeys(row["category"].strip() for row in categories))
    category_rank = {category: index for index, category in enumerate(category_order)}
    category_color = {
        category: CATEGORY_COLORS[index] if index < len(CATEGORY_COLORS) else mpl.colormaps["tab20"](index % 20)
        for index, category in enumerate(category_order)
    }
    categories_of: dict[str, list[str]] = {}
    for row in categories:
        gene, category = row["gene"].strip(), row["category"].strip()
        categories_of.setdefault(gene, [])
        if category not in categories_of[gene]:
            categories_of[gene].append(category)
        categories_of[gene.upper()] = categories_of[gene]
    genes_of_interest = list(dict.fromkeys(row["gene"].strip() for row in categories))
    mouse_interest = set(genes_of_interest)
    human_interest = {gene.upper() for gene in genes_of_interest}

    blocks = [
        block for block in original_blocks
        if not mouse_interest.isdisjoint(block["mouseGenes"])
        or not human_interest.isdisjoint(block["humanGenes"])
    ]
    if not blocks:
        return
    omitted = len(original_blocks) - len(blocks)
    y_edges = np.concatenate([[0], np.cumsum([block["nMouse"] for block in blocks])])
    x_edges = np.concatenate([[0], np.cumsum([block["nHuman"] for block in blocks])])
    total_mouse, total_human = int(y_edges[-1]), int(x_edges[-1])
    mouse_pos = {gene: index + 0.5 for index, gene in enumerate(gene for block in blocks for gene in block["mouseGenes"])}
    human_pos = {gene: index + 0.5 for index, gene in enumerate(gene for block in blocks for gene in block["humanGenes"])}
    mouse_rank = {gene: rank for rank, block in enumerate(blocks) for gene in block["mouseGenes"]}
    human_rank = {gene: rank for rank, block in enumerate(blocks) for gene in block["humanGenes"]}
    mouse_goi = [gene for gene in genes_of_interest if gene in mouse_pos]
    human_goi = [gene.upper() for gene in genes_of_interest if gene.upper() in human_pos]
    missing = [gene for gene in genes_of_interest if gene not in mouse_pos]
    cross = {gene for gene in mouse_goi if gene.upper() in human_rank and human_rank[gene.upper()] != mouse_rank[gene]}
    cross_h = {gene.upper() for gene in cross}

    label_kw = {"fontsize": FS, "fontstyle": "italic", "fontweight": "semibold"}
    mark_kw = {"fontsize": FS * 0.8}
    temporary = plt.figure()
    renderer = temporary.canvas.get_renderer()

    def width(text, **kwargs):
        item = temporary.text(0, 0, text, **kwargs)
        result = item.get_window_extent(renderer).width / temporary.dpi
        item.remove()
        return result

    def display(gene, human):
        return gene + ("†" if (gene in cross_h if human else gene in cross) else "")

    name_width = {("m", gene): width(display(gene, False), **label_kw) for gene in mouse_goi}
    name_width.update({("h", gene): width(display(gene, True), **label_kw) for gene in human_goi})
    mark_width = width("■", **mark_kw) + 0.01
    full_width = {
        key: value + (len(categories_of[key[1]]) - 1) * mark_width + (0.01 if len(categories_of[key[1]]) > 1 else 0)
        for key, value in name_width.items()
    }
    plt.close(temporary)
    sort_key = lambda gene: (category_rank[categories_of[gene][0]], gene.lower())

    y_groups = []
    x_groups = []
    for rank in range(len(blocks)):
        genes = sorted([gene for gene in mouse_goi if mouse_rank[gene] == rank], key=sort_key)
        if genes:
            top, bottom = AX_H * (1 - y_edges[rank] / total_mouse), AX_H * (1 - y_edges[rank + 1] / total_mouse)
            columns = 1
            while math.ceil(len(genes) / columns) * LH > max((top - bottom) * 0.92, LH) and columns < 6:
                columns += 1
            rows = math.ceil(len(genes) / columns)
            column_width = max(full_width[("m", gene)] for gene in genes) + PAD
            y_groups.append(dict(k=rank, genes=genes, a=bottom, b=top, cols=columns, rows=rows, size=rows * LH, colw=column_width))
        genes = sorted([gene for gene in human_goi if human_rank[gene] == rank], key=sort_key)
        if genes:
            left, right = AX_W * x_edges[rank] / total_human, AX_W * x_edges[rank + 1] / total_human
            column_width = max(full_width[("h", gene)] for gene in genes) + PAD
            columns = max(1, min(len(genes), int((right - left) * 0.92 // column_width)))
            rows = math.ceil(len(genes) / columns)
            if rows > 14:
                columns = math.ceil(len(genes) / 14)
                rows = math.ceil(len(genes) / columns)
            x_groups.append(dict(k=rank, genes=genes, a=left, b=right, cols=columns, rows=rows, size=columns * column_width, colw=column_width))

    left_margin = BRACKET + CONN + max([group["cols"] * group["colw"] for group in y_groups] + [1]) + 0.9
    bottom_margin = BRACKET + CONN + max([group["rows"] for group in x_groups] + [1]) * LH + 0.85
    right_margin, top_margin, extra_bottom = 4.2, 0.85, 0.55
    figure_width = left_margin + AX_W + right_margin
    figure_height = bottom_margin + AX_H + top_margin + extra_bottom
    figure = plt.figure(figsize=(figure_width, figure_height))
    axis_left, axis_bottom = left_margin, bottom_margin + extra_bottom
    axis = figure.add_axes([axis_left / figure_width, axis_bottom / figure_height, AX_W / figure_width, AX_H / figure_height])

    def figure_point(x, y):
        return (axis_left + x) / figure_width, (axis_bottom + y) / figure_height

    def figure_line(xs, ys, **kwargs):
        points = [figure_point(x, y) for x, y in zip(xs, ys)]
        figure.add_artist(Line2D([point[0] for point in points], [point[1] for point in points], transform=figure.transFigure, **kwargs))

    def figure_text(x, y, text, **kwargs):
        px, py = figure_point(x, y)
        figure.text(px, py, text, **kwargs)

    def label(x, y, gene, human, ha="left", rotation=0):
        gene_categories = categories_of[gene]
        name = display(gene, human)
        base_width = name_width[("h" if human else "m", gene)]
        extra = (len(gene_categories) - 1) * mark_width + (0.01 if len(gene_categories) > 1 else 0)
        if rotation == 0:
            start = x - (base_width + extra) if ha == "right" else x
            figure_text(start, y, name, ha="left", va="center", color=category_color[gene_categories[0]], **label_kw)
            for index, category in enumerate(gene_categories[1:]):
                figure_text(start + base_width + 0.01 + index * mark_width, y, "■", ha="left", va="center", color=category_color[category], **mark_kw)
        else:
            start = y - extra
            figure_text(x, start, name, ha="center", va="top", rotation=90, color=category_color[gene_categories[0]], **label_kw)
            for index, category in enumerate(gene_categories[1:]):
                figure_text(x, y - index * mark_width, "■", ha="center", va="top", rotation=90, color=category_color[category], **mark_kw)

    axis.set_facecolor("#f2f2f2")
    axis.set_xlim(0, total_human)
    axis.set_ylim(total_mouse, 0)
    for edge in y_edges[1:-1]:
        axis.axhline(edge, color="white", lw=0.6, zorder=1)
    for edge in x_edges[1:-1]:
        axis.axvline(edge, color="white", lw=0.6, zorder=1)
    for rank, block in enumerate(blocks):
        axis.add_patch(Rectangle((x_edges[rank], y_edges[rank]), block["nHuman"], block["nMouse"], color="black", lw=0, zorder=2))
        block_width = AX_W * block["nHuman"] / total_human
        block_height = AX_H * block["nMouse"] / total_mouse
        if block_width > 0.55 and block_height > 0.5:
            font_size = 9 if block_width > 1 and block_height > 0.9 else 6.5
            axis.text(x_edges[rank] + block["nHuman"] / 2, y_edges[rank] + block["nMouse"] / 2,
                      f"cluster {block['cluster']}\nmass {block['transportMass']:.3g}\n{block['nMouse']}×{block['nHuman']}",
                      ha="center", va="center", fontsize=font_size, color="white", zorder=3)
    axis.set_xticks([])
    axis.set_yticks([])

    tick = 0.07
    for gene in mouse_goi:
        y = AX_H * (1 - mouse_pos[gene] / total_mouse)
        figure_line([0, -tick], [y, y], color=category_color[categories_of[gene][0]], lw=0.8)
    for gene in human_goi:
        x = AX_W * human_pos[gene] / total_human
        figure_line([x, x], [0, -tick], color=category_color[categories_of[gene][0]], lw=0.8)

    y_centers = spread_intervals([(group["a"] + group["b"]) / 2 for group in y_groups][::-1], [group["size"] for group in y_groups][::-1], LH * 0.8, -0.6, AX_H + 0.2)[::-1]
    bracket_x, text_right = -BRACKET, -(BRACKET + CONN)
    for group, center in zip(y_groups, y_centers):
        inset = min(0.02, (group["b"] - group["a"]) * 0.2)
        figure_line([bracket_x + 0.03, bracket_x, bracket_x, bracket_x + 0.03], [group["b"] - inset, group["b"] - inset, group["a"] + inset, group["a"] + inset], color=GUIDE, lw=0.9)
        middle = (group["a"] + group["b"]) / 2
        figure_line([bracket_x, bracket_x - 0.08, text_right + 0.06, text_right + 0.01], [middle, middle, center, center], color=GUIDE, lw=0.5)
        top = center + group["size"] / 2
        start_x = text_right - group["cols"] * group["colw"]
        for index, gene in enumerate(group["genes"]):
            column, row = divmod(index, group["rows"])
            label(start_x + column * group["colw"] + PAD * 0.5, top - (row + 0.5) * LH, gene, False)

    x_centers = spread_intervals([(group["a"] + group["b"]) / 2 for group in x_groups], [group["size"] for group in x_groups], 0.15, -0.3, AX_W + 0.3)
    bracket_y, text_top = -BRACKET, -(BRACKET + CONN)
    for group, center in zip(x_groups, x_centers):
        inset = min(0.02, (group["b"] - group["a"]) * 0.2)
        figure_line([group["a"] + inset, group["a"] + inset, group["b"] - inset, group["b"] - inset], [bracket_y + 0.03, bracket_y, bracket_y, bracket_y + 0.03], color=GUIDE, lw=0.9)
        middle = (group["a"] + group["b"]) / 2
        figure_line([middle, middle, center, center], [bracket_y, bracket_y - 0.08, text_top + 0.06, text_top + 0.01], color=GUIDE, lw=0.5)
        start_x = center - group["size"] / 2
        for index, gene in enumerate(group["genes"]):
            column, row = divmod(index, group["rows"])
            label(start_x + column * group["colw"] + PAD * 0.5, text_top - (row + 0.5) * LH, gene, True)

    y_label_x = text_right - max([group["cols"] * group["colw"] for group in y_groups] + [1]) - 0.35
    x_label_y = text_top - max([group["rows"] for group in x_groups] + [1]) * LH - 0.3
    figure_text(y_label_x, AX_H / 2, f"Mouse genes (n = {total_mouse:,}; {len(mouse_goi)} genes of interest)", rotation=90, ha="center", va="center", fontsize=11)
    figure_text(AX_W / 2, x_label_y, f"Human genes (n = {total_human:,}; {len(human_goi)} genes of interest)", ha="center", va="top", fontsize=11)
    figure_text(AX_W / 2, AX_H + 0.2, "Block-diagonal bicluster heatmap — transport mass per cluster\nGenes coloured by Categorie", ha="center", va="bottom", fontsize=13)

    mouse_counts = {category: sum(category in categories_of[gene] for gene in mouse_goi) for category in category_order}
    human_counts = {category: sum(category in categories_of[gene] for gene in human_goi) for category in category_order}
    handles = [Patch(color=category_color[category], label=f"{category}:  {mouse_counts[category]} / {human_counts[category]}") for category in category_order]
    legend_x, legend_y = figure_point(AX_W + 0.25, AX_H * 0.75)
    legend = figure.legend(handles=handles, loc="center left", bbox_to_anchor=(legend_x, legend_y), frameon=False, fontsize=8.5, title="Gene category\n(n mouse / n human)", title_fontsize=10, handlelength=1.2, handleheight=1, labelspacing=0.7)
    legend._legend_box.align = "left"
    for text, category in zip(legend.get_texts(), category_order):
        text.set_color(category_color[category])
        text.set_fontweight("semibold")
    key_lines = []
    if any(len(categories_of[gene]) > 1 for gene in mouse_goi + human_goi):
        key_lines.append("■ = also in the category\n     of that colour")
    if cross:
        key_lines.append("† = mouse and human copies\n     in different clusters")
    key_lines.append("names sorted by category,\nthen alphabetically")
    figure_text(AX_W + 0.3, AX_H * 0.75 - 2.2, "\n".join(key_lines), ha="left", va="top", fontsize=8, color="0.25", linespacing=1.5)

    notes = []
    if omitted:
        notes.append(f"{omitted} cluster(s) without genes of interest omitted")
    if missing:
        notes.append(f"{len(missing)} genes of interest not in this bicluster set")
    small = [block for block in blocks if block["nMouse"] <= 150 or block["nHuman"] <= 150]
    if small:
        notes.append("small blocks (bottom-right): " + ", ".join(f"cluster {block['cluster']} ({block['nMouse']}×{block['nHuman']}, mass {block['transportMass']:.2g})" for block in small))
    figure.text(0.5, 0.12 / figure_height, "\n".join(notes), ha="center", va="bottom", fontsize=8, color="0.3", linespacing=1.4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_format = destination.suffix.lstrip(".") or "svg"
    figure.savefig(destination, format=output_format, dpi=120 if output_format == "png" else None)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    for rho in manifest["rhos"]:
        source = args.data / rho["file"]
        destination = args.output / f"{source.stem}-bicluster.svg"
        render(source, manifest["categories"], destination)
        print(destination)


if __name__ == "__main__":
    main()

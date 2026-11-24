"""Draw four lattice-topology figures matching the paper-style propagation grid."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'propagation_topologies'

# Visible rows/cols (ellipsis rows/cols omitted from arrow chains, as in the reference).
ROW_LABELS = ('1', 'L-3', 'L-2', 'L-1')
COL_LABELS = ('1', '2', '3', 'N-1', 'N')
ROW_IDX = (0, 2, 3, 4)
COL_IDX = (0, 1, 2, 3, 4)
NCOLS = 5
NROWS = 5
NODE_R = 0.11
DX = 1.35
DY = 1.15


def node_xy(col: int, row: int) -> tuple[float, float]:
    return col * DX, row * DY


def draw_nodes(ax) -> None:
    for row in range(NROWS):
        for col in range(NCOLS):
            x, y = node_xy(col, row)
            ax.add_patch(Circle((x, y), NODE_R, fill=True, facecolor='#e8e8e8', edgecolor='#333', lw=1.2))


def arrow(ax, start, end, *, mutation_scale=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>',
            mutation_scale=mutation_scale,
            linewidth=1.2,
            color='#333',
            shrinkA=NODE_R * 72,
            shrinkB=NODE_R * 72,
        )
    )


def label_axes(ax, title: str) -> None:
    for col, label in zip(COL_IDX, COL_LABELS):
        x, _ = node_xy(col, 0)
        ax.text(x, -0.55, label, ha='center', va='top', fontsize=11)
    for row, label in zip(ROW_IDX, ROW_LABELS):
        _, y = node_xy(0, row)
        ax.text(-0.45, y, label, ha='right', va='center', fontsize=11)
    ax.text(DX * (NCOLS - 1) / 2, -1.15, r'token $x$', ha='center', fontsize=11)
    ax.text(-1.05, DY * (NROWS - 1) / 2, r'layer $\ell$', ha='center', va='center', rotation=90, fontsize=11)
    ax.text(DX * (NCOLS - 1) / 2, DY * (NROWS - 1) + 0.65, title, ha='center', fontsize=12)
    for row in ROW_IDX:
        _, y = node_xy(0, row)
        ax.text(-0.15, (node_xy(0, 1)[1] + node_xy(0, 2)[1]) / 2, '⋮', ha='center', va='center', fontsize=14)
    mid_x = (node_xy(2, 0)[0] + node_xy(3, 0)[0]) / 2
    ax.text(mid_x, -0.15, '⋯', ha='center', va='top', fontsize=14)


def style_axes(ax) -> None:
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-1.3, DX * (NCOLS - 1) + 0.5)
    ax.set_ylim(-1.45, DY * (NROWS - 1) + 0.9)


def v1_reversed_chain(ax) -> None:
    """Horizontal arrows left; vertical arrows up in column 1 only."""
    draw_nodes(ax)
    for row in ROW_IDX:
        for left, right in zip(COL_IDX[1:], COL_IDX[:-1]):
            arrow(ax, node_xy(right, row), node_xy(left, row))
    for low, high in zip(ROW_IDX[:-1], ROW_IDX[1:]):
        arrow(ax, node_xy(0, low), node_xy(0, high))
    label_axes(ax, '(1) Reversed chain')


def v2_star_to_col1(ax) -> None:
    """Each node (except column 1) sends an arrow to column 1 in the same layer."""
    draw_nodes(ax)
    for row in ROW_IDX:
        for col in COL_IDX[1:]:
            arrow(ax, node_xy(col, row), node_xy(0, row), mutation_scale=10)
    label_axes(ax, '(2) Star to column 1')


def v3_column1_only(ax) -> None:
    """Only column 1 with vertical links."""
    draw_nodes(ax)
    for low, high in zip(ROW_IDX[:-1], ROW_IDX[1:]):
        arrow(ax, node_xy(0, low), node_xy(0, high))
    label_axes(ax, '(3) Column 1 only')


def v4_all_leftwards(ax) -> None:
    """Each node points to every node to its left in the same layer."""
    draw_nodes(ax)
    for row in ROW_IDX:
        for col in COL_IDX[1:]:
            for target in COL_IDX:
                if target < col:
                    arrow(ax, node_xy(col, row), node_xy(target, row), mutation_scale=8)
    label_axes(ax, '(4) All leftward pairs')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = (
        ('v1_reversed_chain', v1_reversed_chain),
        ('v2_star_to_column1', v2_star_to_col1),
        ('v3_column1_vertical', v3_column1_only),
        ('v4_all_leftward', v4_all_leftwards),
    )
    for name, draw in specs:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        draw(ax)
        style_axes(ax)
        path = OUT_DIR / f'{name}.png'
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(path)


if __name__ == '__main__':
    main()
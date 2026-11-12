"""Biểu đồ dùng chung cho ứng dụng Streamlit và báo cáo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .solver import Problem, Solution

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
MONEY = "triệu đồng"


def color(i: int) -> str:
    return SERIES[i % len(SERIES)]


def _axes(size=(7.2, 3.8), ncols=1, nrows=1, **kwargs):
    fig = Figure(figsize=size, facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False, **kwargs)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.xaxis.label.set(color=MUTED, size=10)
        ax.yaxis.label.set(color=MUTED, size=10)
    return (fig, axes[0, 0]) if ncols * nrows == 1 else (fig, axes)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, **kwargs)


def response_curves(p: Problem, s: Solution) -> Figure:
    fig, ax = _axes()
    top = max(float(s.x.max()) * 1.8, p.budget * 0.8)
    x = np.linspace(0, top, 300)
    for i, name in enumerate(p.names):
        ax.plot(x, p.a[i] * np.log1p(x / p.b[i]), color=color(i), linewidth=2, label=name)
        ax.plot(s.x[i], p.revenue(s.x)[i], "o", color=color(i), markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.set_xlim(0, top)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"Chi phí x ({MONEY})")
    ax.set_ylabel(f"Doanh thu R(x) ({MONEY})")
    _legend(ax, loc="upper left", ncols=2)
    return fig


def marginal_curves(p: Problem, s: Solution) -> Figure:
    """Doanh thu biên của từng kênh; tại nghiệm các kênh tự do cùng nằm trên đường lambda."""
    fig, ax = _axes()
    top = max(float(s.x.max()) * 1.8, p.budget * 0.8)
    x = np.linspace(0, top, 300)
    for i, name in enumerate(p.names):
        ax.plot(x, p.a[i] / (p.b[i] + x), color=color(i), linewidth=2, label=name)
        ax.plot(s.x[i], p.marginal(s.x)[i], "o", color=color(i), markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.axhline(s.lam, color=INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"λ = {s.lam:.3f}".replace(".", ","), (top, s.lam), xytext=(0, 5),
                textcoords="offset points", ha="right", color=INK, fontsize=10)
    ax.set_xlim(0, top)
    ax.set_ylim(0, min(float((p.a / p.b).max()) * 1.05, s.lam * 3.2))
    ax.set_xlabel(f"Chi phí x ({MONEY})")
    ax.set_ylabel("Doanh thu biên R'(x)")
    _legend(ax, loc="upper right", ncols=2)
    return fig


def allocation_compare(p: Problem, plans: dict[str, np.ndarray]) -> Figure:
    """So sánh phương án chia ngân sách theo từng kênh."""
    fig, ax = _axes()
    width = 0.8 / len(plans)
    centers = np.arange(len(p.names))
    for k, (label, x) in enumerate(plans.items()):
        offset = (k - (len(plans) - 1) / 2) * width
        ax.bar(centers + offset, x, width * 0.92, color=color(k), label=label)
    ax.set_xticks(centers, p.names)
    ax.grid(False, axis="x")
    ax.set_ylabel(f"Ngân sách ({MONEY})")
    _legend(ax, loc="upper right")
    return fig


def shadow_price(table: pd.DataFrame, budget: float) -> Figure:
    """So sánh lambda với đạo hàm số của doanh thu tối ưu theo ngân sách."""
    fig, ax = _axes()
    slope = np.gradient(table["revenue"], table["budget"])
    ax.plot(table["budget"], table["lam"], color=color(0), linewidth=2, label="λ từ bộ giải")
    ax.plot(table["budget"][3::6], slope[3::6], "o", color=color(1), markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="Đạo hàm số dR*/dB")
    ax.axvline(budget, color=MUTED, linewidth=1, linestyle=":")
    ax.annotate("B hiện tại", (budget, ax.get_ylim()[1]), xytext=(4, -12),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_xlabel(f"Ngân sách B ({MONEY})")
    ax.set_ylabel("Giá bóng λ")
    _legend(ax, loc="upper right")
    return fig


def allocation_path(table: pd.DataFrame, names: list[str], budget: float) -> Figure:
    fig, ax = _axes()
    for i, name in enumerate(names):
        ax.plot(table["budget"], table[name], color=color(i), linewidth=2, label=name)
    ax.axvline(budget, color=MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel(f"Ngân sách B ({MONEY})")
    ax.set_ylabel(f"Phân bổ tối ưu x ({MONEY})")
    _legend(ax, loc="upper left")
    return fig


def tangency(p: Problem, s: Solution, i: int, j: int) -> Figure:
    """Mặt cắt hai kênh: đường đồng mức tiếp xúc đường ngân sách tại nghiệm."""
    fig, ax = _axes((6.0, 4.6))
    ax.grid(False)
    pair = float(s.x[i] + s.x[j])
    top = pair * 1.35
    u, v = np.meshgrid(np.linspace(0, top, 300), np.linspace(0, top, 300))
    value = p.a[i] * np.log1p(u / p.b[i]) + p.a[j] * np.log1p(v / p.b[j])
    best = float(p.revenue(s.x)[[i, j]].sum())

    levels = sorted(best * np.array([0.55, 0.7, 0.85, 1.12, 1.25]))
    lines = ax.contour(u, v, value, levels=levels, colors=GRID, linewidths=1.1)
    ax.clabel(lines, fmt=lambda t: f"{t:.0f}", fontsize=8, colors=MUTED)
    ax.contour(u, v, value, levels=[best], colors=[color(0)], linewidths=2)
    ax.plot([0, pair], [pair, 0], color=color(1), linewidth=2)

    arrow = pair * 0.2
    ax.annotate("", (s.x[i] + arrow, s.x[j] + arrow), (s.x[i], s.x[j]),
                arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 1.5})
    ax.annotate("∇f = λ∇g", (s.x[i] + arrow, s.x[j] + arrow), xytext=(5, 3),
                textcoords="offset points", color=INK, fontsize=10)
    ax.plot(s.x[i], s.x[j], "o", color=INK, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.plot([], [], color=color(0), linewidth=2, label=f"Đường đồng mức f = {best:.1f}".replace(".", ","))
    ax.plot([], [], color=color(1), linewidth=2, label=f"Ngân sách x₁ + x₂ = {pair:.1f}".replace(".", ","))
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel(f"{p.names[i]} ({MONEY})")
    ax.set_ylabel(f"{p.names[j]} ({MONEY})")
    _legend(ax, loc="upper right")
    return fig


def fitted_curves(history: pd.DataFrame, fits: pd.DataFrame) -> Figure:
    """Mỗi kênh một ô: điểm dữ liệu lịch sử và đường cong ước lượng."""
    n = len(fits)
    cols = min(n, 3)
    rows = -(-n // cols)
    fig, axes = _axes((7.2, 2.3 * rows), ncols=cols, nrows=rows)
    cells = np.atleast_1d(axes).ravel()
    for k, row in enumerate(fits.itertuples(index=False)):
        ax = cells[k]
        data = history[history["kenh"] == row[0]]
        x = np.linspace(0, float(data["chi_phi"].max()) * 1.05, 200)
        ax.plot(data["chi_phi"], data["doanh_thu"], "o", color=color(k), markersize=4.5,
                alpha=0.75, markeredgewidth=0)
        ax.plot(x, row.a * np.log1p(x / row.b), color=INK, linewidth=1.5)
        ax.set_title(row[0], fontsize=10, color=INK, loc="left")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    for ax in cells[n:]:
        ax.set_visible(False)
    fig.supxlabel(f"Chi phí tháng ({MONEY})", fontsize=10, color=MUTED)
    fig.supylabel(f"Doanh thu tháng ({MONEY})", fontsize=10, color=MUTED)
    return fig

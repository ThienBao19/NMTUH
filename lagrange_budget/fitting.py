"""Ước lượng tham số a, b của đường cong R(x) = a * ln(1 + x / b) từ dữ liệu lịch sử."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REQUIRED_COLUMNS = {"kenh", "chi_phi", "doanh_thu"}


@dataclass
class Fit:
    a: float
    b: float
    r2: float
    rmse: float
    n: int


def fit_channel(spend, revenue) -> Fit:
    """Bình phương tối thiểu phi tuyến; với mỗi b, a tối ưu có dạng đóng nên duyệt b để lấy điểm đầu."""
    x = np.asarray(spend, dtype=float)
    y = np.asarray(revenue, dtype=float)
    if len(x) < 3:
        raise ValueError("Mỗi kênh cần ít nhất 3 quan sát.")
    if (x < 0).any():
        raise ValueError("Chi phí không được âm.")

    def best_a(b: float) -> float:
        z = np.log1p(x / b)
        return float(z @ y / (z @ z))

    scale = max(float(x.mean()), 1e-6)
    grid = scale * np.logspace(-2, 2, 81)
    errors = [np.sum((best_a(b) * np.log1p(x / b) - y) ** 2) for b in grid]
    b0 = float(grid[int(np.argmin(errors))])

    def residual(theta):
        a, b = np.exp(theta)
        return a * np.log1p(x / b) - y

    theta = least_squares(residual, np.log([max(best_a(b0), 1e-6), b0])).x
    a, b = np.exp(theta)
    res = residual(theta)
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(res @ res) / total if total > 0 else 0.0
    return Fit(float(a), float(b), r2, float(np.sqrt(np.mean(res ** 2))), len(x))


def fit_history(history: pd.DataFrame) -> pd.DataFrame:
    """Ước lượng cho từng kênh trong bảng lịch sử có các cột kenh, chi_phi, doanh_thu."""
    missing = REQUIRED_COLUMNS - set(history.columns)
    if missing:
        raise ValueError(f"Thiếu cột: {', '.join(sorted(missing))}.")
    rows = []
    for name, group in history.groupby("kenh", sort=False):
        fit = fit_channel(group["chi_phi"], group["doanh_thu"])
        rows.append({"Kênh": name, "a": fit.a, "b": fit.b, "R²": fit.r2,
                     "RMSE": fit.rmse, "Số tháng": fit.n})
    return pd.DataFrame(rows)

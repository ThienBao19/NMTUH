"""Bộ giải bài toán phân bổ ngân sách bằng phương pháp nhân tử Lagrange.

Bài toán:
    max   sum_i a_i * ln(1 + x_i / b_i)
    s.t.  sum_i x_i = B
          l_i <= x_i <= u_i

Hàm Lagrange:
    L = sum_i R_i(x_i) - lambda * (sum_i x_i - B)
        + sum_i mu_i * (x_i - l_i) + sum_i nu_i * (u_i - x_i)

Điều kiện dừng khi kênh i không chạm cận: R_i'(x_i) = a_i / (b_i + x_i) = lambda,
suy ra x_i = a_i / lambda - b_i. Thế vào ràng buộc ngân sách cho lambda dạng đóng.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FREE, AT_LOWER, AT_UPPER = "tự do", "chạm cận dưới", "chạm cận trên"


class InfeasibleError(ValueError):
    """Dữ liệu vào sai hoặc không tồn tại phương án thoả các ràng buộc."""


@dataclass
class Problem:
    names: list[str]
    a: np.ndarray
    b: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    budget: float

    def __post_init__(self):
        self.a = np.asarray(self.a, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.budget = float(self.budget)
        n = len(self.names)
        if n == 0:
            raise InfeasibleError("Cần ít nhất một kênh.")
        if len(set(self.names)) != n:
            raise InfeasibleError("Tên kênh không được trùng nhau.")
        if not all(len(v) == n for v in (self.a, self.b, self.lower, self.upper)):
            raise InfeasibleError("Các cột tham số phải có cùng số dòng.")
        if np.isnan(np.concatenate([self.a, self.b, self.lower, self.upper])).any():
            raise InfeasibleError("Bảng tham số còn ô trống.")
        if (self.a <= 0).any() or (self.b <= 0).any():
            raise InfeasibleError("Tham số a và b phải dương.")
        if (self.lower < 0).any() or (self.lower > self.upper).any():
            raise InfeasibleError("Cần có 0 <= cận dưới <= cận trên ở mọi kênh.")
        if self.budget < self.lower.sum():
            raise InfeasibleError(
                f"Ngân sách {self.budget:g} nhỏ hơn tổng cận dưới {self.lower.sum():g}."
            )
        if self.budget > self.upper.sum():
            raise InfeasibleError(
                f"Ngân sách {self.budget:g} lớn hơn tổng cận trên {self.upper.sum():g}."
            )

    def revenue(self, x) -> np.ndarray:
        """Doanh thu từng kênh R_i(x_i)."""
        return self.a * np.log1p(np.asarray(x, dtype=float) / self.b)

    def marginal(self, x) -> np.ndarray:
        """Doanh thu biên từng kênh R_i'(x_i)."""
        return self.a / (self.b + np.asarray(x, dtype=float))

    def total(self, x) -> float:
        return float(self.revenue(x).sum())


@dataclass
class Iteration:
    """Một vòng của thuật toán tập tích cực, lưu lại để hiển thị từng bước."""

    index: int
    free: list[int]
    budget_left: float
    lam: float
    x_trial: np.ndarray
    lower_violation: float
    upper_violation: float
    pegged: list[int]
    pegged_to: str | None


@dataclass
class Solution:
    x: np.ndarray
    lam: float
    mu: np.ndarray
    nu: np.ndarray
    status: list[str]
    revenue: float
    iterations: list[Iteration] = field(default_factory=list)


def multipliers(p: Problem, x: np.ndarray, lam: float, tol: float = 1e-9):
    """Tính mu, nu từ điều kiện dừng và phân loại trạng thái từng kênh."""
    gap = lam - p.marginal(x)
    mu = np.zeros(len(x))
    nu = np.zeros(len(x))
    status = []
    for i in range(len(x)):
        at_lower = x[i] <= p.lower[i] + tol
        at_upper = x[i] >= p.upper[i] - tol
        if at_lower and not (at_upper and gap[i] < 0):
            mu[i] = max(gap[i], 0.0)
            status.append(AT_LOWER)
        elif at_upper:
            nu[i] = max(-gap[i], 0.0)
            status.append(AT_UPPER)
        else:
            status.append(FREE)
    return mu, nu, status


def solve_lagrange(p: Problem) -> Solution:
    """Giải chính xác bằng nghiệm dạng đóng kết hợp ghim biến (tập tích cực).

    Mỗi vòng giải hệ Lagrange trên tập kênh tự do. Nếu có kênh vượt cận thì ghim
    các kênh ở phía có tổng vi phạm lớn hơn rồi giải lại. Vì hàm mục tiêu tách
    được và lõm chặt nên kênh đã ghim không cần xét lại, số vòng không quá n.
    """
    n = len(p.names)
    x = np.zeros(n)
    free = list(range(n))
    iterations: list[Iteration] = []
    lam = float("nan")

    while free:
        idx = np.array(free)
        left = p.budget - (x.sum() - x[idx].sum())
        lam = p.a[idx].sum() / (left + p.b[idx].sum())
        trial = p.a[idx] / lam - p.b[idx]

        below = np.maximum(p.lower[idx] - trial, 0.0)
        above = np.maximum(trial - p.upper[idx], 0.0)
        low_sum, up_sum = float(below.sum()), float(above.sum())

        x_trial = np.full(n, np.nan)
        x_trial[idx] = trial
        step = Iteration(len(iterations) + 1, list(free), float(left), float(lam),
                         x_trial, low_sum, up_sum, [], None)
        iterations.append(step)

        if low_sum <= 1e-12 and up_sum <= 1e-12:
            x[idx] = trial
            free = []
            break

        if low_sum >= up_sum:
            hit = idx[below > 0]
            x[hit] = p.lower[hit]
            step.pegged_to = AT_LOWER
        else:
            hit = idx[above > 0]
            x[hit] = p.upper[hit]
            step.pegged_to = AT_UPPER
        step.pegged = [int(i) for i in hit]
        free = [i for i in free if i not in step.pegged]
    else:
        # Mọi kênh đều bị ghim nên lambda chỉ cần nằm trong khoảng thoả KKT.
        slope = p.marginal(x)
        at_low = x <= p.lower + 1e-12
        at_up = x >= p.upper - 1e-12
        floor = slope[at_low & ~at_up].max(initial=0.0)
        ceiling = slope[at_up & ~at_low].min(initial=np.inf)
        both = slope[at_low & at_up]
        lam = floor if np.isinf(ceiling) else (floor + ceiling) / 2
        if not np.isfinite(lam) or lam == 0.0:
            lam = float(both.mean()) if both.size else float(slope.mean())

    mu, nu, status = multipliers(p, x, lam)
    return Solution(x, float(lam), mu, nu, status, p.total(x), iterations)


def solve_bisection(p: Problem, tol: float = 1e-10, max_iter: int = 500):
    """Kiểm chứng: tìm lambda bằng chia đôi trên hàm g(lambda) = sum x_i(lambda) - B."""

    def alloc(lam: float) -> np.ndarray:
        return np.clip(p.a / lam - p.b, p.lower, p.upper)

    high = float((p.a / (p.b + p.lower)).max()) * 2
    low = float((p.a / (p.b + np.minimum(p.upper, p.budget + p.lower))).min()) / 2
    while alloc(low).sum() < p.budget and low > 1e-300:
        low /= 2

    steps = 0
    for steps in range(1, max_iter + 1):
        mid = (low + high) / 2
        if alloc(mid).sum() > p.budget:
            low = mid
        else:
            high = mid
        if high - low <= tol * max(1.0, high):
            break
    lam = (low + high) / 2
    x = alloc(lam)
    return x, float(lam), steps


def solve_slsqp(p: Problem):
    """Kiểm chứng: gọi bộ giải tổng quát SLSQP của SciPy, không dùng công thức Lagrange."""
    start = np.clip(np.full(len(p.names), p.budget / len(p.names)), p.lower, p.upper)
    upper = [None if np.isinf(u) else u for u in p.upper]
    result = minimize(
        lambda x: -p.total(x),
        start,
        jac=lambda x: -p.marginal(x),
        bounds=list(zip(p.lower, upper)),
        constraints=[{"type": "eq", "fun": lambda x: x.sum() - p.budget,
                      "jac": lambda x: np.ones_like(x)}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    return result.x, result


def kkt_report(p: Problem, s: Solution) -> dict[str, float]:
    """Sai số của bốn nhóm điều kiện KKT tại nghiệm tìm được."""
    stationarity = p.marginal(s.x) - s.lam + s.mu - s.nu
    upper_gap = np.where(np.isinf(p.upper), 0.0, p.upper - s.x)
    return {
        "Ràng buộc ngân sách |Σx − B|": abs(float(s.x.sum() - p.budget)),
        "Vi phạm cận": float(max(0.0, np.max(p.lower - s.x), np.max(s.x - p.upper))),
        "Điều kiện dừng max|R' − λ + μ − ν|": float(np.abs(stationarity).max()),
        "Bù yếu max|μ(x−l)|, |ν(u−x)|": float(max(
            np.abs(s.mu * (s.x - p.lower)).max(), np.abs(s.nu * upper_gap).max())),
        "Nhân tử âm": float(max(0.0, -s.mu.min(), -s.nu.min())),
    }


def _fit_to_budget(p: Problem, shape: np.ndarray) -> np.ndarray:
    """Co giãn vector shape theo hệ số t sao cho clip(t * shape) có tổng bằng B."""
    low, high = 0.0, 1.0
    while np.clip(high * shape, p.lower, p.upper).sum() < p.budget and high < 1e15:
        high *= 2
    for _ in range(200):
        mid = (low + high) / 2
        if np.clip(mid * shape, p.lower, p.upper).sum() < p.budget:
            low = mid
        else:
            high = mid
    return np.clip(high * shape, p.lower, p.upper)


def baselines(p: Problem) -> dict[str, np.ndarray]:
    """Các cách chia theo kinh nghiệm để so sánh với nghiệm tối ưu."""
    return {
        "Chia đều": _fit_to_budget(p, np.ones(len(p.names))),
        "Theo tỉ lệ a": _fit_to_budget(p, p.a),
    }


def sweep_budget(p: Problem, budgets) -> pd.DataFrame:
    """Giải lại bài toán với nhiều mức ngân sách để phân tích độ nhạy."""
    rows = []
    for budget in budgets:
        q = Problem(p.names, p.a, p.b, p.lower, p.upper, float(budget))
        s = solve_lagrange(q)
        rows.append({"budget": float(budget), "lam": s.lam, "revenue": s.revenue,
                     **dict(zip(p.names, s.x))})
    return pd.DataFrame(rows)

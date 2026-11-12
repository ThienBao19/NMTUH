"""Kịch bản mẫu dùng chung cho ứng dụng, kiểm thử và báo cáo. Số liệu là giả lập."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .solver import Problem

DEFAULT_BUDGET = 300.0

DEFAULT_CHANNELS = pd.DataFrame(
    {
        "Kênh": ["Google Ads", "Facebook Ads", "TikTok Ads", "Email", "KOL"],
        "a": [400.0, 300.0, 180.0, 60.0, 150.0],
        "b": [120.0, 80.0, 50.0, 10.0, 200.0],
        "Tối thiểu": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Tối đa": [300.0, 300.0, 300.0, 20.0, 300.0],
    }
)


def default_problem(budget: float = DEFAULT_BUDGET) -> Problem:
    t = DEFAULT_CHANNELS
    return Problem(list(t["Kênh"]), t["a"], t["b"], t["Tối thiểu"], t["Tối đa"], budget)


def make_history(months: int = 24, noise: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Sinh lịch sử chi tiêu theo tháng từ tham số mặc định, có thêm nhiễu nhân tính."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, c in DEFAULT_CHANNELS.iterrows():
        spend = c["b"] * rng.uniform(0.1, 3.0, months)
        revenue = c["a"] * np.log1p(spend / c["b"]) * (1 + noise * rng.standard_normal(months))
        for month, (x, y) in enumerate(zip(spend, revenue), start=1):
            rows.append({"thang": month, "kenh": c["Kênh"],
                         "chi_phi": round(float(x), 2), "doanh_thu": round(float(y), 2)})
    return pd.DataFrame(rows)

import numpy as np
import pytest

from lagrange_budget.fitting import fit_channel, fit_history
from lagrange_budget.sample import DEFAULT_CHANNELS, default_problem, make_history
from lagrange_budget.solver import (
    AT_LOWER,
    AT_UPPER,
    FREE,
    InfeasibleError,
    Problem,
    baselines,
    kkt_report,
    solve_bisection,
    solve_lagrange,
    solve_slsqp,
    sweep_budget,
)


def random_problem(rng):
    n = int(rng.integers(1, 12))
    a = rng.uniform(1, 500, n)
    b = rng.uniform(1, 300, n)
    lower = rng.uniform(0, 30, n) * (rng.random(n) < 0.5)
    upper = lower + rng.uniform(0.5, 200, n)
    upper[rng.random(n) < 0.3] = np.inf
    top = upper.sum() if np.isfinite(upper).all() else lower.sum() + 1000
    budget = rng.uniform(lower.sum(), top)
    return Problem([f"k{i}" for i in range(n)], a, b, lower, upper, budget)


def test_interior_matches_closed_form():
    p = Problem(["A", "B"], [100, 200], [10, 20], [0, 0], [np.inf, np.inf], 90)
    s = solve_lagrange(p)
    assert s.lam == pytest.approx(300 / 120)
    assert s.x == pytest.approx([30, 60])
    assert len(s.iterations) == 1


def test_default_scenario_hand_calculation():
    s = solve_lagrange(default_problem())
    assert [len(i.free) for i in s.iterations] == [5, 4, 3]
    assert s.iterations[0].lam == pytest.approx(1090 / 760)
    assert s.iterations[1].lam == pytest.approx(940 / 560)
    assert s.lam == pytest.approx(880 / 530)
    assert s.status == [FREE, FREE, FREE, AT_UPPER, AT_LOWER]
    assert s.x[3] == 20 and s.x[4] == 0
    assert s.x.sum() == pytest.approx(300)


@pytest.mark.parametrize("seed", range(300))
def test_random_problems_satisfy_kkt_and_match_bisection(seed):
    p = random_problem(np.random.default_rng(seed))
    s = solve_lagrange(p)
    for name, value in kkt_report(p, s).items():
        assert value < 1e-7, name
    x, _, _ = solve_bisection(p)
    assert p.total(x) == pytest.approx(s.revenue, rel=1e-7)


@pytest.mark.parametrize("seed", range(40))
def test_no_worse_than_slsqp_or_random_feasible_points(seed):
    rng = np.random.default_rng(1000 + seed)
    p = random_problem(rng)
    s = solve_lagrange(p)
    x, result = solve_slsqp(p)
    if result.success:
        assert s.revenue >= p.total(x) - 1e-5 * max(1, abs(s.revenue))
    for name, candidate in baselines(p).items():
        assert candidate.sum() == pytest.approx(p.budget, rel=1e-8), name
        assert s.revenue >= p.total(candidate) - 1e-6


def test_lambda_equals_derivative_of_optimal_value():
    p = default_problem()
    h = 1e-3
    up = solve_lagrange(default_problem(300 + h)).revenue
    down = solve_lagrange(default_problem(300 - h)).revenue
    assert (up - down) / (2 * h) == pytest.approx(solve_lagrange(p).lam, rel=1e-6)


def test_all_channels_pegged_at_bounds():
    p = Problem(["A", "B"], [100, 100], [10, 10], [5, 5], [50, 50], 10)
    s = solve_lagrange(p)
    assert s.x == pytest.approx([5, 5])
    assert np.isfinite(s.lam)
    assert all(v < 1e-9 for v in kkt_report(p, s).values())

    p = Problem(["A", "B"], [100, 50], [10, 10], [0, 0], [20, 20], 40)
    s = solve_lagrange(p)
    assert s.x == pytest.approx([20, 20])
    assert s.status == [AT_UPPER, AT_UPPER]
    assert all(v < 1e-9 for v in kkt_report(p, s).values())


def test_single_channel_and_fixed_channel():
    s = solve_lagrange(Problem(["A"], [10], [5], [0], [100], 7))
    assert s.x == pytest.approx([7])
    p = Problem(["A", "B"], [100, 100], [10, 10], [5, 0], [5, 100], 30)
    s = solve_lagrange(p)
    assert s.x == pytest.approx([5, 25])
    assert all(v < 1e-9 for v in kkt_report(p, s).values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"budget": 5, "lower": [10, 0]},
        {"budget": 500, "upper": [10, 10]},
        {"a": [-1, 2]},
        {"b": [0, 2]},
        {"lower": [20, 0], "upper": [10, 10]},
        {"a": [np.nan, 2]},
    ],
)
def test_invalid_input_rejected(kwargs):
    base = {"a": [1, 2], "b": [1, 2], "lower": [0, 0], "upper": [100, 100], "budget": 50}
    with pytest.raises(InfeasibleError):
        Problem(["A", "B"], **{**base, **kwargs})


def test_duplicate_channel_names_rejected():
    with pytest.raises(InfeasibleError):
        Problem(["A", "A"], [1, 2], [1, 2], [0, 0], [10, 10], 5)


def test_sweep_budget_lambda_decreases():
    table = sweep_budget(default_problem(), np.linspace(10, 900, 60))
    assert (np.diff(table["lam"]) <= 1e-9).all()
    assert (np.diff(table["revenue"]) > 0).all()


def test_fit_recovers_exact_curve():
    x = np.linspace(5, 300, 30)
    fit = fit_channel(x, 250 * np.log1p(x / 70))
    assert fit.a == pytest.approx(250, rel=1e-5)
    assert fit.b == pytest.approx(70, rel=1e-5)
    assert fit.r2 == pytest.approx(1)


def test_fit_sample_history():
    table = fit_history(make_history())
    assert list(table["Kênh"]) == list(DEFAULT_CHANNELS["Kênh"])
    assert (table["R²"] > 0.8).all()


def test_fit_rejects_missing_columns():
    with pytest.raises(ValueError):
        fit_history(make_history().rename(columns={"chi_phi": "cost"}))

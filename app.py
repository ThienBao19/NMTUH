"""Ứng dụng phân bổ ngân sách marketing bằng phương pháp nhân tử Lagrange.

Chạy: streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from lagrange_budget import charts
from lagrange_budget.fitting import fit_history
from lagrange_budget.sample import DEFAULT_BUDGET, DEFAULT_CHANNELS, make_history
from lagrange_budget.solver import (
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

st.set_page_config(page_title="Phân bổ ngân sách - Lagrange", layout="wide")


def num(value: float, digits: int = 2) -> str:
    """Định dạng số kiểu Việt Nam: dấu phẩy thập phân."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def tex(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", "{,}")


def show(fig) -> None:
    st.pyplot(fig, width="stretch")


def build_problem(table: pd.DataFrame, budget: float) -> Problem:
    table = table.dropna(how="all")
    return Problem(
        [str(v) for v in table["Kênh"]],
        table["a"].astype(float),
        table["b"].astype(float),
        table["Tối thiểu"].astype(float).fillna(0.0),
        table["Tối đa"].astype(float).fillna(np.inf),
        budget,
    )


def apply_fits(fits: pd.DataFrame, current: pd.DataFrame) -> None:
    """Thay a, b trong bảng tham số; kênh đã có thì giữ nguyên cận."""
    old = current.dropna(subset=["Kênh"]).drop_duplicates("Kênh").set_index("Kênh")
    rows = []
    for row in fits.itertuples(index=False):
        name = row[0]
        known = name in old.index
        rows.append({
            "Kênh": name, "a": round(row.a, 2), "b": round(row.b, 2),
            "Tối thiểu": old.loc[name, "Tối thiểu"] if known else 0.0,
            "Tối đa": old.loc[name, "Tối đa"] if known else np.nan,
        })
    st.session_state.params = pd.DataFrame(rows)
    st.session_state.editor_version += 1


if "params" not in st.session_state:
    st.session_state.params = DEFAULT_CHANNELS.copy()
    st.session_state.editor_version = 0

with st.sidebar:
    st.header("Ngân sách")
    budget = st.number_input("Tổng ngân sách B (triệu đồng/tháng)", min_value=1.0,
                             value=DEFAULT_BUDGET, step=10.0)
    if st.button("Khôi phục kịch bản mẫu", width="stretch"):
        st.session_state.params = DEFAULT_CHANNELS.copy()
        st.session_state.editor_version += 1
        st.rerun()
    st.caption("Số liệu mẫu là số liệu giả lập, dùng để minh hoạ phương pháp.")

st.title("Phân bổ ngân sách marketing bằng nhân tử Lagrange")
st.latex(r"\max_{x}\ \sum_{i=1}^{n} a_i\ln\!\left(1+\frac{x_i}{b_i}\right)"
         r"\quad\text{với}\quad \sum_{i=1}^{n}x_i=B,\quad l_i\le x_i\le u_i")

tab_input, tab_result, tab_steps, tab_sensitivity, tab_geometry = st.tabs(
    ["1. Dữ liệu vào", "2. Kết quả", "3. Các bước tính", "4. Độ nhạy", "5. Hình học"]
)

with tab_input:
    st.subheader("Tham số các kênh")
    st.caption("a: mức doanh thu của kênh, b: độ bão hoà, a/b: doanh thu biên tại đồng đầu tiên. "
               "Để trống ô Tối đa nếu kênh không có giới hạn. Có thể thêm hoặc xoá dòng.")
    table = st.data_editor(
        st.session_state.params,
        key=f"editor_{st.session_state.editor_version}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "a": st.column_config.NumberColumn("a", min_value=0.0, format="%.2f"),
            "b": st.column_config.NumberColumn("b", min_value=0.0, format="%.2f"),
            "Tối thiểu": st.column_config.NumberColumn("Tối thiểu", min_value=0.0, format="%.2f"),
            "Tối đa": st.column_config.NumberColumn("Tối đa", min_value=0.0, format="%.2f"),
        },
    )

    st.subheader("Ước lượng a, b từ dữ liệu lịch sử")
    st.caption("Tệp CSV cần các cột kenh, chi_phi, doanh_thu; mỗi dòng là một tháng của một kênh.")
    upload = st.file_uploader("Tải tệp CSV", type="csv")
    use_sample = st.toggle("Dùng dữ liệu lịch sử mẫu (giả lập, 24 tháng)")
    history = None
    if upload is not None:
        try:
            history = pd.read_csv(upload)
        except (ValueError, UnicodeDecodeError) as error:
            st.error(f"Không đọc được tệp: {error}")
    elif use_sample:
        history = make_history()

    if history is not None:
        try:
            fits = fit_history(history)
        except (ValueError, TypeError) as error:
            st.error(f"Không ước lượng được: {error}")
        else:
            st.dataframe(fits.round(3), hide_index=True, width="stretch")
            show(charts.fitted_curves(history, fits))
            if st.button("Dùng các tham số này"):
                apply_fits(fits, table)
                st.rerun()

try:
    problem = build_problem(table, budget)
except InfeasibleError as error:
    st.error(f"Dữ liệu chưa hợp lệ: {error}")
    st.stop()

solution = solve_lagrange(problem)
plans = {"Tối ưu Lagrange": solution.x, **baselines(problem)}
names = problem.names

with tab_result:
    equal_total = problem.total(plans["Chia đều"])
    gain = (solution.revenue / equal_total - 1) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Doanh thu (triệu đồng)", num(solution.revenue, 1))
    c2.metric("Giá bóng λ", num(solution.lam, 3))
    c3.metric("Doanh thu / chi phí", num(solution.revenue / problem.budget, 2))
    c4.metric("Tăng so với chia đều", f"{num(gain, 1)}%")

    st.info(f"Tăng ngân sách thêm 1 triệu đồng thì doanh thu tối ưu tăng khoảng "
            f"{num(solution.lam, 2)} triệu đồng.")
    if solution.lam < 1:
        st.warning("λ nhỏ hơn 1: đồng ngân sách cuối cùng đem về ít hơn một đồng doanh thu, "
                   "nên cân nhắc giảm tổng ngân sách.")

    st.subheader("Phương án phân bổ")
    st.dataframe(
        pd.DataFrame({
            "Kênh": names,
            "Ngân sách": solution.x.round(2),
            "Tỉ trọng (%)": (solution.x / problem.budget * 100).round(1),
            "Doanh thu": problem.revenue(solution.x).round(2),
            "Doanh thu biên": problem.marginal(solution.x).round(3),
            "Trạng thái": solution.status,
        }),
        hide_index=True, width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("So sánh với cách chia khác")
        show(charts.allocation_compare(problem, plans))
    with right:
        st.subheader("Tổng doanh thu")
        st.dataframe(
            pd.DataFrame({
                "Phương án": list(plans),
                "Doanh thu": [round(problem.total(x), 2) for x in plans.values()],
                "Chênh lệch": [round(problem.total(x) - solution.revenue, 2)
                                        for x in plans.values()],
            }),
            hide_index=True, width="stretch",
        )
        export = pd.DataFrame({"kenh": names, "ngan_sach": solution.x.round(4),
                               "doanh_thu": problem.revenue(solution.x).round(4)})
        st.download_button("Tải kết quả CSV", export.to_csv(index=False).encode("utf-8-sig"),
                           "phan_bo_toi_uu.csv", "text/csv")

    left, right = st.columns(2)
    with left:
        st.subheader("Đường cong doanh thu")
        show(charts.response_curves(problem, solution))
    with right:
        st.subheader("Doanh thu biên tại nghiệm")
        show(charts.marginal_curves(problem, solution))

with tab_steps:
    st.subheader("Bước 1. Lập hàm Lagrange")
    st.latex(r"\mathcal{L}(x,\lambda,\mu,\nu)=\sum_i a_i\ln\!\left(1+\frac{x_i}{b_i}\right)"
             r"-\lambda\left(\sum_i x_i-B\right)+\sum_i\mu_i(x_i-l_i)+\sum_i\nu_i(u_i-x_i)")

    st.subheader("Bước 2. Điều kiện dừng")
    st.write("Với kênh không chạm cận thì μ = ν = 0, đạo hàm riêng bằng 0 cho:")
    st.latex(r"\frac{a_i}{b_i+x_i}=\lambda\ \Rightarrow\ x_i=\frac{a_i}{\lambda}-b_i")
    st.write("Thế vào ràng buộc ngân sách trên tập kênh tự do F, với B' là ngân sách còn lại:")
    st.latex(r"\lambda=\frac{\sum_{i\in F}a_i}{B'+\sum_{i\in F}b_i}")

    st.subheader("Bước 3. Giải lặp với tập kênh tự do")
    for step in solution.iterations:
        with st.expander(f"Vòng {step.index}: {len(step.free)} kênh tự do, "
                         f"λ = {num(step.lam, 4)}", expanded=True):
            free = np.array(step.free)
            st.latex(rf"\lambda=\frac{{{tex(problem.a[free].sum(), 2)}}}"
                     rf"{{{tex(step.budget_left, 2)}+{tex(problem.b[free].sum(), 2)}}}"
                     rf"={tex(step.lam, 4)}")
            trial = step.x_trial[free]
            lower, upper = problem.lower[free], problem.upper[free]
            st.dataframe(
                pd.DataFrame({
                    "Kênh": [names[i] for i in step.free],
                    "x = a/λ − b": trial.round(3),
                    "Cận dưới": lower,
                    "Cận trên": upper,
                    "Kiểm tra": np.where(trial < lower - 1e-12, "dưới cận",
                                         np.where(trial > upper + 1e-12, "vượt cận", "đạt")),
                }),
                hide_index=True, width="stretch",
            )
            st.write(f"Tổng vi phạm cận dưới: {num(step.lower_violation)}; "
                     f"tổng vi phạm cận trên: {num(step.upper_violation)}.")
            if step.pegged:
                listed = ", ".join(names[i] for i in step.pegged)
                st.write(f"Phía vi phạm lớn hơn được ghim: **{listed}** ({step.pegged_to}), "
                         f"sau đó giải lại với các kênh còn lại.")
            else:
                st.success("Mọi kênh đều thoả cận, thuật toán dừng.")

    st.subheader("Bước 4. Kiểm tra điều kiện KKT")
    st.dataframe(
        pd.DataFrame({
            "Kênh": names,
            "x*": solution.x.round(3),
            "R'(x*)": problem.marginal(solution.x).round(4),
            "λ": round(solution.lam, 4),
            "μ": solution.mu.round(4),
            "ν": solution.nu.round(4),
            "Trạng thái": solution.status,
        }),
        hide_index=True, width="stretch",
    )
    report = kkt_report(problem, solution)
    st.dataframe(
        pd.DataFrame({"Điều kiện": list(report),
                      "Sai số": [f"{v:.2e}" for v in report.values()],
                      "Kết quả": ["đạt" if v < 1e-6 else "không đạt" for v in report.values()]}),
        hide_index=True, width="stretch",
    )

    st.subheader("Bước 5. Đối chiếu với phương pháp độc lập")
    x_bisection, lam_bisection, bisection_steps = solve_bisection(problem)
    x_slsqp, slsqp = solve_slsqp(problem)
    methods = {
        "Lagrange dạng đóng": (solution.x, solution.lam, len(solution.iterations)),
        "Chia đôi trên λ": (x_bisection, lam_bisection, bisection_steps),
        "SciPy SLSQP": (x_slsqp, np.nan, slsqp.nit),
    }
    st.dataframe(
        pd.DataFrame([
            {"Phương pháp": label, "Tổng doanh thu": round(problem.total(x), 4),
             "λ": None if np.isnan(lam) else round(lam, 4), "Số vòng lặp": steps,
             **dict(zip(names, x.round(3)))}
            for label, (x, lam, steps) in methods.items()
        ]),
        hide_index=True, width="stretch",
    )

with tab_sensitivity:
    st.subheader("Khi ngân sách thay đổi")
    floor = float(problem.lower.sum())
    ceiling = float(min(problem.upper.sum(), max(problem.budget, 1.0) * 2.5))
    if ceiling - floor < 1e-6:
        st.info("Ngân sách không thể thay đổi vì tổng cận dưới bằng tổng cận trên.")
    else:
        sweep = sweep_budget(problem, np.linspace(floor + (ceiling - floor) * 0.01, ceiling, 121))
        left, right = st.columns(2)
        with left:
            st.write("**Giá bóng λ so với đạo hàm số của doanh thu tối ưu**")
            show(charts.shadow_price(sweep, problem.budget))
        with right:
            st.write("**Phân bổ tối ưu theo ngân sách**")
            show(charts.allocation_path(sweep, names, problem.budget))
        with st.expander("Xem bảng số liệu"):
            st.dataframe(sweep.iloc[::10].round(3), hide_index=True, width="stretch")

with tab_geometry:
    st.subheader("Ý nghĩa hình học: ∇f song song ∇g")
    free_channels = [i for i, state in enumerate(solution.status) if state == FREE]
    if len(free_channels) < 2:
        st.info("Cần ít nhất hai kênh tự do để vẽ mặt cắt tiếp xúc.")
    else:
        c1, c2 = st.columns(2)
        first = c1.selectbox("Kênh trục ngang", free_channels, format_func=lambda i: names[i])
        others = [i for i in free_channels if i != first]
        second = c2.selectbox("Kênh trục dọc", others, format_func=lambda i: names[i])
        st.write("Giữ nguyên các kênh khác tại nghiệm tối ưu. Trên mặt cắt này, đường đồng mức "
                 "tốt nhất chỉ vừa chạm đường ngân sách tại nghiệm.")
        left, _ = st.columns([3, 2])
        with left:
            show(charts.tangency(problem, solution, first, second))

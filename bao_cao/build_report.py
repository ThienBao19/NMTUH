"""Dựng báo cáo đồ án. Mọi số liệu được tính từ bộ giải tại thời điểm chạy.

Chạy từ thư mục gốc: python bao_cao/build_report.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "bao_cao")]

from docx_helpers import Report  # noqa: E402
from lagrange_budget import charts  # noqa: E402
from lagrange_budget.fitting import fit_history  # noqa: E402
from lagrange_budget.sample import DEFAULT_CHANNELS, default_problem, make_history  # noqa: E402
from lagrange_budget.solver import (  # noqa: E402
    Problem,
    baselines,
    kkt_report,
    solve_bisection,
    solve_lagrange,
    solve_slsqp,
    sweep_budget,
)

OUTPUT = ROOT / "bao_cao" / "MaSV_Ho va ten SV_Lagrange_BaoCao.docx"
FIGURES = ROOT / "bao_cao" / "hinh"


def vi(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def save(fig, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=200)
    return path


def crop(name: str, top: int, bottom: int) -> Path:
    """Cắt ảnh chụp màn hình, bỏ thanh bên để chữ trong báo cáo lớn hơn."""
    path = FIGURES / f"{name}_cat.jpg"
    with Image.open(FIGURES / f"{name}.jpg") as image:
        image.crop((262, top, 750, bottom)).save(path, quality=95)
    return path


def count_tests() -> int:
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                         cwd=ROOT, capture_output=True, text=True, check=True)
    return int(run.stdout.strip().splitlines()[-1].split()[0])


def timing_rows() -> list[list[str]]:
    rng = np.random.default_rng(0)
    rows = []
    for n in (10, 100, 500):
        a, b = rng.uniform(10, 500, n), rng.uniform(5, 200, n)
        p = Problem([str(i) for i in range(n)], a, b, np.zeros(n), np.full(n, np.inf), 30 * n)
        start = time.perf_counter()
        s = solve_lagrange(p)
        fast = time.perf_counter() - start
        start = time.perf_counter()
        x, _ = solve_slsqp(p)
        slow = time.perf_counter() - start
        rows.append([str(n), vi(fast * 1000, 2), vi(slow * 1000, 1),
                     f"{abs(s.revenue - p.total(x)) / s.revenue:.0e}"])
    return rows


# ---------------------------------------------------------------- số liệu
problem = default_problem()
sol = solve_lagrange(problem)
names = problem.names
plans = {"Tối ưu Lagrange": sol.x, **baselines(problem)}
equal_total = problem.total(plans["Chia đều"])
ratio_total = problem.total(plans["Theo tỉ lệ a"])
gain_equal = (sol.revenue / equal_total - 1) * 100
gain_ratio = (sol.revenue / ratio_total - 1) * 100

x_bis, lam_bis, steps_bis = solve_bisection(problem)
x_sqp, res_sqp = solve_slsqp(problem)
kkt = kkt_report(problem, sol)

sweep = sweep_budget(problem, np.linspace(30, 750, 121))
break_even = brentq(lambda B: solve_lagrange(default_problem(B)).lam - 1, 100, 1000)
scenarios = {B: solve_lagrange(default_problem(B)) for B in (150, 300, 500, 700)}

history = make_history()
fits = fit_history(history)
estimated = Problem(names, fits["a"], fits["b"], problem.lower, problem.upper, problem.budget)
x_hat = solve_lagrange(estimated).x
regret = sol.revenue - problem.total(x_hat)

tests_passed = count_tests()
speed = timing_rows()

FIGURES.mkdir(exist_ok=True)
fig_tangency = save(charts.tangency(problem, sol, 0, 1), "tiep_xuc")
fig_response = save(charts.response_curves(problem, sol), "duong_cong_doanh_thu")
fig_marginal = save(charts.marginal_curves(problem, sol), "doanh_thu_bien")
fig_compare = save(charts.allocation_compare(problem, plans), "so_sanh_phuong_an")
fig_shadow = save(charts.shadow_price(sweep, problem.budget), "gia_bong")
fig_path = save(charts.allocation_path(sweep, names, problem.budget), "duong_phan_bo")
fig_fit = save(charts.fitted_curves(history, fits), "uoc_luong")

# ---------------------------------------------------------------- bìa
r = Report()
r.centered("TRƯỜNG ĐẠI HỌC VĂN LANG", 14, True, 2)
r.centered("KHOA CÔNG NGHỆ THÔNG TIN", 14, True, 90)
r.centered("BÁO CÁO ĐỒ ÁN MÔN HỌC", 18, True, 8)
r.centered("NHẬP MÔN TỐI ƯU HOÁ", 16, True, 50)
r.centered("ĐỀ TÀI", 13, True, 6)
r.centered("TÌM HIỂU VÀ CÀI ĐẶT ỨNG DỤNG", 16, True, 2)
r.centered("BÀI TOÁN TỐI ƯU NHÂN TỬ LAGRANGE", 16, True, 8)
r.centered("Ứng dụng: phân bổ ngân sách marketing đa kênh", 14, False, 70)
for line in ("Giảng viên: Trần Ngọc Việt",
             "Mã học phần: 71AICR30203",
             "Nhóm lớp học phần: 261_71AICR30203_01",
             "Sinh viên 1: [Họ và tên] – [Mã số sinh viên]",
             "Sinh viên 2: [Họ và tên] – [Mã số sinh viên]"):
    r.centered(line, 13, False, 3)
r.centered("", 13, False, 70)
r.centered("TP. Hồ Chí Minh, học kỳ 1, năm học 2026–2027", 13, False, 0)
r.page_break()

r.centered("MỤC LỤC", 14, True, 10)
r.toc()
r.page_break()

# ---------------------------------------------------------------- chương 1
r.h1("CHƯƠNG 1. ĐẶT VẤN ĐỀ")
r.h2("1.1. Bối cảnh")
r.p("Doanh nghiệp thường có một khoản ngân sách marketing cố định mỗi tháng và phải chia khoản đó "
    "cho nhiều kênh như quảng cáo tìm kiếm, mạng xã hội, email hay người ảnh hưởng (KOL). Cách làm "
    "phổ biến là chia đều hoặc giữ tỉ lệ của tháng trước. Cả hai cách đều bỏ qua một đặc điểm quan "
    "trọng: hiệu quả của mỗi kênh giảm dần khi chi nhiều hơn. Đồng đầu tiên vào một kênh thường đem "
    "lại nhiều doanh thu, còn đồng thứ một trăm triệu thì ít hơn hẳn vì tệp khách hàng dễ tiếp cận "
    "đã được khai thác.")
r.p("Câu hỏi đặt ra là: với tổng ngân sách đã cho, nên chi cho mỗi kênh bao nhiêu để tổng doanh thu "
    "lớn nhất? Đây là bài toán tối ưu có ràng buộc, và phương pháp nhân tử Lagrange là công cụ tự "
    "nhiên để giải. Phương pháp này không chỉ cho phương án chia tiền mà còn cho biết giá trị của "
    "một đồng ngân sách tăng thêm, tức thông tin giúp người quản lý quyết định có nên xin thêm ngân "
    "sách hay không.")
r.h2("1.2. Phát biểu bài toán")
r.p("**Đầu vào:** tổng ngân sách $B$; với mỗi kênh $i=1,…,n$ có hai tham số đường cong doanh thu "
    "$a_i,b_i$ và hai cận chi tiêu $l_i,u_i$. Các tham số đường cong có thể nhập trực tiếp hoặc ước "
    "lượng từ dữ liệu chi tiêu lịch sử.", indent=False)
r.p("**Đầu ra:** mức chi $x_i^{*}$ cho từng kênh, tổng doanh thu tối ưu, nhân tử $λ$ cùng các nhân "
    "tử của cận, và các bước tính dẫn tới kết quả.", indent=False)
r.h2("1.3. Mục tiêu và phạm vi")
r.bullets([
    "Tìm hiểu phương pháp nhân tử Lagrange, điều kiện KKT và ý nghĩa của nhân tử.",
    "Xây dựng mô hình toán cho bài toán phân bổ ngân sách và suy ra nghiệm dạng đóng.",
    "Cài đặt ứng dụng web có đầu vào, đầu ra và hiển thị các bước tính.",
    "Kiểm chứng kết quả bằng các phương pháp độc lập và phân tích độ nhạy.",
])
r.p("Số liệu trong báo cáo là **số liệu giả lập** do nhóm tự xây dựng để minh hoạ phương pháp, không "
    "phải số liệu của doanh nghiệp thực. Đơn vị tiền là triệu đồng mỗi tháng.")

# ---------------------------------------------------------------- chương 2
r.h1("CHƯƠNG 2. CƠ SỞ LÝ THUYẾT")
r.h2("2.1. Bài toán tối ưu có ràng buộc đẳng thức")
r.p("Xét bài toán tìm cực đại của hàm khả vi $f(x)$ với $x∈ℝ^n$ dưới ràng buộc $g(x)=c$. Không thể "
    "chỉ giải $∇f(x)=0$ vì điểm đó có thể không thoả ràng buộc. Cách thế biến để khử ràng buộc chỉ "
    "thuận tiện với bài toán rất đơn giản. Phương pháp nhân tử Lagrange giữ nguyên mọi biến và đưa "
    "thêm một ẩn mới $λ$ cho ràng buộc.")
r.h2("2.2. Phương pháp nhân tử Lagrange")
r.p("**Định lý (điều kiện cần bậc nhất).** Nếu $x^{*}$ là cực trị địa phương của $f$ trên tập "
    "$g(x)=c$, các hàm $f,g$ khả vi liên tục và $∇g(x^{*})≠0$, thì tồn tại số $λ$ sao cho:", indent=False, keep=True)
r.eq(r"∇f(x^{*})=λ∇g(x^{*})", "2.1")
r.p("Đặt hàm Lagrange như sau, khi đó điều kiện (2.1) cùng ràng buộc chính là điều kiện mọi đạo hàm "
    "riêng của $\\t{L}$ theo $x$ và $λ$ bằng 0:", keep=True)
r.eq(r"\t{L}(x,λ)=f(x)-λ(g(x)-c)", "2.2")
r.p("Quy trình giải gồm ba bước: lập hàm Lagrange; giải hệ $n+1$ phương trình "
    "$∇_{x}\\t{L}=0$, $g(x)=c$; so sánh giá trị $f$ tại các nghiệm tìm được.")
r.p("**Ý nghĩa hình học.** Di chuyển dọc đường ràng buộc mà $f$ còn tăng được thì điểm đó chưa tối "
    "ưu. Tại nghiệm, đường đồng mức của $f$ chỉ vừa chạm đường ràng buộc, nên hai pháp tuyến "
    "$∇f$ và $∇g$ cùng phương. Hình 2.1 cho thấy điều này trên chính bài toán của đồ án, với mặt cắt "
    "qua hai kênh Google Ads và Facebook Ads khi các kênh khác giữ nguyên.")
r.figure(fig_tangency, "Hình 2.1. Đường đồng mức tiếp xúc đường ngân sách tại nghiệm tối ưu", 10.5)
r.p("**Ví dụ.** Tìm cực đại $f=xy$ với $x+y=10$. Ta có $\\t{L}=xy-λ(x+y-10)$. Các đạo hàm riêng cho "
    "$y=λ$, $x=λ$ và $x+y=10$, suy ra $x=y=λ=5$ và $f=25$. Nếu ràng buộc đổi thành $x+y=c$ thì "
    "$f^{*}=c^2/4$, nên $df^{*}/dc=c/2=5$ tại $c=10$, đúng bằng $λ$. Tính chất này được nêu ở mục 2.4.")
r.h2("2.3. Mở rộng cho ràng buộc bất đẳng thức: điều kiện KKT")
r.p("Bài toán thực tế thường có ràng buộc bất đẳng thức như mức chi không âm. Điều kiện "
    "Karush–Kuhn–Tucker (KKT) mở rộng phương pháp Lagrange cho trường hợp này. Với bài toán cực đại "
    "$f(x)$ dưới các ràng buộc $g(x)=c$ và $h_j(x)≥0$, hàm Lagrange là:", keep=True)
r.eq(r"\t{L}=f(x)-λ(g(x)-c)+\sum_{j}{μ_j h_j(x)}", "2.3")
r.p("Bốn nhóm điều kiện KKT tại nghiệm gồm:", keep=True)
r.bullets([
    "Điều kiện dừng: $∇_{x}\\t{L}=0$.",
    "Khả thi nguyên thuỷ: $g(x)=c$ và $h_j(x)≥0$.",
    "Khả thi đối ngẫu: $μ_j≥0$.",
    "Bù yếu: $μ_j h_j(x)=0$, nghĩa là ràng buộc nào không chạm thì nhân tử của nó bằng 0.",
])
r.p("Nếu $f$ lõm và các ràng buộc đều tuyến tính thì KKT vừa là điều kiện cần vừa là điều kiện đủ "
    "cho cực đại toàn cục. Nếu $f$ lõm chặt thì nghiệm là duy nhất. Bài toán của đồ án thuộc đúng "
    "trường hợp này, nên điểm thoả KKT chắc chắn là nghiệm tối ưu toàn cục.")
r.h2("2.4. Ý nghĩa của nhân tử: giá bóng")
r.p("Gọi $f^{*}(c)$ là giá trị tối ưu khi vế phải của ràng buộc là $c$. Dưới các điều kiện chính "
    "quy, ta có:", keep=True)
r.eq(r"\frac{df^{*}}{dc}=λ", "2.4")
r.p("Vì vậy $λ$ được gọi là giá bóng (shadow price): nó đo mức tăng của giá trị tối ưu khi nới ràng "
    "buộc thêm một đơn vị. Trong bài toán ngân sách, $λ$ là doanh thu tăng thêm khi có thêm một "
    "triệu đồng ngân sách, nên là một đại lượng có ý nghĩa kinh doanh trực tiếp.")

# ---------------------------------------------------------------- chương 3
r.h1("CHƯƠNG 3. MÔ HÌNH HOÁ VÀ PHƯƠNG PHÁP GIẢI")
r.h2("3.1. Hàm doanh thu của một kênh")
r.p("Nhóm chọn hàm logarit để mô tả doanh thu của kênh $i$ khi chi $x_i$:", keep=True)
r.eq(r"R_i(x_i)=a_i\t{ ln}\p{1+\frac{x_i}{b_i}}", "3.1")
r.p("Hàm này có ba tính chất phù hợp với thực tế: $R_i(0)=0$; tăng theo $x_i$; và hiệu quả giảm dần "
    "vì đạo hàm cấp hai âm. Tham số $a_i$ quy định quy mô doanh thu, còn $b_i$ quy định tốc độ bão "
    "hoà. Doanh thu biên, tức doanh thu thu thêm từ đồng kế tiếp, là:", keep=True)
r.eq(r"R_i^{′}(x_i)=\frac{a_i}{b_i+x_i}", "3.2")
r.p("Tại đồng đầu tiên, doanh thu biên bằng $a_i/b_i$. Hình 3.1 vẽ đường cong của năm kênh trong "
    "kịch bản mẫu, các chấm tròn là mức chi tối ưu.")
r.figure(fig_response, "Hình 3.1. Đường cong doanh thu của các kênh trong kịch bản mẫu", 13)
r.h2("3.2. Mô hình tối ưu")
r.eq(r"\t{max }\sum_{i=1}^{n}{a_i\t{ ln}\p{1+\frac{x_i}{b_i}}}\quad\t{với }"
     r"\sum_{i=1}^{n}{x_i}=B,\quad l_i≤x_i≤u_i", "3.3")
r.p("Hàm mục tiêu lõm chặt vì là tổng các hàm lõm chặt theo từng biến, còn các ràng buộc đều tuyến "
    "tính. Theo mục 2.3, nếu tập khả thi khác rỗng, tức $\\sum_{i}{l_i}≤B≤\\sum_{i}{u_i}$, thì bài "
    "toán có nghiệm duy nhất và KKT là điều kiện đủ.")
r.h2("3.3. Lập hàm Lagrange và nghiệm dạng đóng")
r.p("Với nhân tử $λ$ cho ngân sách, $μ_i≥0$ cho cận dưới và $ν_i≥0$ cho cận trên:", keep=True)
r.eq(r"\t{L}=\sum_{i}{R_i(x_i)}-λ\p{\sum_{i}{x_i}-B}+\sum_{i}{μ_i(x_i-l_i)}+\sum_{i}{ν_i(u_i-x_i)}", "3.4")
r.p("Điều kiện dừng theo từng $x_i$ là $R_i^{′}(x_i)-λ+μ_i-ν_i=0$. Kết hợp với bù yếu, mỗi kênh rơi "
    "vào một trong ba trường hợp:", keep=True)
r.bullets([
    "Kênh tự do ($l_i<x_i<u_i$): $μ_i=ν_i=0$ nên $R_i^{′}(x_i)=λ$.",
    "Kênh chạm cận dưới ($x_i=l_i$): $μ_i=λ-R_i^{′}(l_i)≥0$. Kênh này không đủ hấp dẫn dù ở mức chi thấp nhất.",
    "Kênh chạm cận trên ($x_i=u_i$): $ν_i=R_i^{′}(u_i)-λ≥0$. Kênh này còn hấp dẫn nhưng bị giới hạn.",
])
r.p("Trường hợp thứ nhất cho nguyên lý kinh tế của nghiệm: **doanh thu biên của mọi kênh tự do phải "
    "bằng nhau**, và giá trị chung đó chính là $λ$. Nếu một kênh có doanh thu biên cao hơn kênh khác, chuyển tiền sang kênh "
    "đó sẽ làm tổng doanh thu tăng. Từ (3.2) suy ra:", keep=True)
r.eq(r"x_i=\frac{a_i}{λ}-b_i", "3.5")
r.p("Gọi $F$ là tập kênh tự do và $B′$ là ngân sách còn lại sau khi trừ phần của các kênh đã cố định. "
    "Thế (3.5) vào ràng buộc $\\sum_{i∈F}{x_i}=B′$ ta có $λ$ dưới dạng đóng:", keep=True)
r.eq(r"λ=\frac{\sum_{i∈F}{a_i}}{B′+\sum_{i∈F}{b_i}}", "3.6")
r.h2("3.4. Xử lý cận bằng thuật toán ghim biến")
r.p("Công thức (3.5) có thể cho kết quả ngoài đoạn $[l_i,u_i]$. Nhóm dùng thuật toán tập tích cực "
    "theo tư tưởng ghim biến (variable pegging) cho bài toán lồi tách được [4]:", keep=True)
r.bullets([
    "Khởi tạo $F$ gồm mọi kênh, $B′=B$.",
    "Tính $λ$ theo (3.6) và $x_i$ theo (3.5) với $i∈F$.",
    "Tính tổng vi phạm cận dưới $Δ^{-}=\\sum_{i∈F}{\\t{max}(l_i-x_i,0)}$ và tổng vi phạm cận trên "
    "$Δ^{+}=\\sum_{i∈F}{\\t{max}(x_i-u_i,0)}$.",
    "Nếu $Δ^{-}=Δ^{+}=0$ thì dừng. Nếu $Δ^{-}≥Δ^{+}$ thì ghim các kênh dưới cận tại $l_i$; "
    "ngược lại ghim các kênh vượt cận tại $u_i$.",
    "Bỏ các kênh đã ghim khỏi $F$, cập nhật $B′$ rồi quay lại bước 2.",
    "Tính $μ_i,ν_i$ từ điều kiện dừng và kiểm tra KKT.",
], "List Number")
r.p("**Vì sao ghim là đúng.** Đặt $x_i(λ)$ là giá trị (3.5) đã cắt vào đoạn $[l_i,u_i]$, hàm này không "
    "tăng theo $λ$. Giả sử $Δ^{-}≥Δ^{+}$ tại $λ$ hiện tại. Tổng các giá trị đã cắt bằng "
    "$B′+Δ^{-}-Δ^{+}≥B′$, tức đang chi vượt ngân sách, nên nhân tử đúng $λ^{*}$ không nhỏ hơn $λ$ hiện tại. Khi đó "
    "mức chi của mọi kênh chỉ giảm, nên kênh đang dưới cận chắc chắn vẫn ở cận dưới tại nghiệm. Lập "
    "luận đối xứng cho trường hợp còn lại. Do mỗi vòng ghim ít nhất một kênh và không phải xét lại, "
    "thuật toán dừng sau không quá $n$ vòng, độ phức tạp xấu nhất là $O(n^2)$.")
r.h2("3.5. Ví dụ tính tay với kịch bản mẫu")
r.p(f"Với $B={vi(problem.budget, 0)}$ và tham số ở Bảng 5.1, thuật toán chạy {len(sol.iterations)} vòng:", keep=True)
for step in sol.iterations:
    free = np.array(step.free)
    a_sum, b_sum = problem.a[free].sum(), problem.b[free].sum()
    formula = (f"$λ={vi(a_sum, 0)}/({vi(step.budget_left, 0)}+{vi(b_sum, 0)})"
               f"={vi(step.lam, 4)}$")
    trial = "; ".join(f"{names[i]} {vi(step.x_trial[i], 1)}" for i in step.free)
    if step.pegged:
        pegged = ", ".join(names[i] for i in step.pegged)
        side = "dưới" if "dưới" in step.pegged_to else "trên"
        result = (f"$Δ^{{-}}={vi(step.lower_violation, 1)}$, $Δ^{{+}}={vi(step.upper_violation, 1)}$ "
                  f"nên ghim {pegged} tại cận {side}.")
    else:
        result = "Mọi kênh thoả cận, thuật toán dừng."
    r.p(f"**Vòng {step.index}.** {formula}. Mức chi thử: {trial}. {result}", indent=False)
r.p(f"Kiểm tra: {' + '.join(vi(v, 2) for v in sol.x[:4])} + {vi(sol.x[4], 0)} = {vi(sol.x.sum(), 0)}. "
    f"Doanh thu biên của ba kênh tự do đều bằng ${vi(sol.lam, 4).replace(',', ',')}$. "
    f"Email có $R′(20)=60/30={vi(problem.marginal(sol.x)[3], 2)}$ lớn hơn $λ$ nên $ν={vi(sol.nu[3], 4)}$. "
    f"KOL có $R′(0)=150/200={vi(problem.marginal(sol.x)[4], 2)}$ nhỏ hơn $λ$ nên $μ={vi(sol.mu[4], 4)}$. "
    "Mọi nhân tử đều không âm nên nghiệm thoả KKT.")
r.h2("3.6. Ước lượng tham số từ dữ liệu lịch sử")
r.p("Trong thực tế $a_i,b_i$ chưa biết trước. Với dữ liệu các tháng $(x_{it},y_{it})$ của kênh $i$, "
    "nhóm ước lượng bằng bình phương tối thiểu phi tuyến:", keep=True)
r.eq(r"\t{min}_{a,b>0}\sum_{t}{\p{y_{it}-a\t{ ln}\p{1+\frac{x_{it}}{b}}}^2}", "3.7")
r.p("Với mỗi $b$ cố định, $a$ tối ưu có dạng đóng vì mô hình tuyến tính theo $a$. Chương trình duyệt "
    "lưới $b$ để lấy điểm khởi đầu tốt, sau đó tinh chỉnh bằng hàm least_squares của SciPy.")

# ---------------------------------------------------------------- chương 4
r.h1("CHƯƠNG 4. CÀI ĐẶT ỨNG DỤNG")
r.h2("4.1. Công nghệ và cấu trúc chương trình")
r.p("Chương trình viết bằng Python, dùng NumPy và SciPy cho tính toán, Matplotlib cho biểu đồ, "
    "Streamlit cho giao diện web và pytest cho kiểm thử. Phần tính toán tách riêng khỏi giao diện để "
    "có thể kiểm thử độc lập.")
r.table("Bảng 4.1. Các thành phần của chương trình",
        ["Tệp", "Vai trò"],
        [["app.py", "Giao diện web gồm năm tab"],
         ["lagrange_budget/solver.py", "Bộ giải Lagrange, chia đôi trên λ, SLSQP, kiểm tra KKT"],
         ["lagrange_budget/fitting.py", "Ước lượng a, b từ dữ liệu lịch sử"],
         ["lagrange_budget/charts.py", "Biểu đồ dùng chung cho ứng dụng và báo cáo"],
         ["lagrange_budget/sample.py", "Kịch bản mẫu và sinh dữ liệu giả lập"],
         ["tests/test_solver.py", "Các ca kiểm thử tự động"]],
        [6.5, 9.5], left_columns=2)
r.h2("4.2. Phần lõi của thuật toán")
r.p("Đoạn mã sau trong solver.py thực hiện công thức (3.6), (3.5) và phép tính vi phạm cận:", keep=True)
r.code([
    "left = p.budget - (x.sum() - x[idx].sum())       # B'",
    "lam = p.a[idx].sum() / (left + p.b[idx].sum())   # (3.6)",
    "trial = p.a[idx] / lam - p.b[idx]                # (3.5)",
    "below = np.maximum(p.lower[idx] - trial, 0.0)",
    "above = np.maximum(trial - p.upper[idx], 0.0)",
    "low_sum, up_sum = float(below.sum()), float(above.sum())",
])
r.h2("4.3. Chức năng của ứng dụng")
r.table("Bảng 4.2. Năm tab của ứng dụng",
        ["Tab", "Nội dung"],
        [["1. Dữ liệu vào", "Sửa bảng tham số, thêm bớt kênh, tải CSV lịch sử để ước lượng a, b"],
         ["2. Kết quả", "Phương án tối ưu, λ, so sánh với cách chia khác, tải kết quả CSV"],
         ["3. Các bước tính", "Hàm Lagrange, từng vòng lặp, kiểm tra KKT, đối chiếu ba phương pháp"],
         ["4. Độ nhạy", "λ và phương án tối ưu thay đổi theo ngân sách"],
         ["5. Hình học", "Hình tiếp xúc giữa đường đồng mức và đường ngân sách"]],
        [4.0, 12.0], left_columns=2)
r.p("Dữ liệu sai được chặn và báo rõ lý do, ví dụ tham số không dương, tên kênh trùng, cận dưới lớn "
    "hơn cận trên, hoặc ngân sách nằm ngoài khoảng từ tổng cận dưới đến tổng cận trên.")
r.h2("4.4. Cách chạy chương trình")
r.p("Cài thư viện một lần rồi khởi động ứng dụng từ thư mục đồ án:", keep=True)
r.code(["pip install -r requirements.txt", "streamlit run app.py"])
r.p("Ứng dụng mở tại địa chỉ http://localhost:8501. Lệnh pytest tests chạy toàn bộ kiểm thử.")

# ---------------------------------------------------------------- chương 5
r.h1("CHƯƠNG 5. KẾT QUẢ CHẠY CHƯƠNG TRÌNH")
r.h2("5.1. Đầu vào")
r.p(f"Kịch bản mẫu gồm năm kênh với tổng ngân sách $B={vi(problem.budget, 0)}$ triệu đồng/tháng. Email "
    "bị giới hạn 20 triệu đồng để mô phỏng danh sách khách hàng có hạn.")
t = DEFAULT_CHANNELS
r.table("Bảng 5.1. Tham số đầu vào của kịch bản mẫu (giả lập)",
        ["Kênh", "a", "b", "a/b", "Tối thiểu", "Tối đa"],
        [[row["Kênh"], vi(row["a"], 0), vi(row["b"], 0), vi(row["a"] / row["b"], 2),
          vi(row["Tối thiểu"], 0), vi(row["Tối đa"], 0)] for _, row in t.iterrows()],
        [4.0, 2.2, 2.2, 2.2, 2.4, 2.4])
r.figure(crop("app_du_lieu_vao", 245, 560), "Hình 5.1. Màn hình nhập dữ liệu của ứng dụng", 13)
r.h2("5.2. Các bước tính hiển thị trên chương trình")
r.p("Tab Các bước tính trình bày đúng mô hình ở chương 3: hàm Lagrange, điều kiện dừng, từng vòng lặp "
    "với công thức đã thay số, bảng kiểm tra cận và quyết định ghim kênh. Kết quả trùng với phần tính "
    "tay ở mục 3.5.")
r.figure(crop("app_cac_buoc", 195, 555), "Hình 5.2. Màn hình các bước tính, vòng lặp thứ nhất", 13)
r.h2("5.3. Đầu ra")
r.table("Bảng 5.2. Phương án phân bổ tối ưu",
        ["Kênh", "Ngân sách", "Tỉ trọng (%)", "Doanh thu", "R′(x*)", "Trạng thái"],
        [[names[i], vi(sol.x[i]), vi(sol.x[i] / problem.budget * 100, 1),
          vi(problem.revenue(sol.x)[i]), vi(problem.marginal(sol.x)[i], 3), sol.status[i]]
         for i in range(len(names))],
        [3.2, 2.2, 2.2, 2.2, 2.0, 3.4], left_last=True)
r.figure(crop("app_ket_qua", 62, 392), "Hình 5.3. Màn hình kết quả của ứng dụng", 13)
r.p(f"Tổng doanh thu tối ưu đạt {vi(sol.revenue, 1)} triệu đồng, cao hơn cách chia đều {vi(gain_equal, 1)}% "
    f"và cao hơn cách chia theo tỉ lệ $a_i$ {vi(gain_ratio, 1)}% với cùng ngân sách.")
r.table("Bảng 5.3. So sánh các phương án (triệu đồng)",
        ["Phương án"] + names + ["Doanh thu"],
        [[label] + [vi(v, 1) for v in x] + [vi(problem.total(x), 1)] for label, x in plans.items()],
        [3.4, 1.9, 2.4, 1.9, 1.5, 1.4, 2.1])
r.figure(fig_compare, "Hình 5.4. Mức chi từng kênh theo ba phương án", 12.5)
r.p(f"Hình 5.5 minh hoạ nguyên lý của nghiệm: ba kênh tự do nằm đúng trên đường $λ={vi(sol.lam, 3)}$. "
    "Email nằm phía trên đường vì bị giới hạn, còn KOL nằm phía dưới nên không được cấp tiền. Các "
    "nhân tử cho thông tin quản trị cụ thể:", keep=True)
r.figure(fig_marginal, "Hình 5.5. Doanh thu biên của các kênh tại nghiệm tối ưu", 12.5)
r.bullets([
    f"$λ={vi(sol.lam, 3)}$: thêm 1 triệu đồng ngân sách đem lại khoảng {vi(sol.lam, 2)} triệu đồng doanh thu.",
    f"$ν_{{\t{{Email}}}}={vi(sol.nu[3], 3)}$: nới trần Email thêm 1 triệu đồng, chuyển từ kênh khác, tăng doanh thu khoảng {vi(sol.nu[3], 2)} triệu đồng.",
    f"$μ_{{\t{{KOL}}}}={vi(sol.mu[4], 3)}$: nếu buộc chi 1 triệu đồng cho KOL thì doanh thu giảm khoảng {vi(sol.mu[4], 2)} triệu đồng.",
])
r.h2("5.4. Kiểm chứng kết quả")
r.p("Nhóm đối chiếu với hai phương pháp không dùng công thức (3.6): chia đôi trên $λ$ và bộ giải "
    "tổng quát SLSQP của SciPy.", keep=True)
r.table("Bảng 5.4. Đối chiếu ba phương pháp trên kịch bản mẫu",
        ["Phương pháp", "Doanh thu", "λ", "Số vòng lặp"],
        [["Lagrange dạng đóng", vi(sol.revenue, 4), vi(sol.lam, 4), str(len(sol.iterations))],
         ["Chia đôi trên λ", vi(problem.total(x_bis), 4), vi(lam_bis, 4), str(steps_bis)],
         ["SciPy SLSQP", vi(problem.total(x_sqp), 4), "không có", str(res_sqp.nit)]],
        [6.0, 3.4, 3.0, 3.0])
r.p(f"Sai số lớn nhất của các điều kiện KKT tại nghiệm là {max(kkt.values()):.1e}, ở mức sai số làm "
    f"tròn của máy tính. Bộ kiểm thử tự động gồm {tests_passed} ca đều đạt, trong đó có 300 bài toán "
    "ngẫu nhiên được kiểm tra KKT và so với phương pháp chia đôi, các ca biên như mọi kênh đều bị "
    "ghim, và việc kiểm tra số vòng lặp cùng giá trị $λ$ của phần tính tay ở mục 3.5.")
r.table("Bảng 5.5. Thời gian giải với số kênh tăng dần (mili giây)",
        ["Số kênh", "Lagrange dạng đóng", "SLSQP", "Sai lệch doanh thu tương đối"],
        speed, [2.4, 4.2, 3.2, 5.6])
r.p("Thời gian đo trên máy của nhóm nên chỉ có ý nghĩa so sánh. Vì khai thác được cấu trúc tách biến, "
    "lời giải dạng đóng nhanh hơn bộ giải tổng quát nhiều bậc khi số kênh lớn, trong khi cho cùng "
    "kết quả.")
r.h2("5.5. Phân tích độ nhạy theo ngân sách")
r.p("Chương trình giải lại bài toán với nhiều mức ngân sách. Trên Hình 5.6, đạo hàm số của doanh thu "
    "tối ưu trùng với $λ$ từ bộ giải, xác nhận công thức (2.4) bằng thực nghiệm. $λ$ giảm dần khi "
    "ngân sách tăng, phản ánh hiệu quả giảm dần.")
r.figure(fig_shadow, "Hình 5.6. Giá bóng λ so với đạo hàm số của doanh thu tối ưu", 12.5)
r.figure(fig_path, "Hình 5.7. Phân bổ tối ưu khi ngân sách thay đổi", 12.5)
r.table("Bảng 5.6. Nghiệm tối ưu ở một số mức ngân sách",
        ["B"] + names + ["λ", "Doanh thu"],
        [[vi(B, 0)] + [vi(v, 1) for v in s.x] + [vi(s.lam, 3), vi(s.revenue, 1)]
         for B, s in scenarios.items()],
        [1.3, 1.9, 2.4, 1.9, 1.5, 1.3, 1.6, 2.1], left_columns=0)
r.p(f"Tại $B={vi(break_even, 0)}$ ta có $λ=1$. Vượt mức này, một đồng ngân sách đem về ít hơn một đồng "
    "doanh thu. Mức hoà vốn thực tế còn thấp hơn vì doanh thu chưa trừ giá vốn. Đây là thông tin mà "
    "chỉ nhìn vào phương án chia tiền thì không thấy được.")
r.h2("5.6. Ước lượng tham số từ dữ liệu")
r.p("Tệp data/lich_su_chi_tieu_mau.csv chứa 24 tháng dữ liệu giả lập, sinh từ tham số ở Bảng 5.1 và "
    "thêm nhiễu 5%. Nhóm giả định không biết tham số thật và ước lượng lại từ tệp này.", keep=True)
r.table("Bảng 5.7. Tham số ước lượng so với tham số dùng để sinh dữ liệu",
        ["Kênh", "a thật", "a ước lượng", "b thật", "b ước lượng", "R²"],
        [[row["Kênh"], vi(t["a"][k], 0), vi(row["a"], 1), vi(t["b"][k], 0), vi(row["b"], 1), vi(row["R²"], 3)]
         for k, row in fits.iterrows()],
        [3.6, 2.0, 2.8, 2.0, 2.8, 1.8])
r.figure(fig_fit, "Hình 5.8. Dữ liệu lịch sử và đường cong ước lượng", 13)
r.p(f"Tham số ước lượng lệch tối đa khoảng {vi(float(np.max(np.abs(fits['a'].values / t['a'].values - 1))) * 100, 0)}% với $a$ "
    f"và {vi(float(np.max(np.abs(fits['b'].values / t['b'].values - 1))) * 100, 0)}% với $b$, "
    f"nhưng khi tối ưu theo tham số ước lượng rồi đánh giá bằng tham số thật, doanh thu chỉ kém nghiệm "
    f"lý tưởng {vi(regret, 2)} triệu đồng ({vi(regret / sol.revenue * 100, 2)}%). Lý do là gần điểm tối "
    "ưu, hàm mục tiêu khá phẳng nên sai số vừa phải trong tham số ít ảnh hưởng đến doanh thu.")

# ---------------------------------------------------------------- chương 6
r.h1("CHƯƠNG 6. KẾT LUẬN")
r.h2("6.1. Kết quả đạt được")
r.bullets([
    "Trình bày phương pháp nhân tử Lagrange, điều kiện KKT, ý nghĩa hình học và ý nghĩa giá bóng.",
    "Xây dựng mô hình phân bổ ngân sách, suy ra nghiệm dạng đóng và thuật toán xử lý cận có lập luận tính đúng.",
    "Cài đặt ứng dụng web chạy được, thể hiện đầu vào, đầu ra và các bước tính.",
    f"Kết quả được kiểm chứng bằng KKT, hai phương pháp độc lập và {tests_passed} ca kiểm thử. "
    f"Trên kịch bản mẫu, nghiệm tối ưu cao hơn chia đều {vi(gain_equal, 1)}%.",
])
r.h2("6.2. Hạn chế")
r.bullets([
    "Số liệu là giả lập nên các con số chỉ minh hoạ phương pháp, chưa phản ánh một doanh nghiệp cụ thể.",
    "Mô hình giả định các kênh độc lập, trong khi thực tế các kênh có thể hỗ trợ lẫn nhau.",
    "Mô hình tĩnh, chưa xét tính mùa vụ và tác động kéo dài của quảng cáo sang các tháng sau.",
    "Ước lượng từ dữ liệu quan sát chỉ phản ánh tương quan giữa chi tiêu và doanh thu, chưa chắc là quan hệ nhân quả.",
])
r.h2("6.3. Hướng phát triển")
r.bullets([
    "Thử nghiệm trên dữ liệu thật và so sánh các dạng đường cong khác như hàm Hill.",
    "Đưa tương tác giữa các kênh vào mô hình, khi đó cần bộ giải số thay cho công thức đóng.",
    "Mở rộng thành bài toán nhiều kỳ và đổi mục tiêu từ doanh thu sang lợi nhuận.",
])

r.h1("TÀI LIỆU THAM KHẢO")
for k, ref in enumerate([
    "S. Boyd, L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.",
    "J. Nocedal, S. J. Wright. Numerical Optimization, 2nd ed. Springer, 2006.",
    "D. P. Bertsekas. Nonlinear Programming, 3rd ed. Athena Scientific, 2016.",
    "G. R. Bitran, A. C. Hax. Disaggregation and resource allocation using convex knapsack "
    "problems with bounded variables. Management Science, 27(4), 431–441, 1981.",
    "P. Virtanen et al. SciPy 1.0: fundamental algorithms for scientific computing in Python. "
    "Nature Methods, 17, 261–272, 2020.",
    "Streamlit Documentation. https://docs.streamlit.io",
], start=1):
    r.p(f"[{k}] {ref}", indent=False)

r.save(OUTPUT)
print("saved", OUTPUT)

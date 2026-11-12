# Phân bổ ngân sách marketing bằng nhân tử Lagrange

Đồ án môn Nhập môn Tối ưu hoá, đề tài 3. Số liệu mẫu là số liệu giả lập.

## Chạy ứng dụng

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng mở tại http://localhost:8501.

## Kiểm thử

```bash
pytest tests
```

## Thư mục

- `app.py`: giao diện Streamlit.
- `lagrange_budget/`: bộ giải, ước lượng tham số, biểu đồ, kịch bản mẫu.
- `data/lich_su_chi_tieu_mau.csv`: dữ liệu lịch sử giả lập để thử chức năng tải CSV.
- `tests/`: kiểm thử tự động.
- `bao_cao/`: báo cáo Word, PDF và script dựng báo cáo.

Dựng lại báo cáo (cần thêm `pip install python-docx`): `python bao_cao/build_report.py`, sau đó chạy `bao_cao/finalize.ps1` để cập nhật mục lục và xuất PDF bằng Word.

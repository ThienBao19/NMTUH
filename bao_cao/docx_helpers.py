"""Tiện ích dựng tệp Word: chuyển cú pháp kiểu LaTeX sang công thức Word (OMML), bảng, hình."""

from __future__ import annotations

import re
from xml.sax.saxutils import escape

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

FONT = "Times New Roman"
NS = ('xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
      'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"')
MATH_FONT = '<w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math"/></w:rPr>'


def _run(text: str, plain: bool = False) -> str:
    style = '<m:rPr><m:nor/></m:rPr>' if plain else ""
    font = f'<w:rPr><w:rFonts w:ascii="{FONT}" w:hAnsi="{FONT}"/></w:rPr>' if plain else MATH_FONT
    return f'<m:r>{style}{font}<m:t xml:space="preserve">{escape(text)}</m:t></m:r>'


def _group(s: str, i: int) -> tuple[str, int]:
    """Đọc một đối số: {...} hoặc một ký tự. Trả về nội dung và vị trí kế tiếp."""
    if s[i] != "{":
        return s[i], i + 1
    depth, j = 0, i
    while True:
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
        j += 1


def tex(s: str, inline: bool = False) -> str:
    """Chuyển tập con LaTeX (_, ^, frac, sum, t, p, quad) sang OMML."""
    atoms: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == " ":
            i += 1
        elif c == "{":
            body, i = _group(s, i)
            atoms.append(tex(body, inline))
        elif c == "\\":
            name = re.match(r"[a-zA-Z]+", s[i + 1:]).group()
            i += 1 + len(name)
            if name == "frac":
                top, i = _group(s, i)
                bottom, i = _group(s, i)
                atoms.append(f"<m:f><m:num>{tex(top, inline)}</m:num>"
                             f"<m:den>{tex(bottom, inline)}</m:den></m:f>")
            elif name == "t":
                body, i = _group(s, i)
                atoms.append(_run(body, plain=True))
            elif name == "p":
                body, i = _group(s, i)
                atoms.append(f"<m:d><m:e>{tex(body, inline)}</m:e></m:d>")
            elif name == "quad":
                atoms.append(_run("  ", plain=True))
            elif name == "sum":
                lower = upper = ""
                while i < len(s) and s[i] in "_^":
                    mark = s[i]
                    arg, i = _group(s, i + 1)
                    if mark == "_":
                        lower = arg
                    else:
                        upper = arg
                body, i = _group(s, i)
                place = "subSup" if inline else "undOvr"
                hide = "" if upper else '<m:supHide m:val="1"/>'
                atoms.append(
                    f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="{place}"/>{hide}'
                    f"</m:naryPr><m:sub>{tex(lower, inline)}</m:sub>"
                    f"<m:sup>{tex(upper, inline)}</m:sup><m:e>{tex(body, inline)}</m:e></m:nary>"
                )
            else:
                raise ValueError(f"Lệnh không hỗ trợ: {name}")
        elif c in "_^":
            base = atoms.pop()
            scripts = {}
            while i < len(s) and s[i] in "_^":
                mark = s[i]
                scripts[mark], i = _group(s, i + 1)
            sub = tex(scripts.get("_", ""), inline)
            sup = tex(scripts.get("^", ""), inline)
            if len(scripts) == 2:
                atoms.append(f"<m:sSubSup><m:e>{base}</m:e><m:sub>{sub}</m:sub>"
                             f"<m:sup>{sup}</m:sup></m:sSubSup>")
            elif "_" in scripts:
                atoms.append(f"<m:sSub><m:e>{base}</m:e><m:sub>{sub}</m:sub></m:sSub>")
            else:
                atoms.append(f"<m:sSup><m:e>{base}</m:e><m:sup>{sup}</m:sup></m:sSup>")
        else:
            atoms.append(_run(c))
            i += 1
    return "".join(atoms)


def math_inline(expression: str):
    return parse_xml(f"<m:oMath {NS}>{tex(expression, inline=True)}</m:oMath>")


def math_display(expression: str):
    return parse_xml(f'<m:oMathPara {NS}><m:oMathParaPr><m:jc m:val="center"/></m:oMathParaPr>'
                     f"<m:oMath>{tex(expression)}</m:oMath></m:oMathPara>")


def _fonts(style, size: float, bold=None, italic=None):
    style.font.name = FONT
    style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    style.font.color.rgb = RGBColor(0, 0, 0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    for key in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(key), FONT)
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(key), None)


class Report:
    def __init__(self, line_spacing: float = 1.2):
        self.doc = Document()
        self.line_spacing = line_spacing
        section = self.doc.sections[0]
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.left_margin, section.right_margin = Cm(3), Cm(2)
        section.top_margin, section.bottom_margin = Cm(2.5), Cm(2.2)
        section.different_first_page_header_footer = True

        styles = self.doc.styles
        normal = styles["Normal"]
        _fonts(normal, 13)
        normal.paragraph_format.line_spacing = line_spacing
        normal.paragraph_format.space_after = Pt(6)
        lang = OxmlElement("w:lang")
        lang.set(qn("w:val"), "vi-VN")
        normal.element.get_or_add_rPr().append(lang)

        for name, size, before, after in (("Heading 1", 14, 14, 8), ("Heading 2", 13, 10, 5),
                                          ("Heading 3", 13, 8, 4)):
            style = styles[name]
            _fonts(style, size, bold=True, italic=name == "Heading 3")
            fmt = style.paragraph_format
            fmt.space_before, fmt.space_after, fmt.keep_with_next = Pt(before), Pt(after), True
            fmt.line_spacing = 1.15

        for name in ("toc 1", "toc 2"):
            style = styles.add_style(name, 1, builtin=True)
            _fonts(style, 13, bold=name == "toc 1")
            style.paragraph_format.space_after = Pt(2)
            style.paragraph_format.line_spacing = 1.0
            if name == "toc 2":
                style.paragraph_format.left_indent = Cm(0.6)

        self._page_numbers(section)

    def _page_numbers(self, section):
        p = section.footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for kind, text in (("begin", None), (None, " PAGE "), ("end", None)):
            run = p.add_run()
            if kind:
                mark = OxmlElement("w:fldChar")
                mark.set(qn("w:fldCharType"), kind)
                run._r.append(mark)
            else:
                code = OxmlElement("w:instrText")
                code.set(qn("xml:space"), "preserve")
                code.text = text
                run._r.append(code)

    def h1(self, text):
        return self.doc.add_heading(text, 1)

    def h2(self, text):
        return self.doc.add_heading(text, 2)

    def h3(self, text):
        return self.doc.add_heading(text, 3)

    def _rich(self, p, text: str):
        """Viết đoạn có $công thức$ và **chữ đậm**."""
        for k, part in enumerate(re.split(r"\$(.+?)\$", text)):
            if k % 2:
                p._p.append(math_inline(part))
                continue
            for j, piece in enumerate(re.split(r"\*\*(.+?)\*\*", part)):
                if piece:
                    p.add_run(piece).bold = bool(j % 2)
        return p

    def p(self, text: str, indent: bool = True, keep: bool = False):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        if indent:
            para.paragraph_format.first_line_indent = Cm(1)
        para.paragraph_format.keep_with_next = keep
        return self._rich(para, text)

    def bullets(self, items: list[str], style: str = "List Bullet"):
        for item in items:
            para = self.doc.add_paragraph(style=style)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            para.paragraph_format.space_after = Pt(3)
            self._rich(para, item)

    def eq(self, expression: str, number: str | None = None):
        """Công thức hiển thị; có số thứ tự thì đặt trong bảng hai cột không viền."""
        if number is None:
            para = self.doc.add_paragraph()
            para._p.append(math_display(expression))
            return
        table = self.doc.add_table(rows=1, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        left, right = table.rows[0].cells
        left.width, right.width = Cm(14.2), Cm(1.8)
        left.vertical_alignment = right.vertical_alignment = 1
        left.paragraphs[0]._p.append(math_display(expression))
        tag = right.paragraphs[0]
        tag.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        tag.add_run(f"({number})")
        for cell in (left, right):
            cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        spacer = self.doc.add_paragraph()
        spacer.paragraph_format.space_after = Pt(0)
        spacer.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        spacer.paragraph_format.line_spacing = Pt(4)

    def code(self, lines: list[str]):
        para = self.doc.add_paragraph()
        para.paragraph_format.left_indent = Cm(0.4)
        para.paragraph_format.line_spacing = 1.0
        para.paragraph_format.space_after = Pt(8)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:val"), "clear")
        shade.set(qn("w:color"), "auto")
        shade.set(qn("w:fill"), "F2F2F2")
        para._p.get_or_add_pPr().append(shade)
        for k, line in enumerate(lines):
            run = para.add_run(line)
            run.font.name = "Consolas"
            run.font.size = Pt(10)
            if k < len(lines) - 1:
                run.add_break()

    def figure(self, path, caption: str, width_cm: float = 14.0):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = True
        para.paragraph_format.space_after = Pt(3)
        para.add_run().add_picture(str(path), width=Cm(width_cm))
        self.caption(caption)

    def caption(self, text: str, keep: bool = False):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.keep_with_next = keep
        para.paragraph_format.space_after = Pt(4 if keep else 10)
        run = para.add_run(text)
        run.italic = True
        run.font.size = Pt(12)

    def table(self, caption: str, headers: list[str], rows: list[list[str]],
              widths: list[float], left_columns: int = 1, left_last: bool = False):
        """Bảng số liệu; left_columns cột đầu (và cột cuối nếu left_last) căn trái, còn lại căn phải."""
        self.caption(caption, keep=True)
        table = self.doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for r, values in enumerate([headers] + rows):
            row = table.rows[r]
            props = row._tr.get_or_add_trPr()
            props.append(OxmlElement("w:cantSplit"))
            if r == 0:
                props.append(OxmlElement("w:tblHeader"))
            for c, value in enumerate(values):
                cell = row.cells[c]
                cell.width = Cm(widths[c])
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(1)
                para.paragraph_format.space_before = Pt(1)
                para.paragraph_format.line_spacing = 1.0
                para.paragraph_format.keep_with_next = r < len(rows)
                para.alignment = (WD_ALIGN_PARAGRAPH.CENTER if r == 0
                                  else WD_ALIGN_PARAGRAPH.LEFT
                                  if c < left_columns or (left_last and c == len(headers) - 1)
                                  else WD_ALIGN_PARAGRAPH.RIGHT)
                run = para.add_run(str(value))
                run.font.size = Pt(11.5)
                run.bold = r == 0
                if r == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:val"), "clear")
                    shade.set(qn("w:color"), "auto")
                    shade.set(qn("w:fill"), "DCE6F2")
                    cell._tc.get_or_add_tcPr().append(shade)
        gap = self.doc.add_paragraph()
        gap.paragraph_format.space_after = Pt(4)
        gap.paragraph_format.line_spacing_rule = WD_LINE_SPACING.EXACTLY
        gap.paragraph_format.line_spacing = Pt(6)

    def toc(self):
        para = self.doc.add_paragraph()
        run = para.add_run()
        for kind, text in (("begin", None), (None, r' TOC \o "1-2" \h \z \u '),
                           ("separate", None), ("text", "Mục lục"), ("end", None)):
            if kind == "text":
                run.add_text(text)
                continue
            if kind:
                mark = OxmlElement("w:fldChar")
                mark.set(qn("w:fldCharType"), kind)
                run._r.append(mark)
            else:
                code = OxmlElement("w:instrText")
                code.set(qn("xml:space"), "preserve")
                code.text = text
                run._r.append(code)

    def centered(self, text: str, size: float = 13, bold: bool = False, after: float = 6):
        para = self.doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.space_after = Pt(after)
        run = para.add_run(text)
        run.bold = bold
        run.font.size = Pt(size)
        return para

    def page_break(self):
        self.doc.add_page_break()

    def save(self, path):
        self.doc.save(str(path))

"""Build docs/Support_Investigation_Agent_Design.pdf (the design document) with ReportLab.

    python docs/build_design_pdf.py            # from the project root, inside the .venv

Edit the content below when the code changes and re-run; the figures are drawn here as well.
Fonts: Segoe UI + Consolas from C:/Windows/Fonts when present, otherwise Helvetica / Courier.
"""
from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Flowable, Frame, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

OUT = Path(__file__).resolve().parent / "Support_Investigation_Agent_Design.pdf"
DATE = "6 November 2026"

# ------------------------------------------------------------------ fonts
FONT_DIR = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
F = {"reg": "Helvetica", "bold": "Helvetica-Bold", "semi": "Helvetica-Bold", "it": "Helvetica-Oblique",
     "light": "Helvetica", "mono": "Courier"}
try:
    for name, file in (("SegoeUI", "segoeui.ttf"), ("SegoeUI-Bold", "segoeuib.ttf"),
                       ("SegoeUI-Italic", "segoeuii.ttf"), ("SegoeUI-Semibold", "seguisb.ttf"),
                       ("SegoeUI-Semilight", "segoeuisl.ttf"), ("Consolas", "consola.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
    pdfmetrics.registerFontFamily("SegoeUI", normal="SegoeUI", bold="SegoeUI-Bold", italic="SegoeUI-Italic",
                                  boldItalic="SegoeUI-Bold")
    F = {"reg": "SegoeUI", "bold": "SegoeUI-Bold", "semi": "SegoeUI-Semibold", "it": "SegoeUI-Italic",
         "light": "SegoeUI-Semilight", "mono": "Consolas"}
except Exception as e:  # noqa: BLE001
    print(f"Segoe UI not available ({e}); falling back to Helvetica")

# ------------------------------------------------------------------ palette
INDIGO = colors.HexColor("#4338CA")
INDIGO_2 = colors.HexColor("#6366F1")
INDIGO_BG = colors.HexColor("#EEF2FF")
INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E5E7EB")
ZEBRA = colors.HexColor("#F8FAFC")
CODE_BG = colors.HexColor("#F3F4F6")
GREY_FILL, GREY_STROKE = colors.HexColor("#F3F4F6"), colors.HexColor("#9CA3AF")
LLM_FILL, LLM_STROKE = colors.HexColor("#EEF2FF"), colors.HexColor("#6366F1")
HITL_FILL, HITL_STROKE = colors.HexColor("#FEF3C7"), colors.HexColor("#D97706")
BLUE_FILL, BLUE_STROKE = colors.HexColor("#E0F2FE"), colors.HexColor("#0EA5E9")
GREEN_FILL, GREEN_STROKE = colors.HexColor("#DCFCE7"), colors.HexColor("#16A34A")
RED_FILL, RED_STROKE = colors.HexColor("#FEE2E2"), colors.HexColor("#DC2626")
LEAD_FILL, LEAD_STROKE = colors.HexColor("#F3E8FF"), colors.HexColor("#9333EA")
EDGE = colors.HexColor("#4B5563")

PAGE_W, PAGE_H = A4
MARGIN_X = 51
CONTENT_W = PAGE_W - 2 * MARGIN_X

# ------------------------------------------------------------------ styles
BODY = ParagraphStyle("body", fontName=F["reg"], fontSize=9.3, leading=13.6, textColor=INK, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.9, leading=10.6, spaceAfter=0)
SMALL_B = ParagraphStyle("smallb", parent=SMALL, fontName=F["bold"])
TH = ParagraphStyle("th", parent=SMALL, fontName=F["bold"], textColor=colors.white)
CAPTION = ParagraphStyle("cap", parent=BODY, fontSize=7.8, leading=10, textColor=MUTED, spaceBefore=4, spaceAfter=8)
H1 = ParagraphStyle("h1", fontName=F["bold"], fontSize=19, leading=24, textColor=INDIGO, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName=F["semi"], fontSize=12.5, leading=16, textColor=INK, spaceBefore=8, spaceAfter=5)
H3 = ParagraphStyle("h3", fontName=F["bold"], fontSize=9.6, leading=13, textColor=INK, spaceBefore=6, spaceAfter=3)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=2.5)
MONO = ParagraphStyle("mono", fontName=F["mono"], fontSize=7.6, leading=10.2, textColor=INK)
TOC0 = ParagraphStyle("toc0", fontName=F["semi"], fontSize=9.6, leading=15, textColor=INK, leftIndent=0)
TOC1 = ParagraphStyle("toc1", fontName=F["reg"], fontSize=8.6, leading=12.5, textColor=MUTED, leftIndent=14)


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def h1(text: str) -> Paragraph:
    return Paragraph(text, H1)


def h2(text: str) -> Paragraph:
    return Paragraph(text, H2)


def h3(text: str) -> Paragraph:
    return Paragraph(text, H3)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(t, BULLET, bulletText="•") for t in items]


def table(header: list[str], rows: list[list[str]], widths: list[float], bold_first: bool = False) -> Table:
    """Indigo header, zebra rows; widths are fractions of the content width."""
    data = [[Paragraph(h, TH) for h in header]]
    for r in rows:
        data.append([Paragraph(c, SMALL_B if (bold_first and i == 0) else SMALL) for i, c in enumerate(r)])
    t = Table(data, colWidths=[w * CONTENT_W for w in widths], repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), INDIGO), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.6),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
             ("LINEBELOW", (0, 1), (-1, -1), 0.5, LINE)]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def callout(text: str) -> Table:
    t = Table([[Paragraph(text, BODY)]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INDIGO_BG), ("LINEBEFORE", (0, 0), (0, -1), 3, INDIGO),
                           ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def code(text: str) -> Table:
    lines = [Paragraph(l.replace(" ", "&nbsp;") or "&nbsp;", MONO) for l in text.strip("\n").split("\n")]
    t = Table([[lines]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), CODE_BG), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def stats(items: list[tuple[str, str]]) -> Table:
    num = ParagraphStyle("num", fontName=F["bold"], fontSize=21, leading=25, textColor=INDIGO)
    lab = ParagraphStyle("lab", fontName=F["reg"], fontSize=7.4, leading=9.5, textColor=MUTED)
    cells = [[[Paragraph(n, num), Paragraph(l, lab)] for n, l in items]]
    t = Table(cells, colWidths=[CONTENT_W / len(items)] * len(items))
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), INDIGO_BG), ("LINEAFTER", (0, 0), (-2, -1), 1, colors.white),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10)]))
    return t


# ------------------------------------------------------------------ drawing helpers for the figures
class Figure(Flowable):
    """A fixed-size drawing; subclasses implement paint(c) in a top-left coordinate system (y grows down)."""

    def __init__(self, height: float, scale: float = 1.0):
        super().__init__()
        self.width, self.height, self.scale = CONTENT_W, height, scale

    def wrap(self, aw, ah):
        return self.width, self.height * self.scale

    def draw(self):
        self.canv.saveState()
        self.canv.translate(self.width * (1 - self.scale) / 2, 0)
        self.canv.scale(self.scale, self.scale)
        self.paint()
        self.canv.restoreState()

    def paint(self):
        raise NotImplementedError

    # y measured from the top of the figure
    def Y(self, y: float) -> float:
        return self.height - y

    def box(self, cx, cy, w, h, title, sub="", fill=GREY_FILL, stroke=GREY_STROKE, r=5, title_size=8.6):
        c = self.canv
        c.setFillColor(fill)
        c.setStrokeColor(stroke)
        c.setLineWidth(0.9)
        c.roundRect(cx - w / 2, self.Y(cy + h / 2), w, h, r, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont(F["bold"], title_size)
        ty = cy - (2 if sub else -3)
        c.drawCentredString(cx, self.Y(ty), title)
        if sub:
            c.setFillColor(MUTED)
            c.setFont(F["reg"], 6.4)
            c.drawCentredString(cx, self.Y(cy + 9.5), sub)

    def pill(self, cx, cy, w, h, text):
        c = self.canv
        c.setFillColor(colors.white)
        c.setStrokeColor(EDGE)
        c.setLineWidth(0.9)
        c.roundRect(cx - w / 2, self.Y(cy + h / 2), w, h, h / 2, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont(F["bold"], 7.6)
        c.drawCentredString(cx, self.Y(cy + 2.7), text)

    def path(self, pts, dashed=False, arrow=True):
        """Polyline through pts (top-left coords) with an arrow head at the last point."""
        c = self.canv
        c.setStrokeColor(EDGE)
        c.setLineWidth(0.85)
        c.setDash(2.6, 2.2) if dashed else c.setDash()
        pth = c.beginPath()
        pth.moveTo(pts[0][0], self.Y(pts[0][1]))
        for x, y in pts[1:]:
            pth.lineTo(x, self.Y(y))
        c.drawPath(pth, stroke=1, fill=0)
        c.setDash()
        if arrow:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            self.head(x1, y1, x2, y2)

    def head(self, x1, y1, x2, y2, size=4.6):
        import math
        c = self.canv
        ang = math.atan2(self.Y(y2) - self.Y(y1), x2 - x1)
        tip = (x2, self.Y(y2))
        left = (x2 - size * math.cos(ang - 0.42), self.Y(y2) - size * math.sin(ang - 0.42))
        right = (x2 - size * math.cos(ang + 0.42), self.Y(y2) - size * math.sin(ang + 0.42))
        c.setFillColor(EDGE)
        pth = c.beginPath()
        pth.moveTo(*tip)
        pth.lineTo(*left)
        pth.lineTo(*right)
        pth.close()
        c.drawPath(pth, stroke=0, fill=1)

    def label(self, x, y, text, align="left", size=6.4, color=MUTED, font=None):
        c = self.canv
        c.setFillColor(color)
        c.setFont(font or F["it"], size)
        for i, line in enumerate(text.split("\n")):
            yy = self.Y(y + i * (size + 1.6))
            if align == "center":
                c.drawCentredString(x, yy, line)
            elif align == "right":
                c.drawRightString(x, yy, line)
            else:
                c.drawString(x, yy, line)


class GraphFigure(Figure):
    """Figure 2: the compiled 10-node graph."""

    def __init__(self):
        super().__init__(612, scale=0.86)

    def paint(self):
        cx, W = 232, 196  # main column centre / node width
        L, R = cx - W / 2, cx + W / 2
        y = {"start": 14, "intake": 56, "retrieve": 114, "triage": 172, "inv": 244, "respond": 334,
             "evaluate": 404, "review": 478, "fin": 548, "end": 600}
        h = 34
        top = lambda k: y[k] - h / 2  # noqa: E731
        bot = lambda k: y[k] + h / 2  # noqa: E731

        # legend (top right)
        lx, ly = 392, 6
        for i, (fill, stroke, text) in enumerate(((GREY_FILL, GREY_STROKE, "code node"), (LLM_FILL, LLM_STROKE, "LLM node"),
                                                  (HITL_FILL, HITL_STROKE, "human-in-the-loop (interrupt)"))):
            self.canv.setFillColor(fill)
            self.canv.setStrokeColor(stroke)
            self.canv.roundRect(lx, self.Y(ly + 9 + i * 12), 12, 8, 2, stroke=1, fill=1)
            self.label(lx + 17, ly + 7 + i * 12, text, font=F["reg"], color=INK)
        self.path([(lx, ly + 44), (lx + 12, ly + 44)], arrow=False)
        self.label(lx + 17, ly + 46, "static edge", font=F["reg"], color=INK)
        self.path([(lx, ly + 56), (lx + 12, ly + 56)], dashed=True, arrow=False)
        self.label(lx + 17, ly + 58, "conditional edge", font=F["reg"], color=INK)

        # edges first (boxes are drawn on top)
        self.path([(cx, y["start"] + 9), (cx, top("intake"))])
        self.path([(cx, bot("intake")), (cx, top("retrieve"))], dashed=True)
        self.label(cx + 5, bot("intake") + 14, "new ticket")
        self.path([(cx, bot("retrieve")), (cx, top("triage"))])
        self.path([(cx, bot("triage")), (cx, top("inv"))])
        # follow-up turn: intake -> investigate
        self.path([(R, y["intake"]), (372, y["intake"]), (372, top("inv") - 14), (R - 30, top("inv") - 14),
                   (R - 30, top("inv"))], dashed=True)
        self.label(377, 128, "follow-up turn:\nevidence + classification\nin state, no re-retrieval")
        # investigate <-> tools
        tx, tw = 428, 128
        self.path([(R, y["inv"] - 6), (tx - tw / 2, y["inv"] - 6)], dashed=True)
        self.label((R + tx - tw / 2) / 2, y["inv"] - 10, "tool calls", align="center")
        self.path([(tx - tw / 2, y["inv"] + 7), (R, y["inv"] + 7)], dashed=True)
        self.label((R + tx - tw / 2) / 2, y["inv"] + 17, "budget left", align="center")
        # investigate -> respond
        self.path([(cx, bot("inv")), (cx, top("respond"))], dashed=True)
        self.label(cx + 5, bot("inv") + 22, "done (no tool call)")
        # tools -> respond (budget spent)
        self.path([(tx, y["inv"] + h / 2), (tx, top("respond") - 14), (R - 30, top("respond") - 14),
                   (R - 30, top("respond"))], dashed=True)
        self.label(tx + 4, y["inv"] + h / 2 + 18, "tool budget\nspent")
        # respond -> evaluate
        self.path([(cx, bot("respond")), (cx, top("evaluate"))])
        # evaluate -> investigate (retry)
        self.path([(L, y["evaluate"]), (L - 40, y["evaluate"]), (L - 40, y["inv"] + 8), (L, y["inv"] + 8)], dashed=True)
        self.label(L - 36, 300, "check failed\n→ retry ×1")
        # evaluate -> respond (retries exhausted)
        self.path([(R, y["evaluate"] + 2), (R + 22, y["evaluate"] + 2), (R + 22, y["respond"] + 4), (R, y["respond"] + 4)],
                  dashed=True)
        self.label(R + 27, 366, "retries exhausted\n→ escalate")
        # evaluate -> review
        self.path([(cx, bot("evaluate")), (cx, top("review"))], dashed=True)
        self.label(cx + 5, bot("evaluate") + 22, "passed / escalation")
        # review -> investigate (rejected)
        self.path([(L, y["review"]), (L - 82, y["review"]), (L - 82, y["inv"] - 8), (L, y["inv"] - 8)], dashed=True)
        self.label(L - 78, 380, "rejected\n+ feedback")
        # review -> respond (lead diagnosis / amend)
        ax = R + 98
        self.path([(R, y["review"]), (ax, y["review"]), (ax, y["respond"] - 7), (R, y["respond"] - 7)], dashed=True)
        self.label(ax + 4, 410, "lead diagnosis\n(amend):\nrewrite reply", color=LEAD_STROKE)
        # review -> finalize / update_memory
        fx, mx = cx - 92, cx + 102
        self.path([(cx - 8, bot("review")), (fx, top("fin"))], dashed=True)
        self.path([(cx + 8, bot("review")), (mx, top("fin"))], dashed=True)
        self.label(cx, bot("review") + 27, "accepted → parallel ‖", align="center")
        self.path([(fx, bot("fin")), (cx - 10, y["end"] - 9)])
        self.path([(mx, bot("fin")), (cx + 10, y["end"] - 9)])

        # nodes
        self.pill(cx, y["start"], 66, 18, "START")
        self.box(cx, y["intake"], W, h, "intake", "code · persist open case + load memory / history")
        self.box(cx, y["retrieve"], W, h, "retrieve", "code · Qdrant hybrid search (+ trusted new cases)")
        self.box(cx, y["triage"], W, h, "triage", "LLM 20b · classify + pick skills", LLM_FILL, LLM_STROKE)
        self.box(cx, y["inv"], W, h, "investigate", "LLM 120b · ReAct: decides the next action", LLM_FILL, LLM_STROKE)
        self.box(tx, y["inv"], tw, h, "tools (ToolNode)", "7 tools · ask_customer interrupts", HITL_FILL, HITL_STROKE)
        self.box(cx, y["respond"], W, h, "respond", "LLM 120b · grounded answer or escalation", LLM_FILL, LLM_STROKE)
        self.box(cx, y["evaluate"], W, h, "evaluate", "rules + LLM judge 20b · confidence", LLM_FILL, LLM_STROKE)
        self.box(cx, y["review"], W, h, "review", "interrupt · support lead: accept / amend / reject", HITL_FILL, HITL_STROKE)
        self.box(fx, y["fin"], 160, h, "finalize", "code · persist case + index if trusted")
        self.box(mx, y["fin"], 160, h, "update_memory", "LLM 20b · durable customer facts", LLM_FILL, LLM_STROKE)
        self.pill(cx, y["end"], 66, 18, "END")


class ArchitectureFigure(Figure):
    """Figure 1: components and data flow."""

    def __init__(self):
        super().__init__(262)

    def paint(self):
        cx = CONTENT_W / 2
        self.box(cx, 20, 320, 34, "Browser UI  (frontend/)",
                 "live workflow graph · trace · clarify / review + lead diagnosis · results", BLUE_FILL, BLUE_STROKE)
        self.path([(cx - 30, 37), (cx - 30, 77)])
        self.label(cx - 36, 60, "POST ticket / resume / lead-solution", align="right")
        self.path([(cx + 30, 77), (cx + 30, 37)])
        self.label(cx + 36, 60, "SSE events")
        self.box(cx, 94, 380, 34, "FastAPI  (app/api.py · app/service.py)",
                 "REST + Server-Sent Events · event log · run / resume / follow-up / retry / lead-solution")
        self.path([(cx, 111), (cx, 147)])
        self.box(cx, 166, CONTENT_W - 30, 38, "LangGraph engine  (graph.py · nodes.py · tools.py · skills/)",
                 "10-node StateGraph · SqliteSaver checkpointer · SqliteStore long-term memory · Groq usage accounting · "
                 "Qdrant throttle", LLM_FILL, LLM_STROKE)
        bw, by = 112, 238
        xs = [bw / 2 + 4, bw / 2 + 4 + (CONTENT_W - bw - 8) / 3, bw / 2 + 4 + 2 * (CONTENT_W - bw - 8) / 3,
              CONTENT_W - bw / 2 - 4]
        specs = [("Groq API", "gpt-oss-120b · gpt-oss-20b", HITL_FILL, HITL_STROKE),
                 ("Qdrant Cloud", "23,608 + accepted live cases", GREEN_FILL, GREEN_STROKE),
                 ("SQLite (local)", "app.db · checkpoints.db · memory.db", GREY_FILL, GREY_STROKE),
                 ("fastembed (local CPU)", "bge-small-en-v1.5 + BM25", GREY_FILL, GREY_STROKE)]
        for x, (t, s, fl, st) in zip(xs, specs):
            self.path([(cx + (x - cx) * 0.55, 185), (x, by - 17)])
        for x, (t, s, fl, st) in zip(xs, specs):
            self.box(x, by, bw, 34, t, s, fl, st, title_size=8.2)


class LifecycleFigure(Figure):
    """Figure 3: case lifecycle in the SQLite cases table."""

    def __init__(self):
        super().__init__(118)

    def paint(self):
        y, w, h = 52, 92, 32
        xs = [48, 160, 272]
        self.box(xs[0], y, w, h, "open", "intake persists")
        self.box(xs[1], y, w, h, "awaiting_customer", "ask_customer paused", HITL_FILL, HITL_STROKE, title_size=7.8)
        self.box(xs[2], y, w, h, "open", "resumed / lead-solution")
        self.path([(xs[0] + w / 2, y), (xs[1] - w / 2, y)])
        self.path([(xs[1] + w / 2, y), (xs[2] - w / 2, y)])
        rx = 392
        self.path([(xs[2] + w / 2, y - 4), (rx - 52, 20)])
        self.path([(xs[2] + w / 2, y + 4), (rx - 52, 86)])
        self.box(rx, 18, 104, 32, "resolved", "indexed if accepted + check passed", GREEN_FILL, GREEN_STROKE)
        self.box(rx, 88, 104, 32, "escalated", "indexed only with a lead diagnosis", RED_FILL, RED_STROKE)
        self.label(rx + 55, 15, "lead-verified\nwhen the lead\ndiagnosed it", color=LEAD_STROKE, size=6)
        self.label(rx + 55, 85, "rejected drafts\nare never indexed", size=6)


# ------------------------------------------------------------------ document template with TOC + header/footer
class DesignDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN_X, rightMargin=MARGIN_X, topMargin=58,
                         bottomMargin=56, title="Autonomous Support Investigation Agent - Design Document",
                         author="Support Investigation Agent project", subject="Design document")
        cover = Frame(MARGIN_X, 56, CONTENT_W, PAGE_H - 300 - 56, id="cover")
        main = Frame(MARGIN_X, 56, CONTENT_W, PAGE_H - 58 - 56, id="main")
        self.addPageTemplates([PageTemplate("cover", [cover], onPage=self.draw_cover),
                               PageTemplate("main", [main], onPage=self.draw_chrome)])

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
            level = 0 if f.style.name == "h1" else 1
            text = f.getPlainText()
            key = f"h-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))

    @staticmethod
    def draw_cover(c, doc):
        c.saveState()
        band = 258
        c.setFillColor(INDIGO)
        c.rect(0, PAGE_H - band, PAGE_W, band, stroke=0, fill=1)
        c.setFillColor(INDIGO_2)
        c.rect(0, PAGE_H - band - 9, PAGE_W, 9, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont(F["bold"], 27)
        c.drawString(MARGIN_X, PAGE_H - 135, "Autonomous Support")
        c.drawString(MARGIN_X, PAGE_H - 168, "Investigation Agent")
        c.setFillColor(colors.HexColor("#C7D2FE"))
        c.setFont(F["light"], 12.5)
        c.drawString(MARGIN_X, PAGE_H - 197, "Design document · architecture, techniques and capabilities")
        c.restoreState()

    @staticmethod
    def draw_chrome(c, doc):
        c.saveState()
        c.setFont(F["reg"], 7)
        c.setFillColor(MUTED)
        c.drawString(MARGIN_X, PAGE_H - 32, "Autonomous Support Investigation Agent · Design Document")
        c.drawRightString(PAGE_W - MARGIN_X, PAGE_H - 32, "LangGraph · Groq · Qdrant")
        c.setStrokeColor(LINE)
        c.setLineWidth(0.6)
        c.line(MARGIN_X, PAGE_H - 38, PAGE_W - MARGIN_X, PAGE_H - 38)
        c.line(MARGIN_X, 40, PAGE_W - MARGIN_X, 40)
        c.drawString(MARGIN_X, 28, "DCKAP AI Engineer Assessment · Level 2")
        c.drawRightString(PAGE_W - MARGIN_X, 28, f"Page {doc.page}")
        c.restoreState()


# ------------------------------------------------------------------ content
def story() -> list:
    s: list = []

    # ---------------- cover
    s += [p("A locally runnable LangGraph agent that investigates new customer-support tickets using ~28K historical "
            "English tickets. It retrieves and compares past cases, decides which tools to use, loads skills on demand, "
            "asks the customer for clarification when needed, verifies its own answer against the evidence, lets a "
            "support lead correct it with their own diagnosis, and adds accepted resolutions back into its knowledge base.",
            ParagraphStyle("intro", parent=BODY, fontSize=10.2, leading=15.5)),
          Spacer(1, 12),
          stats([("23,608", "historical tickets indexed"), ("10", "graph nodes"), ("7", "LLM-callable tools"),
                 ("4", "dynamic skills"), ("7", "LangSmith evaluators")]),
          Spacer(1, 16),
          table(["Layer", "Choice"], [
              ["Orchestration", "LangGraph 1.2 · StateGraph · SqliteSaver checkpointer · SqliteStore · interrupt / "
                                "Command(resume)"],
              ["LLM (cloud)", "Groq · openai/gpt-oss-120b (investigation, response) + openai/gpt-oss-20b (triage, judge, "
                              "memory)"],
              ["Vector database", "Qdrant Cloud (free tier, strict mode) or embedded local mode · named dense + sparse "
                                  "vectors · RRF fusion"],
              ["Embeddings (local)", "fastembed · BAAI/bge-small-en-v1.5 (384-d dense) + Qdrant/bm25 (sparse)"],
              ["Structured storage", "SQLite · app.db (historical tickets, cases, event log, LLM usage) · checkpoints.db "
                                     "· memory.db"],
              ["Human in the loop", "Clarification questions to the customer · support-lead review: accept, amend with "
                                    "own root cause + solution, or reject"],
              ["Interfaces", "FastAPI REST + Server-Sent Events · single-page web UI · CLI · demo script"],
              ["Evaluation", "LangSmith evaluate() with 7 evaluators on a 40-ticket held-out split"],
          ], [0.21, 0.79]),
          Spacer(1, 14),
          p(f"Prepared for: DCKAP AI Engineer Assessment (Level 2) · Date: {DATE} · Revised for the support-lead "
            "knowledge feature, the lead-solution API and the demo reset script.", CAPTION),
          NextPageTemplate("main"), PageBreak()]

    # ---------------- contents
    toc = TableOfContents(dotsMinLevel=0)
    toc.levelStyles = [TOC0, TOC1]
    s += [p("Contents", ParagraphStyle("toctitle", parent=H1)), toc, PageBreak()]

    # ---------------- 1
    s += [h1("1. Executive summary"),
          p("The agent turns the brief's 9-step behaviour into a <b>ReAct investigation loop wrapped in a "
            "verify-and-commit pipeline</b>. Steps every ticket needs (persist, load memory, retrieve, triage, verify, "
            "persist, index) are fixed nodes, so they are predictable and auditable. The open-ended part (which evidence "
            "to gather, whether to ask the customer, whether to escalate) is decided by the model inside a tool loop."),
          p("Every design choice was made against two real constraints that were <b>measured before anything was loaded "
            "or tested</b>: the Groq free tier of the time (8K tokens/minute and 200K tokens/day per model) and Qdrant "
            "Cloud's strict mode (filters on payload fields that have no index are rejected). The result is a lean "
            "10-node graph that uses one fewer LLM call per ticket than the first version. The Groq account has since "
            "moved to a paid plan: usage is recorded per key and model, and no client-side limits are enforced."),
          p("The review step is also a <b>knowledge channel</b>. When the historical base has no answer, the support "
            "lead can enter the real root cause and fix. The reply is rewritten from it and checked again, and the case "
            "is indexed as <b>lead-verified</b>, so similar future tickets get the lead's fix (section 11). This added "
            "no node: it reuses review → respond → evaluate."),
          callout("<b>Verified on real services.</b> Normal resolution, clarification with pause and resume, long-term "
                  "memory reuse and retrieval of a newly resolved case were all demonstrated end to end on Groq and "
                  "Qdrant Cloud (section 16). After the token trims a resolved ticket costs about 7.1–10.6K tokens in "
                  "5–6 Groq calls, about 25% less input than before."),
          h2("What the agent can do"),
          table(["Ability", "What it does"], [
              ["Intake", "Persists every new query as an <b>open case</b> before doing anything else, and re-opens it "
                         "for follow-up turns."],
              ["Context", "Loads the customer's <b>long-term memory</b> (durable facts) and <b>case history</b>, "
                          "including still-open cases."],
              ["Retrieval", "<b>Hybrid search</b> (meaning + keywords) over 23,608 historical tickets plus accepted new "
                            "cases, with the query enriched by remembered facts. Lead-verified cases rank first among "
                            "equally similar ones."],
              ["Triage", "Classifies queue, type, priority and tags using the ticket and the labels of similar past "
                         "cases; output is normalised to the dataset's vocabulary."],
              ["Skills", "Picks 1–2 investigation <b>skills</b> from a catalog and loads only their full instructions "
                         "(progressive disclosure)."],
              ["Investigation", "Decides <b>by itself</b> which tools to call: targeted filtered searches, opening full "
                                "records, customer history, category statistics, re-classification. Several tools per "
                                "round, budgeted."],
              ["Clarification", "Asks the customer <b>one focused question</b> when information only they have is "
                                "missing; the run pauses, survives restarts and resumes with the answer. No fixed limit, "
                                "but repeated questions, questions after \"no more details\" and requests for secrets "
                                "are blocked."],
              ["Evidence reasoning", "Compares resolutions across <b>multiple cases</b>, reports consensus and conflicts, "
                                     "and never just copies the nearest answer."],
              ["Answer or escalate", "Writes a grounded customer reply with cited cases, or an <b>escalation hand-off</b> "
                                     "(target team, urgency, internal summary, holding reply)."],
              ["Self-verification", "Checks its own draft (rules + an independent LLM judge), scores <b>confidence</b>, "
                                    "retries with a targeted search, and escalates instead of sending an unsupported "
                                    "answer."],
              ["Human trust gate", "A support lead <b>accepts</b>, <b>rejects</b> with feedback, or <b>amends</b> with "
                                   "their own root cause and solution before the answer becomes knowledge (auto-accept "
                                   "mode for demos/evals)."],
              ["Support-lead knowledge", "The lead's diagnosis becomes authoritative evidence (LEAD-&lt;case&gt;), the "
                                         "reply is rewritten from it and re-checked, and the case is indexed as "
                                         "<b>lead-verified</b>. Also works on finished or escalated cases."],
              ["Growing knowledge", "Indexes accepted, verified resolutions and lead-diagnosed cases into Qdrant so "
                                    "<b>future tickets retrieve them</b>."],
              ["Learning about customers", "Extracts durable customer facts, updates or forgets them, blocks secrets "
                                           "and payment data, and caps memory size."],
              ["Multi-turn", "Handles <b>follow-up messages</b> in the same thread with the full prior context; a "
                             "follow-up sent while a question is open is taken as its answer."],
              ["Usage-aware", "Records Groq tokens per key, model and day (paid plan: no client-side limits), throttles "
                              "and retries Qdrant calls, and resumes a failed run from its checkpoint."],
              ["Observable", "Streams every step with token counts, persists an event log for replay, and supports "
                             "LangSmith tracing and evaluation."],
          ], [0.21, 0.79], bold_first=True)]

    # ---------------- 2
    s += [CondPageBreak(200), h1("2. Requirements coverage"),
          p("Each requirement from the assessment brief and where it is implemented."),
          table(["Requirement (brief)", "Implementation", "Where"], [
              ["A. LangGraph orchestration: multiple nodes", "10-node StateGraph", "app/graph.py"],
              ["Conditional routing", "5 routers: after intake, after investigate, after tools, after evaluate, "
                                      "after review", "app/nodes.py"],
              ["At least one meaningful loop / retry", "4 loops: tool loop · evidence-check retry · review rejection · "
                                                       "lead amend (review → respond)", "graph.py"],
              ["Interrupt / resume for more information", "interrupt() inside the ask_customer tool and in review "
                                                          "(accept / amend / reject); Command(resume=...)",
               "tools.py, nodes.py"],
              ["B. Index tickets in a vector DB", "Qdrant Cloud, named dense (bge-small) + sparse (BM25) vectors",
               "vectorstore.py"],
              ["Use metadata (queue, type, priority, tags)", "Payload filters chosen by the model + strict-mode payload "
                                                             "indexes; neighbour label votes for triage",
               "vectorstore.py, labels.py"],
              ["Compare evidence across cases, handle conflicts", "Prompts require consensus vs conflicts; ≥2 cited "
                                                                  "cases (or the lead's diagnosis) enforced by the "
                                                                  "evidence check", "nodes.py"],
              ["Follow-up / targeted retrieval", "search_similar_tickets with filters, chosen by the model; "
                                                 "judge-suggested search on retry", "tools.py"],
              ["C. Short-term memory", "SqliteSaver checkpointer per thread_id; follow-ups keep messages, evidence, "
                                       "clarifications, lead guidance", "graph.py"],
              ["Long-term customer memory, decide what to keep", "SqliteStore namespace per customer; LLM memory policy "
                                                                 "+ PII filter + cap", "memory.py"],
              ["D. ≥4 meaningful tools, model decides", "7 tools bound to the model; no fixed sequence", "tools.py"],
              ["≥3 reusable skills, loaded dynamically", "4 markdown skills; catalog shown, full text loaded only when "
                                                         "selected", "skills/"],
              ["E. LangSmith evaluators + datasets", "7 evaluators (rules + LLM judges), held-out dataset, evaluate()",
               "evals/"],
              ["3. Store every query; store final resolution; make it retrievable",
               "SQLite cases table from intake; finalize stores the result; accepted cases indexed (source=live, "
               "lead_verified when the lead diagnosed them)", "db.py, nodes.py"],
              ["New/open cases used as customer history", "customer_cases() reads all statuses; shown to the model; "
                                                          "citable as evidence", "db.py, nodes.py"],
              ["5. Output: classification, evidence, analysis, resolution, escalation, confidence",
               "finalize builds exactly these fields (plus evaluation, clarifications, skills, review, lead guidance)",
               "nodes.finalize"],
              ["6. Local, Python + LangGraph, cloud LLM allowed, dataset-only knowledge, secrets outside VCS",
               "Runs locally; Groq cloud LLM; only the dataset (and the lead's own input) is knowledge; .env is "
               "gitignored", "config.py, .gitignore"],
              ["7. Code + deps + README; ingest script; demo of 4 paths",
               "requirements.txt, README.md, scripts/ingest.py, sync_qdrant.py, demo.py (4 required paths + a lead "
               "scenario), reset_live_cases.py", "repo"],
              ["8. Parallel execution (plus)", "finalize ‖ update_memory fan-out; ToolNode runs a round's tool calls in "
                                               "parallel", "graph.py"],
              ["Human expertise as knowledge (beyond the brief)", "Lead's root cause + solution become authoritative "
                                                                  "evidence and lead-verified cases",
               "nodes.review_update, finalize"],
          ], [0.33, 0.48, 0.19]),
          h2("Expected agent behaviour (brief §4) mapped to the graph"),
          table(["#", "Brief step", "Graph"], [
              ["1", "Receive ticket, persist as open case", "intake"],
              ["2", "Load short-term state and long-term memory", "checkpointer (thread) + intake (store, history)"],
              ["3", "Classify / triage with ticket and historical evidence", "retrieve → triage (neighbour label votes "
                                                                             "as prior)"],
              ["4", "Retrieve multiple cases, inspect previous resolutions", "retrieve + search_similar_tickets / "
                                                                             "get_ticket_by_id"],
              ["5", "Use tools and relevant skills", "investigate ↔ tools with dynamically loaded skills"],
              ["6", "Ask a focused clarification question and resume", "ask_customer tool → interrupt → "
                                                                        "Command(resume)"],
              ["7", "Grounded resolution or escalation", "respond"],
              ["8", "Evaluation / evidence check; if unsupported, investigate again",
               "evaluate → investigate (retry) or → respond (escalate)"],
              ["9", "Persist final result; trusted cases searchable", "review (accept / amend / reject) → finalize "
                                                                      "(persist + index) ‖ update_memory"],
          ], [0.05, 0.48, 0.47])]

    # ---------------- 3
    s += [CondPageBreak(330), h1("3. System architecture"), ArchitectureFigure(),
          p("Figure 1 · Components and data flow. Embeddings are computed locally, so only the LLM is a paid cloud API.",
            CAPTION),
          h2("Components"),
          table(["Component", "Responsibility"], [
              ["frontend/ (index.html, app.js, styles.css)", "Single page without a build step. Draws the real compiled "
               "graph (Mermaid), streams the trace, shows clarification and review panels (with the lead's root cause / "
               "solution fields), results, memory and case replay."],
              ["app/api.py", "FastAPI endpoints incl. lead-solution; SSE streaming; startup ensures Qdrant payload "
                             "indexes exist."],
              ["app/service.py", "Runs / resumes the graph, applies a lead's solution to finished cases, converts "
                                 "LangGraph updates and custom events into UI events, persists them."],
              ["app/graph.py", "Graph topology, SqliteSaver (short-term) and SqliteStore (long-term memory)."],
              ["app/nodes.py", "All node logic, prompts and pydantic output schemas; evidence check, confidence and the "
                               "support-lead rules."],
              ["app/tools.py", "7 LLM-callable tools; state-changing tools return LangGraph Command updates; lead-safe "
                               "answer clipping."],
              ["app/skills/*.md", "4 conditional instruction modules + registry (catalog, render, default selection)."],
              ["app/vectorstore.py", "Qdrant hybrid search, upsert, payload indexes, throttled and retried calls."],
              ["app/llm.py + app/ratelimit.py", "Groq models, structured output with local repair + fallback, token "
                                                "accounting (no Groq limits), Qdrant throttle."],
              ["app/memory.py", "Long-term memory on the LangGraph store with PII guard and size cap."],
              ["app/db.py · app/labels.py", "SQLite app.db (historical tickets, cases, events) · label vocabulary, "
                                            "normalisation, neighbour votes."],
              ["scripts/ · evals/", "ingest, cloud sync, demo, CLI, reset of live cases · LangSmith evaluators and "
                                    "runner."],
          ], [0.3, 0.7], bold_first=True),
          h2("Project layout"),
          code("""
backend/
  app/        api.py  service.py  graph.py  nodes.py  tools.py  state.py  llm.py  ratelimit.py
              vectorstore.py  memory.py  db.py  labels.py  samples.py  config.py  skills/*.md
  scripts/    ingest.py  sync_qdrant.py  demo.py  cli.py  reset_live_cases.py
  evals/      evaluators.py  run_evals.py
  data/       raw CSVs, app.db, checkpoints.db, memory.db, local qdrant/  (gitignored)
frontend/     index.html  app.js  styles.css
docs/         Support_Investigation_Agent_Design.pdf  build_design_pdf.py
.env          secrets (gitignored)      .env.example  template      README.md
""")]

    # ---------------- 4
    s += [CondPageBreak(250), h1("4. Data and ingestion"),
          h2("Files taken from Hugging Face"),
          p("Dataset: Tobi-Bueck/customer-support-tickets. Columns used: subject, body, answer, queue, type, priority, "
            "tag_1 … tag_8 (the language column is used only for filtering)."),
          table(["File", "Rows", "English rows used"], [
              ["aa_dataset-tickets-multi-lang-5-2-50-version.csv", "28,587", "16,338"],
              ["dataset-tickets-multi-lang-4-20k.csv", "20,000", "11,923"],
              ["dataset-tickets-german_normalized_50_5_2.csv", "–", "not used (German only)"],
              ["<b>Total English subset</b>", "", "<b>28,261</b> (the brief's ~28K)"],
          ], [0.56, 0.16, 0.28]),
          h2("Pipeline (scripts/ingest.py)"),
          table(["Step", "Technique", "Result"], [
              ["Download", "huggingface_hub; skipped if files already exist", "backend/data/raw/"],
              ["Filter", "language == en", "28,261 rows"],
              ["Clean", "unescape literal \\n; drop rows with body or answer ≤ 20 chars", "–"],
              ["De-duplicate", "exact duplicates on (subject, body); the two files overlap heavily", "23,648 unique"],
              ["Hold-out", "40 random tickets (seed 42) never indexed, used for evaluation", "eval_holdout.jsonl"],
              ["Store", "SQLite historical_tickets (ids HT-00000…) for get_ticket_by_id and statistics", "23,608 rows"],
              ["Embed", "dense: first 1,200 chars of subject+body, batch 32; sparse BM25 on full text", "local CPU"],
              ["Speed-up", "records sorted by length so ONNX batches have little padding (≈ +40% throughput)",
               "~58 min total"],
              ["Index", "named vectors dense + bm25 (IDF); deterministic uuid5 point ids = idempotent re-runs",
               "23,608 points"],
              ["Cloud copy", "scripts/sync_qdrant.py copies vectors local → Qdrant Cloud without re-embedding",
               "72 s, 0 retries"],
          ], [0.16, 0.62, 0.22], bold_first=True),
          Spacer(1, 6),
          callout("<b>What is embedded and what is only stored.</b> Only the problem text (subject + body) is embedded, "
                  "because a new ticket describes a problem and is matched against past problems. Everything else rides "
                  "in the Qdrant payload: case_id, subject, body, answer, queue, type, priority, tags, source "
                  "(historical / live), and for live cases resolved_at, customer_id, lead_verified, root_cause and "
                  "lead_solution. The past resolution is compared later, in the prompts.")]

    # ---------------- 5
    s += [CondPageBreak(250), h1("5. LangGraph workflow design"),
          h2("Pattern and rationale"),
          p("Support work is mostly a fixed procedure with one open-ended part. A single free-running agent is hard to "
            "control and audit; a fully fixed chain cannot adapt to vague tickets, conflicting evidence or escalations. "
            "The chosen pattern keeps the procedure deterministic and gives the model full control only where judgement "
            "is needed: inside the <b>investigate ↔ tools</b> loop. Two gates follow it: an automated <b>evidence "
            "check</b> and a human <b>trust gate</b>. When the lead supplies their own diagnosis at that gate, the run "
            "goes back to <b>respond</b>, not to investigate, because the lead has already given the answer. Figure 2 "
            "shows the whole graph."),
          KeepTogether([GraphFigure(),
                        p("Figure 2 · The compiled 10-node graph. The UI draws this from the live graph object, so the "
                          "picture always matches the code.", CAPTION)]),
          h2("Nodes"),
          table(["Node", "Kind / model", "Responsibility"], [
              ["intake", "code", "Create the open case (or re-open it for a follow-up), reset per-turn fields, load "
                                 "long-term facts and the last 5 customer cases. A follow-up turn with evidence + "
                                 "classification in state routes straight to investigate."],
              ["retrieve", "code", "Hybrid search (k=8) with a memory-enriched query, no filters; if live cases exist, a "
                                   "dedicated source=live search for trusted new cases (similarity ≥ 0.6); neighbour "
                                   "label votes."],
              ["triage", "LLM · 20b", "Queue, type, priority, tags + 1–2 skills from the catalog. Sees only labels + "
                                      "subjects of the 6 nearest cases and compact label shares. Labels normalised; "
                                      "escalation skill added for high priority or repeat incidents."],
              ["investigate", "LLM · 120b", "ReAct step with 7 tools bound. Starts from the evidence pool (top 5 cases, "
                                            "a pool-strength line) and searches only if it is weak. Gets every point the "
                                            "lead made on this case and must keep each one. Tool budget 2 rounds; "
                                            "escalation skill criteria only."],
              ["tools", "ToolNode", "Executes the round's tool calls (in parallel); tools update state through Command; "
                                    "ask_customer can interrupt."],
              ["respond", "LLM · 120b", "One schema for both outcomes: grounded resolution or escalation hand-off with "
                                        "holding reply. Top 5 cases, the ones the notes cite first; the lead's diagnosis "
                                        "and feedback verbatim when present; hand-off format loaded only when "
                                        "escalating."],
              ["evaluate", "rules + LLM · 20b", "Evidence check, confidence score, retry or forced escalation. The judge "
                                                "treats support-lead input as authoritative."],
              ["review", "interrupt", "Support lead: accept, amend (own root cause + solution) or reject with feedback; "
                                      "auto-accept configurable per run."],
              ["finalize", "code", "Build and persist the structured result; index the case in Qdrant if trusted (as "
                                   "lead-verified when the lead diagnosed it)."],
              ["update_memory", "LLM · 20b", "Extract up to 4 durable customer facts per case; runs in parallel with "
                                             "finalize."],
          ], [0.15, 0.15, 0.70], bold_first=True),
          h2("Routing"),
          table(["Router", "Condition", "Next"], [
              ["after intake", "evidence pool and classification already in state (follow-up)", "investigate · otherwise "
                                                                                                "retrieve"],
              ["after investigate", "last AI message has tool calls", "tools · otherwise respond"],
              ["after tools", "tool budget spent (rounds ≥ 2)", "respond (skips a notes-only LLM call) · otherwise "
                                                                "investigate"],
              ["after evaluate", "passed, or the draft is an escalation", "review"],
              ["", "failed and retries left (1)", "investigate with judge feedback and one targeted search round"],
              ["", "failed and retries exhausted", "respond with forced escalation"],
              ["after review", "rejected", "investigate with the lead's feedback"],
              ["", "amended (lead's root cause / solution)", "respond: rewrite the reply from the diagnosis"],
              ["", "accepted", "[finalize, update_memory] in parallel"],
          ], [0.18, 0.42, 0.40]),
          h2("Loops, interrupts and parallelism"),
          *bullets([
              "<b>Tool loop</b> (investigate ↔ tools): model-decided, capped at 2 rounds; asking the customer refunds "
              "its round so the answer can be used.",
              "<b>Evidence-check retry</b> (evaluate → investigate): one retry with the judge's feedback and suggested "
              "query, then escalation.",
              "<b>Review rejection</b> (review → investigate): the lead's feedback becomes a message (marked as the "
              "lead's, not the customer's); budgets reset and stale notes are cleared.",
              "<b>Lead amend</b> (review → respond): the diagnosis enters the evidence pool as LEAD-&lt;case_id&gt;; the "
              "reply is rewritten, evidence-checked and shown for review again.",
              "<b>Interrupts</b>: interrupt() inside the ask_customer tool (in the ToolNode) and in review. State is "
              "checkpointed, so a paused case survives a server restart; resumption uses Command(resume=…).",
              "<b>Parallelism</b>: a conditional edge returning [finalize, update_memory] fans out two independent side "
              "effects; ToolNode executes multiple tool calls of one round concurrently.",
          ])]

    # ---------------- 6
    s += [CondPageBreak(250), h1("6. Lean design: from 17 nodes to 10"),
          p("The first version mirrored the brief's steps one-to-one (17 nodes, 6 skills). It was reviewed against "
            "measured cost and value, and every node that did not justify itself was merged or removed. Rule used: a node "
            "must do something no other node can, or be a quality gate."),
          table(["Removed / changed", "Replaced by", "Why"], [
              ["load_memory ‖ retrieve_cases", "memory loaded in intake, then retrieve", "Memory load is a millisecond "
               "SQLite read, so parallelism bought nothing; loading it first lets remembered facts improve the search "
               "query."],
              ["select_skills", "merged into triage", "It only validated names triage had already chosen."],
              ["focused_retrieval", "the agent's filtered search tool", "Duplicated model-decided retrieval and inflated "
                                                                        "every prompt."],
              ["assess (resolve / clarify / escalate)", "ask_customer and escalate_ticket tools", "The investigating "
               "model already holds the evidence; a separate decision call cost ~3K tokens per ticket against an "
               "8K/min limit."],
              ["ask_clarification node", "interrupt inside the ask_customer tool", "Same pause/resume semantics, no "
                                                                                   "extra node; the answer returns as "
                                                                                   "a tool result."],
              ["draft_resolution + escalate", "respond (one schema)", "Same output structure with an escalation flag."],
              ["index_case", "merged into finalize", "Both are deterministic persistence."],
              ["notes call after the last tool round", "tools → respond when budget spent", "Saved one 120b call "
                                                                                             "(~3.5K tokens) per "
                                                                                             "ticket."],
              ["skills ticket_triage, customer_response", "moved into the triage / respond prompts", "Loaded on every "
                                                                                                    "run, so they were "
                                                                                                    "prompts, not "
                                                                                                    "skills."],
          ], [0.27, 0.27, 0.46]),
          Spacer(1, 6),
          p("Kept on purpose: evaluate (the brief's evidence check with retry), review (prevents unchecked LLM output from "
            "entering the knowledge base) and the finalize ‖ update_memory fan-out (two independent side effects, one of "
            "them an LLM call). The support-lead feature followed the same rule and added <b>no node</b>: an amend "
            "decision is handled inside review and routed to respond, and the normal evidence check and review run "
            "again.")]

    # ---------------- 7
    s += [CondPageBreak(250), h1("7. State and memory"),
          h2("Graph state"),
          table(["Field", "Reducer", "Purpose"], [
              ["messages", "add_messages", "Investigation scratchpad and conversation: ticket, tool calls/results, "
                                           "clarifications, judge and lead feedback."],
              ["evidence", "custom merge_evidence", "Evidence pool keyed by case id; repeated retrievals merge (best "
                                                    "similarity kept, every retrieval path recorded). Also holds the "
                                                    "lead's diagnosis as LEAD-&lt;case_id&gt;."],
              ["retrieval_log, clarifications", "operator.add", "Audit trail of every search and every question/answer."],
              ["classification, skills, label_votes", "last value", "Triage output, loaded skills, neighbour label "
                                                                    "distribution."],
              ["customer_memory, customer_history", "last value", "Loaded per turn by intake."],
              ["tool_rounds, escalation_requested, investigation_notes", "last value", "Tool budget, escalation flag, "
                                                                                       "notes for the writer."],
              ["lead_guidance", "last value", "The lead's root cause, solution and note (who, when); kept across "
                                              "follow-up turns and saved to the case row at once."],
              ["draft, evaluation, review, final", "last value", "Answer; evidence-check result (also counts failed "
                                                                 "attempts for the retry loop); review decision "
                                                                 "(accept / amend / reject); final structured output "
                                                                 "(incl. the index result)."],
              ["memory_update", "last value", "What the parallel memory node saved or forgot."],
          ], [0.33, 0.17, 0.50]),
          p("State fields are not sent to the LLM as such: each node builds its prompt from a trimmed view of them "
            "(evidence digest, memory digest, the last 40 messages / 60K characters).", CAPTION),
          h2("Short-term memory"),
          p("The SqliteSaver checkpointer writes the full state to checkpoints.db after every node, keyed by thread_id. "
            "At the start of a run LangGraph loads the latest checkpoint of that thread (a local read, not an API call); "
            "a new ticket gets a new thread and starts empty. This gives interrupt/resume across process restarts, retry "
            "from the last checkpoint, replay, and multi-turn follow-ups: a follow-up re-enters at intake, keeps "
            "messages, evidence, classification, skills, clarifications and lead guidance, skips retrieve and triage, "
            "and resets only per-turn fields (draft, evaluation, budgets). A follow-up sent while a clarification "
            "question is open is treated as its answer."),
          h2("Long-term memory"),
          table(["Aspect", "Technique"], [
              ["Storage", "LangGraph SqliteStore in memory.db, namespace (\"customers\", customer_id, \"facts\"), one item "
                          "per fact key."],
              ["What to remember", "LLM step with an explicit policy: durable customer-level facts only (product, "
                                   "environment, plan, integrations, preferences, recurring issues, constraints). Not "
                                   "stored: one-off details, resolution steps, speculation."],
              ["Safety", "Deterministic filter rejects card-like numbers, password/PIN/CVV/secret/token/API-key mentions, "
                         "IBAN- and SSN-like patterns, and values over 300 chars."],
              ["Freshness", "Upsert by key (changed facts overwrite); obsolete keys explicitly forgotten; ≤4 new facts per "
                            "case; hard cap of 10 per customer (oldest dropped)."],
              ["Use", "Loaded at intake; enriches the retrieval query; shown to triage, investigate and respond; "
                      "available via get_customer_history; viewable and deletable through the memory API."],
          ], [0.2, 0.8], bold_first=True),
          h2("Customer history"),
          p("Every case lives in the SQLite cases table from the moment it arrives (open → awaiting_customer → resolved / "
            "escalated), so open and new cases count as customer history immediately. History cases are also citable "
            "evidence, which is what justifies a \"recurring issue → escalate\" decision."),
          h2("Storage map"),
          table(["Store", "Type", "Holds", "Used by"], [
              ["Qdrant collection support_cases", "Vector DB: dense + BM25 vectors, keyword payload indexes",
               "Historical and accepted live cases: vectors of subject + body, payload with answer, labels, tags, "
               "source and lead fields", "retrieve, search_similar_tickets, finalize (upsert), reset_live_cases.py"],
              ["data/app.db", "SQLite (sqlite3, WAL)", "historical_tickets · cases (full lifecycle, lead_guidance, "
                                                       "review, indexed flag) · events · llm_usage_by_key",
               "get_ticket_by_id, category statistics, label vocabulary, intake / finalize, customer history, replay, "
               "/api/health"],
              ["data/checkpoints.db", "SQLite via LangGraph SqliteSaver", "Full graph state per thread_id after every "
                                                                         "node, pending interrupts",
               "every run: resume, follow-up, retry, lead-solution on finished cases"],
              ["data/memory.db", "SQLite via LangGraph SqliteStore", "Customer facts", "intake, update_memory, "
                                                                                       "get_customer_history, memory "
                                                                                       "API"],
          ], [0.19, 0.2, 0.33, 0.28], bold_first=True),
          p("Both LangGraph stores are on disk (not InMemorySaver / InMemoryStore), so paused cases and customer memory "
            "survive a restart. Historical tickets are deliberately in Qdrant and in app.db: Qdrant answers \"which "
            "cases are similar\", SQLite answers exact look-ups and counts.", CAPTION)]

    # ---------------- 8
    s += [CondPageBreak(250), h1("8. Case-based RAG"),
          table(["Technique", "Detail"], [
              ["Hybrid retrieval", "Dense (bge-small, cosine) and BM25 (sparse, IDF) prefetches fused with Reciprocal Rank "
                                   "Fusion inside Qdrant's Query API. BM25 catches error codes and product names; dense "
                                   "catches paraphrases."],
              ["Interpretable scores", "RRF scores are rank-based, so the cosine similarity of every hit is recomputed "
                                       "from the returned dense vector for display and confidence."],
              ["Near-duplicate removal", "Hits with the same subject/body signature are dropped (the dataset contains "
                                         "many near-copies)."],
              ["Memory-enriched query", "Ticket + clarifications + follow-up + remembered product/environment facts, so a "
                                        "vague \"it is broken again\" still finds cases about the customer's system."],
              ["Metadata filters", "queue / type / priority / tag / source / exclude-ids. The initial retrieval is "
                                   "unfiltered because it runs before triage; filtered searches are chosen by the model "
                                   "through search_similar_tickets (often with the triaged labels)."],
              ["Filter relaxation", "If a filtered search returns nothing, it is retried once without filters."],
              ["Strict-mode indexes", "Qdrant Cloud rejects filters on non-indexed fields, so keyword payload indexes "
                                      "exist on queue, type, priority, tags, source, case_id (created at startup if "
                                      "missing)."],
              ["Trusted new cases", "A dedicated source=live search (only when live cases exist) so a few new resolutions "
                                    "are not out-ranked by 23K historical near-duplicates."],
              ["Lead-verified preference", "Lead-verified cases get +0.10 in evidence ranking; the lead's own diagnosis "
                                           "of this case (LEAD-…) always ranks first; clip_answer() never cuts the "
                                           "lead's block when a resolution is shortened for a prompt."],
              ["Neighbour label votes", "Similarity-weighted label distribution of retrieved cases, used as a prior for "
                                        "triage."],
              ["Evidence pool", "All retrieval paths merge into one pool; the model sees a compact digest and full "
                                "records on demand (get_ticket_by_id, also for LEAD- ids)."],
              ["No repeated evidence", "search_similar_tickets excludes case ids already in the pool (Qdrant must_not), "
                                       "so a tool search only ever adds new cases."],
              ["Multi-case comparison", "Prompts require consensus vs conflicts; respond reports conflicts; the evidence "
                                        "check requires ≥2 valid citations for a resolution (or the lead's diagnosis)."],
              ["No leakage in evaluation", "Evaluation tickets were held out at ingestion and never indexed."],
          ], [0.24, 0.76], bold_first=True)]

    # ---------------- 9
    s += [CondPageBreak(250), h1("9. Tools and skills"),
          h2("Tools (the model decides if, when and which)"),
          table(["Tool", "Purpose", "State effect"], [
              ["search_similar_tickets", "Targeted hybrid search with optional queue / type / priority / tag filters "
                                         "(k ≤ 6). Cases already in the evidence pool are excluded in Qdrant, so it "
                                         "returns only new evidence; results flag new and lead-verified cases.",
               "adds evidence + retrieval log"],
              ["get_ticket_by_id", "Full record of a historical (HT-…) or live (CASE-…) case, or a lead's diagnosis "
                                   "(LEAD-CASE-…), before relying on it.", "adds evidence"],
              ["get_customer_history", "Previous / open cases and remembered facts.", "–"],
              ["get_category_statistics", "Volume, type/priority distribution and top tags for a queue, tag or type.",
               "–"],
              ["update_classification", "Correct triage when evidence disagrees, with a reason.",
               "updates classification"],
              ["escalate_ticket", "Flag escalation with reason, target team, urgency.", "respond produces an "
                                                                                       "escalation"],
              ["ask_customer", "One focused clarification question; pauses the graph until answered. No fixed limit: "
                               "repeats, asks after \"no more details\" and requests for secrets are rejected.",
               "adds Q/A; answer returned as tool result"],
          ], [0.24, 0.52, 0.24], bold_first=True),
          h3("Tool techniques"),
          *bullets([
              "State-changing tools return LangGraph Command updates (evidence, classification, escalation flag, "
              "clarifications) together with the ToolMessage.",
              "Injected arguments (InjectedToolCallId, InjectedState, InjectedStore) are hidden from the model's schema.",
              "Model inputs are normalised (\"null\"/\"any\" → none; fuzzy match to valid labels with difflib).",
              "interrupt() inside a tool: the ToolNode pauses, and on resume the tool returns the customer's answer.",
              "Tool errors become ToolMessages (handle_tool_errors), so the model can recover instead of crashing the run.",
          ]),
          h2("Skills (progressive disclosure)"),
          p("A module is a skill only if it is loaded conditionally. Only the catalog (name + description) is shown when "
            "choosing; the full instructions enter the prompt after selection."),
          table(["Skill", "Loaded when", "Key guidance"], [
              ["incident_investigation", "triage picks it (Incident/Problem, technical queues)", "hypothesis → targeted "
               "search → open records → compare fixes; security/outage escalation"],
              ["request_fulfillment", "triage picks it (Request/Change, sales/HR/general)", "identify what is requested "
                                                                                           "and prerequisites; ask "
                                                                                           "rather than guess on "
                                                                                           "ambiguity"],
              ["billing_and_accounts", "triage picks it (billing, returns)", "never promise refunds/timelines not in "
                                                                            "evidence; never ask for card numbers"],
              ["escalation", "high priority, repeat Incident/Problem, or escalation flagged", "criteria + hand-off format "
                                                                                             "+ holding reply; used by "
                                                                                             "investigate and respond"],
          ], [0.24, 0.36, 0.40], bold_first=True)]

    # ---------------- 10
    s += [CondPageBreak(250), h1("10. Response, evidence check and confidence"),
          h2("respond"),
          p("One structured output (pydantic) for both outcomes: analysis, conflicts, resolution (customer reply), "
            "internal_actions, evidence_used (case id + how used), escalation_required, escalation_reason, target_team, "
            "urgency, handoff_summary, self_confidence. The <b>grounding rule</b>: every step or promise must appear in "
            "at least one cited case; no generic steps from the model's own knowledge; if the cases only ask for details "
            "or describe a standard process, the reply does the same."),
          p("<b>Support-lead rule.</b> When the lead gave a diagnosis or rejection feedback, respond receives it "
            "verbatim. It overrides conflicting cases and counts as support for the grounding rule; every point the lead "
            "made becomes a regular step, never \"optional\", and investigation notes that doubt it are ignored. The "
            "\"mark single-source steps optional\" rule applies to historical cases only."),
          h2("evaluate"),
          table(["Check", "Type", "Rule"], [
              ["cites_existing_evidence", "rule", "every cited id exists in the evidence pool or customer history"],
              ["compares_multiple_cases", "rule", "≥ 2 valid citations, or the lead's diagnosis cited (resolutions "
                                                  "only)"],
              ["has_customer_reply", "rule", "reply ≥ 60 characters"],
              ["classification_valid", "rule", "queue, type and priority set"],
              ["has_escalation_reason, valid_target_team", "rule", "escalations only (no LLM judge needed: a holding "
                                                                   "reply makes no factual claims)"],
              ["groundedness, addresses_issue", "LLM judge (20b)", "share of substantive claims supported by cited "
                                                                   "cases, customer info or support-lead input; "
                                                                   "information requests and process statements need "
                                                                   "no support"],
          ], [0.33, 0.16, 0.51]),
          p("Pass = judge says grounded <b>and</b> groundedness ≥ 0.7 <b>and</b> all rule checks pass. The judge runs on a "
            "different model from the writer to reduce self-preference."),
          h3("Confidence score"),
          code("""
confidence = 0.40 · groundedness
           + 0.20 · addresses_issue
           + 0.20 · retrieval_strength   (1.0 if the lead's diagnosis is cited; else mean cosine of the
                                          top-3 cited cases, mapped 0.55..0.90 → 0..1)
           + 0.10 · label_agreement      (share of cited cases in the predicted queue)
           + 0.10 · model self-confidence
           − 0.05 per unresolved conflict (max 0.15)  − 0.15 if any citation is invalid   → clamped to 0..1
"""),
          h3("Retry and escalation policy"),
          *bullets([
              "First failure: back to investigate with the judge's feedback and suggested query, one tool round, then "
              "straight to respond.",
              "Second failure: forced escalation (\"evidence check failed\"), never an unsupported answer.",
              "review: a support lead accepts, rejects with feedback, or amends with a diagnosis (auto-accept for "
              "demos/evals).",
          ])]

    # ---------------- 11
    s += [CondPageBreak(300), h1("11. Continuously growing case knowledge"), LifecycleFigure(),
          p("Figure 3 · Case lifecycle in the SQLite cases table, and when a case becomes searchable knowledge.", CAPTION),
          *bullets([
              "Every query is stored at intake as an <b>open</b> case; status moves through awaiting_customer, open, then "
              "resolved or escalated.",
              "finalize stores classification, evidence, analysis, resolution, escalation, confidence, evaluation, "
              "clarifications, review and lead guidance.",
              "The case is indexed into Qdrant (source = live) <b>only if</b> the reviewer accepted it and either it is a "
              "resolution that passed the evidence check, or it carries a support-lead diagnosis (even an escalation). "
              "Escalated cases without a diagnosis and rejected drafts stay as history but are not trusted knowledge "
              "(prevents knowledge-base poisoning).",
              "Future tickets find these cases through normal hybrid search and the trusted-new-cases lookup; the UI "
              "labels them <i>new case</i> or <i>lead-verified</i>.",
              "<b>scripts/reset_live_cases.py</b> removes live points from Qdrant (all, or one --case-id) and marks the "
              "rows indexed=0; historical tickets, memory and checkpoints are untouched. Run it before a demo so rehearsal "
              "answers do not change the next run.",
          ]),
          h2("Support-lead knowledge (lead-verified cases)"),
          p("The historical base cannot contain fixes nobody has written down yet. When the reviewing lead knows the real "
            "cause, that knowledge is used for this customer and for every similar ticket after it."),
          table(["Step", "What happens"], [
              ["Input", "The review card has Root cause and Solution fields with an <b>Apply my diagnosis</b> button "
                        "(action amend). A diagnosis typed before clicking Accept is applied too, never dropped; feedback "
                        "given with Accept is saved as a review note."],
              ["Authoritative evidence", "review stores the diagnosis in lead_guidance (and in the case row right away) "
                                         "and adds it to the evidence pool as LEAD-&lt;case_id&gt;, ranked first. The "
                                         "graph goes review → respond; no new investigation is needed."],
              ["Grounded on the lead", "respond and the judge get the diagnosis plus every rejection comment verbatim. "
                                       "Lead input overrides conflicting cases and counts as support; citing LEAD-… "
                                       "satisfies the multiple-cases check. The rewritten reply returns to review; "
                                       "Accept finalises it."],
              ["Indexed as lead-verified", "finalize upserts the case with lead_verified=true, root_cause and "
                                           "lead_solution in the payload. Its answer starts with \"[SUPPORT-LEAD "
                                           "VERIFIED] Root cause … Verified solution …\", followed by \"Reply sent: "
                                           "…\" (for an escalation only the lead's part)."],
              ["Reused, never cut", "+0.10 ranking over equally similar historical cases; prompts label them "
                                    "LEAD-VERIFIED CASE; investigate and respond prefer their fix over conflicting older "
                                    "cases; clip_answer() keeps the lead's block whole and shortens only the reply "
                                    "text."],
              ["Never downgraded", "The lead's points need no backing from other cases. investigate lists each as "
                                   "required; respond turns each into a regular step and ignores notes that doubt them. "
                                   "A note like \"X might be the issue, check it too\" becomes a normal step next to the "
                                   "diagnosis."],
              ["Finished or escalated cases", "POST /api/threads/{id}/lead-solution (also under the result in the UI) "
                                              "applies a diagnosis afterwards: the state is updated as if review had "
                                              "decided amend, then respond → evaluate → review → finalize. An escalation "
                                              "becomes a resolution built on the lead's fix and is indexed as "
                                              "lead-verified."],
          ], [0.24, 0.76], bold_first=True)]

    # ---------------- 12
    s += [CondPageBreak(250), h1("12. LLM layer, usage accounting and limits"),
          p("The Groq account is on a paid plan, so the app enforces no Groq limits: it never waits for, trims or refuses "
            "a call because of token or request counts. On the earlier free tier (8K TPM, 200K TPD per model) a "
            "per-model token bucket mirrored Groq's refill and daily guards failed fast; both were removed after the "
            "upgrade. The token trims were kept because fewer tokens also means faster answers."),
          table(["Service", "Limit", "Handling"], [
              ["Groq (paid plan)", "reported per key and model in x-ratelimit-* headers", "Usage recorded per day, API key "
               "(SHA-256 fingerprint) and model in app.db; limits shown in /api/health, never enforced; the Groq SDK "
               "retries 429/5xx (max_retries 3, honours retry-after); a remaining failure can be retried from the "
               "checkpoint."],
              ["Single request", "model context window", "Generous output caps (4,096 main / 2,048 fast) only against "
                                                         "runaway replies; no prompt shrinking; history keeps the last 40 "
                                                         "messages / 60K characters."],
              ["Qdrant Cloud (free, strict mode)", "no per-collection RPS cap; unindexed filters rejected; 1 GB RAM",
               "Payload indexes; client-side throttle 5 req/s with exponential backoff; uploads batched 128 "
               "points/request; collection-exists and counts cached."],
          ], [0.2, 0.28, 0.52], bold_first=True),
          h2("Token-efficiency techniques"),
          table(["Technique", "Effect"], [
              ["Role split: main model for investigation and the final answer; fast model for triage, judge, memory",
               "Cheaper, faster steps; judge independent from writer"],
              ["Reasoning effort low by default (configurable)", "Lower latency; reasoning tokens count as output"],
              ["Removed the separate decide (assess) node", "≈ 3K tokens saved per ticket"],
              ["Skip the notes-only call when the tool budget is spent", "≈ 3.5K tokens and one round trip saved per "
                                                                        "ticket"],
              ["Several tool calls per round, 2-round budget", "Fewer full-context re-sends"],
              ["Static-first prompts; stable evidence block across rounds", "Enables Groq prompt caching (cached tokens "
                                                                           "observed)"],
              ["Agent starts from the evidence pool; searches only when it is weak", "Skips a redundant search round (was "
                                                                                    "the first action in 27 of 29 logged "
                                                                                    "runs)"],
              ["Follow-up turns reuse evidence + classification from state", "No re-retrieval and no re-triage on a "
                                                                            "reopened case"],
              ["Grounding rule + 1 retry instead of 2", "Stopped ~27K-token failing tickets caused by generic, "
                                                        "unsupported steps"],
              ["Search returns only cases not yet in the evidence pool", "No case is sent twice in one prompt (was 4 of 5 "
                                                                        "in a measured run)"],
              ["Top 5 cases per prompt instead of 8 near-copies; triage gets labels + subjects only",
               "≈ 1.6K tokens per ticket"],
              ["Short tool descriptions; compact label shares instead of JSON", "≈ 0.5K tokens per ticket"],
              ["Escalation skill split: criteria while investigating, hand-off format only when escalating",
               "≈ 0.3K tokens per ticket"],
              ["Lead amend goes review → respond, not back to investigate", "No investigation rounds when the lead "
                                                                            "already gave the answer"],
              ["Structured-output slips repaired locally before any retry", "Avoids a second call (≈ 0.6K tokens) when "
                                                                           "the reply is only mis-formatted"],
          ], [0.6, 0.4]),
          h2("Structured output"),
          p("Every decision node returns a pydantic object via tool-style function calling. If the model mis-formats, the "
            "reply is first repaired locally (tool-call args, JSON in the text, or Groq's failed_generation); only if "
            "that fails is the call retried once in JSON mode with the schema attached. API errors (rate limit, network, "
            "auth) are surfaced instead of being retried in another format. Each call emits an llm_call event (input, "
            "output, cached and counted tokens) shown per node in the UI."),
          h2("Measured cost per ticket"),
          table(["Call", "Model", "Typical tokens"], [
              ["triage", "fast", "0.82–0.92K in / 0.12–0.19K out (was 1.33K in)"],
              ["investigate (per round)", "main", "1.9–2.8K in / 0.05–0.5K out (was 2.6–3.6K in)"],
              ["respond", "main", "1.3–2.0K in / 0.3–0.5K out (was 2.4K in)"],
              ["evaluate (judge)", "fast", "0.9–1.5K in / 0.06–0.3K out"],
              ["update_memory", "fast", "0.4–0.5K in / 0.05–0.23K out (one call)"],
              ["<b>Resolved ticket total</b>", "", "≈ 7.1–10.6K tokens in 5–6 calls (up to ~15K with an evidence-check "
                                                   "retry); each extra clarification round adds ≈ 2.2K; a lead amend "
                                                   "adds one respond and one judge call"],
          ], [0.25, 0.1, 0.65])]

    # ---------------- 13
    s += [CondPageBreak(250), h1("13. Evaluation (LangSmith)"),
          p("<font name='%s'>python -m evals.run_evals --n 3</font> uploads held-out tickets as the LangSmith dataset "
            "support-agent-heldout and runs langsmith.evaluate() on the full agent (clarifications auto-answered, review "
            "auto-accepted). Without a LangSmith key the same evaluators run locally and write data/eval_report.json. "
            "LANGSMITH_TRACING=true traces every graph run." % F["mono"]),
          table(["Evaluator", "Type", "Measures"], [
              ["classification_accuracy", "rule", "queue / type / priority exact match + tag Jaccard vs the reference"],
              ["retrieval_queue_precision", "rule", "share of retrieved cases in the reference queue"],
              ["retrieval_relevance", "LLM judge", "relevance of the top-5 retrieved cases to the ticket"],
              ["response_quality", "LLM judge", "agent reply vs the dataset's reference reply (correctness, helpfulness, "
                                                "tone)"],
              ["groundedness", "LLM judge", "reply supported by the cited cases"],
              ["tool_usage", "rule", "tools used, only valid tools, within budget"],
              ["agent_behavior", "rule", "all output fields present, confidence in range, escalation reason, evidence "
                                         "check ran, case persisted, ≥2 cases"],
          ], [0.27, 0.13, 0.60], bold_first=True)]

    # ---------------- 14
    s += [CondPageBreak(250), h1("14. API, streaming and frontend"),
          table(["Method", "Path", "Purpose"], [
              ["POST", "/api/tickets", "New ticket {customer_id, subject, body, auto_accept} → SSE stream"],
              ["POST", "/api/threads/{id}/resume", "{answer} for a clarification, or {action: accept / amend / reject, "
                                                   "feedback, root_cause, solution} for review → SSE"],
              ["POST", "/api/threads/{id}/lead-solution", "{root_cause, solution, note}: the lead's diagnosis at review "
                                                          "or for a finished / escalated case → SSE (404 unknown thread, "
                                                          "409 case busy)"],
              ["POST", "/api/threads/{id}/followup", "New turn on the same thread (or the answer to an open question) → "
                                                     "SSE"],
              ["POST", "/api/threads/{id}/retry", "Resume from the last checkpoint after an error"],
              ["GET", "/api/threads/{id}", "State snapshot + persisted event timeline"],
              ["GET", "/api/cases · /api/tickets/{id} · /api/customers/{id}/memory", "Case list · full ticket or case · "
                                                                                     "memory and history"],
              ["DELETE", "/api/customers/{id}/memory", "Forget all facts of a customer"],
              ["GET", "/api/graph · /api/health · /api/samples", "Graph topology · status incl. Groq usage and vector "
                                                                 "counts · demo scenarios"],
          ], [0.1, 0.42, 0.48]),
          p("Event types streamed: run_started, node (with summary and data), llm_call, llm_note, interrupt, completed, "
            "error. Every event is stored in SQLite (events table) so any case can be replayed later."),
          h2("Web UI features"),
          *bullets([
              "Live workflow graph drawn from the compiled graph; visited, last-step and waiting nodes highlighted; loop "
              "counts (e.g. investigate ×2).",
              "Live trace with per-node summaries, token counts per call and expandable details (tool calls, results, "
              "retrieved cases).",
              "Clarification panel (question + answer box) and review panel (draft, evidence-check result, the lead's "
              "Root cause / Solution fields, Accept · Apply my diagnosis · Reject &amp; re-investigate).",
              "Result card: status, confidence ring, classification chips, reply, internal actions, escalation, the lead's "
              "diagnosis, the index result, evidence table with similarity bars and <i>new case</i> / "
              "<i>lead-verified</i> / <i>support lead</i> labels, evidence-check metrics.",
              "Under a finished case: \"Resolve this escalation with the lead's solution\" (or \"Correct this case\"), "
              "which calls the lead-solution endpoint.",
              "Customer memory panel, case list with replay, ticket viewer (also for LEAD- ids), follow-up box, "
              "demo-scenario loader with fresh customer ids, header with Groq usage today and Qdrant counts.",
          ])]

    # ---------------- 15
    s += [CondPageBreak(250), h1("15. Techniques catalogue"),
          p("Every technique used in the implementation, grouped by area."),
          table(["Area", "Techniques"], [
              ["Agent design", "ReAct loop inside a verify-and-commit pipeline · Model-decided tool use (no fixed "
                               "sequence) · Human-in-the-loop via tool-level interrupt · Self-verification with an "
                               "independent LLM judge · Escalation as a safe fallback · Human trust gate before knowledge "
                               "grows · Expert input as authoritative evidence (lead-verified cases) · Lean graph: every "
                               "node justified by cost/value"],
              ["LangGraph", "StateGraph with typed state and reducers (add_messages, custom merge, operator.add) · "
                            "Conditional edges incl. list return for parallel fan-out · ToolNode with Command-returning "
                            "tools · InjectedState / InjectedStore / InjectedToolCallId · interrupt() + Command(resume) · "
                            "update_state(as_node=\"review\") to continue a finished thread · SqliteSaver checkpointer, "
                            "SqliteStore long-term memory · custom stream mode via get_stream_writer · Thread-based "
                            "multi-turn with per-turn state reset · recursion limit and tool budgets"],
              ["Retrieval", "Hybrid dense + BM25 with server-side RRF · Named vectors in one collection · Problem-only "
                            "embedding · Memory-enriched queries · Metadata filters + relaxation fallback · Strict-mode "
                            "payload indexes · Near-duplicate removal · Cosine re-scoring from returned vectors · "
                            "Neighbour label voting · Trusted-new-case lookup · Lead-verified ranking boost · Evidence "
                            "pool merging · Exclusion of known ids · Hold-out split against leakage"],
              ["Prompting and LLM", "Role-based model split · Pydantic structured output with JSON-mode fallback · "
                                    "Grounding rule against outside knowledge · Lead-input rules (overrides cases, never "
                                    "downgraded) · Lead-safe answer clipping · Hyphen normalisation of cited ids · "
                                    "Progressive-disclosure skills · Static-first prompts for caching · History "
                                    "trimming that never orphans tool results · Low reasoning effort + output caps"],
              ["Memory", "LLM-curated memory policy · Deterministic PII / secret filter · Upsert-by-key and explicit "
                         "forgetting · Per-customer cap · Case history from the structured store, citable as evidence"],
              ["Reliability and limits", "Measured limits before testing · Per-key token accounting (paid plan: no "
                                         "client limits) · Plan limits read from response headers · SDK retries on "
                                         "429/5xx, checkpoint retry · Local repair of structured-output slips · Persisted "
                                         "daily usage · Qdrant throttle with exponential backoff · Batched, idempotent "
                                         "uuid5 upserts · Cached counts/exists · Resumable local → cloud vector sync · "
                                         "Demo reset of live cases"],
              ["Data engineering", "Language filtering, cleaning, exact de-duplication · Length-sorted embedding batches "
                                   "· Dense-text truncation, full-text BM25 · Local CPU embeddings (no embedding API "
                                   "limits)"],
              ["Observability and UX", "SSE streaming of node and LLM-call events · Persisted event log and replay · "
                                       "Graph diagram generated from the compiled graph · Per-node token accounting · "
                                       "Optional LangSmith tracing"],
              ["Evaluation", "LangSmith datasets and experiments · Rule evaluators + LLM-as-judge evaluators · Trajectory "
                             "capture (nodes, tool calls, retrieved cases) · Local fallback report"],
              ["Security and config", "Secrets only in a gitignored .env · Env merging with precedence and aliases · "
                                      "Duplicate-key warning for .env · No secrets or payment data in memory · Fresh demo "
                                      "customers to avoid state leakage"],
          ], [0.2, 0.8], bold_first=True)]

    # ---------------- 16
    s += [CondPageBreak(250), h1("16. Verified demo runs"),
          p("Run with scripts/demo.py on the Groq free tier and Qdrant Cloud (auto-accept review, scripted customer "
            "answers)."),
          table(["Required path", "Result"], [
              ["Normal resolution", "Duplicate-charge ticket → resolved, groundedness 1.0, confidence 0.996; 5 consistent "
                                    "cases cited; indexed as trusted. Memory correctly stored nothing (no durable facts; "
                                    "the customer's name was not kept)."],
              ["Clarification + resume", "Vague \"Not working\" → the agent called ask_customer (\"which product or "
                                         "service, any error messages…\"); the graph paused, resumed with the answer, "
                                         "searched again and resolved at groundedness 1.0, confidence 0.76."],
              ["Long-term memory reuse", "Ticket 1 stored 4 facts (Enterprise plan, Azure, EMR integration module, "
                                         "telemedicine platform). A separate conversation, \"Sync delays are back\", "
                                         "loaded them and needed no clarification."],
              ["Retrieval of a newly resolved case", "In that second conversation the top evidence was the case resolved "
                                                     "minutes earlier (similarity 0.82, labelled new case) and it was "
                                                     "cited in the answer."],
              ["Support lead teaches a new fix (scenario 5, added later)", "Available with <font name='%s'>python "
               "scripts/demo.py --only lead_knowledge</font>. The lead's diagnosis (Premium dashboard over its QuickBooks "
               "RPS limit; shared cache + 15-minute refresh, or a quota increase) is applied at review and the case is "
               "indexed as lead-verified; a similar ticket from another customer should retrieve and cite it. The script "
               "prints a CHECK line for each of these. Not part of the original verified runs above." % F["mono"]],
          ], [0.28, 0.72], bold_first=True),
          h2("Behaviour observed during testing"),
          *bullets([
              "Memory and history change decisions correctly: with reused customer ids the agent skipped clarification "
              "(memory already held the answer) and escalated recurring issues. The demo therefore uses fresh customer "
              "ids per run (--reuse-customers to disable).",
              "Accepted demo answers are indexed and change the next run of the same scenario (a vague ticket can be "
              "answered from its earlier clarified copy instead of asking). scripts/reset_live_cases.py --yes restores "
              "a clean knowledge base before a demo.",
              "The evidence check works: before the grounding rule, the 120b model added generic steps not in the "
              "dataset (e.g. \"re-authorize the integration\"); the judge rejected them and the agent escalated rather "
              "than answer without support.",
              "Rate limiting (free tier, before the paid plan): the UI and the demo showed TPM waits, and when both daily "
              "budgets ran out, runs stopped with a clear message instead of looping. That limiter has since been "
              "removed.",
              "Cloud copy: 23,608 points copied in 72 s at ~330 points/s with zero retries under the 5 req/s throttle.",
          ])]

    # ---------------- 17
    s += [CondPageBreak(330), h1("17. How to run"),
          code(r"""
# backend + frontend (one command; open http://localhost:8000)
cd D:\Agent_Latest\backend
D:\Agent_Latest\.venv\Scripts\python.exe -m uvicorn app.api:app --port 8000

# scripted demo of the four required paths + the lead scenario (fresh customers each run)
D:\Agent_Latest\.venv\Scripts\python.exe scripts\demo.py
D:\Agent_Latest\.venv\Scripts\python.exe scripts\demo.py --only lead_knowledge

# before a live demo: drop resolutions indexed by rehearsals (list first, then remove)
D:\Agent_Latest\.venv\Scripts\python.exe scripts\reset_live_cases.py
D:\Agent_Latest\.venv\Scripts\python.exe scripts\reset_live_cases.py --yes

# evaluations (LangSmith if LANGSMITH_API_KEY is set, otherwise local report)
D:\Agent_Latest\.venv\Scripts\python.exe -m evals.run_evals --n 3

# rebuild the index from scratch (optional)
D:\Agent_Latest\.venv\Scripts\python.exe scripts\ingest.py
D:\Agent_Latest\.venv\Scripts\python.exe scripts\sync_qdrant.py

# regenerate this document
D:\Agent_Latest\.venv\Scripts\python.exe D:\Agent_Latest\docs\build_design_pdf.py
""")]

    # ---------------- 18
    s += [CondPageBreak(200), h1("18. Limitations and next steps"),
          *bullets([
              "<b>Dataset nature</b>: synthetic tickets; many historical answers ask for more details rather than "
              "solving, so grounded replies often follow that pattern. Lead-verified cases are the way to add real fixes.",
              "<b>No customer ids in the dataset</b>: customer history exists only for new tickets.",
              "<b>Lead input is trusted as authoritative</b> and there is no reviewer authentication yet: anyone with UI "
              "access can act as the support lead, and a wrong diagnosis would be indexed as lead-verified (it can be "
              "removed with reset_live_cases.py --case-id).",
              "<b>Groq limits</b>: none enforced by the app (paid plan); the plan's own limits are only reported. On the "
              "free tier the removed limiter would be needed again.",
              "<b>LLM-as-judge bias</b>: mitigated by rule checks and a different judge model, not eliminated.",
              "<b>Concurrency</b>: sync graph with SQLite is fine locally, and embedded Qdrant is single-process; an async "
              "graph with a Postgres checkpointer and a Qdrant server would be the production path.",
              "<b>Next steps</b>: cross-encoder re-ranking, a post-triage filtered retrieval pass, clustering resolutions "
              "into approaches, confidence calibration on the eval set, reviewer authentication and an audit trail for "
              "lead edits.",
          ]),
          CondPageBreak(330), h1("Appendix · Configuration reference"),
          table(["Variable", "Default", "Meaning"], [
              ["GROQ_API_KEY", "–", "Groq key (required)"],
              ["GROQ_MODEL / LLM_MAIN", "openai/gpt-oss-120b", "investigation and response model"],
              ["GROQ_FAST_MODEL / LLM_FAST", "openai/gpt-oss-20b", "triage, judge, memory model"],
              ["GROQ_MAX_OUT_MAIN · GROQ_MAX_OUT_FAST", "4096 · 2048", "output caps (against runaway replies only)"],
              ["GROQ_REASONING_EFFORT", "low", "gpt-oss reasoning effort"],
              ["QDRANT_URL · QDRANT_API_KEY · QDRANT_COLLECTION", "– · – · support_cases", "Qdrant server (empty URL = "
                                                                                        "embedded local mode)"],
              ["QDRANT_PATH", "backend/data/qdrant", "embedded store location"],
              ["QDRANT_MAX_RPS · QDRANT_UPLOAD_BATCH", "5 · 128", "Qdrant throttle and batch size"],
              ["DENSE_MODEL · SPARSE_MODEL", "BAAI/bge-small-en-v1.5 · Qdrant/bm25", "local embedding models"],
              ["RETRIEVAL_K · MAX_TOOL_ROUNDS", "6 · 2", "initial retrieval size (k+2 fetched) · tool rounds"],
              ["MAX_EVAL_RETRIES · MAX_MEMORY_FACTS", "1 · 10", "quality loop, memory cap"],
              ["LANGSMITH_API_KEY · LANGSMITH_TRACING · LANGSMITH_PROJECT", "– · false · support-investigation-agent",
               "evaluation and tracing"],
          ], [0.42, 0.28, 0.30])]
    return s


def no_orphan_headings(flowables: list) -> list:
    """A sub-heading never ends a page: start a new page if less than ~110pt is left below it."""
    out = []
    for f in flowables:
        if isinstance(f, Paragraph) and f.style.name in ("h2", "h3"):
            out.append(CondPageBreak(110))
        out.append(f)
    return out


def main() -> None:
    doc = DesignDoc(str(OUT))
    doc.multiBuild(no_orphan_headings(story()))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {doc.page} pages)")


if __name__ == "__main__":
    main()

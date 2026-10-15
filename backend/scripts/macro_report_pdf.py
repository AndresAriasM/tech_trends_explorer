"""
backend/scripts/macro_report_pdf.py

Informe PDF de resultados de la prueba macro: curvas en S, matriz
patentes/papers y Hype Cycle (Google Trends) de todas las tecnologías de
`macro_results.json`. Centrado en resultados; el método va en un recuadro.

Necesita matplotlib y reportlab, que no están en el venv del backend:

    cd backend
    uv run --with matplotlib --with reportlab --with numpy --with scipy \
        python -m scripts.macro_report_pdf ../data/macro_test
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.s_curve_models import model_curve  # noqa: E402

# ── Identidad ────────────────────────────────────────────────────────────────
INK = "#131B2A"
BODY = "#3D4757"
MUTED = "#6B7585"
RULE = "#DCE1E8"
PAPERS = "#2B6CC4"
PATENTS = "#D9622B"
ALT = "#B5BDC9"
AI = "#5A3FB0"
QUAD = {
    "maturity": ("#2E7D4F", "#E1F0E6", "Maturity and optimization"),
    "industrial": ("#B2481A", "#FBE8DD", "Industrial exploitation and applied science"),
    "disruption": ("#5A3FB0", "#EEE7F8", "Disruption opportunity"),
    "watch": ("#2B5FB0", "#E2EAF7", "Technology watch and future bets"),
}
FONTS = Path("/System/Library/Fonts/Supplemental")

ORDER = ["facial_emotion_recognition", "m2m_commerce", "fresh_water", "personal_health_dx", "synthetic_pandemic"]
SHORT = {
    "facial_emotion_recognition": ("T1", "Facial recognition"),
    "m2m_commerce": ("T2", "M2M commerce"),
    "fresh_water": ("T3", "Fresh water"),
    "personal_health_dx": ("T4", "Health diagnostics"),
    "synthetic_pandemic": ("T5", "Synthetic pandemic"),
}
# Lectura de IA por tecnología (la misma del deck), marcada como tal en el PDF.
AI_READING = {
    "facial_emotion_recognition": "Identification is mature; the frontier has moved. Patents peaked in 2020 as deep learning was industrialized, while papers still grew 12% in 2025, likely in emotion recognition and facial synthesis.",
    "m2m_commerce": "The matrix sees the old wave, not the new one. IoT-era M2M payment patents peaked in 2021 and halved by 2025; agentic commerce appears in 2025–26 news with the most positive tone of the five, but not yet in patents.",
    "fresh_water": "Steady infrastructure, not an emerging technology. Patents have run flat at 2,300–3,000 a year since 2016 while papers rise steadily: a linear regime that S-curves read as pre-inflection.",
    "personal_health_dx": "Patents plateaued after the testing boom; research did not. Patents have held near 900 a year since 2021 while papers reached a record 4,833 in 2025.",
    "synthetic_pandemic": "A wild card, not a technology. 335 biosecurity patents in 27 years cannot place a pandemic risk on an innovation matrix; the research trend and the negative news tone are the useful signals.",
}
PHASE_EN = {
    "Pre-inflexión": "Pre-inflection", "Emergente": "Emerging", "Crecimiento": "Growth",
    "Aceleración": "Accelerating", "Maduración": "Maturing", "Saturación": "Saturated",
}


def setup_fonts() -> None:
    for name, f in [("Arial", "Arial.ttf"), ("Arial-Bold", "Arial Bold.ttf"),
                    ("Georgia", "Georgia.ttf"), ("Georgia-Bold", "Georgia Bold.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(FONTS / f)))
        font_manager.fontManager.addfont(str(FONTS / f))
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 9, "axes.edgecolor": "#9AA4B2",
        "axes.labelcolor": BODY, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "axes.titlesize": 10, "axes.titlecolor": INK, "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": "#ECEFF3", "grid.linewidth": 0.6,
    })


def kfmt(v, _=None):
    v = float(v)
    if abs(v) >= 1e6:
        return f"{v/1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def png(fig, width_pt: float) -> Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width_pt, width_pt * ratio
    return img


# ── Gráficas ────────────────────────────────────────────────────────────────

def scurve(ax, series: dict, color: str, label: str, x0=1980, x1=2040):
    yearly = series["yearly"]
    years = [r["year"] for r in yearly]
    cum = [r["cumulative"] for r in yearly]
    t0 = years[0] - 1
    grid = np.arange(years[0], x1 + 1)
    lo, hi = series["inflection_range"] or (None, None)
    if lo and hi:
        ax.axvspan(max(lo, x0), min(hi, x1), color=color, alpha=0.07, lw=0)
    ax.axvspan(2025.5, x1, color="#F1F3F6", lw=0, zorder=0)
    ymax = max(cum)
    for m in series["models"]:
        if m["is_best"]:
            continue
        c = model_curve(m["key"], m["params"], grid, t0)
        ax.plot(grid, c, color=ALT, lw=1)
        ymax = max(ymax, min(max(c), 6 * max(cum)))
    best = next(m for m in series["models"] if m["is_best"])
    c = model_curve(best["key"], best["params"], grid, t0)
    ymax = max(ymax, min(max(c), 6 * max(cum)))
    ax.plot(grid, c, color=INK, lw=1.8, ls="--")
    ax.scatter(years, cum, s=9, color=color, zorder=3)
    med = series["inflection_median"]
    if med and x0 <= med <= x1:
        ax.axvline(med, color=color, lw=1.2)
        near_edge = med > x1 - 9
        ax.text(med - 0.6 if near_edge else med + 0.6, ymax * 0.97, f"median\ninflection\n{med:.0f}",
                color=color, fontsize=7.5, va="top", ha="right" if near_edge else "left")
    elif med:
        ax.text(x0 + 1, ymax * 0.97, f"median inflection {med:.0f}\n(beyond the axis)", color=color,
                fontsize=7.5, va="top", ha="left")
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, ymax * 1.05)
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_title(f"{label} · cumulative", pad=6)
    ax.text(2025.8, ymax * 0.02, "projection", color=MUTED, fontsize=7)


def annual(ax, p: dict, t: dict, x0=1980):
    for s, c, lab in ((p, PAPERS, "Papers"), (t, PATENTS, "Patents")):
        ys = [r["year"] for r in s["yearly"] if r["year"] >= x0]
        vs = [r["annual"] for r in s["yearly"] if r["year"] >= x0]
        ax.plot(ys, vs, color=c, lw=1.8, label=lab)
        i = int(np.argmax(vs))
        ax.scatter([ys[i]], [vs[i]], color=c, s=18, zorder=3)
        ax.annotate(f"{vs[i]:,}", (ys[i], vs[i]), xytext=(-4, 5), textcoords="offset points",
                    ha="right", fontsize=7.5, color=c)
    ax.set_xlim(x0, 2025.5)
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Annual output", pad=6)


def trends(ax, ht: dict):
    term, r = next(iter(ht["terms"].items()))
    mon = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    tt = np.array([int(x["date"][-4:]) + mon[x["date"][:3]] / 12 for x in r["monthly"]])
    vv = np.array([x["value"] for x in r["monthly"]], float)
    ax.plot(tt, vv, color="#C9CFD8", lw=0.7)
    ma = np.convolve(vv, np.ones(12) / 12, mode="same")
    keep = (tt > tt[0] + 0.5) & (tt < tt[-1] - 0.5)
    ax.plot(tt[keep], ma[keep], color=BODY, lw=1.4)
    if r.get("fitted"):
        ax.plot([p["t"] for p in r["fitted"]], [p["y"] for p in r["fitted"]], color=AI, lw=1.8, ls="--")
    brk = ht.get("fit_end") or 2025.58
    ax.axvspan(brk, tt[-1] + 0.1, color="#F1F3F6", lw=0)
    ax.text(brk + 0.1, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 90, "excluded\n(Trends break)",
            fontsize=6.5, color=MUTED, va="top")
    ax.set_xlim(2004, tt[-1] + 0.1)
    ax.set_title(f'Google Trends · "{term}"', pad=6)
    ph = r["phase"]["phase"]
    ax.text(0.02, 0.95, ph, transform=ax.transAxes, color=AI, fontsize=8.5, fontweight="bold", va="top")


def tech_figure(t: dict) -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 4.9), gridspec_kw={"hspace": 0.45, "wspace": 0.16})
    scurve(axs[0, 0], t["papers"], PAPERS, "Papers (Scopus)")
    scurve(axs[0, 1], t["patents"], PATENTS, "Patents (EPO OPS)")
    annual(axs[1, 0], t["papers"], t["patents"])
    trends(axs[1, 1], t["hype_trends"])
    return fig


def matrix_figure(techs: list[dict], ref=2025) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    X0, X1, Y0, Y1 = 2010, 2100, 2010, 2040
    for q, (x_a, x_b, y_a, y_b) in {
        "maturity": (X0, ref, Y0, ref), "industrial": (ref, X1, Y0, ref),
        "disruption": (X0, ref, ref, Y1), "watch": (ref, X1, ref, Y1),
    }.items():
        ax.add_patch(plt.Rectangle((x_a, y_a), x_b - x_a, y_b - y_a, color=QUAD[q][1], lw=0, zorder=0))
    ax.axvline(ref, color=INK, lw=1, ls="--")
    ax.axhline(ref, color=INK, lw=1, ls="--")
    ax.text(X0 + 0.8, Y1 - 0.6, "Disruption\nopportunity", color=QUAD["disruption"][0], fontsize=8.5, fontweight="bold", va="top")
    ax.text(X1 - 0.8, Y1 - 0.6, "Technology watch\nand future bets", color=QUAD["watch"][0], fontsize=8.5, fontweight="bold", va="top", ha="right")
    ax.text(X0 + 0.8, Y0 + 0.6, "Maturity", color=QUAD["maturity"][0], fontsize=8.5, fontweight="bold")
    ax.text(X1 - 0.8, Y0 + 0.6, "Industrial exploitation\nand applied science", color=QUAD["industrial"][0], fontsize=8.5, fontweight="bold", ha="right")
    offsets = {"T1": (-10, 10), "T2": (-10, 10), "T3": (-10, -16), "T4": (8, 8), "T5": (10, -10)}
    for t in techs:
        m = t["matrix"]
        tid, name = SHORT[t["slug"]]
        x, y = m["papers_inflection"], m["patents_inflection"]
        (xl, xh), (yl, yh) = m["papers_range"], m["patents_range"]
        col = QUAD[m["quadrant"]][0]
        ax.plot([max(xl, X0), min(xh, X1)], [y, y], color=col, lw=1.6, alpha=0.6)
        ax.plot([x, x], [max(yl, Y0), min(yh, Y1)], color=col, lw=1.6, alpha=0.6)
        if xh > X1:
            ax.annotate("", (X1, y), (X1 - 2.5, y), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        if yh > Y1:
            ax.annotate("", (x, Y1), (x, Y1 - 1.2), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.scatter([x], [y], s=110, color=col, edgecolor="white", lw=1.8, zorder=4)
        dx, dy = offsets[tid]
        ax.annotate(f"{tid} {name}\n{x:.0f} / {y:.0f}", (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color=INK, fontweight="bold", ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top")
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.grid(False)
    ax.set_xlabel("Papers inflection year (median of five models)")
    ax.set_ylabel("Patents inflection year")
    ax.set_xticks([2010, 2025, 2040, 2055, 2070, 2085, 2100])
    ax.set_yticks([2010, 2015, 2020, 2025, 2030, 2035, 2040])
    return fig


# ── Documento ───────────────────────────────────────────────────────────────

def styles():
    return {
        "eyebrow": ParagraphStyle("eyebrow", fontName="Arial-Bold", fontSize=8, textColor=colors.HexColor("#0E6E77"), leading=10),
        "h1": ParagraphStyle("h1", fontName="Georgia-Bold", fontSize=24, leading=28, textColor=colors.HexColor(INK)),
        "h2": ParagraphStyle("h2", fontName="Georgia-Bold", fontSize=16, leading=20, textColor=colors.HexColor(INK)),
        "body": ParagraphStyle("body", fontName="Arial", fontSize=9.5, leading=13.5, textColor=colors.HexColor(BODY)),
        "small": ParagraphStyle("small", fontName="Arial", fontSize=7.8, leading=10.5, textColor=colors.HexColor(MUTED)),
        "ai": ParagraphStyle("ai", fontName="Arial", fontSize=9.2, leading=13, textColor=colors.HexColor(BODY),
                             backColor=colors.HexColor("#F1EFFA"), borderPadding=(6, 8, 6, 8)),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.6, leading=11, textColor=colors.HexColor(BODY)),
        "cellb": ParagraphStyle("cellb", fontName="Arial-Bold", fontSize=8.6, leading=11, textColor=colors.HexColor(INK)),
    }


def table(rows, widths, header=True, zebra_total=False):
    tb = Table(rows, colWidths=widths, hAlign="LEFT")
    st = [
        ("FONT", (0, 0), (-1, -1), "Arial", 8.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(BODY)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(RULE)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st += [("FONT", (0, 0), (-1, 0), "Arial-Bold", 7.6), ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(MUTED)),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#9AA4B2"))]
    if zebra_total:
        st += [("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EEF1F5")), ("FONT", (0, -1), (-1, -1), "Arial-Bold", 8.6)]
    tb.setStyle(TableStyle(st))
    return tb


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    w, _ = landscape(A4)
    canvas.drawString(15 * mm, 9 * mm, "Five futures on the patent–paper matrix · Prospecta · runs of 13–14 October 2026")
    canvas.drawRightString(w - 15 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


def build(root: Path) -> Path:
    setup_fonts()
    data = json.loads((root / "macro_results.json").read_text())
    techs = {t["slug"]: t for t in data["technologies"]}
    techs = [techs[s] for s in ORDER if s in techs]
    S = styles()
    W = landscape(A4)[0] - 30 * mm

    out = root / "Five_Futures_Results.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=landscape(A4), leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=13 * mm, bottomMargin=15 * mm, title="Five futures on the patent–paper matrix",
                            author="Andrés Arias Medina")
    story = []

    # ── 1. Resumen ──
    tp = sum(t["papers"]["total"] for t in techs)
    tt = sum(t["patents"]["total"] for t in techs)
    tn = sum(t["hype"]["news_total"] for t in techs)
    story += [
        Paragraph("PROSPECTA · TECHNOLOGY FORESIGHT TEST · RESULTS", S["eyebrow"]), Spacer(1, 4),
        Paragraph("Five futures on the patent–paper matrix", S["h1"]), Spacer(1, 6),
        Paragraph(f"S-curves of <b>{tp:,}</b> scientific papers (Scopus) and <b>{tt:,}</b> patent publications (EPO OPS), "
                  f"1960–2025, plus Google Trends attention and the tone of <b>{tn:,}</b> news items, for five technologies "
                  "rated by a foresight panel.", S["body"]),
        Spacer(1, 10),
    ]
    rows = [["ID", "Technology", "Panel", "Papers", "Patents", "Papers infl.", "Patents infl.", "Quadrant (ref. 2025)", "Attention phase"]]
    for t in techs:
        tid, _ = SHORT[t["slug"]]
        m = t["matrix"]
        ph = next(iter(t["hype_trends"]["terms"].values()))["phase"]["phase"]
        rows.append([
            Paragraph(f"<b>{tid}</b>", S["cellb"]), Paragraph(t["spec"]["name"], S["cell"]),
            f"{t['spec'].get('expert_priority_pct', '')}%", f"{t['papers']['total']:,}", f"{t['patents']['total']:,}",
            f"{m['papers_inflection']:.0f}", f"{m['patents_inflection']:.0f}",
            Paragraph(f"<font color='{QUAD[m['quadrant']][0]}'><b>{QUAD[m['quadrant']][2]}</b></font>"
                      + ("" if m["robust"] else " <font color='#6B7585'>(model-sensitive)</font>"), S["cell"]),
            Paragraph(ph + (" (proxy)" if t["slug"] == "synthetic_pandemic" else ""), S["cell"]),
        ])
    rows.append(["", "All five", "", f"{tp:,}", f"{tt:,}", "", "", "", ""])
    tb = table(rows, [26, 175, 36, 54, 54, 56, 60, 186, 110], zebra_total=True)
    tb.setStyle(TableStyle([("ALIGN", (2, 1), (6, -1), "RIGHT"), ("ALIGN", (2, 0), (6, 0), "RIGHT")]))
    story += [tb, Spacer(1, 12)]
    story.append(Table([[
        Paragraph("<b>Key results.</b> Every paper series still peaks in its last year, while four of five patent series "
                  "peaked between 2017 and 2022. No technology sits in the disruption quadrant. Only T3 keeps its "
                  "quadrant under all five S-curve models; the others are sensitive to model choice.", S["body"]),
        Paragraph("<b>Method in brief.</b> Annual counts from Scopus and EPO OPS (66 years each); five 3-parameter "
                  "S-curve models per series; the matrix uses the median inflection year of the five, split at 2025. "
                  "Attention: Google Trends, monthly 2004–Jul 2025, fitted with an expectation bell plus an adoption curve.",
                  S["small"]),
    ]], colWidths=[W * 0.56, W * 0.44], style=TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 18),
        ("BOX", (1, 0), (1, 0), 0.6, colors.HexColor(RULE)), ("LEFTPADDING", (1, 0), (1, 0), 10),
        ("TOPPADDING", (1, 0), (1, 0), 8), ("BOTTOMPADDING", (1, 0), (1, 0), 8),
    ])))
    story.append(PageBreak())

    # ── 2. Matriz ──
    side = [
        Paragraph("Where the five land", S["h2"]), Spacer(1, 8),
        Paragraph("Each point is the median inflection year of five S-curve models (papers on X, patents on Y). "
                  "Bars span the range across models; arrows mark ranges that run past the axis.", S["body"]), Spacer(1, 10),
    ]
    for q, txt in [
        ("watch", "<b>T3, T4.</b> Neither curve has turned. Technology watch, exploratory research, small pilots."),
        ("industrial", "<b>T1, T5.</b> Patents turned first. Scale patented technology, applied research with industry. T5 placement rests on a weak proxy."),
        ("maturity", "<b>T2.</b> Both turned. Cost efficiency, incremental and process innovation."),
        ("disruption", "<b>None.</b> Mature science with technology still to build: transfer and prototyping."),
    ]:
        side += [Paragraph(f"<font color='{QUAD[q][0]}'><b>{QUAD[q][2]}</b></font>", S["cellb"]),
                 Paragraph(txt, S["cell"]), Spacer(1, 7)]
    side += [Spacer(1, 4), Paragraph("Quadrants and recommendations follow Zartha et al., “Toma de decisiones en "
                                     "Agroindustria apoyada en Cuadrantes Matriz Patentes – Papers”.", S["small"])]
    story += [
        Paragraph("PATENT–PAPER MATRIX · REFERENCE YEAR 2025", S["eyebrow"]), Spacer(1, 6),
        Table([[png(matrix_figure(techs), W * 0.64), side]], colWidths=[W * 0.68, W * 0.32],
              style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)])),
        PageBreak(),
    ]

    # ── 3–7. Una página por tecnología ──
    for t in techs:
        tid, _ = SHORT[t["slug"]]
        m = t["matrix"]
        p, pt = t["papers"], t["patents"]
        hv = t["hype"]
        ph_p = PHASE_EN.get(p["phase"]["phase"], p["phase"]["phase"])
        ph_t = PHASE_EN.get(pt["phase"]["phase"], pt["phase"]["phase"])
        facts = [[
            Paragraph(f"<b>{p['total']:,}</b> papers · peak {p['peak_annual']:,} in {p['peak_annual_year']} · {ph_p}", S["cell"]),
            Paragraph(f"<b>{pt['total']:,}</b> patents · peak {pt['peak_annual']:,} in {pt['peak_annual_year']} · {ph_t}", S["cell"]),
            Paragraph(f"Inflection <b>{m['papers_inflection']:.0f}</b> / <b>{m['patents_inflection']:.0f}</b> "
                      f"(ranges {m['papers_range'][0]:.0f}–{m['papers_range'][1]:.0f} / {m['patents_range'][0]:.0f}–{m['patents_range'][1]:.0f})", S["cell"]),
            Paragraph(f"News {hv['news_total']} · tone {hv['metrics']['avg_sentiment']:+.2f}", S["cell"]),
        ]]
        story += [KeepTogether([
            Paragraph(f"{tid} · PANEL SCORE {t['spec'].get('expert_priority_pct')}%", S["eyebrow"]), Spacer(1, 3),
            Paragraph(f"{t['spec']['name']} &nbsp;<font size=11 color='{QUAD[m['quadrant']][0]}'>{QUAD[m['quadrant']][2]}</font>", S["h2"]),
            Spacer(1, 6),
            table(facts, [W * 0.25, W * 0.25, W * 0.33, W * 0.17], header=False),
            Spacer(1, 4),
            png(tech_figure(t), W),
            Spacer(1, 2),
            Paragraph(f"<font color='{AI}'><b>AI-generated reading.</b></font> {AI_READING[t['slug']]}", S["ai"]),
            Spacer(1, 3),
            Paragraph("Dots: observed cumulative counts · dashed black: AICc-selected model · grey: other four models · "
                      "shaded band: inflection range across models · Trends: monthly index, 12-month average (dark), fitted curve (violet).",
                      S["small"]),
        ]), PageBreak()]

    # ── 8. Síntesis ──
    story += [
        Paragraph("AI-GENERATED SYNTHESIS · LIMITATIONS", S["eyebrow"]), Spacer(1, 6),
        Paragraph("What the results mean for the panel", S["h2"]), Spacer(1, 8),
    ]
    left = [
        Paragraph(f"<font color='{AI}'><b>AI-generated analysis</b></font> — written by Claude (Anthropic) standing in for "
                  "the platform assistant, from the figures in this report. Not reviewed by the panel.", S["small"]), Spacer(1, 6),
        Paragraph("<b>No disruption opportunity.</b> None of the five combines mature science with technology still to be exploited.", S["body"]), Spacer(1, 5),
        Paragraph("<b>The top score goes to the most industrial.</b> T1 (63%) has the clearest patent peak; T3 and T4 have the most open science.", S["body"]), Spacer(1, 5),
        Paragraph("<b>Tone and patents diverge.</b> T2 and T4 carry the most positive news (+0.39, +0.31) while their patent curves have flattened.", S["body"]), Spacer(1, 9),
        Paragraph("<b>For the next Delphi round</b>", S["cellb"]), Spacer(1, 3),
        Paragraph("• Split T1 into face identification, emotion recognition and facial synthesis.<br/>"
                  "• Add agentic commerce as its own variable next to T2.<br/>"
                  "• Reframe T5 as a wild card; track DNA-synthesis screening rules and biosecurity research.<br/>"
                  "• Add atmospheric water harvesting under T3.", S["body"]),
    ]
    right = [
        Paragraph("<b>Limitations</b>", S["cellb"]), Spacer(1, 4),
        Paragraph("• <b>Model uncertainty.</b> Inflection ranges span decades where a curve has not turned; Durbin-Watson "
                  "values are well below 2 for most series.<br/>"
                  "• <b>Recent patent years</b> (2024–2025) are understated by publication and classification lags.<br/>"
                  "• <b>Counting units.</b> Scopus counts all document types; EPO counts publications, not families.<br/>"
                  "• <b>Proxies.</b> T5 is measured through biosecurity work; each Trends term is one phrase for a whole technology.<br/>"
                  "• <b>Trends break.</b> All five series jumped about 2.3× in August 2025; fits stop in July 2025.", S["body"]),
        Spacer(1, 10),
        Paragraph("Data: data/macro_test/macro_results.json · Code: backend/scripts/macro_test.py, macro_consolidate.py, "
                  "macro_report_pdf.py, backend/core/hype_trends.py", S["small"]),
    ]
    story.append(Table([[left, right]], colWidths=[W * 0.5, W * 0.5], style=TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), 20),
    ])))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "../data/macro_test").resolve()))

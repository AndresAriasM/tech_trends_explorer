"""
LACCEI 2025 - Hype Cycle PDF Generator
Loads the survey workbook, computes the dominant Hype Cycle phase per country
per technology, and writes a single multi-page PDF with several charts on each
page. All output text is in English.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
EXCEL_FILE = HERE / (
    "HYPECYCLE EN TECNOLOGÍAS E INNOVACIONES EMERGENTES EN IA. "
    "PARTICIPANTES PARA CERTIFICACIONES (1).xlsx"
)
OUTPUT_PDF = HERE / "Hype_Cycle_LACCEI_2025_AI_Technologies.pdf"

# ── Configuration ──────────────────────────────────────────────────────────
DROP_COLS = [0, 1, 2, 3, 4, 5, 7, 34]
TARGET_COUNTRIES = ["Peru", "Colombia", "Ecuador", "Mexico", "Costa Rica", "Honduras"]
MIN_RESPONSES_PER_CELL = 3
CHARTS_PER_PAGE = 4  # 2 columns x 2 rows per page

PHASES = [
    "Innovation Trigger",
    "Peak of Inflated Expectations",
    "Trough of Disillusionment",
    "Slope of Enlightenment",
    "Plateau of Productivity",
]

# The survey captures "time-to-adoption" estimates, not Hype Cycle phases
# directly. We translate each time bucket to a Hype Cycle phase using the
# canonical Gartner reading: the further the adoption horizon, the earlier in
# the cycle. This mapping is documented on the methodology page of the PDF.
TIME_PHASE_MAP = {
    "adoption has already been achieved":   "Plateau of Productivity",
    "under 2 years":                        "Slope of Enlightenment",
    "between 2 and 5 years":                "Trough of Disillusionment",
    "between 5 and 10 years":               "Peak of Inflated Expectations",
    "more than 10 years":                   "Innovation Trigger",
}

# Human-readable English labels for each bucket — used on the methodology page.
TIME_BUCKET_LABELS = [
    ("Adoption already achieved",  "Plateau of Productivity"),
    ("Under 2 years",              "Slope of Enlightenment"),
    ("Between 2 and 5 years",      "Trough of Disillusionment"),
    ("Between 5 and 10 years",     "Peak of Inflated Expectations"),
    ("More than 10 years",         "Innovation Trigger"),
]


def normalize_phase_value(value):
    """Map a raw survey cell to a Hype Cycle phase, or None if not applicable."""
    if pd.isna(value):
        return None
    text = str(value).replace("\xa0", " ").strip().lower()
    if "dk/nr" in text or "ns/nr" in text:
        return None
    # The survey cell is bilingual (English - Spanish); we match on the English
    # portion which always appears before the hyphen.
    english = text.split(" - ")[0].strip()
    for key, phase in TIME_PHASE_MAP.items():
        if english.startswith(key):
            return phase
    return None

TECH_NAMES = [
    "Country",
    "Conversational AI agents",
    "Robotic Process Automation (RPA) with AI",
    "Autonomous business operations",
    "Intelligent virtual assistants",
    "AI Agentic workflow systems",
    "Large Language Models (LLMs)",
    "Text-to-image generation",
    "Code generation tools",
    "Multimodal AI systems",
    "Content creation platforms",
    "AI ethics and compliance platforms",
    "Model bias detection and mitigation",
    "AI audit and monitoring tools",
    "Responsible AI frameworks",
    "AI security and privacy solutions",
    "AI-powered business process management",
    "Workflow optimization platforms",
    "Human-AI collaboration interfaces",
    "Process mining with AI",
    "Decision automation systems",
    "Multi-model orchestration",
    "Model lifecycle management",
    "AI model marketplaces",
    "Federated learning systems",
    "Model performance optimization",
    "Comments",
]

COUNTRY_COLORS = {
    "Peru":       "#E74C3C",
    "Colombia":   "#F39C12",
    "Ecuador":    "#3498DB",
    "Mexico":     "#27AE60",
    "Costa Rica": "#9B59B6",
    "Honduras":   "#E67E22",
}

# ── Step 1: Load and normalise data ────────────────────────────────────────
def normalize_country(name):
    if pd.isna(name):
        return name
    name = str(name).strip()
    name = name.title()
    name = re.sub(r"\s+", " ", name)
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name


COUNTRY_CORRECTIONS = {
    "Peru ": "Peru",
    "Usa": "USA",
    "United States": "USA",
    "Puerto RicoOa": "Puerto Rico",
    "Puerto Rico0A": "Puerto Rico",
    "Lima _ Peru": "Peru",
    "Lima _Peru": "Peru",
    "Lima- Peru": "Peru",
    "Lima Peru": "Peru",
    "Lima": "Peru",
    "Costa Eica": "Costa Rica",
    "Costa Rica ": "Costa Rica",
    "Telus University & Florida Atlantic University": "USA",
    "Telus University": "USA",
    "Florida Atlantic University": "USA",
    "Teluq University & Florida Atlantic University": "USA",
    "Brasil": "Brazil",
    "Venezuela ": "Venezuela",
    "Panama": "Panama",
    "Puerto Rico/Us": "Puerto Rico",
}


def load_and_clean(excel_path: Path) -> pd.DataFrame:
    df = pd.read_excel(excel_path)
    df = df.drop(df.columns[DROP_COLS], axis=1)
    df = df.rename(columns={df.columns[0]: "Country"})
    df["Country"] = df["Country"].apply(normalize_country).replace(COUNTRY_CORRECTIONS)
    df.columns = TECH_NAMES[: len(df.columns)]
    return df


# ── Step 2: Build summary table (country x technology) ─────────────────────
def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    df_f = df[df["Country"].isin(TARGET_COUNTRIES)].copy()
    tech_cols = list(df_f.columns[1:26])

    # Map every cell from the raw time-bucket value to the Hype Cycle phase.
    for col in tech_cols:
        df_f[col] = df_f[col].apply(normalize_phase_value)

    rows = []
    for country in TARGET_COUNTRIES:
        df_c = df_f[df_f["Country"] == country]
        for tech in tech_cols:
            valid = df_c[df_c[tech].isin(PHASES)]
            if len(valid) < MIN_RESPONSES_PER_CELL:
                continue
            counts = valid[tech].value_counts()
            dominant = counts.index[0]
            dominant_pct = (counts.iloc[0] / counts.sum()) * 100
            rows.append({
                "Country": country,
                "Technology": tech,
                "Dominant_Phase": dominant,
                "Total_Responses": int(len(valid)),
                "Dominant_Percentage": round(dominant_pct, 1),
            })
    return pd.DataFrame(rows)


# ── Step 3: Drawing helpers ────────────────────────────────────────────────
def hype_cycle_curve():
    control = np.array([
        [0.0,  5],
        [1.5, 25],
        [2.5, 85],
        [4.0, 20],
        [5.5,  8],
        [7.0, 25],
        [8.5, 45],
        [10.0, 50],
    ])
    x_smooth = np.linspace(0, 10, 200)
    spline = make_interp_spline(control[:, 0], control[:, 1], k=3)
    y_smooth = np.maximum(spline(x_smooth), 2)
    return x_smooth, y_smooth


PHASE_CENTERS = {
    "Innovation Trigger":            (1.0, 20),
    "Peak of Inflated Expectations": (2.5, 75),
    "Trough of Disillusionment":     (5.0, 12),
    "Slope of Enlightenment":        (7.0, 30),
    "Plateau of Productivity":       (9.0, 48),
}


def place_country_points(base_x, base_y, n):
    """Return (x[], y[]) for n country dots laid out around the phase center."""
    if n == 1:
        return [base_x], [base_y]
    if n == 2:
        return [base_x, base_x], [base_y + 5, base_y - 5]
    if n == 3:
        return [base_x, base_x - 0.25, base_x + 0.25], [base_y + 5, base_y - 5, base_y - 5]
    if n == 4:
        return ([base_x - 0.25, base_x + 0.25, base_x - 0.25, base_x + 0.25],
                [base_y + 5, base_y + 5, base_y - 5, base_y - 5])
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 0.3
    xs = base_x + radius * np.cos(angles)
    ys = base_y + radius * 5 * np.sin(angles)
    return list(xs), list(ys)


def draw_single_hype_cycle(ax, tech_name: str, countries_by_phase: dict):
    x_curve, y_curve = hype_cycle_curve()
    ax.plot(x_curve, y_curve, linewidth=2.5, color="#2C3E50", alpha=0.85, zorder=2)
    ax.fill_between(x_curve, y_curve, alpha=0.07, color="#3498DB", zorder=1)

    for x_pos in [0.5, 2.0, 4.0, 6.0, 8.0]:
        ax.axvline(x=x_pos, color="gray", linestyle=":", alpha=0.3, linewidth=0.8, zorder=1)

    legend_seen = {}
    for phase, entries in countries_by_phase.items():
        if not entries:
            continue
        base_x, base_y = PHASE_CENTERS[phase]
        xs, ys = place_country_points(base_x, base_y, len(entries))
        for (country, responses), x, y in zip(entries, xs, ys):
            color = COUNTRY_COLORS.get(country, "#95A5A6")
            size = 110 + min(responses * 5, 70)
            ax.scatter(x, y, s=size, c=color, alpha=0.92,
                       edgecolors="white", linewidth=2.0, zorder=5)
            legend_seen.setdefault(country, (color, responses))

    # Phase labels at the bottom (inside the plot area, above the X axis line)
    phase_short = [
        "Innovation\nTrigger",
        "Peak of\nInflated Exp.",
        "Trough of\nDisillusionment",
        "Slope of\nEnlightenment",
        "Plateau of\nProductivity",
    ]
    for label, (x_pos, _) in zip(phase_short, PHASE_CENTERS.values()):
        ax.text(x_pos, -8, label, ha="center", va="top",
                fontsize=7, fontweight="bold", color="#334155",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#ECEFF1", edgecolor="none", alpha=0.85))

    # Title
    title = tech_name if len(tech_name) <= 50 else tech_name[:47] + "..."
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1a2332")

    # Compact country legend
    if legend_seen:
        items = sorted(legend_seen.items())
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color, markersize=7,
                   markeredgecolor="white", markeredgewidth=1.2,
                   label=f"{country} (n={resp})")
            for country, (color, resp) in items
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7,
                  frameon=True, fancybox=True, framealpha=0.92,
                  borderpad=0.4, handletextpad=0.4)

    # Axis limits and labels. Each chart is meant to stand on its own when the
    # research team extracts it into another document, so axes carry explicit
    # direction-of-meaning labels with an arrow.
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-22, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Time / Maturity →",
                  fontsize=9, fontweight="bold", color="#1a2332", labelpad=20)
    ax.set_ylabel("Expectations / Adoption →",
                  fontsize=9, fontweight="bold", color="#1a2332", labelpad=6)

    # Subtle horizontal grid lines for visual reference (matches the Y axis).
    for y_ref in [25, 50, 75]:
        ax.axhline(y=y_ref, color="#CBD5E0", linestyle="-",
                   linewidth=0.4, alpha=0.4, zorder=0)

    ax.set_facecolor("#FAFBFC")

    # Clean axis frame: keep bottom + left as visible, hide top + right.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_edgecolor("#475569")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_edgecolor("#475569")
    ax.spines["bottom"].set_linewidth(1.0)


def organize_phases(summary: pd.DataFrame, tech: str) -> dict:
    by_phase = {p: [] for p in PHASES}
    sub = summary[summary["Technology"] == tech]
    for _, r in sub.iterrows():
        if r["Dominant_Phase"] in by_phase:
            by_phase[r["Dominant_Phase"]].append(
                (r["Country"], int(r["Total_Responses"]))
            )
    return by_phase


# ── Step 4: PDF layout ─────────────────────────────────────────────────────
def draw_cover(pdf: PdfPages, summary: pd.DataFrame, technologies: list[str]):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.93, "Hype Cycle of Emerging AI Technologies",
            ha="center", va="top", fontsize=24, fontweight="bold", color="#1a2332")
    ax.text(0.5, 0.875, "LACCEI 2025 - Country positioning by expert survey",
            ha="center", va="top", fontsize=13, color="#4a5568")
    ax.text(0.5, 0.83, datetime.now().strftime("Generated on %B %d, %Y"),
            ha="center", va="top", fontsize=10, color="#94a3b8")

    countries = sorted(summary["Country"].unique())
    n_combos = len(summary)
    modal = summary["Dominant_Phase"].mode().iloc[0]
    summary_text = (
        f"Technologies analysed:  {len(technologies)}\n"
        f"Countries included:     {len(countries)}  ({', '.join(countries)})\n"
        f"Country-technology cells:  {n_combos}\n"
        f"Global modal phase:     {modal}"
    )
    ax.text(0.5, 0.74, summary_text, ha="center", va="top", fontsize=11,
            color="#2d3748", family="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f7fafc",
                      edgecolor="#cbd5e0", linewidth=1))

    # Index of technologies
    ax.text(0.5, 0.555, "Technology Index", ha="center", va="top",
            fontsize=13, fontweight="bold", color="#1a2332")
    mid = math.ceil(len(technologies) / 2)
    col1, col2 = technologies[:mid], technologies[mid:]
    line_h = 0.024
    y0 = 0.515
    for i, t in enumerate(col1):
        ax.text(0.08, y0 - i * line_h, f"{i + 1:2d}. {t}",
                ha="left", va="top", fontsize=9, color="#2d3748")
    for i, t in enumerate(col2):
        ax.text(0.55, y0 - i * line_h, f"{mid + i + 1:2d}. {t}",
                ha="left", va="top", fontsize=9, color="#2d3748")

    ax.text(0.5, 0.05,
            "Dominant phase = the phase most voted by experts in each country "
            "(minimum 3 valid responses per cell).\n"
            "Marker size is proportional to the number of valid responses; "
            "color identifies the country.",
            ha="center", va="bottom", fontsize=8, style="italic", color="#718096")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_methodology_page(pdf: PdfPages):
    """Explicit page documenting the time-bucket -> Hype Cycle phase mapping."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")

    ax.text(0.5, 0.93, "Methodology",
            ha="center", va="top", fontsize=22, fontweight="bold", color="#1a2332")
    ax.text(0.5, 0.88,
            "How survey responses map onto the Gartner Hype Cycle",
            ha="center", va="top", fontsize=12, color="#4a5568")

    explanation = (
        "The survey asked experts in each country to estimate the time-to-adoption "
        "of every technology, choosing one of five buckets. Those buckets are not\n"
        "Hype Cycle phases by themselves; we translate them following Gartner's "
        "convention that a longer time-to-adoption corresponds to an earlier phase\n"
        "of the cycle, and shorter horizons to later phases. The dominant phase per "
        "country-technology cell is the most-voted bucket (minimum 3 responses)."
    )
    ax.text(0.5, 0.79, explanation, ha="center", va="top",
            fontsize=10, color="#2d3748")

    # Mapping table
    ax.text(0.5, 0.66, "Survey bucket  ->  Hype Cycle phase",
            ha="center", va="top", fontsize=12, fontweight="bold", color="#1a2332")
    y = 0.60
    row_h = 0.055
    for bucket, phase in TIME_BUCKET_LABELS:
        ax.text(0.20, y, bucket, ha="left", va="center",
                fontsize=11, color="#2d3748", fontweight="bold")
        ax.text(0.50, y, "->", ha="center", va="center",
                fontsize=11, color="#94a3b8")
        ax.text(0.55, y, phase, ha="left", va="center",
                fontsize=11, color="#1a4d8f", fontweight="bold")
        y -= row_h

    ax.text(0.5, 0.20,
            "Notes for the research team:",
            ha="center", va="top", fontsize=11, fontweight="bold", color="#1a2332")
    notes = (
        "- DK/NR (Don't Know / No Reply) answers are excluded from every count.\n"
        "- A country-technology cell needs at least 3 valid responses to be shown.\n"
        "- The chart marker size is proportional to the response count for that cell.\n"
        "- Country color is consistent across all pages (see Color Legend page)."
    )
    ax.text(0.5, 0.16, notes, ha="center", va="top",
            fontsize=10, color="#2d3748", family="monospace")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_country_legend_page(pdf: PdfPages):
    """A small reference page so readers know which color is which country."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.9, "Country Color Legend",
            ha="center", va="top", fontsize=20, fontweight="bold", color="#1a2332")
    ax.text(0.5, 0.85, "The same color scheme is used across every chart in this document.",
            ha="center", va="top", fontsize=11, color="#4a5568")

    y_top = 0.7
    spacing = 0.08
    for i, (country, color) in enumerate(COUNTRY_COLORS.items()):
        y = y_top - i * spacing
        ax.scatter(0.38, y, s=420, c=color, edgecolors="white", linewidth=2.5)
        ax.text(0.42, y, country, ha="left", va="center",
                fontsize=14, color="#1a2332", fontweight="bold")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def render_charts(pdf: PdfPages, summary: pd.DataFrame, technologies: list[str]):
    per_page = CHARTS_PER_PAGE
    n_pages = math.ceil(len(technologies) / per_page)

    for page_idx in range(n_pages):
        chunk = technologies[page_idx * per_page:(page_idx + 1) * per_page]
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        axes = axes.flatten()

        for slot, ax in enumerate(axes):
            if slot < len(chunk):
                tech = chunk[slot]
                draw_single_hype_cycle(ax, tech, organize_phases(summary, tech))
            else:
                ax.set_visible(False)

        # Page header
        fig.suptitle("Hype Cycle Positioning by Country - Emerging AI Technologies",
                     fontsize=12, fontweight="bold", color="#1a2332", y=0.985)
        # Page footer
        fig.text(0.5, 0.015,
                 f"Page {page_idx + 4} of {n_pages + 3}   |   "
                 f"Technologies {page_idx * per_page + 1}-"
                 f"{page_idx * per_page + len(chunk)} of {len(technologies)}   |   "
                 "LACCEI 2025 - Hype Cycle Survey",
                 ha="center", va="bottom", fontsize=8, color="#718096")

        plt.tight_layout(rect=(0, 0.03, 1, 0.965))
        plt.subplots_adjust(hspace=0.4, wspace=0.18)
        pdf.savefig(fig, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print(f"Loading workbook: {EXCEL_FILE.name}")
    df = load_and_clean(EXCEL_FILE)
    print(f"  rows loaded: {len(df)}")

    print("Building country x technology summary...")
    summary = build_summary(df)
    print(f"  cells (country-technology): {len(summary)}")
    technologies = sorted(summary["Technology"].unique())
    print(f"  technologies with data:   {len(technologies)}")
    print(f"  countries:                {sorted(summary['Country'].unique())}")

    print(f"\nWriting PDF: {OUTPUT_PDF.name}")
    with PdfPages(OUTPUT_PDF) as pdf:
        info = pdf.infodict()
        info["Title"] = "Hype Cycle of Emerging AI Technologies - LACCEI 2025"
        info["Subject"] = "Country positioning per technology in the Gartner Hype Cycle"
        info["Author"] = "LACCEI 2025 Research Team"
        info["Keywords"] = "Hype Cycle, LACCEI, AI, Latin America"
        info["CreationDate"] = datetime.now()

        draw_cover(pdf, summary, technologies)
        draw_methodology_page(pdf)
        draw_country_legend_page(pdf)
        render_charts(pdf, summary, technologies)

    size_kb = OUTPUT_PDF.stat().st_size / 1024
    n_chart_pages = math.ceil(len(technologies) / CHARTS_PER_PAGE)
    print(f"  pages: 3 prefatory + {n_chart_pages} chart pages "
          f"= {3 + n_chart_pages} total")
    print(f"  size:  {size_kb:.1f} KB")
    print(f"\nDone -> {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

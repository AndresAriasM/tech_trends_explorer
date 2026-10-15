"use client";

import { useMemo } from "react";
import dynamic from "next/dynamic";
import type { HypeCycleQuery, HypePhase } from "@/types";

// Dynamic import to avoid SSR issues with Plotly
const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const PHASE_COLORS: Record<HypePhase, string> = {
  "Innovation Trigger": "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment": "#ef4444",
  "Slope of Enlightenment": "#34d399",
  "Plateau of Productivity": "#818cf8",
  Unknown: "#9ca3af",
};

// Canonical Gartner phase boundaries on the [0, 100] x axis. Used to draw
// subtle background bands and dashed dividers so users can locate each phase
// at a glance.
const PHASE_BANDS: Array<{ phase: HypePhase; x0: number; x1: number }> = [
  { phase: "Innovation Trigger",            x0: 0,  x1: 15 },
  { phase: "Peak of Inflated Expectations", x0: 15, x1: 35 },
  { phase: "Trough of Disillusionment",     x0: 35, x1: 65 },
  { phase: "Slope of Enlightenment",        x0: 65, x1: 85 },
  { phase: "Plateau of Productivity",       x0: 85, x1: 100 },
];

interface Props {
  queries: HypeCycleQuery[];
  highlightId?: string;
  /** Show Plotly's built-in legend. Disable when an external legend is rendered. */
  showLegend?: boolean;
  /** Chart height. Number is treated as px; pass "100%" to fill the parent. */
  height?: number | string;
  /** Override bubble labels (typically with short codes like "T1"). Keyed by query_id. */
  labelMap?: Record<string, string>;
}

export function HypeCycleChart({
  queries,
  highlightId,
  showLegend = true,
  height = 500,
  labelMap,
}: Props) {
  // Build the background curve (same shape as master visualization)
  const curveX = Array.from({ length: 400 }, (_, i) => i * 0.25);
  const curveY = curveX.map((x) => hypeCurve(x));

  // Density modes: with a labelMap (short numeric codes) the bubble carries the
  // code *inside* itself like a subway station, so we can pack them tightly.
  const hasCodes = !!labelMap;
  const maxPerPhase = useMemo(() => {
    const counts: Record<string, number> = {};
    queries.forEach((q) => {
      counts[q.hype_metrics.phase] = (counts[q.hype_metrics.phase] ?? 0) + 1;
    });
    return Math.max(0, ...Object.values(counts));
  }, [queries]);
  const isDense = hasCodes && (queries.length > 12 || maxPerPhase > 4);
  const markerSize = !hasCodes ? 24 : isDense ? 20 : 26;
  const markerHighlight = !hasCodes ? 30 : isDense ? 26 : 32;
  const insideText = hasCodes;
  const labelFontSize = insideText
    ? Math.round(markerSize * 0.46)
    : isDense
    ? 11
    : 17;

  // Two-pass layout: (1) spread bubbles horizontally within each phase, then
  // (2) run iterative repulsion so clusters get separated. **Every bubble is
  // clamped to its canonical phase band on every iteration** so the visual
  // never lies about which Gartner phase a technology belongs to. If a phase
  // is crowded, bubbles stack vertically inside their band instead of leaking
  // into a neighbouring phase.
  const positions = useMemo(() => {
    const phaseCounts: Record<string, number> = {};
    queries.forEach((q) => {
      phaseCounts[q.hype_metrics.phase] = (phaseCounts[q.hype_metrics.phase] ?? 0) + 1;
    });
    const indexInPhase: Record<string, number> = {};
    const EDGE_PAD = 0.6; // breathing room from each phase divider

    const result = queries.map((q) => {
      const phase = q.hype_metrics.phase;
      const band = PHASE_BANDS.find((b) => b.phase === phase);
      const xMin = (band?.x0 ?? 0) + EDGE_PAD;
      const xMax = (band?.x1 ?? 100) - EDGE_PAD;
      const halfBandWidth = (xMax - xMin) / 2;

      const siblings = phaseCounts[phase] ?? 1;
      const idx = (indexInPhase[phase] = (indexInPhase[phase] ?? -1) + 1);

      let px = q.hype_metrics.hype_cycle_position_x;
      if (siblings > 1) {
        const capRaw = hasCodes ? (isDense ? 18 : 14) : 12;
        // Never let the spread exceed the half-width of the band — that's
        // what was leaking bubbles into adjacent phases.
        const cap = Math.min(capRaw, halfBandWidth);
        const factor = hasCodes ? (isDense ? 2.4 : 2.8) : 3.5;
        const spread = Math.min(factor * (siblings - 1), cap);
        const offset = (idx / (siblings - 1) - 0.5) * spread * 2;
        px += offset;
      }
      // Hard clamp to the canonical band before snapping to the curve.
      px = Math.min(Math.max(px, xMin), xMax);

      return { px, py: hypeCurve(px), xMin, xMax };
    });

    // Per-axis minimum separation. The chart aspect ratio is roughly 2:1
    // (wider than tall) on our embedded view, so 1 X unit ≈ 2 pixels per Y
    // unit. Using a larger Y gap keeps the visual spacing roughly equal.
    const minGapX = hasCodes ? (isDense ? 2.7 : 3.4) : 4.0;
    const minGapY = hasCodes ? (isDense ? 5.6 : 6.8) : 8.0;

    for (let iter = 0; iter < 160; iter++) {
      let moved = false;
      for (let i = 0; i < result.length; i++) {
        for (let j = i + 1; j < result.length; j++) {
          let dx = result[i].px - result[j].px;
          let dy = result[i].py - result[j].py;
          // Break ties on exact coincidences with a tiny deterministic jiggle.
          if (dx === 0 && dy === 0) {
            result[i].px += 0.1;
            result[j].py += 0.1;
            dx = result[i].px - result[j].px;
            dy = result[i].py - result[j].py;
          }
          const overlapX = minGapX - Math.abs(dx);
          const overlapY = minGapY - Math.abs(dy);
          if (overlapX > 0 && overlapY > 0) {
            moved = true;
            // Different bands? Only push in Y — never let repulsion cross a
            // phase boundary, otherwise we'd misrepresent the phase.
            const sameBand =
              result[i].xMin === result[j].xMin && result[i].xMax === result[j].xMax;
            const ratioX = overlapX / minGapX;
            const ratioY = overlapY / minGapY;
            if (sameBand && ratioX < ratioY) {
              const push = overlapX * 0.5 * (dx >= 0 ? 1 : -1);
              result[i].px += push;
              result[j].px -= push;
            } else {
              const push = overlapY * 0.5 * (dy >= 0 ? 1 : -1);
              result[i].py += push;
              result[j].py -= push;
            }
          }
        }
      }
      // Per-bubble clamp: X stays inside its own phase band, Y stays inside
      // the visible plot area (above phase labels, below the top edge).
      for (const p of result) {
        if (p.px < p.xMin) p.px = p.xMin;
        if (p.px > p.xMax) p.px = p.xMax;
        if (p.py < 2) p.py = 2;
        if (p.py > 95) p.py = 95;
      }
      if (!moved) break;
    }

    return result;
  }, [queries, hasCodes, isDense]);

  // Technology bubbles
  const techTraces = queries.map((q, i) => {
    const phase = q.hype_metrics.phase;
    const { px, py } = positions[i];

    const isHighlighted = q.query_id === highlightId;
    const fullLabel = q.technology_name || q.search_query.slice(0, 30);
    const shortLabel = labelMap?.[q.query_id] ?? makeShortLabel(fullLabel);

    return {
      x: [px],
      y: [py],
      mode: "markers+text" as const,
      type: "scatter" as const,
      name: fullLabel,
      text: [shortLabel],
      textposition: (insideText ? "middle center" : "top center") as
        | "middle center"
        | "top center",
      marker: {
        size: isHighlighted ? markerHighlight : markerSize,
        color: PHASE_COLORS[phase] ?? "#9ca3af",
        line: {
          color: isHighlighted ? "#ffffff" : "rgba(0,0,0,0.45)",
          width: isHighlighted ? 2.5 : 1.2,
        },
        opacity: 0.95,
      },
      textfont: {
        size: labelFontSize,
        color: insideText ? "#0b0f1a" : "#f1f5f9",
        family: "Inter, sans-serif",
      },
      hovertemplate:
        `<b>${fullLabel}</b><br>` +
        `Fase: ${phase}<br>` +
        `Confianza: ${(q.hype_metrics.confidence * 100).toFixed(0)}%<br>` +
        `Menciones: ${q.hype_metrics.total_mentions}<br>` +
        `<extra></extra>`,
    };
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    // Soft glow under the curve for print contrast
    {
      x: curveX,
      y: curveY,
      type: "scatter",
      mode: "lines",
      line: { color: "rgba(148, 163, 184, 0.25)", width: 7, shape: "spline" },
      hoverinfo: "skip",
      showlegend: false,
      name: "curve-glow",
    },
    // Main Hype Cycle curve — high contrast on dark background
    {
      x: curveX,
      y: curveY,
      type: "scatter",
      mode: "lines",
      line: { color: "#cbd5e1", width: 2.5, shape: "spline" },
      hoverinfo: "skip",
      showlegend: false,
      name: "curve",
    },
    ...techTraces,
  ];

  // Phase guides: very subtle background bands tinted with each phase color
  // plus dotted vertical dividers at the canonical boundaries.
  const phaseShapes = [
    ...PHASE_BANDS.map((b) => ({
      type: "rect" as const,
      xref: "x" as const,
      yref: "y" as const,
      x0: b.x0,
      x1: b.x1,
      y0: -8,
      y1: 98,
      fillcolor: PHASE_COLORS[b.phase],
      opacity: 0.05,
      line: { width: 0 },
      layer: "below" as const,
    })),
    ...PHASE_BANDS.slice(1).map((b) => ({
      type: "line" as const,
      xref: "x" as const,
      yref: "y" as const,
      x0: b.x0,
      x1: b.x0,
      y0: 0,
      y1: 96,
      line: {
        dash: "dot" as const,
        width: 1,
        color: "rgba(148, 163, 184, 0.30)",
      },
      layer: "below" as const,
    })),
  ];

  const layout = {
    showlegend: showLegend,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb", family: "Inter, sans-serif" },
    autosize: true,
    shapes: phaseShapes,
    xaxis: {
      title: "Tiempo →",
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      range: [0, 100],
    },
    yaxis: {
      title: "Expectativas →",
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      // Negative lower bound reserves space under the curve for phase labels
      // so they never cross the trough line.
      range: [-8, 98],
    },
    margin: { t: 24, b: 28, l: 36, r: 24 },
    legend: {
      bgcolor: "rgba(6, 9, 15, 0.82)",
      bordercolor: "rgba(148, 163, 184, 0.22)",
      borderwidth: 1,
      font: { size: 17, color: "#e2e8f0" },
      itemsizing: "constant" as const,
      itemwidth: 40,
      x: 0.985,
      xanchor: "right" as const,
      y: 0.97,
      yanchor: "top" as const,
    },
    // Phase labels below the curve, centred on each canonical band.
    annotations: [
      label(7.5,  -3, "Innovation<br>Trigger"),
      label(25,   -3, "Peak of Inflated<br>Expectations"),
      label(50,   -3, "Trough of<br>Disillusionment"),
      label(75,   -3, "Slope of<br>Enlightenment"),
      label(92.5, -3, "Plateau of<br>Productivity"),
    ],
  };

  const cssHeight = typeof height === "number" ? `${height}px` : height;

  return (
    <Plot
      data={data}
      layout={layout as object}
      config={{ displayModeBar: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: cssHeight }}
    />
  );
}

// ── Helpers ──────────────────────────────────────────────────

// Approximates the canonical Gartner Hype Cycle shape on the [0, 100] domain.
// Tuned so that canonical phase centers yield:
//   Innovation Trigger (x≈8) → y≈26
//   Peak of Inflated Expectations (x≈25) → y≈88
//   Trough of Disillusionment (x≈55) → y≈6
//   Slope of Enlightenment (x≈75) → y≈45
//   Plateau of Productivity (x≈90) → y≈62
function hypeCurve(x: number): number {
  const hype    =  70 * Math.exp(-0.5 * Math.pow((x - 25) / 8, 2));
  const dip     = -14 * Math.exp(-0.5 * Math.pow((x - 55) / 9, 2));
  const plateau =  45 * (1 / (1 + Math.exp(-0.18 * (x - 72))));
  return Math.max(5, 18 + hype + dip + plateau);
}

function label(x: number, y: number, text: string) {
  return {
    x,
    y,
    text,
    showarrow: false,
    font: { size: 13, color: "#cbd5e1", family: "Inter, sans-serif" },
    xanchor: "center" as const,
    yanchor: "top" as const,
  };
}

// Extract a compact code (e.g. "M1", "C3", "TEC-01") from the full technology
// name so crowded charts stay readable. Falls back to the first few characters
// when no code pattern is present.
function makeShortLabel(fullLabel: string): string {
  const trimmed = fullLabel.trim();
  // Match patterns like "M1", "C3", "TEC-01", "EQ7", "X1:"
  const m = trimmed.match(/^([A-Za-zÑñ]{1,5}[-_]?\d{1,3})[\s:.\-–—]/);
  if (m) return m[1].toUpperCase();
  // Otherwise use the first word (up to 10 chars)
  const firstWord = trimmed.split(/[\s:.\-–—]/)[0];
  return firstWord.length <= 10 ? firstWord : firstWord.slice(0, 10) + "…";
}

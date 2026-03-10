"use client";

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

interface Props {
  queries: HypeCycleQuery[];
  highlightId?: string;
}

export function HypeCycleChart({ queries, highlightId }: Props) {
  // Build the background curve (same shape as master visualization)
  const curveX = Array.from({ length: 200 }, (_, i) => i * 0.5);
  const curveY = curveX.map((x) => hypeCurve(x));

  // Technology bubbles
  const techTraces = queries.map((q) => {
    const phase = q.hype_metrics.phase;
    const px = q.hype_metrics.hype_cycle_position_x;
    const py = q.hype_metrics.hype_cycle_position_y;
    const isHighlighted = q.query_id === highlightId;
    const label = q.technology_name || q.search_query.slice(0, 20);

    return {
      x: [px],
      y: [py],
      mode: "markers+text" as const,
      type: "scatter" as const,
      name: label,
      text: [label],
      textposition: "top center" as const,
      marker: {
        size: isHighlighted ? 18 : 12,
        color: PHASE_COLORS[phase] ?? "#9ca3af",
        line: { color: isHighlighted ? "#ffffff" : "transparent", width: 2 },
        opacity: 0.9,
      },
      textfont: { size: 10, color: "#e5e7eb" },
      hovertemplate:
        `<b>${label}</b><br>` +
        `Fase: ${phase}<br>` +
        `Confianza: ${(q.hype_metrics.confidence * 100).toFixed(0)}%<br>` +
        `Menciones: ${q.hype_metrics.total_mentions}<br>` +
        `<extra></extra>`,
    };
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const data: any[] = [
    // Background Hype Cycle curve
    {
      x: curveX,
      y: curveY,
      type: "scatter",
      mode: "lines",
      line: { color: "#374151", width: 2 },
      hoverinfo: "skip",
      showlegend: false,
      name: "curve",
    },
    ...techTraces,
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb", family: "Inter, sans-serif" },
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
      range: [0, 100],
    },
    margin: { t: 20, b: 50, l: 50, r: 20 },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { size: 11 } },
    // Phase labels
    annotations: [
      label(8, 5, "Innovation<br>Trigger"),
      label(25, 5, "Peak of Inflated<br>Expectations"),
      label(55, 5, "Trough of<br>Disillusionment"),
      label(75, 5, "Slope of<br>Enlightenment"),
      label(90, 5, "Plateau of<br>Productivity"),
    ],
  };

  return (
    <Plot
      data={data}
      layout={layout as object}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "420px" }}
    />
  );
}

// ── Helpers ──────────────────────────────────────────────────

// Approximates the classic Hype Cycle bell-trough shape in [0,100] domain
function hypeCurve(x: number): number {
  const peak = 85 * Math.exp(-0.5 * Math.pow((x - 25) / 8, 2));
  const trough = -30 * Math.exp(-0.5 * Math.pow((x - 55) / 12, 2));
  const slope = 40 * (1 / (1 + Math.exp(-0.15 * (x - 70))));
  return Math.max(0, 5 + peak + trough + slope);
}

function label(x: number, y: number, text: string) {
  return {
    x,
    y,
    text,
    showarrow: false,
    font: { size: 9, color: "#6b7280" },
    xanchor: "center" as const,
    yanchor: "bottom" as const,
  };
}

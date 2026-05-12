"use client";

import dynamic from "next/dynamic";
import type { KeywordStat } from "@/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  data: KeywordStat[];
}

export function KeywordsChart({ data }: Props) {
  const sorted = [...data].sort((a, b) => a.count - b.count);

  const trace = {
    type: "bar" as const,
    orientation: "h" as const,
    x: sorted.map((d) => d.count),
    y: sorted.map((d) => d.keyword),
    marker: {
      color: sorted.map((d) => {
        const max = sorted[sorted.length - 1].count;
        const intensity = d.count / max;
        return `rgba(96, 165, 250, ${0.35 + intensity * 0.65})`;
      }),
    },
    hovertemplate: "<b>%{y}</b><br>Apariciones: %{x}<extra></extra>",
  };

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb", size: 11, family: "Inter, sans-serif" },
    xaxis: { showgrid: true, gridcolor: "#1f2937", zeroline: false, title: "Apariciones" },
    yaxis: { showgrid: false, automargin: true },
    margin: { t: 10, b: 40, l: 120, r: 20 },
  };

  return (
    <Plot
      data={[trace]}
      layout={layout as object}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: `${Math.max(240, sorted.length * 26 + 60)}px` }}
    />
  );
}

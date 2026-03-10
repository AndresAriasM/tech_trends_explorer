"use client";

import dynamic from "next/dynamic";
import type { YearlyStat } from "@/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  data: YearlyStat[];
  technologyName?: string;
}

export function MentionsTimelineChart({ data, technologyName }: Props) {
  const years = data.map((d) => d.year);
  const mentions = data.map((d) => d.mentions);
  const sentiments = data.map((d) => d.sentiment);

  const traces = [
    {
      x: years,
      y: mentions,
      type: "bar" as const,
      name: "Menciones",
      marker: { color: "#3b82f6", opacity: 0.8 },
      yaxis: "y",
    },
    {
      x: years,
      y: sentiments,
      type: "scatter" as const,
      mode: "lines+markers" as const,
      name: "Sentimiento",
      line: { color: "#10b981", width: 2 },
      marker: { size: 5 },
      yaxis: "y2",
    },
  ];

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#e5e7eb", size: 11 },
    title: {
      text: technologyName ? `Menciones: ${technologyName}` : "Menciones por año",
      font: { size: 13 },
    },
    xaxis: { showgrid: false, zeroline: false, tickmode: "linear" as const },
    yaxis: { title: "Menciones", showgrid: true, gridcolor: "#1f2937", zeroline: false },
    yaxis2: {
      title: "Sentimiento",
      overlaying: "y" as const,
      side: "right" as const,
      showgrid: false,
      zeroline: false,
      range: [-1, 1],
    },
    legend: { bgcolor: "rgba(0,0,0,0)" },
    margin: { t: 40, b: 40, l: 50, r: 50 },
  };

  return (
    <Plot
      data={traces}
      layout={layout as object}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "300px" }}
    />
  );
}

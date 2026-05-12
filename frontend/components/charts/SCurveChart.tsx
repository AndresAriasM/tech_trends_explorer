"use client";

import dynamic from "next/dynamic";
import type { SCurveResult } from "@/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface Props {
  result: SCurveResult;
  height?: number;
}

export function SCurveChart({ result, height = 420 }: Props) {
  const { yearly_data, sigmoid_curve, phase } = result;

  const years = yearly_data.map((d) => d.year);
  const annual = yearly_data.map((d) => d.annual);
  const cumulative = yearly_data.map((d) => d.cumulative);

  const sigX = sigmoid_curve.map((d) => d.year);
  const sigY = sigmoid_curve.map((d) => d.value);

  const phaseColor = phase.color;

  const data: any[] = [
    // Annual bars
    {
      type: "bar",
      name: "Publicaciones anuales",
      x: years,
      y: annual,
      yaxis: "y",
      marker: {
        color: "rgba(79,142,247,0.55)",
        line: { color: "rgba(79,142,247,0.9)", width: 1 },
      },
      hovertemplate: "<b>%{x}</b><br>Anuales: %{y}<extra></extra>",
    },
    // Cumulative scatter dots
    {
      type: "scatter",
      mode: "markers",
      name: "Acumulado (datos reales)",
      x: years,
      y: cumulative,
      yaxis: "y2",
      marker: {
        color: phaseColor,
        size: 7,
        line: { color: "#fff", width: 1 },
      },
      hovertemplate: "<b>%{x}</b><br>Acumulado: %{y:,.0f}<extra></extra>",
    },
    // Sigmoid fit line
    ...(sigX.length > 0
      ? [
          {
            type: "scatter",
            mode: "lines",
            name: "Curva S (ajuste sigmoidal)",
            x: sigX,
            y: sigY,
            yaxis: "y2",
            line: {
              color: phaseColor,
              width: 3,
              dash: "solid",
            },
            hovertemplate: "<b>%{x:.1f}</b><br>Proyección: %{y:,.0f}<extra></extra>",
          },
        ]
      : []),
    // Inflection point vertical line
    ...(result.inflection_year
      ? [
          {
            type: "scatter",
            mode: "lines+text",
            name: "Punto de inflexión",
            x: [result.inflection_year, result.inflection_year],
            y: [0, Math.max(...cumulative) * 1.1],
            yaxis: "y2",
            line: { color: "rgba(251,191,36,0.7)", width: 2, dash: "dash" },
            text: ["", `↑ ${result.inflection_year}`],
            textposition: "top right",
            textfont: { color: "#fbbf24", size: 11 },
            hoverinfo: "skip",
            showlegend: false,
          },
        ]
      : []),
  ];

  const layout: any = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "Inter, system-ui, sans-serif", color: "#9ca3af", size: 12 },
    margin: { t: 20, r: 70, b: 60, l: 70 },
    height,
    legend: {
      orientation: "h",
      y: -0.18,
      x: 0.5,
      xanchor: "center",
      font: { size: 11, color: "#9ca3af" },
      bgcolor: "transparent",
    },
    xaxis: {
      title: { text: "Año", font: { size: 12, color: "#6b7280" } },
      gridcolor: "rgba(255,255,255,0.04)",
      linecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "#6b7280" },
      tickformat: "d",
    },
    yaxis: {
      title: { text: "Publicaciones anuales", font: { size: 12, color: "rgba(79,142,247,0.8)" } },
      gridcolor: "rgba(255,255,255,0.04)",
      linecolor: "rgba(255,255,255,0.08)",
      tickfont: { color: "rgba(79,142,247,0.7)" },
      rangemode: "tozero",
    },
    yaxis2: {
      title: { text: "Acumulado", font: { size: 12, color: phaseColor } },
      overlaying: "y",
      side: "right",
      gridcolor: "transparent",
      tickfont: { color: phaseColor },
      rangemode: "tozero",
    },
    hovermode: "x unified",
    hoverlabel: {
      bgcolor: "rgba(15,20,30,0.95)",
      bordercolor: "rgba(255,255,255,0.12)",
      font: { color: "#e5e7eb", size: 12 },
    },
    bargap: 0.3,
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const config: any = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
    responsive: true,
  };

  return (
    <div style={{ width: "100%", height: height }}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

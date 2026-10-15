"use client";

/**
 * El mapa de variables en cuatro cuadrantes (§7) — el producto del módulo.
 *
 * «El resultado útil no es una hoja coloreada por panelista, sino una
 * clasificación de cada variable en dos ejes: cuánto acuerdo hay, y cuánto
 * se movió.»
 *
 * Decisión de visualización: **la posición es la categoría**. Cuatro tonos
 * categóricos no superaban la separación bajo daltonismo en todos los pares,
 * y aquí el color sería redundante de todos modos: cada cuadrante lleva su
 * nombre escrito dentro. Los puntos usan un solo tono y el fondo sólo matiza.
 */

import { useState } from "react";
import type { Cuadrante, VariableMap } from "./types";
import {
  DATA_HUE,
  QUADRANT_LABEL,
  QUADRANT_TINT,
  QuadrantBadge,
  fmt,
  pct,
} from "./primitives";

const QUADRANT_HINT: Record<Cuadrante, string> = {
  CERRADA: "acuerdo estable → no repreguntar",
  CONVERGIENDO: "el panel se movió hacia el acuerdo → una ronda más",
  DISENSO_PERSISTENTE: "desacuerdo firme → no lo resuelve otra ronda",
  INESTABLE: "mucho movimiento, sin dirección → revisar la redacción",
};

const W = 560;
const H = 420;
const PAD = { t: 26, r: 22, b: 44, l: 52 };

export function QuadrantMap({ map }: { map: VariableMap }) {
  const [hover, setHover] = useState<string | null>(null);

  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;
  const xT = map.axes.x.threshold;
  const yT = map.axes.y.threshold;

  const sx = (v: number) => PAD.l + v * plotW;
  const sy = (v: number) => PAD.t + (1 - v) * plotH;

  const points = map.variables.filter(
    (v) => v.stability != null && v.consensus != null
  );

  // Cuadrantes como rectángulos, con su nombre dentro: la etiqueta directa
  // que hace innecesario el color.
  const regions: {
    q: Cuadrante;
    x: number;
    y: number;
    w: number;
    h: number;
    anchor: "start" | "end";
  }[] = [
    { q: "CONVERGIENDO", x: PAD.l, y: PAD.t, w: xT * plotW, h: (1 - yT) * plotH, anchor: "start" },
    { q: "CERRADA", x: sx(xT), y: PAD.t, w: (1 - xT) * plotW, h: (1 - yT) * plotH, anchor: "end" },
    { q: "INESTABLE", x: PAD.l, y: sy(yT), w: xT * plotW, h: yT * plotH, anchor: "start" },
    { q: "DISENSO_PERSISTENTE", x: sx(xT), y: sy(yT), w: (1 - xT) * plotW, h: yT * plotH, anchor: "end" },
  ];

  const hovered = points.find((p) => p.variable_id === hover);

  return (
    <section className="card p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-bold text-[var(--text-1)]">
            Mapa de variables
          </h3>
          <p className="mt-0.5 max-w-xl text-[12px] leading-relaxed text-[var(--text-3)]">
            Dos ejes: cuánto acuerdo hay (consenso ponderado por K) y cuánto se
            movió el panel (estabilidad de la matriz de transición). Los cuatro
            cuadrantes piden acciones distintas.
          </p>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {(Object.keys(QUADRANT_LABEL) as Cuadrante[]).map((q) => (
            <span
              key={q}
              className="rounded-lg border border-[var(--border)] px-2.5 py-1 text-[11px] text-[var(--text-2)]"
            >
              {QUADRANT_LABEL[q]}{" "}
              <strong className="tabular-nums text-[var(--text-1)]">
                {map.counts[q] ?? 0}
              </strong>
            </span>
          ))}
        </div>
      </header>

      <div className="flex flex-col gap-5 xl:flex-row">
        {/* Ancho reservado: dentro de un flex, un svg al 100 % de un elemento
            de ancho automático colapsa. */}
        <div className="shrink-0" style={{ width: W, maxWidth: "100%" }}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            width="100%"
            role="img"
            aria-label="Mapa de variables por consenso y estabilidad"
          >
            {regions.map((r) => (
              <g key={r.q}>
                <rect
                  x={r.x}
                  y={r.y}
                  width={Math.max(0, r.w)}
                  height={Math.max(0, r.h)}
                  fill={QUADRANT_TINT[r.q]}
                />
                <text
                  x={r.anchor === "start" ? r.x + 10 : r.x + r.w - 10}
                  y={r.y + 18}
                  textAnchor={r.anchor}
                  fontSize="11"
                  fontWeight="700"
                  letterSpacing="0.06em"
                  fill="var(--text-2)"
                >
                  {QUADRANT_LABEL[r.q].toUpperCase()}
                </text>
                <text
                  x={r.anchor === "start" ? r.x + 10 : r.x + r.w - 10}
                  y={r.y + 32}
                  textAnchor={r.anchor}
                  fontSize="9.5"
                  fill="var(--text-4)"
                >
                  {QUADRANT_HINT[r.q]}
                </text>
              </g>
            ))}

            {/* Ejes de corte: recesivos, discontinuos */}
            <line
              x1={sx(xT)}
              y1={PAD.t}
              x2={sx(xT)}
              y2={PAD.t + plotH}
              stroke="var(--border-2)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <line
              x1={PAD.l}
              y1={sy(yT)}
              x2={PAD.l + plotW}
              y2={sy(yT)}
              stroke="var(--border-2)"
              strokeWidth="1"
              strokeDasharray="3 3"
            />
            <rect
              x={PAD.l}
              y={PAD.t}
              width={plotW}
              height={plotH}
              fill="none"
              stroke="var(--border)"
            />

            {/* Puntos: un solo tono. Anillo de 2px en la superficie para que
                no se fundan al solaparse. */}
            {points.map((p) => {
              const cx = sx(Math.max(0, Math.min(1, p.stability!)));
              const cy = sy(Math.max(0, Math.min(1, p.consensus!)));
              const on = hover === p.variable_id;
              return (
                <g key={p.variable_id}>
                  <circle
                    cx={cx}
                    cy={cy}
                    r={on ? 7.5 : 5.5}
                    fill={DATA_HUE}
                    stroke="var(--surface)"
                    strokeWidth="2"
                  />
                  <circle
                    cx={cx}
                    cy={cy}
                    r={13}
                    fill="transparent"
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHover(p.variable_id)}
                    onMouseLeave={() => setHover(null)}
                  >
                    <title>{`${p.variable_name}\nconsenso ${fmt(p.consensus)} · estabilidad ${fmt(p.stability)}`}</title>
                  </circle>
                  {(on || points.length <= 6) && (
                    <text
                      x={cx + 10}
                      y={cy + 4}
                      fontSize="10.5"
                      fill="var(--text-1)"
                      style={{ pointerEvents: "none" }}
                    >
                      {p.variable_name.length > 26
                        ? `${p.variable_name.slice(0, 25)}…`
                        : p.variable_name}
                    </text>
                  )}
                </g>
              );
            })}

            {/* Ejes */}
            <text
              x={PAD.l + plotW / 2}
              y={H - 10}
              textAnchor="middle"
              fontSize="11"
              fill="var(--text-3)"
            >
              {map.axes.x.label} → (corte {pct(xT, 0)})
            </text>
            <text
              x={-(PAD.t + plotH / 2)}
              y={14}
              transform="rotate(-90)"
              textAnchor="middle"
              fontSize="11"
              fill="var(--text-3)"
            >
              {map.axes.y.label} → (corte {pct(yT, 0)})
            </text>
            <text x={PAD.l} y={H - 26} fontSize="9.5" fill="var(--text-4)">
              0
            </text>
            <text
              x={PAD.l + plotW}
              y={H - 26}
              textAnchor="end"
              fontSize="9.5"
              fill="var(--text-4)"
            >
              1
            </text>
          </svg>
        </div>

        {/* Vista de tabla: la identidad nunca depende de leer el gráfico. */}
        <div className="min-w-0 flex-1">
          {hovered && (
            <div className="mb-3 rounded-lg border border-[var(--accent-border)] bg-[var(--accent-sub)] px-3 py-2 text-[12px]">
              <strong className="text-[var(--text-1)]">
                {hovered.variable_name}
              </strong>
              <span className="ml-2 text-[var(--text-2)]">{hovered.action}</span>
            </div>
          )}
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full min-w-[540px] border-collapse text-[12px]">
              <thead className="sticky top-0 bg-[var(--surface)]">
                <tr className="border-b border-[var(--border)] text-[10px] uppercase tracking-wider text-[var(--text-3)]">
                  <th className="px-2 py-2 text-left">Variable</th>
                  <th className="px-2 py-2 text-right">Consenso</th>
                  <th className="px-2 py-2 text-right">Estab.</th>
                  <th className="px-2 py-2 text-right" title="c − b">
                    Neta
                  </th>
                  <th className="px-2 py-2 text-left">Cuadrante</th>
                  <th className="px-2 py-2 text-left" title="Criterio de parada del estudio">
                    Parada
                  </th>
                </tr>
              </thead>
              <tbody>
                {map.variables.map((v) => (
                  <tr
                    key={v.variable_id}
                    onMouseEnter={() => setHover(v.variable_id)}
                    onMouseLeave={() => setHover(null)}
                    className="border-b border-[var(--border)] last:border-0"
                    style={{
                      background:
                        hover === v.variable_id ? "var(--surface-2)" : undefined,
                    }}
                  >
                    <td className="max-w-[220px] truncate px-2 py-1.5 text-[var(--text-1)]">
                      {v.variable_name}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[var(--text-2)]">
                      {fmt(v.consensus)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[var(--text-2)]">
                      {fmt(v.stability)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[var(--text-2)]">
                      {v.direccion_neta == null
                        ? "—"
                        : `${v.direccion_neta > 0 ? "+" : ""}${v.direccion_neta}`}
                    </td>
                    <td className="px-2 py-1.5">
                      <QuadrantBadge value={v.quadrant} />
                    </td>
                    <td
                      className="px-2 py-1.5 text-[11px] text-[var(--text-3)]"
                      title={v.stopping.reason}
                    >
                      {v.stopping.closed ? "cerrada" : "abierta"}
                    </td>
                  </tr>
                ))}
                {map.sin_clasificar.map((v) => (
                  <tr
                    key={v.variable_id}
                    className="border-b border-[var(--border)] last:border-0 opacity-70"
                  >
                    <td className="max-w-[220px] truncate px-2 py-1.5 text-[var(--text-2)]">
                      {v.variable_name}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[var(--text-3)]">
                      {fmt(v.consensus)}
                    </td>
                    <td
                      colSpan={4}
                      className="px-2 py-1.5 text-[11px] text-[var(--text-3)]"
                    >
                      Sin par de rondas: no se le inventa un cuadrante.
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {map.next_round_variables.length > 0 && (
            <p className="mt-3 rounded-lg border border-[var(--border)] px-3 py-2 text-[12px] text-[var(--text-2)]">
              <strong className="text-[var(--text-1)]">
                {map.next_round_variables.length}
              </strong>{" "}
              variable{map.next_round_variables.length > 1 ? "s" : ""} no
              cumple{map.next_round_variables.length > 1 ? "n" : ""} el criterio
              de parada. La ronda siguiente puede abrirse sólo sobre ellas:
              menos carga para el panel, que es la causa principal de la
              deserción.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}

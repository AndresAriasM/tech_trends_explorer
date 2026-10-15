"use client";

/**
 * Matriz de correlaciones de Spearman entre variables de una ronda.
 *
 * Es polaridad, no magnitud: par divergente con **gris neutro en el cero**
 * (nunca un tono en el punto medio). Azul #4f8ef7 para ρ positivo, rojo
 * #d9534f para negativo; ambos validados contra la superficie oscura, con
 * ΔE 25.4 bajo protanopía. Cada celda lleva su valor en el tooltip, así que
 * el color nunca es la única vía.
 */

import { useState } from "react";
import type { SpearmanResult } from "./types";
import { DIVERGING_NEG, DIVERGING_POS } from "./primitives";

function cellColor(rho: number | null): string {
  if (rho == null) return "var(--surface-3)";
  const a = Math.min(1, Math.abs(rho));
  // Gris neutro en el cero: la opacidad tiende a 0 y asoma la superficie.
  const hue = rho >= 0 ? DIVERGING_POS : DIVERGING_NEG;
  return `color-mix(in oklab, ${hue} ${Math.round(a * 88)}%, #2a3548)`;
}

export function SpearmanMatrix({
  data,
  roundNumber,
}: {
  data: SpearmanResult;
  roundNumber: number;
}) {
  const [hover, setHover] = useState<{
    i: number;
    j: number;
    rho: number | null;
  } | null>(null);

  if (!data.labels.length) return null;

  const n = data.labels.length;
  const cell = n > 14 ? 18 : n > 9 ? 24 : 30;

  return (
    <section className="card p-5">
      <h3 className="text-[15px] font-bold text-[var(--text-1)]">
        Correlaciones entre preguntas (Spearman, Ronda {roundNumber})
      </h3>
      <p className="mt-0.5 text-[12px] text-[var(--text-3)]">
        Azul = correlación positiva · rojo = negativa · gris = nula. Pares
        completos por celda: una abstención sólo descuenta en su par.
      </p>

      <div className="mt-4 flex flex-col gap-6 lg:flex-row">
        <div className="min-w-0 flex-1 overflow-x-auto">
          <div className="flex">
            <div className="shrink-0 pr-2" style={{ paddingTop: 0 }}>
              {data.labels.map((label, i) => (
                <div
                  key={data.variable_ids[i]}
                  className="flex items-center justify-end whitespace-nowrap text-[10px] text-[var(--text-3)]"
                  style={{ height: cell, width: 170 }}
                  title={label}
                >
                  {/* Se recorta por la derecha, que es donde sobra: un
                      «…confiabilidad del sitio» no se distingue del vecino. */}
                  {label.length > 30 ? `${label.slice(0, 29)}…` : label}
                </div>
              ))}
            </div>
            <div>
              {data.matrix.map((row, i) => (
                <div key={i} className="flex" style={{ gap: 2, marginBottom: 2 }}>
                  {row.map((rho, j) => (
                    <div
                      key={j}
                      onMouseEnter={() => setHover({ i, j, rho })}
                      onMouseLeave={() => setHover(null)}
                      title={
                        rho == null
                          ? `${data.labels[i]} ↔ ${data.labels[j]}: sin pares suficientes`
                          : `${data.labels[i]} ↔ ${data.labels[j]}: ρ ${rho > 0 ? "+" : ""}${rho.toFixed(2)}`
                      }
                      style={{
                        width: cell,
                        height: cell,
                        borderRadius: 3,
                        background: cellColor(rho),
                        outline:
                          hover && hover.i === i && hover.j === j
                            ? "2px solid var(--text-2)"
                            : "none",
                        outlineOffset: -1,
                        cursor: "default",
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>

          {/* Leyenda de la escala divergente */}
          <div className="mt-3 flex items-center gap-2 text-[10px] text-[var(--text-3)]">
            <span>−1</span>
            <div
              className="h-2 w-40 rounded-full"
              style={{
                background: `linear-gradient(90deg, ${DIVERGING_NEG}, #2a3548, ${DIVERGING_POS})`,
              }}
            />
            <span>+1</span>
            {hover && (
              <span className="ml-3 text-[var(--text-2)]">
                {data.labels[hover.i]} ↔ {data.labels[hover.j]}:{" "}
                <strong className="tabular-nums">
                  {hover.rho == null
                    ? "sin datos"
                    : `ρ ${hover.rho > 0 ? "+" : ""}${hover.rho.toFixed(2)}`}
                </strong>
              </span>
            )}
          </div>
        </div>

        <aside className="w-full shrink-0 lg:w-[340px]">
          <p className="label-sm mb-2">Pares más correlacionados</p>
          <ul className="space-y-1.5">
            {data.top_pairs.slice(0, 6).map((p) => (
              <li
                key={`${p.variable_a}-${p.variable_b}`}
                className="rounded-lg border border-[var(--border)] px-3 py-2 text-[12px] leading-snug"
              >
                <span
                  className="mr-2 font-bold tabular-nums"
                  style={{ color: p.rho >= 0 ? DIVERGING_POS : DIVERGING_NEG }}
                >
                  ρ {p.rho > 0 ? "+" : ""}
                  {p.rho.toFixed(2)}
                </span>
                <span className="text-[var(--text-2)]">
                  {p.label_a} ↔ {p.label_b}
                </span>
                <span className="ml-2 text-[10px] text-[var(--text-4)]">
                  n={p.n}
                  {p.p_value != null && ` · p=${p.p_value.toFixed(3)}`}
                </span>
              </li>
            ))}
            {!data.top_pairs.length && (
              <li className="text-[12px] text-[var(--text-3)]">
                Hacen falta más respuestas por par para calcular correlaciones.
              </li>
            )}
          </ul>
        </aside>
      </div>
    </section>
  );
}

"use client";

/**
 * La tabla por agrupador: una fila por variable con n, NS/NR, media, mediana,
 * σ, RIQ, % sobre el umbral y la clasificación.
 *
 * Reproduce la vista que existía antes de perderse, con tres arreglos:
 *   · NS/NR es una **columna propia** y está fuera de la distribución, no un
 *     6 dentro de la escala (§3);
 *   · se muestra el consenso ponderado por K junto al % crudo (§5.2);
 *   · la cabecera declara sobre qué población está calculada (§6).
 */

import type { VariableRoundStats } from "./types";
import {
  ClassificationBadge,
  DistributionBars,
  PopulationTag,
  fmt,
  pct,
} from "./primitives";

function HeadCell({
  children,
  title,
  align = "right",
}: {
  children: React.ReactNode;
  title?: string;
  align?: "left" | "right" | "center";
}) {
  return (
    <th
      title={title}
      className={`whitespace-nowrap px-3 py-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-3)] text-${align}`}
      style={{ textAlign: align }}
    >
      {children}
    </th>
  );
}

export function VariableTable({
  family,
  familyDescription,
  variables,
  threshold,
  population,
}: {
  family: string;
  familyDescription?: string;
  variables: VariableRoundStats[];
  threshold: number;
  population: number;
}) {
  return (
    <section className="card overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-4">
        <div>
          <h3 className="text-[15px] font-bold text-[var(--text-1)]">
            Agrupador · {family}
          </h3>
          {familyDescription && (
            <p className="mt-0.5 text-[12px] text-[var(--text-3)]">
              {familyDescription}
            </p>
          )}
        </div>
        <PopulationTag kind="agregado" n={population} />
      </header>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <HeadCell align="left">Pregunta</HeadCell>
              <HeadCell align="left" title="Reparto de respuestas en la escala. NS/NR va aparte.">
                Distribución
              </HeadCell>
              <HeadCell title="Respuestas con juicio numérico">n</HeadCell>
              <HeadCell title="Abstenciones declaradas. Nunca se cuentan como un valor de la escala.">
                NS/NR
              </HeadCell>
              <HeadCell>Media</HeadCell>
              <HeadCell>Mediana</HeadCell>
              <HeadCell title="Desviación típica">σ</HeadCell>
              <HeadCell title="Rango intercuartílico: la medida de consenso clásica en Delphi">
                RIQ
              </HeadCell>
              <HeadCell title={`Proporción de respuestas por encima del umbral de dicotomía (≥ ${threshold})`}>
                % ≥ {threshold}
              </HeadCell>
              <HeadCell title="Acuerdo del panel sobre el lado del umbral, ponderado por el coeficiente de competencia experta K (§5.2)">
                Consenso K
              </HeadCell>
              <HeadCell align="left">Clasificación</HeadCell>
            </tr>
          </thead>
          <tbody>
            {variables.map((v) => (
              <tr
                key={v.variable_id}
                className="border-b border-[var(--border)] last:border-0 transition-colors hover:bg-[var(--surface-2)]"
              >
                <td className="max-w-[320px] px-3 py-2.5 text-[var(--text-1)]">
                  {v.variable_name ?? v.variable_id}
                </td>
                <td className="px-3 py-2.5">
                  <DistributionBars distribution={v.distribution} />
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]">
                  {v.n}
                </td>
                <td
                  className="px-3 py-2.5 text-right tabular-nums"
                  style={{
                    color: v.n_abstentions ? "#c9821a" : "var(--text-4)",
                  }}
                >
                  {v.n_abstentions}
                </td>
                <td className="px-3 py-2.5 text-right font-semibold tabular-nums text-[var(--text-1)]">
                  {fmt(v.mean)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]">
                  {fmt(v.median, 1)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]">
                  {fmt(v.std)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]">
                  {fmt(v.iqr, 2)}
                </td>
                <td className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]">
                  {pct(v.pct_above_threshold)}
                </td>
                <td
                  className="px-3 py-2.5 text-right tabular-nums text-[var(--text-2)]"
                  title={
                    v.pct_above_threshold_weighted != null
                      ? `Proporción ponderada por K sobre el umbral: ${pct(v.pct_above_threshold_weighted)}`
                      : undefined
                  }
                >
                  {fmt(v.consensus)}
                </td>
                <td className="px-3 py-2.5">
                  <ClassificationBadge value={v.classification} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

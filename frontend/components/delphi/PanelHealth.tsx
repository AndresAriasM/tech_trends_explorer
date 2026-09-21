"use client";

/**
 * Salud del panel (§6, §9): deserción por ronda, K medio y —lo importante—
 * la comparación **R1-total vs R1-restringido** de §6.2.
 *
 * «Es una comparación barata y nadie la hizo.» Si el agregado cambia al
 * quedarse sólo con quienes siguieron, la deserción no fue aleatoria y la
 * estabilidad medida sobre la intersección está sesgada. El módulo lo
 * calcula sin que nadie lo pida.
 */

import type { AttritionCheck, DelphiParticipant, PanelStats } from "./types";
import { BarList, Notice, PopulationTag, fmt, pct } from "./primitives";

export function PanelHealth({
  panel,
  attrition,
  participants,
}: {
  panel: PanelStats;
  attrition: AttritionCheck[];
  participants?: DelphiParticipant[];
}) {
  const biased = attrition.filter((a) => a.attrition_non_random);

  return (
    <div className="space-y-4">
      {/* ── Participación y deserción por ronda ── */}
      <section className="card p-5">
        <h3 className="text-[15px] font-bold text-[var(--text-1)]">
          Participación y deserción
        </h3>
        <p className="mt-0.5 text-[12px] text-[var(--text-3)]">
          Las dos poblaciones de §6, siempre separadas: quienes contestaron la
          ronda, y la intersección con la anterior.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-[var(--border)] text-[10px] uppercase tracking-wider text-[var(--text-3)]">
                <th className="px-3 py-2 text-left">Ronda</th>
                <th className="px-3 py-2 text-right" title="Quienes contestaron esta ronda">
                  Pobl. de agregado
                </th>
                <th className="px-3 py-2 text-right">Tasa de respuesta</th>
                <th
                  className="px-3 py-2 text-right"
                  title="Intersección con la ronda anterior: la única población sobre la que se puede hablar de movimiento"
                >
                  Pobl. de cambio
                </th>
                <th className="px-3 py-2 text-right">Deserción</th>
                <th className="px-3 py-2 text-right">Nuevos</th>
              </tr>
            </thead>
            <tbody>
              {panel.rounds.map((r) => (
                <tr
                  key={r.round_number}
                  className="border-b border-[var(--border)] last:border-0"
                >
                  <td className="px-3 py-2 font-semibold text-[var(--text-1)]">
                    Ronda {r.round_number}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[var(--text-1)]">
                    {r.poblacion_agregado}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[var(--text-2)]">
                    {pct(r.response_rate, 0)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[var(--text-2)]">
                    {r.poblacion_cambio ?? "—"}
                  </td>
                  <td
                    className="px-3 py-2 text-right tabular-nums"
                    style={{
                      color:
                        (r.attrition_rate ?? 0) >= 0.3
                          ? "#e0a33a"
                          : "var(--text-2)",
                    }}
                  >
                    {pct(r.attrition_rate, 1)}
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-[var(--text-3)]">
                    {r.new_panelists ?? "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-[var(--border)] px-3 py-2">
            <p className="label-sm">K medio del panel</p>
            <p className="mt-1 text-[18px] font-bold tabular-nums text-[var(--text-1)]">
              {fmt(panel.mean_k_all)}
            </p>
          </div>
          <div className="rounded-lg border border-[var(--border)] px-3 py-2">
            <p className="label-sm" title="Sólo quienes contestaron la última ronda">
              K medio del panel activo
            </p>
            <p className="mt-1 text-[18px] font-bold tabular-nums text-[var(--text-1)]">
              {fmt(panel.mean_k_active)}
            </p>
          </div>
          <div className="rounded-lg border border-[var(--border)] px-3 py-2">
            <p className="label-sm" title={`estabilidad ≥ ${panel.stopping_rule.stability_min} o RIQ ≤ ${panel.stopping_rule.iqr_max}`}>
              Criterio de parada
            </p>
            <p className="mt-1 text-[18px] font-bold tabular-nums text-[var(--text-1)]">
              {panel.stopping_rule.n_closed}{" "}
              <span className="text-[12px] font-normal text-[var(--text-3)]">
                cerradas · {panel.stopping_rule.n_open} abiertas
              </span>
            </p>
          </div>
        </div>
      </section>

      {/* ── §6.2 ── */}
      <section className="card p-5">
        <header className="mb-3 flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-[15px] font-bold text-[var(--text-1)]">
              ¿Fue aleatoria la deserción?
            </h3>
            <p className="mt-0.5 max-w-2xl text-[12px] leading-relaxed text-[var(--text-3)]">
              El agregado de la ronda anterior sobre todos los que la
              contestaron, junto al mismo agregado restringido a quienes
              siguieron. Si difieren, la estabilidad medida sobre la
              intersección está sesgada.
            </p>
          </div>
          <div className="flex gap-2">
            <PopulationTag kind="agregado" />
            <PopulationTag kind="agregado_restringido" />
          </div>
        </header>

        <Notice tone={biased.length ? "warn" : "info"}>
          {panel.attrition_summary.verdict}
        </Notice>

        {attrition.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full border-collapse text-[13px]">
              <thead>
                <tr className="border-b border-[var(--border)] text-[10px] uppercase tracking-wider text-[var(--text-3)]">
                  <th className="px-3 py-2 text-left">Variable</th>
                  <th className="px-3 py-2 text-right">% ≥ umbral (todos)</th>
                  <th className="px-3 py-2 text-right">
                    % ≥ umbral (los que siguieron)
                  </th>
                  <th className="px-3 py-2 text-right">Δ</th>
                  <th className="px-3 py-2 text-left">Lectura</th>
                </tr>
              </thead>
              <tbody>
                {[...attrition]
                  .sort(
                    (a, b) =>
                      Math.abs(b.delta_pct_above_threshold ?? 0) -
                      Math.abs(a.delta_pct_above_threshold ?? 0)
                  )
                  .map((a) => (
                    <tr
                      key={`${a.variable_id}-${a.round_to}`}
                      className="border-b border-[var(--border)] last:border-0"
                    >
                      <td className="max-w-[260px] truncate px-3 py-2 text-[var(--text-1)]">
                        {a.variable_name}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-[var(--text-2)]">
                        {pct(a.total.pct_above_threshold)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-[var(--text-2)]">
                        {pct(a.restricted.pct_above_threshold)}
                      </td>
                      <td
                        className="px-3 py-2 text-right tabular-nums font-semibold"
                        style={{
                          color: a.attrition_non_random
                            ? "#e0a33a"
                            : "var(--text-3)",
                        }}
                      >
                        {a.delta_pct_above_threshold == null
                          ? "—"
                          : `${a.delta_pct_above_threshold > 0 ? "+" : ""}${(a.delta_pct_above_threshold * 100).toFixed(1)} pp`}
                      </td>
                      <td className="px-3 py-2 text-[11px] text-[var(--text-3)]">
                        {a.attrition_non_random
                          ? "Se fueron los que opinaban distinto"
                          : "Sin señal de selección"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {/* ── Perfil del panel ── */}
      {participants && participants.length > 0 && (
        <section className="card p-5">
          <h3 className="text-[15px] font-bold text-[var(--text-1)]">
            Perfil del panel
          </h3>
          <p className="mt-0.5 text-[12px] text-[var(--text-3)]">
            Formación y afiliación se recogen al alta y sirven para
            estratificar. El instrumento original ya las preguntaba y el
            análisis las descartaba.
          </p>
          <div className="mt-4 grid gap-6 lg:grid-cols-2">
            <div>
              <p className="label-sm mb-2">
                Nivel de formación{" "}
                <span className="normal-case tracking-normal text-[var(--text-4)]">
                  n={participants.length}
                </span>
              </p>
              <BarList
                items={countBy(participants, (p) => p.education)}
                total={participants.length}
                emptyLabel="Sin formación registrada en el panel."
              />
            </div>
            <div>
              <p className="label-sm mb-2">
                Afiliación{" "}
                <span className="normal-case tracking-normal text-[var(--text-4)]">
                  n={participants.length}
                </span>
              </p>
              <BarList
                items={countBy(participants, (p) => p.affiliation)}
                total={participants.length}
                emptyLabel="Sin afiliación registrada en el panel."
              />
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function countBy<T>(
  rows: T[],
  key: (row: T) => string | null | undefined
): { label: string; count: number }[] {
  const acc = new Map<string, number>();
  for (const r of rows) {
    const v = (key(r) ?? "").trim();
    if (!v) continue;
    acc.set(v, (acc.get(v) ?? 0) + 1);
  }
  return [...acc.entries()]
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

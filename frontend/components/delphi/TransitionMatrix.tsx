"use client";

/**
 * Lo que distingue un Delphi de un taller: el análisis ENTRE rondas.
 *
 * La versión perdida de esta pantalla no lo tenía, y el notebook original
 * tampoco: coloreaba las celdas `a` y `d` y dejaba `b` y `c` en blanco. Pero
 * «b y c son la respuesta a la pregunta del ejercicio»: son quienes cambiaron
 * de opinión y las dos celdas discordantes de McNemar (§2).
 *
 * Aquí las cuatro celdas se cuentan, las dos discordantes van resaltadas y el
 * contraste se muestra con su veredicto en palabras. Todo sobre la población
 * de cambio, etiquetada como tal.
 */

import type { VariableTransition } from "./types";
import { PopulationTag, fmt, pct } from "./primitives";

const CELL = {
  a: { label: "Mantiene alto", key: "a", tint: "rgba(217, 83, 79, 0.10)", border: "rgba(217,83,79,0.28)" },
  b: { label: "Bajó", key: "b", tint: "rgba(201, 130, 26, 0.16)", border: "rgba(201,130,26,0.45)" },
  c: { label: "Subió", key: "c", tint: "rgba(79, 142, 247, 0.16)", border: "rgba(79,142,247,0.45)" },
  d: { label: "Mantiene bajo", key: "d", tint: "rgba(15, 158, 82, 0.10)", border: "rgba(15,158,82,0.28)" },
} as const;

function Cell({
  spec,
  count,
  total,
  discordant,
}: {
  spec: (typeof CELL)[keyof typeof CELL];
  count: number;
  total: number;
  discordant?: boolean;
}) {
  return (
    <div
      className="rounded-lg border px-3 py-2.5"
      style={{ background: spec.tint, borderColor: spec.border }}
      title={
        discordant
          ? "Celda discordante: aquí está el cambio de opinión. El notebook la dejaba en blanco."
          : "Celda concordante: mantiene su posición respecto al umbral."
      }
    >
      <p className="flex items-baseline gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-3)]">
        <span className="font-mono text-[11px] text-[var(--text-2)]">
          {spec.key}
        </span>
        {spec.label}
        {discordant && (
          <span className="ml-auto rounded-full bg-[var(--surface-3)] px-1.5 py-0.5 text-[9px] normal-case tracking-normal text-[var(--text-2)]">
            discordante
          </span>
        )}
      </p>
      <p className="mt-1 text-[22px] font-extrabold leading-none tabular-nums text-[var(--text-1)]">
        {count}
        <span className="ml-1.5 text-[11px] font-normal text-[var(--text-3)]">
          {total ? `${((count / total) * 100).toFixed(0)}%` : ""}
        </span>
      </p>
    </div>
  );
}

export function TransitionCard({ t }: { t: VariableTransition }) {
  const m = t.matrix;
  const sig = t.mcnemar.significant_05;

  return (
    <article className="card p-5">
      <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-[15px] font-bold text-[var(--text-1)]">
            {t.variable_name ?? t.variable_id}
          </h3>
          <p className="mt-0.5 text-[12px] text-[var(--text-3)]">
            Ronda {t.round_from} → {t.round_to}
            {t.family && ` · ${t.family}`}
          </p>
        </div>
        <PopulationTag kind="cambio" n={t.n_change_population} />
      </header>

      <div className="grid grid-cols-2 gap-2">
        <Cell spec={CELL.a} count={m.a_mantiene_alto} total={m.pareados} />
        <Cell spec={CELL.b} count={m.b_bajo} total={m.pareados} discordant />
        <Cell spec={CELL.c} count={m.c_subio} total={m.pareados} discordant />
        <Cell spec={CELL.d} count={m.d_mantiene_bajo} total={m.pareados} />
      </div>

      {m.no_comparables > 0 && (
        <p className="mt-2 text-[11px] text-[var(--text-3)]">
          {m.no_comparables} no comparable{m.no_comparables > 1 ? "s" : ""} —
          abstención o ausencia, con estado propio: nunca una celda en blanco.
        </p>
      )}

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 border-t border-[var(--border)] pt-4 text-[12px] sm:grid-cols-4">
        <div>
          <dt className="label-sm" title="(a + d) / pareados">
            Estabilidad
          </dt>
          <dd className="mt-0.5 text-[16px] font-bold tabular-nums text-[var(--text-1)]">
            {t.estabilidad == null ? "—" : pct(t.estabilidad, 0)}
          </dd>
        </div>
        <div>
          <dt className="label-sm" title="c − b: cuántos subieron de más">
            Dirección neta
          </dt>
          <dd
            className="mt-0.5 text-[16px] font-bold tabular-nums"
            style={{
              color:
                t.direccion_neta > 0
                  ? "#4f8ef7"
                  : t.direccion_neta < 0
                    ? "#d9534f"
                    : "var(--text-2)",
            }}
          >
            {t.direccion_neta > 0 ? "+" : ""}
            {t.direccion_neta}
          </dd>
        </div>
        <div>
          <dt className="label-sm" title="Corrimiento de la mediana entre rondas">
            Δ mediana
          </dt>
          <dd className="mt-0.5 text-[16px] font-bold tabular-nums text-[var(--text-2)]">
            {t.median_shift == null
              ? "—"
              : `${t.median_shift > 0 ? "+" : ""}${fmt(t.median_shift, 1)}`}
          </dd>
        </div>
        <div>
          <dt className="label-sm" title="Reducción del rango intercuartílico: el panel se estrecha">
            Δ RIQ
          </dt>
          <dd className="mt-0.5 text-[16px] font-bold tabular-nums text-[var(--text-2)]">
            {t.iqr_reduction == null
              ? "—"
              : `${t.iqr_reduction > 0 ? "−" : "+"}${fmt(Math.abs(t.iqr_reduction), 2)}`}
          </dd>
        </div>
      </dl>

      <div
        className="mt-4 rounded-lg border px-3 py-2.5 text-[12px] leading-relaxed"
        style={{
          borderColor: sig ? "rgba(79,142,247,0.32)" : "var(--border)",
          background: sig ? "rgba(79,142,247,0.07)" : "var(--surface-2)",
        }}
      >
        <p className="mb-1 flex flex-wrap items-center gap-2">
          <strong className="text-[var(--text-1)]">McNemar</strong>
          <span className="badge badge-gray">
            {t.mcnemar.method.replace(/_/g, " ")}
          </span>
          <span className="tabular-nums text-[var(--text-2)]">
            χ²cc = {fmt(t.mcnemar.statistic, 3)} · p = {fmt(t.mcnemar.p_value, 4)}
          </span>
          <span className={`badge ${sig ? "badge-blue" : "badge-gray"}`}>
            {sig ? "asimétrico" : "simétrico"}
          </span>
        </p>
        <p className="text-[var(--text-2)]">{t.mcnemar.interpretation}</p>
        {t.wilcoxon.p_value != null && t.wilcoxon.method !== "sin_cambios" && (
          <p className="mt-1.5 text-[11px] text-[var(--text-3)]">
            Wilcoxon de rangos con signo (usa la magnitud, no sólo el signo):
            p = {fmt(t.wilcoxon.p_value, 4)} sobre {t.wilcoxon.n_nonzero} pares
            con cambio.
          </p>
        )}
      </div>
    </article>
  );
}

export function TransitionList({
  transitions,
}: {
  transitions: VariableTransition[];
}) {
  if (!transitions.length) {
    return (
      <div className="card px-5 py-10 text-center">
        <p className="text-[14px] font-medium text-[var(--text-2)]">
          Todavía no hay dos rondas que comparar
        </p>
        <p className="mx-auto mt-1 max-w-md text-[13px] text-[var(--text-3)]">
          El análisis entre rondas es lo que distingue un Delphi de un taller.
          Aparecerá en cuanto la segunda ronda tenga entregas.
        </p>
      </div>
    );
  }
  // Primero lo que se movió: las variables con más discordantes.
  const sorted = [...transitions].sort(
    (a, b) =>
      b.matrix.b_bajo + b.matrix.c_subio - (a.matrix.b_bajo + a.matrix.c_subio)
  );
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      {sorted.map((t) => (
        <TransitionCard key={`${t.variable_id}-${t.round_to}`} t={t} />
      ))}
    </div>
  );
}

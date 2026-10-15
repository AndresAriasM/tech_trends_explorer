"use client";

/**
 * Piezas visuales compartidas del módulo Delphi.
 *
 * Paleta: la del proyecto (`app/globals.css`). Los colores que codifican datos
 * están validados con el script de la guía de visualización contra la
 * superficie oscura `#0c1120`:
 *
 *   · Par divergente del mapa de calor de Spearman — #4f8ef7 (+) ↔ #d9534f (−)
 *     con gris neutro en el cero: todas las comprobaciones pasan, incluido
 *     ΔE 25.4 bajo protanopía.
 *   · El mapa de cuadrantes **no codifica identidad por color**: la posición
 *     ES la categoría, y cada cuadrante lleva su nombre escrito. Cuatro tonos
 *     categóricos no superaban la separación CVD en todos los pares, así que
 *     los puntos usan un solo tono y el fondo sólo matiza.
 *   · Las clasificaciones son insignias con texto: nunca color a solas.
 */

import type { Cuadrante } from "./types";

/* ── Formato ─────────────────────────────────────────────────── */

export const fmt = (v: number | null | undefined, digits = 2): string =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const pct = (v: number | null | undefined, digits = 1): string =>
  v == null || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const int = (v: number | null | undefined): string =>
  v == null ? "—" : String(v);

/* ── Colores de datos ────────────────────────────────────────── */

export const DIVERGING_POS = "#4f8ef7";
export const DIVERGING_NEG = "#d9534f";
export const DATA_HUE = "#4f8ef7";

/** Tinte de fondo de cada cuadrante. Decorativo: el nombre va escrito encima. */
export const QUADRANT_TINT: Record<Cuadrante, string> = {
  CERRADA: "rgba(15, 158, 82, 0.09)",
  CONVERGIENDO: "rgba(79, 142, 247, 0.09)",
  DISENSO_PERSISTENTE: "rgba(201, 130, 26, 0.09)",
  INESTABLE: "rgba(157, 123, 245, 0.09)",
};

export const QUADRANT_LABEL: Record<Cuadrante, string> = {
  CERRADA: "Cerrada",
  CONVERGIENDO: "Convergiendo",
  DISENSO_PERSISTENTE: "Disenso persistente",
  INESTABLE: "Inestable",
};

/* ── Tarjeta de métrica ──────────────────────────────────────── */

export function StatTile({
  label,
  value,
  sub,
  hint,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  hint?: string;
  tone?: "default" | "warn" | "good";
}) {
  const valueColor =
    tone === "warn" ? "#e0a33a" : tone === "good" ? "#3ec07c" : "var(--text-1)";
  return (
    <div className="card px-5 py-4" title={hint}>
      <p className="label-sm mb-2">{label}</p>
      <p
        className="text-[30px] font-extrabold leading-none tracking-tight"
        style={{ color: valueColor, fontVariantNumeric: "tabular-nums" }}
      >
        {value}
      </p>
      {sub && <p className="mt-2 text-[11px] text-[var(--text-3)]">{sub}</p>}
    </div>
  );
}

/* ── Insignias ───────────────────────────────────────────────── */

export function ClassificationBadge({ value }: { value: string | null }) {
  if (!value) return <span className="text-[var(--text-4)]">—</span>;
  const cls =
    value === "PRIORITARIA"
      ? "badge-green"
      : value === "EN DISCUSIÓN"
        ? "badge-yellow"
        : "badge-gray";
  return <span className={`badge ${cls}`}>{value}</span>;
}

export function QuadrantBadge({ value }: { value: Cuadrante | null }) {
  if (!value) return <span className="badge badge-gray">Sin clasificar</span>;
  const cls =
    value === "CERRADA"
      ? "badge-green"
      : value === "CONVERGIENDO"
        ? "badge-blue"
        : value === "DISENSO_PERSISTENTE"
          ? "badge-yellow"
          : "badge-purple";
  return (
    <span className={`badge ${cls} whitespace-nowrap`}>
      {QUADRANT_LABEL[value]}
    </span>
  );
}

/* ── Etiqueta de población (§6) ──────────────────────────────── */

/**
 * La etiqueta que impide el error de la §6: *«la prioridad de X subió del
 * 60 % al 75 %» puede ser que nadie cambió de opinión y simplemente
 * desertaron los que decían que no.* Ninguna cifra del módulo se muestra sin
 * decir sobre qué población está calculada.
 */
export function PopulationTag({
  kind,
  n,
}: {
  kind: "agregado" | "cambio" | "agregado_restringido";
  n?: number | null;
}) {
  const text =
    kind === "cambio"
      ? "Población de cambio"
      : kind === "agregado_restringido"
        ? "Agregado restringido"
        : "Población de agregado";
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider"
      style={{
        borderColor: kind === "cambio" ? "rgba(79,142,247,0.28)" : "var(--border)",
        color: kind === "cambio" ? "#7fb0ff" : "var(--text-3)",
        background: kind === "cambio" ? "rgba(79,142,247,0.07)" : "transparent",
      }}
      title={
        kind === "cambio"
          ? "La intersección de las dos rondas: la única población sobre la que se puede hablar de movimiento de opinión (§6)."
          : kind === "agregado_restringido"
            ? "El agregado de la ronda anterior, restringido a quienes siguieron (§6.2)."
            : "Quienes contestaron esta ronda. Es el resultado de la ronda, no una medida de cambio (§6)."
      }
    >
      {text}
      {n != null && <strong className="text-[var(--text-2)]">n={n}</strong>}
    </span>
  );
}

/* ── Distribución en la escala (mini-barras) ─────────────────── */

/**
 * Una sola serie, magnitud: un solo tono, extremos redondeados de 4px
 * anclados al eje y 2px de hueco entre barras. NS/NR va **aparte**, separado
 * por una línea: no es un punto de la escala (§3).
 */
export function DistributionBars({
  distribution,
  height = 30,
  barWidth = 11,
}: {
  distribution: Record<string, number>;
  height?: number;
  barWidth?: number;
}) {
  const points = Object.keys(distribution)
    .filter((k) => k !== "NS/NR")
    .sort((a, b) => Number(a) - Number(b));
  const nsnr = distribution["NS/NR"] ?? 0;
  const max = Math.max(1, ...points.map((p) => distribution[p]), nsnr);

  const bar = (label: string, count: number, muted: boolean) => {
    const h = Math.max(count > 0 ? 3 : 1, (count / max) * height);
    return (
      <div
        key={label}
        className="flex flex-col items-center gap-1"
        title={`${label}: ${count}`}
      >
        <div
          className="flex items-end"
          style={{ height, width: barWidth }}
          aria-hidden
        >
          <div
            style={{
              width: "100%",
              height: h,
              borderRadius: "4px 4px 0 0",
              background: count === 0 ? "var(--surface-4)" : muted ? "var(--text-4)" : DATA_HUE,
              opacity: count === 0 ? 0.5 : 1,
            }}
          />
        </div>
        <span className="text-[9px] leading-none text-[var(--text-4)]">
          {label}
        </span>
      </div>
    );
  };

  return (
    <div className="flex items-end gap-[2px]">
      {points.map((p) => bar(p, distribution[p], false))}
      {nsnr > 0 && (
        <>
          <div
            className="mx-1 self-stretch"
            style={{ width: 1, background: "var(--border-2)" }}
            aria-hidden
          />
          {bar("NS", nsnr, true)}
        </>
      )}
    </div>
  );
}

/* ── Barras horizontales (perfil del panel) ──────────────────── */

export function BarList({
  items,
  total,
  emptyLabel = "Sin datos",
}: {
  items: { label: string; count: number }[];
  total: number;
  emptyLabel?: string;
}) {
  if (!items.length)
    return <p className="text-[13px] text-[var(--text-3)]">{emptyLabel}</p>;
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div className="space-y-1.5">
      {items.map((i) => (
        <div key={i.label} className="flex items-center gap-3 text-[12px]">
          <span className="w-40 shrink-0 truncate text-right text-[var(--text-2)]">
            {i.label}
          </span>
          <div
            className="h-[13px] flex-1 overflow-hidden rounded-[4px]"
            style={{ background: "var(--surface-3)" }}
          >
            <div
              style={{
                width: `${(i.count / max) * 100}%`,
                height: "100%",
                borderRadius: 4,
                background: DATA_HUE,
              }}
            />
          </div>
          <span
            className="w-24 shrink-0 text-[var(--text-3)]"
            style={{ fontVariantNumeric: "tabular-nums" }}
          >
            {i.count} · {total ? ((i.count / total) * 100).toFixed(1) : "0.0"}%
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── Aviso ───────────────────────────────────────────────────── */

export function Notice({
  tone = "info",
  title,
  children,
}: {
  tone?: "info" | "warn" | "error";
  title?: string;
  children: React.ReactNode;
}) {
  const color =
    tone === "error" ? "#d9534f" : tone === "warn" ? "#c9821a" : "#4f8ef7";
  return (
    <div
      className="rounded-xl border px-4 py-3 text-[13px] leading-relaxed"
      style={{
        borderColor: `${color}44`,
        background: `${color}0f`,
        color: "var(--text-2)",
      }}
      role={tone === "error" ? "alert" : undefined}
    >
      {title && (
        <p className="mb-1 font-semibold" style={{ color }}>
          {title}
        </p>
      )}
      {children}
    </div>
  );
}

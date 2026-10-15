"use client";

/**
 * /delphi/[id] — detalle de un estudio con sus estadísticas.
 *
 * Reproduce la pantalla perdida (tarjetas de invitados/registrados/rondas,
 * pestañas por ronda, tabla por agrupador, correlaciones de Spearman) y añade
 * lo que le faltaba y es la razón de ser del módulo: **el análisis entre
 * rondas**. Sin él, un Delphi es un taller repetido.
 *
 * Tres pestañas:
 *   · Por ronda    — población de agregado (§6): la tabla por agrupador.
 *   · Entre rondas — población de cambio (§6): a/b/c/d, McNemar, deserción.
 *   · Mapa         — los cuatro cuadrantes de §7 y el criterio de parada.
 */

import { useMemo, useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, BarChart3, RefreshCw, SlidersHorizontal } from "lucide-react";
import { useParams } from "next/navigation";

import {
  DelphiApiError,
  getStudy,
  getStudyStats,
  listParticipants,
} from "@/components/delphi/api";
import { Notice, PopulationTag, StatTile, fmt, pct } from "@/components/delphi/primitives";
import { PanelHealth } from "@/components/delphi/PanelHealth";
import { QuadrantMap } from "@/components/delphi/QuadrantMap";
import { SpearmanMatrix } from "@/components/delphi/SpearmanMatrix";
import { TransitionList } from "@/components/delphi/TransitionMatrix";
import { VariableTable } from "@/components/delphi/VariableTable";
import type {
  DelphiParticipant,
  DelphiRound,
  DelphiStudy,
  DelphiVariable,
  StudyReport,
} from "@/components/delphi/types";

type Tab = "rondas" | "entre" | "mapa";

type StudyDetail = DelphiStudy & {
  variables: DelphiVariable[];
  rounds: DelphiRound[];
  participants_summary: {
    n_invited: number;
    n_registered: number;
    n_dropped: number;
    mean_k: number | null;
  };
};

export default function DelphiStudyPage() {
  const params = useParams<{ id: string }>();
  const id = params?.id as string;

  const [tab, setTab] = useState<Tab>("rondas");
  const [roundIdx, setRoundIdx] = useState(0);
  /** §5.3: mover el umbral a posteriori y ver si el mapa aguanta. */
  const [threshold, setThreshold] = useState<number | null>(null);

  /**
   * Una sola consulta con las tres llamadas: el detalle es obligatorio, y el
   * informe y el panel se degradan a vacío si aún no hay respuestas. La clave
   * incluye el umbral, así que moverlo reanaliza sin recargar la página.
   */
  const { data, isFetching, error: queryError, refetch } = useQuery({
    queryKey: ["delphi", "study", id, threshold],
    enabled: Boolean(id),
    retry: false,
    queryFn: async () => {
      const s = (await getStudy(id)) as StudyDetail;
      const [rep, parts] = await Promise.all([
        getStudyStats(id, threshold ?? undefined).catch(() => null),
        listParticipants(id).catch(() => [] as DelphiParticipant[]),
      ]);
      return { study: s, report: rep as StudyReport | null, participants: parts };
    },
  });

  const study = data?.study ?? null;
  const report = data?.report ?? null;
  const participants: DelphiParticipant[] = data?.participants ?? [];
  const loading = isFetching && !data;
  const error =
    queryError instanceof DelphiApiError
      ? queryError
      : queryError
        ? new DelphiApiError(String(queryError), 0)
        : null;

  const activeThreshold =
    threshold ?? study?.dichotomy_threshold ?? report?.config.dichotomy_threshold ?? 4;

  const currentRound = report?.rounds[roundIdx];
  const currentSpearman = useMemo(
    () =>
      report?.spearman.find(
        (s) => s.round_number === currentRound?.round_number
      ) ?? null,
    [report, currentRound]
  );

  if (error && !study) {
    return (
      <div className="mx-auto max-w-3xl px-6 py-12">
        <Back />
        <Notice
          tone={error.schemaMissing ? "warn" : "error"}
          title={
            error.schemaMissing
              ? "El esquema del módulo todavía no está aplicado"
              : "No se pudo cargar el estudio"
          }
        >
          {error.message}
        </Notice>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-6 py-10">
      <Back />

      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2.5">
            <BarChart3 size={22} className="shrink-0 text-[var(--accent)]" />
            <h1 className="page-title truncate">
              {study?.title ?? "Cargando…"}
            </h1>
          </div>
          <p className="mt-1.5 text-[12.5px] text-[var(--text-3)]">
            Medias calculadas excluyendo las abstenciones, que nunca son un
            valor de la escala. Clasificación: media ≥ {activeThreshold} →
            prioritaria. La decisión final es del equipo.
          </p>
        </div>
        <button
          onClick={() => void refetch()}
          className="btn btn-ghost"
          disabled={isFetching}
        >
          <RefreshCw size={14} className={isFetching ? "spin" : undefined} />
          Actualizar
        </button>
      </header>

      {/* ── Tarjetas de cabecera ── */}
      <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Invitados"
          value={study?.participants_summary.n_invited ?? "—"}
        />
        <StatTile
          label="Registrados"
          value={study?.participants_summary.n_registered ?? "—"}
          sub={
            study?.participants_summary.mean_k != null
              ? `K medio ${fmt(study.participants_summary.mean_k)}`
              : "Sin K registrado"
          }
        />
        {(report?.panel.rounds ?? []).slice(0, 2).map((r) => {
          const meta = study?.rounds.find(
            (x) => x.round_number === r.round_number
          );
          return (
            <StatTile
              key={r.round_number}
              label={`Ronda ${r.round_number}${meta?.title ? ` · ${meta.title.replace(/^Ronda \d+\s*·?\s*/i, "")}` : ""}`}
              value={`${r.poblacion_agregado}/${report?.panel.n_invited ?? "—"}`}
              sub={`${pct(r.response_rate, 1)} de respuesta · ${meta?.status ?? "—"}`}
              hint="Población de agregado: quienes contestaron esta ronda (§6)."
            />
          );
        })}
        {report && report.panel.rounds.length > 1 && (
          <StatTile
            label="Deserción última ronda"
            value={pct(
              report.panel.rounds[report.panel.rounds.length - 1].attrition_rate,
              1
            )}
            sub={`Población de cambio: ${report.panel.rounds[report.panel.rounds.length - 1].poblacion_cambio ?? "—"}`}
            tone={
              (report.panel.rounds[report.panel.rounds.length - 1]
                .attrition_rate ?? 0) >= 0.3
                ? "warn"
                : "default"
            }
            hint="La deserción es el patrón normal en Delphi, no una anomalía: hay que diseñar para ella (§6)."
          />
        )}
      </div>

      {!report && !loading && (
        <Notice tone="info" title="Todavía no hay respuestas entregadas">
          Las estadísticas aparecen en cuanto la primera ronda recibe entregas.
          Un borrador no es un juicio y queda fuera del análisis.
        </Notice>
      )}

      {report && (
        <>
          {/* ── Pestañas ── */}
          <nav className="mb-5 flex flex-wrap items-center gap-2">
            <TabButton
              active={tab === "rondas"}
              onClick={() => setTab("rondas")}
              count={report.rounds.length}
            >
              Por ronda
            </TabButton>
            <TabButton
              active={tab === "entre"}
              onClick={() => setTab("entre")}
              count={report.between.length}
            >
              Entre rondas
            </TabButton>
            <TabButton
              active={tab === "mapa"}
              onClick={() => setTab("mapa")}
              count={report.map.variables.length}
            >
              Mapa de variables
            </TabButton>

            <div className="ml-auto flex items-center gap-2 rounded-xl border border-[var(--border)] px-3 py-1.5">
              <SlidersHorizontal size={13} className="text-[var(--text-3)]" />
              <label
                className="text-[11px] text-[var(--text-3)]"
                htmlFor="thr"
                title="El umbral de dicotomía es configuración, no un literal: moverlo recalcula el mapa entero sin migrar nada (§5.3)."
              >
                Umbral
              </label>
              <input
                id="thr"
                type="number"
                min={report.config.scale_min}
                max={report.config.scale_max}
                step={1}
                value={activeThreshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="w-14 bg-transparent text-[13px] font-semibold tabular-nums text-[var(--text-1)] outline-none"
              />
              {threshold != null &&
                threshold !== study?.dichotomy_threshold && (
                  <button
                    onClick={() => setThreshold(null)}
                    className="text-[11px] text-[var(--accent)] hover:underline"
                  >
                    restablecer
                  </button>
                )}
            </div>
          </nav>

          {/* ── Por ronda ── */}
          {tab === "rondas" && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                {report.rounds.map((r, i) => {
                  const meta = study?.rounds.find(
                    (x) => x.round_number === r.round_number
                  );
                  return (
                    <button
                      key={r.round_number}
                      onClick={() => setRoundIdx(i)}
                      className={`rounded-xl px-4 py-2 text-[13px] font-medium transition-colors ${
                        i === roundIdx
                          ? "bg-[var(--accent)] text-white"
                          : "border border-[var(--border)] text-[var(--text-2)] hover:text-[var(--text-1)]"
                      }`}
                    >
                      {meta?.title || `Ronda ${r.round_number}`}
                      <span
                        className={`ml-2 rounded-full px-1.5 py-0.5 text-[10px] tabular-nums ${
                          i === roundIdx
                            ? "bg-white/20"
                            : "bg-[var(--surface-3)] text-[var(--text-3)]"
                        }`}
                      >
                        {r.n_respondents}
                      </span>
                    </button>
                  );
                })}
              </div>

              {currentRound && (
                <>
                  <div className="card flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-3.5 text-[12.5px]">
                    <PopulationTag
                      kind="agregado"
                      n={currentRound.n_respondents}
                    />
                    <span className="text-[var(--text-3)]">
                      Tasa de respuesta{" "}
                      <strong className="text-[var(--text-1)]">
                        {pct(currentRound.response_rate, 1)}
                      </strong>
                    </span>
                    <span className="text-[var(--text-3)]">
                      Prioritarias{" "}
                      <strong className="text-[var(--text-1)]">
                        {currentRound.n_prioritarias}
                      </strong>{" "}
                      · en discusión{" "}
                      <strong className="text-[var(--text-1)]">
                        {currentRound.n_en_discusion}
                      </strong>{" "}
                      · descartadas{" "}
                      <strong className="text-[var(--text-1)]">
                        {currentRound.n_descartadas}
                      </strong>
                    </span>
                    <span className="ml-auto text-[11px] text-[var(--text-4)]">
                      Estas cifras describen esta ronda. No dicen nada sobre
                      cambio de opinión: eso está en «Entre rondas».
                    </span>
                  </div>

                  {currentRound.by_family.map((g) => (
                    <VariableTable
                      key={g.family}
                      family={g.family}
                      variables={g.variables}
                      threshold={activeThreshold}
                      population={currentRound.n_respondents}
                    />
                  ))}

                  {currentSpearman && (
                    <SpearmanMatrix
                      data={currentSpearman}
                      roundNumber={currentRound.round_number}
                    />
                  )}
                </>
              )}
            </div>
          )}

          {/* ── Entre rondas ── */}
          {tab === "entre" && (
            <div className="space-y-4">
              <div className="card px-5 py-4">
                <h2 className="text-[15px] font-bold text-[var(--text-1)]">
                  ¿Cambió el panel de opinión, y hacia dónde?
                </h2>
                <p className="mt-1 max-w-3xl text-[12.5px] leading-relaxed text-[var(--text-3)]">
                  Todo lo de esta pestaña se calcula sobre la población de
                  cambio —los panelistas presentes en ambas rondas— y sólo
                  sobre ella. Las celdas <code>b</code> y <code>c</code> de
                  cada matriz son quienes cambiaron de lado: son la respuesta a
                  la pregunta del ejercicio y las dos celdas discordantes de
                  McNemar.
                </p>
              </div>

              <PanelHealth
                panel={report.panel}
                attrition={report.attrition_check}
                participants={participants}
              />

              <TransitionList transitions={report.between} />
            </div>
          )}

          {/* ── Mapa ── */}
          {tab === "mapa" && (
            <div className="space-y-4">
              <QuadrantMap map={report.map} />
              {report.map.variables.length === 0 && (
                <Notice tone="info">
                  El mapa necesita dos rondas: el eje de estabilidad sale de la
                  matriz de transición. Con una sola ronda hay consenso, pero
                  no hay movimiento que medir.
                </Notice>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Back() {
  return (
    <Link
      href="/delphi"
      className="mb-4 inline-flex items-center gap-1.5 text-[13px] text-[var(--text-3)] transition-colors hover:text-[var(--text-1)]"
    >
      <ArrowLeft size={14} />
      Panel Delphi
    </Link>
  );
}

function TabButton({
  active,
  onClick,
  count,
  children,
}: {
  active: boolean;
  onClick: () => void;
  count?: number;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-xl px-4 py-2 text-[13px] font-medium transition-colors ${
        active
          ? "bg-[var(--surface-3)] text-[var(--text-1)]"
          : "text-[var(--text-3)] hover:text-[var(--text-2)]"
      }`}
    >
      {children}
      {count != null && (
        <span className="ml-2 text-[11px] tabular-nums text-[var(--text-4)]">
          {count}
        </span>
      )}
    </button>
  );
}

"use client";

import { useEffect, useRef } from "react";
import { Zap, RefreshCw, Save, Sparkles, TrendingUp } from "lucide-react";
import { QueryBuilder } from "@/components/query-builder/QueryBuilder";
import { HypeCycleChart } from "@/components/charts/HypeCycleChart";
import { MentionsTimelineChart } from "@/components/charts/MentionsTimelineChart";
import { AnalysisProgressBar } from "@/components/ui/AnalysisProgressBar";
import { useAnalysisStore } from "@/store/analysisStore";
import { startHypeCycleAnalysis, streamTaskProgress } from "@/lib/api";
import type { TaskProgress, HypePhase } from "@/types";
import { cn } from "@/lib/utils";

const PHASE_COLOR: Record<HypePhase, string> = {
  "Innovation Trigger":            "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment":     "#ef4444",
  "Slope of Enlightenment":        "#34d399",
  "Plateau of Productivity":       "#818cf8",
  "Unknown":                       "#6b7280",
};

export default function HypeCyclePage() {
  const {
    searchTerms, technologyName, technologyDescription,
    categoryId, minYear, autoSave,
    isRunning, progress, result, error,
    setTaskId, setProgress, setResult, setRunning, setError, resetAnalysis,
  } = useAnalysisStore();

  const cleanupRef = useRef<(() => void) | null>(null);
  useEffect(() => () => cleanupRef.current?.(), []);

  async function handleAnalyze() {
    if (isRunning) return;
    const filled = searchTerms.filter((t) => t.value.trim());
    if (!filled.length) return;
    resetAnalysis();
    setRunning(true);
    try {
      const res = await startHypeCycleAnalysis({
        search_terms: filled,
        technology_name: technologyName || undefined,
        technology_description: technologyDescription || undefined,
        category_id: categoryId,
        search_parameters: { min_year: minYear, sources_filter: ["Tech News", "Business News"], max_results: 200 },
        auto_save: autoSave,
      });
      setTaskId(res.task_id);
      cleanupRef.current = streamTaskProgress(
        res.task_id,
        (p: TaskProgress) => setProgress(p),
        (r) => { setResult(r); setRunning(false); },
        (e) => { setError(e);  setRunning(false); }
      );
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Error al conectar con el backend");
      setRunning(false);
    }
  }

  const m = result?.hype_metrics;
  const phaseColor = m ? (PHASE_COLOR[m.phase] ?? "#6b7280") : "#3b7cff";

  return (
    <div className="flex h-[calc(100vh-var(--header-h))] overflow-hidden">

      {/* ── LEFT PANEL: Configuration ──────────────────────── */}
      <div
        className="flex flex-col w-[340px] shrink-0 overflow-y-auto"
        style={{ borderRight: "1px solid var(--border)", background: "var(--s1)" }}
      >
        {/* Panel header */}
        <div className="px-5 py-4 shrink-0" style={{ borderBottom: "1px solid var(--border)" }}>
          <div className="flex items-center gap-2 mb-0.5">
            <Sparkles className="w-3.5 h-3.5" style={{ color: "var(--accent)" }} />
            <p className="text-xs font-bold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
              Configurar análisis
            </p>
          </div>
          <p className="text-xs" style={{ color: "var(--t3)" }}>
            Posicionamiento en Hype Cycle de Gartner
          </p>
        </div>

        {/* Form */}
        <div className="flex-1 px-5 py-4">
          <QueryBuilder />
        </div>

        {/* Actions — sticky bottom */}
        <div className="px-5 py-4 shrink-0 space-y-3" style={{ borderTop: "1px solid var(--border)" }}>
          <button
            onClick={handleAnalyze}
            disabled={isRunning || !searchTerms.some((t) => t.value.trim())}
            className="btn btn-primary w-full justify-center"
          >
            {isRunning
              ? <><RefreshCw className="w-4 h-4 animate-spin" /> Analizando...</>
              : <><Zap className="w-4 h-4" /> Iniciar análisis</>
            }
          </button>

          {progress && <AnalysisProgressBar progress={progress} />}

          {error && (
            <div
              className="text-xs px-3 py-2.5 rounded-lg"
              style={{ background: "rgba(239,68,68,.08)", border: "1px solid rgba(239,68,68,.2)", color: "#fca5a5" }}
            >
              {error}
            </div>
          )}
        </div>
      </div>

      {/* ── RIGHT PANEL: Results ───────────────────────────── */}
      <div className="flex-1 overflow-y-auto" style={{ background: "var(--s0)" }}>
        {result && m ? (
          <div className="p-6 space-y-5 fade-in">

            {/* Phase hero banner */}
            <div
              className="relative overflow-hidden rounded-2xl p-6"
              style={{
                background: `linear-gradient(135deg, ${phaseColor}14 0%, ${phaseColor}06 100%)`,
                border: `1px solid ${phaseColor}28`,
              }}
            >
              <div
                className="absolute -top-8 -right-8 w-40 h-40 rounded-full pointer-events-none"
                style={{ background: `radial-gradient(circle, ${phaseColor}18 0%, transparent 70%)` }}
              />
              <div className="relative">
                <div className="flex items-start justify-between gap-4 flex-wrap">
                  <div>
                    <p className="text-xs font-semibold mb-2" style={{ color: phaseColor }}>
                      FASE DETECTADA
                    </p>
                    <h2 className="text-2xl font-bold mb-1" style={{ color: "var(--t1)" }}>
                      {m.phase}
                    </h2>
                    {result.technology_name && (
                      <p className="text-sm" style={{ color: "var(--t2)" }}>
                        {result.technology_name}
                      </p>
                    )}
                  </div>
                  <div className="text-right">
                    <p className="text-4xl font-bold" style={{ color: phaseColor }}>
                      {(m.confidence * 100).toFixed(0)}
                      <span className="text-xl ml-1" style={{ color: "var(--t3)" }}>%</span>
                    </p>
                    <p className="text-xs mt-1" style={{ color: "var(--t3)" }}>confianza</p>
                  </div>
                </div>

                {/* Metric row */}
                <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    { label: "Menciones totales", value: m.total_mentions.toLocaleString(), accent: phaseColor },
                    { label: "Pico de menciones", value: `${m.peak_mentions.toLocaleString()}${m.peak_year ? ` (${m.peak_year})` : ""}`, accent: phaseColor },
                    { label: "Sentimiento",  value: m.sentiment_avg.toFixed(2), accent: m.sentiment_avg > 0 ? "#22c55e" : "#ef4444" },
                    { label: "Al Plateau",   value: m.time_to_plateau,           accent: "#a78bfa" },
                  ].map((s) => (
                    <div
                      key={s.label}
                      className="rounded-xl px-4 py-3"
                      style={{ background: "rgba(255,255,255,.04)", border: "1px solid var(--border)" }}
                    >
                      <p className="text-[10px] mb-1" style={{ color: "var(--t3)" }}>{s.label}</p>
                      <p className="text-sm font-bold" style={{ color: s.accent }}>{s.value}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Hype Cycle chart */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp className="w-4 h-4" style={{ color: "var(--accent)" }} />
                <p className="section-label m-0 p-0">Posición en el ciclo</p>
              </div>
              <HypeCycleChart queries={[result]} highlightId={result.query_id} />
            </div>

            {/* Timeline chart */}
            {result.yearly_stats?.length > 0 && (
              <div className="card p-5">
                <p className="section-label mb-4">Menciones y sentimiento por año</p>
                <MentionsTimelineChart data={result.yearly_stats} technologyName={result.technology_name} />
              </div>
            )}

            {/* Meta footer */}
            <div
              className="flex items-center justify-between px-4 py-3 rounded-xl text-xs"
              style={{ background: "var(--s1)", border: "1px solid var(--border)", color: "var(--t3)" }}
            >
              <span>ID: <code style={{ color: "var(--t2)" }}>{result.query_id}</code></span>
              {result.processing_time && <span>Procesado en {result.processing_time.toFixed(1)}s</span>}
              <button className="flex items-center gap-1.5 hover:text-white transition-colors">
                <Save className="w-3.5 h-3.5" /> Exportar
              </button>
            </div>
          </div>
        ) : (
          /* Empty state */
          <div className="h-full flex flex-col items-center justify-center gap-4 p-10">
            <div
              className="w-20 h-20 rounded-2xl flex items-center justify-center"
              style={{ background: "var(--accent-sub)", border: "1px solid rgba(59,124,255,.15)" }}
            >
              <TrendingUp className="w-9 h-9" style={{ color: "var(--accent)" }} />
            </div>
            <div className="text-center space-y-1.5">
              <p className="text-sm font-semibold" style={{ color: "var(--t1)" }}>
                {isRunning ? "Analizando tecnología..." : "Configura tu análisis"}
              </p>
              <p className="text-xs max-w-xs" style={{ color: "var(--t2)" }}>
                {isRunning
                  ? "Buscando en fuentes web y calculando la posición en el Hype Cycle."
                  : "Agrega los términos de búsqueda en el panel izquierdo e inicia el análisis."
                }
              </p>
            </div>
            {isRunning && (
              <div className="w-48 h-1 rounded-full overflow-hidden" style={{ background: "var(--s3)" }}>
                <div
                  className="h-full rounded-full shimmer"
                  style={{ width: `${progress?.progress ?? 30}%`, background: "var(--accent)" }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

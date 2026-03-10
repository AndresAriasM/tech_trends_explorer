"use client";

import { useState, useEffect, useRef } from "react";
import { Play, RefreshCw } from "lucide-react";
import { QueryBuilder } from "@/components/query-builder/QueryBuilder";
import { HypeCycleChart } from "@/components/charts/HypeCycleChart";
import { MentionsTimelineChart } from "@/components/charts/MentionsTimelineChart";
import { AnalysisProgressBar } from "@/components/ui/AnalysisProgressBar";
import { MetricCard } from "@/components/ui/MetricCard";
import { useAnalysisStore } from "@/store/analysisStore";
import { startHypeCycleAnalysis, streamTaskProgress } from "@/lib/api";
import type { TaskProgress } from "@/types";

export default function HypeCyclePage() {
  const {
    searchTerms,
    technologyName,
    technologyDescription,
    categoryId,
    minYear,
    autoSave,
    isRunning,
    progress,
    result,
    error,
    setTaskId,
    setProgress,
    setResult,
    setRunning,
    setError,
    resetAnalysis,
  } = useAnalysisStore();

  const cleanupRef = useRef<(() => void) | null>(null);

  // Cleanup SSE on unmount
  useEffect(() => () => cleanupRef.current?.(), []);

  async function handleAnalyze() {
    if (isRunning) return;
    const filledTerms = searchTerms.filter((t) => t.value.trim());
    if (filledTerms.length === 0) return;

    resetAnalysis();
    setRunning(true);

    try {
      const res = await startHypeCycleAnalysis({
        search_terms: filledTerms,
        technology_name: technologyName || undefined,
        technology_description: technologyDescription || undefined,
        category_id: categoryId,
        search_parameters: {
          min_year: minYear,
          sources_filter: ["Tech News", "Business News"],
          max_results: 200,
        },
        auto_save: autoSave,
      });

      setTaskId(res.task_id);

      cleanupRef.current = streamTaskProgress(
        res.task_id,
        (p: TaskProgress) => setProgress(p),
        (r) => {
          setResult(r);
          setRunning(false);
        },
        (e) => {
          setError(e);
          setRunning(false);
        }
      );
    } catch (e: unknown) {
      const message = e instanceof Error ? e.message : "Error al conectar con el backend";
      setError(message);
      setRunning(false);
    }
  }

  const metrics = result?.hype_metrics;

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Hype Cycle de Gartner</h1>
        <p className="text-gray-400 text-sm mt-1">
          Determina la posición de una tecnología en el ciclo de sobreexpectativas.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left — Query Builder */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-6 space-y-6">
          <h2 className="font-semibold text-white text-sm uppercase tracking-wider">
            Configurar análisis
          </h2>

          <QueryBuilder />

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={handleAnalyze}
              disabled={isRunning}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors"
            >
              {isRunning ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isRunning ? "Analizando..." : "Iniciar análisis"}
            </button>
          </div>

          {/* Progress */}
          {progress && <AnalysisProgressBar progress={progress} />}

          {/* Error */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3 text-sm text-red-400">
              {error}
            </div>
          )}
        </div>

        {/* Right — Live result */}
        {result && metrics ? (
          <div className="space-y-4">
            {/* Phase badge */}
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
              <p className="text-xs text-gray-500 mb-1">Fase detectada</p>
              <p className="text-xl font-bold text-white">{metrics.phase}</p>
              <p className="text-sm text-gray-400 mt-1">
                Confianza: {(metrics.confidence * 100).toFixed(0)}%
              </p>
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-3">
              <MetricCard
                label="Total menciones"
                value={metrics.total_mentions.toLocaleString()}
                color="blue"
              />
              <MetricCard
                label="Pico de menciones"
                value={metrics.peak_mentions.toLocaleString()}
                sub={metrics.peak_year ? `Año ${metrics.peak_year}` : undefined}
                color="yellow"
              />
              <MetricCard
                label="Sentimiento promedio"
                value={metrics.sentiment_avg.toFixed(2)}
                color={metrics.sentiment_avg > 0 ? "green" : "red"}
              />
              <MetricCard
                label="Tiempo al Plateau"
                value={metrics.time_to_plateau}
                color="purple"
              />
            </div>
          </div>
        ) : (
          <div className="bg-gray-900 border border-dashed border-gray-700 rounded-xl p-8 flex items-center justify-center">
            <p className="text-gray-600 text-sm text-center">
              Los resultados aparecerán aquí<br />cuando el análisis finalice.
            </p>
          </div>
        )}
      </div>

      {/* Bottom — Full charts when result available */}
      {result && (
        <div className="space-y-6">
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
            <h2 className="font-semibold text-white text-sm uppercase tracking-wider mb-4">
              Posición en el Hype Cycle
            </h2>
            <HypeCycleChart queries={[result]} highlightId={result.query_id} />
          </div>

          {result.yearly_stats?.length > 0 && (
            <div className="bg-gray-900 border border-gray-800 rounded-xl p-6">
              <h2 className="font-semibold text-white text-sm uppercase tracking-wider mb-4">
                Menciones y sentimiento por año
              </h2>
              <MentionsTimelineChart
                data={result.yearly_stats}
                technologyName={result.technology_name}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

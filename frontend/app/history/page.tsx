"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Search, Clock, TrendingUp } from "lucide-react";
import { getHypeCycleHistory, getCategories } from "@/lib/api";
import { HypeCycleChart } from "@/components/charts/HypeCycleChart";
import { MentionsTimelineChart } from "@/components/charts/MentionsTimelineChart";
import { MetricCard } from "@/components/ui/MetricCard";
import { useAnalysisStore } from "@/store/analysisStore";
import type { HypeCycleQuery } from "@/types";

export default function HistoryPage() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  const { data: categories } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  const { data: history = [], isLoading } = useQuery({
    queryKey: ["hype-cycle-history", categoryFilter],
    queryFn: () =>
      getHypeCycleHistory(categoryFilter === "all" ? undefined : categoryFilter),
  });

  const { setSelectedQuery } = useAnalysisStore();

  const filtered = history.filter(
    (q) =>
      !search ||
      q.technology_name?.toLowerCase().includes(search.toLowerCase()) ||
      q.search_query.toLowerCase().includes(search.toLowerCase())
  );

  const selected = filtered.find((q) => q.query_id === selectedId) ?? null;

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Historial de análisis</h1>
        <p className="text-gray-400 text-sm mt-1">
          Consultas guardadas en DynamoDB. Reutiliza un análisis para ahorrar créditos de API.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left — list */}
        <div className="lg:col-span-1 space-y-3">
          {/* Filters */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Buscar tecnología..."
                className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-9 pr-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
              />
            </div>

            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
            >
              <option value="all">Todas las categorías</option>
              {categories?.map((c) => (
                <option key={c.category_id} value={c.category_id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          {/* Query list */}
          <div className="space-y-2">
            {isLoading && (
              <p className="text-sm text-gray-500 text-center py-8">Cargando...</p>
            )}
            {!isLoading && filtered.length === 0 && (
              <p className="text-sm text-gray-500 text-center py-8">
                No hay análisis guardados.
              </p>
            )}
            {filtered.map((q) => (
              <QueryCard
                key={q.query_id}
                query={q}
                isActive={q.query_id === selectedId}
                onClick={() => {
                  setSelectedId(q.query_id);
                  setSelectedQuery(q);
                }}
              />
            ))}
          </div>
        </div>

        {/* Right — detail */}
        <div className="lg:col-span-2">
          {selected ? (
            <QueryDetail query={selected} />
          ) : (
            <div className="bg-gray-900 border border-dashed border-gray-700 rounded-xl p-12 flex items-center justify-center h-full">
              <p className="text-gray-600 text-sm">Selecciona un análisis de la lista</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Query Card ───────────────────────────────────────────────

function QueryCard({
  query,
  isActive,
  onClick,
}: {
  query: HypeCycleQuery;
  isActive: boolean;
  onClick: () => void;
}) {
  const date = new Date(query.execution_date).toLocaleDateString("es-CO", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

  return (
    <button
      onClick={onClick}
      className={`w-full text-left bg-gray-900 border rounded-xl p-4 transition-colors ${
        isActive
          ? "border-blue-500 bg-blue-500/5"
          : "border-gray-800 hover:border-gray-700"
      }`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="font-medium text-white text-sm truncate">
            {query.technology_name || query.search_query.slice(0, 30)}
          </p>
          <p className="text-xs text-gray-500 truncate mt-0.5">
            {query.hype_metrics.phase}
          </p>
        </div>
        <TrendingUp className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
      </div>
      <div className="flex items-center gap-1 mt-2 text-xs text-gray-600">
        <Clock className="w-3 h-3" />
        {date}
      </div>
    </button>
  );
}

// ── Query Detail ─────────────────────────────────────────────

function QueryDetail({ query }: { query: HypeCycleQuery }) {
  const m = query.hype_metrics;

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">
              {query.technology_name || "Sin nombre"}
            </h2>
            <p className="text-sm text-gray-400 mt-1">{m.phase}</p>
            {query.technology_description && (
              <p className="text-xs text-gray-500 mt-2 max-w-lg">
                {query.technology_description}
              </p>
            )}
          </div>
          <span className="text-xs bg-blue-500/10 text-blue-400 border border-blue-500/20 rounded-full px-3 py-1">
            {(m.confidence * 100).toFixed(0)}% confianza
          </span>
        </div>
        <p className="text-xs text-gray-600 mt-3 font-mono">{query.search_query}</p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Menciones" value={m.total_mentions.toLocaleString()} color="blue" />
        <MetricCard
          label="Pico"
          value={m.peak_mentions.toLocaleString()}
          sub={m.peak_year ? `${m.peak_year}` : undefined}
          color="yellow"
        />
        <MetricCard
          label="Sentimiento"
          value={m.sentiment_avg.toFixed(2)}
          color={m.sentiment_avg > 0 ? "green" : "red"}
        />
        <MetricCard label="Al Plateau" value={m.time_to_plateau} color="purple" />
      </div>

      {/* Hype Cycle chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-3">
          Posición en el Hype Cycle
        </h3>
        <HypeCycleChart queries={[query]} highlightId={query.query_id} />
      </div>

      {/* Timeline */}
      {query.yearly_stats?.length > 0 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-white uppercase tracking-wider mb-3">
            Menciones por año
          </h3>
          <MentionsTimelineChart
            data={query.yearly_stats}
            technologyName={query.technology_name}
          />
        </div>
      )}
    </div>
  );
}

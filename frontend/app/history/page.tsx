"use client";

import { useEffect, useState, Suspense } from "react";
import { useQuery } from "@tanstack/react-query";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  Search,
  Clock,
  TrendingUp,
  FolderOpen,
  Check,
  X as XIcon,
} from "lucide-react";
import { getHypeCycleHistory, getCategories } from "@/lib/api";
import { HypeCycleChart } from "@/components/charts/HypeCycleChart";
import { MentionsTimelineChart } from "@/components/charts/MentionsTimelineChart";
import { MetricCard } from "@/components/ui/MetricCard";
import { useAnalysisStore } from "@/store/analysisStore";
import type { HypeCycleQuery, HypePhase } from "@/types";

const PHASE_COLORS: Record<HypePhase, string> = {
  "Innovation Trigger": "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment": "#ef4444",
  "Slope of Enlightenment": "#34d399",
  "Plateau of Productivity": "#818cf8",
  Unknown: "#9ca3af",
};

const PHASE_SHORT: Record<HypePhase, string> = {
  "Innovation Trigger": "Lanzamiento",
  "Peak of Inflated Expectations": "Pico de expectativas",
  "Trough of Disillusionment": "Valle de desilusión",
  "Slope of Enlightenment": "Rampa de consolidación",
  "Plateau of Productivity": "Meseta de productividad",
  Unknown: "Desconocida",
};

function HistoryPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const preselectedId = searchParams.get("id");
  const preselectedCategory = searchParams.get("category");

  const [selectedId, setSelectedId] = useState<string | null>(preselectedId);
  const [categoryFilter, setCategoryFilter] = useState<string>(
    preselectedCategory ?? "all"
  );
  const [search, setSearch] = useState("");
  const [multiSelected, setMultiSelected] = useState<Set<string>>(new Set());

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

  // Auto-select first result if preselectedId is set but not yet in view
  useEffect(() => {
    if (preselectedId && !selectedId) setSelectedId(preselectedId);
  }, [preselectedId, selectedId]);

  const selected = filtered.find((q) => q.query_id === selectedId) ?? null;

  function toggleMulti(id: string) {
    setMultiSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearMulti() {
    setMultiSelected(new Set());
  }

  function compareSelected() {
    const ids = Array.from(multiSelected).join(",");
    const base = categoryFilter !== "all" ? `category=${categoryFilter}&` : "";
    router.push(`/hype-cycle/compare?${base}ids=${ids}`);
  }

  return (
    <div style={{ padding: "32px 48px", maxWidth: 1400, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1
          className="heading-glow"
          style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.02em", marginBottom: 6 }}
        >
          Historial de análisis
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-2)", maxWidth: 720 }}>
          Consultas guardadas en DynamoDB. Selecciona múltiples para compararlas
          en un solo Hype Cycle.
        </p>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(320px, 380px) 1fr",
          gap: 24,
          alignItems: "flex-start",
        }}
      >
        {/* ── Left column — list ────────────────────────────────── */}
        <div>
          {/* Filters */}
          <div className="card" style={{ padding: 16, marginBottom: 12 }}>
            <div style={{ position: "relative", marginBottom: 10 }}>
              <Search
                style={{
                  position: "absolute",
                  left: 12,
                  top: "50%",
                  transform: "translateY(-50%)",
                  width: 14,
                  height: 14,
                  color: "var(--text-3)",
                }}
              />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Buscar por nombre o query…"
                className="inp"
                style={{ width: "100%", paddingLeft: 34 }}
              />
            </div>
            <select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              className="inp"
              style={{ width: "100%", cursor: "pointer" }}
            >
              <option value="all">Todas las categorías</option>
              {categories?.map((c) => (
                <option key={c.category_id} value={c.category_id}>
                  {c.name}
                </option>
              ))}
            </select>
          </div>

          {/* Quick action: view category on Hype Cycle */}
          {categoryFilter !== "all" && filtered.length > 0 && (
            <Link
              href={`/hype-cycle/compare?category=${categoryFilter}`}
              className="btn btn-primary"
              style={{
                width: "100%",
                justifyContent: "center",
                marginBottom: 12,
                fontSize: 12.5,
              }}
            >
              <TrendingUp style={{ width: 13, height: 13 }} />
              Graficar categoría completa
            </Link>
          )}

          {/* Query list */}
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {isLoading && (
              <p
                style={{
                  fontSize: 13,
                  color: "var(--text-3)",
                  textAlign: "center",
                  padding: 32,
                }}
              >
                Cargando…
              </p>
            )}
            {!isLoading && filtered.length === 0 && (
              <div className="card" style={{ padding: 40, textAlign: "center" }}>
                <FolderOpen
                  style={{
                    width: 28,
                    height: 28,
                    color: "var(--text-4)",
                    margin: "0 auto 10px",
                  }}
                />
                <p style={{ fontSize: 13, color: "var(--text-3)" }}>
                  No hay análisis guardados.
                </p>
              </div>
            )}
            {filtered.map((q) => (
              <QueryCard
                key={q.query_id}
                query={q}
                isActive={q.query_id === selectedId}
                isChecked={multiSelected.has(q.query_id)}
                onClick={() => {
                  setSelectedId(q.query_id);
                  setSelectedQuery(q);
                }}
                onToggleCheck={() => toggleMulti(q.query_id)}
              />
            ))}
          </div>
        </div>

        {/* ── Right column — detail ─────────────────────────────── */}
        <div>
          {selected ? (
            <QueryDetail query={selected} />
          ) : (
            <div
              className="card"
              style={{
                padding: 60,
                textAlign: "center",
                borderStyle: "dashed",
                minHeight: 300,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <TrendingUp
                style={{ width: 32, height: 32, color: "var(--text-4)", marginBottom: 12 }}
              />
              <p style={{ fontSize: 14, color: "var(--text-2)", marginBottom: 4 }}>
                Selecciona un análisis para ver el detalle
              </p>
              <p style={{ fontSize: 12, color: "var(--text-3)" }}>
                O marca varios con la casilla para compararlos en un solo Hype
                Cycle.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* ── Floating multi-select action bar ─────────────────────── */}
      {multiSelected.size > 0 && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            zIndex: 40,
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "10px 14px 10px 18px",
            background: "rgba(10, 15, 24, 0.95)",
            backdropFilter: "blur(10px)",
            border: "1px solid var(--accent-border)",
            boxShadow: "0 0 32px var(--accent-glow), 0 8px 32px rgba(0,0,0,0.5)",
            borderRadius: 999,
          }}
        >
          <span
            style={{
              fontSize: 13,
              color: "var(--text-1)",
              fontWeight: 600,
            }}
          >
            {multiSelected.size} {multiSelected.size === 1 ? "seleccionada" : "seleccionadas"}
          </span>
          <button
            className="btn btn-primary"
            onClick={compareSelected}
            disabled={multiSelected.size < 2}
            style={{ fontSize: 12.5 }}
            title={
              multiSelected.size < 2
                ? "Selecciona al menos 2 para comparar"
                : "Comparar en Hype Cycle"
            }
          >
            <TrendingUp style={{ width: 13, height: 13 }} />
            Comparar
          </button>
          <button
            onClick={clearMulti}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-3)",
              cursor: "pointer",
              padding: 6,
              borderRadius: 999,
              display: "flex",
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
            onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
            aria-label="Limpiar selección"
          >
            <XIcon style={{ width: 14, height: 14 }} />
          </button>
        </div>
      )}
    </div>
  );
}

export default function HistoryPage() {
  return (
    <Suspense fallback={<div style={{ padding: 48, color: "var(--text-3)" }}>Cargando…</div>}>
      <HistoryPageInner />
    </Suspense>
  );
}

// ── Query Card ───────────────────────────────────────────────

function QueryCard({
  query,
  isActive,
  isChecked,
  onClick,
  onToggleCheck,
}: {
  query: HypeCycleQuery;
  isActive: boolean;
  isChecked: boolean;
  onClick: () => void;
  onToggleCheck: () => void;
}) {
  const date = new Date(query.execution_date).toLocaleDateString("es-CO", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
  const phase = query.hype_metrics.phase;

  return (
    <div
      className="card"
      style={{
        padding: 14,
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
        cursor: "pointer",
        borderColor: isActive ? "var(--accent-border)" : "var(--border)",
        background: isActive ? "var(--accent-sub)" : undefined,
        transition: "all 0.15s",
      }}
      onClick={onClick}
    >
      {/* Checkbox */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onToggleCheck();
        }}
        aria-label={isChecked ? "Deseleccionar" : "Seleccionar"}
        style={{
          width: 18,
          height: 18,
          flexShrink: 0,
          marginTop: 2,
          borderRadius: 4,
          border: isChecked ? "none" : "1.5px solid var(--border-2)",
          background: isChecked ? "var(--accent)" : "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          cursor: "pointer",
          padding: 0,
          transition: "all 0.15s",
        }}
      >
        {isChecked && <Check style={{ width: 11, height: 11, color: "#fff" }} strokeWidth={3} />}
      </button>

      {/* Content */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text-1)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={query.technology_name || query.search_query}
        >
          {query.technology_name || query.search_query.slice(0, 36)}
        </p>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginTop: 4,
            fontSize: 11,
            color: "var(--text-3)",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: PHASE_COLORS[phase],
              flexShrink: 0,
            }}
          />
          <span
            style={{
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              color: "var(--text-2)",
            }}
          >
            {PHASE_SHORT[phase]}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            marginTop: 6,
            fontSize: 10.5,
            color: "var(--text-3)",
          }}
        >
          <span style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
            <Clock style={{ width: 10, height: 10 }} />
            {date}
          </span>
          <span>· {query.hype_metrics.total_mentions} menciones</span>
          <span>· {(query.hype_metrics.confidence * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  );
}

// ── Query Detail ─────────────────────────────────────────────

function QueryDetail({ query }: { query: HypeCycleQuery }) {
  const m = query.hype_metrics;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      {/* Header */}
      <div className="card" style={{ padding: 22 }}>
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          <div style={{ minWidth: 0 }}>
            <h2
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: "var(--text-1)",
                letterSpacing: "-0.01em",
              }}
            >
              {query.technology_name || "Sin nombre"}
            </h2>
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                marginTop: 8,
                fontSize: 13,
                color: "var(--text-2)",
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: PHASE_COLORS[m.phase],
                }}
              />
              {PHASE_SHORT[m.phase]}
            </div>
            {query.technology_description && (
              <p
                style={{
                  fontSize: 12.5,
                  color: "var(--text-3)",
                  marginTop: 12,
                  maxWidth: 560,
                  lineHeight: 1.5,
                }}
              >
                {query.technology_description}
              </p>
            )}
          </div>
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--accent)",
              background: "var(--accent-sub)",
              border: "1px solid var(--accent-border)",
              borderRadius: 999,
              padding: "4px 10px",
              whiteSpace: "nowrap",
            }}
          >
            {(m.confidence * 100).toFixed(0)}% confianza
          </span>
        </div>
        <p
          style={{
            fontSize: 11,
            color: "var(--text-3)",
            marginTop: 14,
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
            background: "rgba(255,255,255,0.02)",
            padding: "6px 10px",
            borderRadius: 6,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={query.search_query}
        >
          {query.search_query}
        </p>
      </div>

      {/* Metrics */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
          gap: 10,
        }}
      >
        <MetricCard
          label="Menciones"
          value={m.total_mentions.toLocaleString()}
          color="blue"
        />
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
      <div className="card" style={{ padding: 22 }}>
        <p
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--text-3)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: 12,
          }}
        >
          Posición en el Hype Cycle
        </p>
        <HypeCycleChart queries={[query]} highlightId={query.query_id} />
      </div>

      {/* Timeline */}
      {query.yearly_stats?.length > 0 && (
        <div className="card" style={{ padding: 22 }}>
          <p
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--text-3)",
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              marginBottom: 12,
            }}
          >
            Menciones por año
          </p>
          <MentionsTimelineChart
            data={query.yearly_stats}
            technologyName={query.technology_name}
          />
        </div>
      )}
    </div>
  );
}

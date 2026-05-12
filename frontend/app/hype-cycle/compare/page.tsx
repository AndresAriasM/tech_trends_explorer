"use client";

import { useMemo, Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { ArrowLeft, TrendingUp, BarChart3, Users, MessageSquare } from "lucide-react";
import { getHypeCycleHistory, getCategories } from "@/lib/api";
import { HypeCycleViewer } from "@/components/charts/HypeCycleViewer";
import type { HypePhase } from "@/types";

const PHASE_LABELS: Record<HypePhase, string> = {
  "Innovation Trigger": "Lanzamiento",
  "Peak of Inflated Expectations": "Pico de expectativas",
  "Trough of Disillusionment": "Valle de desilusión",
  "Slope of Enlightenment": "Rampa de consolidación",
  "Plateau of Productivity": "Meseta de productividad",
  Unknown: "Desconocida",
};

const PHASE_COLORS: Record<HypePhase, string> = {
  "Innovation Trigger": "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment": "#ef4444",
  "Slope of Enlightenment": "#34d399",
  "Plateau of Productivity": "#818cf8",
  Unknown: "#9ca3af",
};

function ComparePageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const categoryId = searchParams.get("category");
  const queryIdsParam = searchParams.get("ids");
  const queryIds = queryIdsParam ? queryIdsParam.split(",").filter(Boolean) : null;

  const { data: categories = [] } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  const { data: allQueries = [], isLoading } = useQuery({
    queryKey: ["hype-cycle-history", categoryId ?? "all-for-compare"],
    queryFn: () => getHypeCycleHistory(categoryId ?? undefined),
  });

  const queries = useMemo(() => {
    if (queryIds && queryIds.length > 0) {
      return allQueries.filter((q) => queryIds.includes(q.query_id));
    }
    return allQueries;
  }, [allQueries, queryIds]);

  const category = categories.find((c) => c.category_id === categoryId);
  const totalMentions = queries.reduce((acc, q) => acc + q.hype_metrics.total_mentions, 0);
  const avgConfidence = queries.length > 0
    ? queries.reduce((acc, q) => acc + q.hype_metrics.confidence, 0) / queries.length
    : 0;

  const phaseDistribution = useMemo(() => {
    const counts: Partial<Record<HypePhase, number>> = {};
    queries.forEach((q) => {
      const p = q.hype_metrics.phase;
      counts[p] = (counts[p] ?? 0) + 1;
    });
    return counts;
  }, [queries]);

  return (
    <div style={{ padding: "32px 48px", maxWidth: 1400, margin: "0 auto" }}>
      {/* Back link */}
      <Link
        href="/categories"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          fontSize: 13,
          color: "var(--text-3)",
          textDecoration: "none",
          marginBottom: 16,
          transition: "color 0.15s",
        }}
        onMouseEnter={(e) => (e.currentTarget.style.color = "var(--text-1)")}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-3)")}
      >
        <ArrowLeft style={{ width: 14, height: 14 }} />
        Volver a Categorías
      </Link>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1
          className="heading-glow"
          style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.02em", marginBottom: 6 }}
        >
          {category?.name ?? "Hype Cycle comparado"}
        </h1>
        {category?.description ? (
          <p style={{ fontSize: 14, color: "var(--text-2)", maxWidth: 720 }}>
            {category.description}
          </p>
        ) : !category && queryIds ? (
          <p style={{ fontSize: 14, color: "var(--text-2)" }}>
            Comparación de {queryIds.length} análisis seleccionados
          </p>
        ) : null}
      </div>

      {/* Stat cards */}
      {queries.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: 12,
            marginBottom: 24,
          }}
        >
          <StatCard icon={Users} label="Tecnologías" value={queries.length.toString()} />
          <StatCard
            icon={MessageSquare}
            label="Menciones totales"
            value={totalMentions.toLocaleString()}
          />
          <StatCard
            icon={BarChart3}
            label="Confianza promedio"
            value={`${(avgConfidence * 100).toFixed(0)}%`}
          />
          <StatCard
            icon={TrendingUp}
            label="Fases distintas"
            value={Object.keys(phaseDistribution).length.toString()}
          />
        </div>
      )}

      {/* Phase distribution badges */}
      {queries.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 24 }}>
          {Object.entries(phaseDistribution).map(([phase, count]) => (
            <span
              key={phase}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                padding: "5px 12px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border)",
                color: "var(--text-2)",
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: PHASE_COLORS[phase as HypePhase],
                }}
              />
              <span style={{ color: "var(--text-1)", fontWeight: 600 }}>{count}</span>
              <span>{PHASE_LABELS[phase as HypePhase]}</span>
            </span>
          ))}
        </div>
      )}

      {/* Hype Cycle chart */}
      <div className="card" style={{ padding: 24, marginBottom: 24 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 16,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <TrendingUp style={{ width: 16, height: 16, color: "var(--accent)" }} />
            <p style={{ fontSize: 15, fontWeight: 700, color: "var(--text-1)" }}>
              Posición consolidada en el Hype Cycle
            </p>
          </div>
        </div>
        {isLoading && (
          <p style={{ color: "var(--text-3)", textAlign: "center", padding: 64 }}>
            Cargando análisis…
          </p>
        )}
        {!isLoading && queries.length === 0 && (
          <div style={{ textAlign: "center", padding: 64 }}>
            <p style={{ color: "var(--text-2)", fontSize: 14, marginBottom: 8 }}>
              No hay análisis para mostrar
            </p>
            <p style={{ color: "var(--text-3)", fontSize: 12 }}>
              Ejecuta análisis asignados a esta categoría para verlos aquí.
            </p>
          </div>
        )}
        {!isLoading && queries.length > 0 && (
          <HypeCycleViewer queries={queries} title={category?.name ?? undefined} />
        )}
      </div>

      {/* Summary table */}
      {queries.length > 0 && (
        <div className="card" style={{ padding: 24 }}>
          <p style={{ fontSize: 14, fontWeight: 700, color: "var(--text-1)", marginBottom: 16 }}>
            Tabla resumen
          </p>
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                fontSize: 13,
                color: "var(--text-2)",
                borderCollapse: "collapse",
              }}
            >
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  <Th>Tecnología</Th>
                  <Th>Fase</Th>
                  <Th align="right">Confianza</Th>
                  <Th align="right">N</Th>
                  <Th align="right">Pico</Th>
                  <Th align="right">Sentimiento</Th>
                </tr>
              </thead>
              <tbody>
                {queries.map((q) => (
                  <tr
                    key={q.query_id}
                    style={{
                      borderBottom: "1px solid var(--border)",
                      cursor: "pointer",
                      transition: "background 0.15s",
                    }}
                    onClick={() => router.push(`/history?id=${q.query_id}`)}
                    onMouseEnter={(e) =>
                      (e.currentTarget.style.background = "rgba(255,255,255,0.02)")
                    }
                    onMouseLeave={(e) =>
                      (e.currentTarget.style.background = "transparent")
                    }
                  >
                    <Td bold>
                      {q.technology_name || q.search_query.slice(0, 40)}
                    </Td>
                    <Td>
                      <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: "50%",
                            background: PHASE_COLORS[q.hype_metrics.phase],
                          }}
                        />
                        {PHASE_LABELS[q.hype_metrics.phase]}
                      </span>
                    </Td>
                    <Td align="right">
                      {(q.hype_metrics.confidence * 100).toFixed(0)}%
                    </Td>
                    <Td align="right">{q.hype_metrics.total_mentions}</Td>
                    <Td align="right">{q.hype_metrics.peak_year ?? "—"}</Td>
                    <Td align="right">
                      <span
                        style={{
                          color:
                            q.hype_metrics.sentiment_avg > 0.1
                              ? "#22c55e"
                              : q.hype_metrics.sentiment_avg < -0.1
                              ? "#ef4444"
                              : "#f59e0b",
                        }}
                      >
                        {q.hype_metrics.sentiment_avg.toFixed(2)}
                      </span>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ComparePage() {
  return (
    <Suspense fallback={<div style={{ padding: 48, color: "var(--text-3)" }}>Cargando…</div>}>
      <ComparePageInner />
    </Suspense>
  );
}

// ── Helpers ──────────────────────────────────────────────────

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ style?: React.CSSProperties }>;
  label: string;
  value: string;
}) {
  return (
    <div className="card" style={{ padding: "16px 20px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <Icon style={{ width: 13, height: 13, color: "var(--text-3)" }} />
        <span
          style={{
            fontSize: 11,
            color: "var(--text-3)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            fontWeight: 600,
          }}
        >
          {label}
        </span>
      </div>
      <p
        className="stat-num"
        style={{ fontSize: 24, fontWeight: 700, color: "var(--text-1)", letterSpacing: "-0.02em" }}
      >
        {value}
      </p>
    </div>
  );
}

function Th({ children, align = "left" }: { children: React.ReactNode; align?: "left" | "right" }) {
  return (
    <th
      style={{
        textAlign: align,
        padding: "10px 12px",
        color: "var(--text-3)",
        fontWeight: 600,
        fontSize: 11,
        textTransform: "uppercase",
        letterSpacing: "0.05em",
      }}
    >
      {children}
    </th>
  );
}

function Td({
  children,
  align = "left",
  bold = false,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  bold?: boolean;
}) {
  return (
    <td
      style={{
        textAlign: align,
        padding: "10px 12px",
        color: bold ? "var(--text-1)" : "var(--text-2)",
        fontWeight: bold ? 500 : 400,
      }}
    >
      {children}
    </td>
  );
}

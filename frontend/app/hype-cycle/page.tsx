"use client";

import { useEffect, useRef } from "react";
import {
  Zap, RefreshCw, Save, Sparkles, TrendingUp,
  BarChart3, Activity, ChevronRight,
} from "lucide-react";
import { QueryBuilder } from "@/components/query-builder/QueryBuilder";
import { HypeCycleChart } from "@/components/charts/HypeCycleChart";
import { MentionsTimelineChart } from "@/components/charts/MentionsTimelineChart";
import { AnalysisProgressBar } from "@/components/ui/AnalysisProgressBar";
import { useAnalysisStore } from "@/store/analysisStore";
import { startHypeCycleAnalysis, streamTaskProgress } from "@/lib/api";
import type { TaskProgress, HypePhase } from "@/types";

/* ── Phase colours ─────────────────────────────────────── */
const PHASE_COLOR: Record<HypePhase, string> = {
  "Innovation Trigger":            "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment":     "#ef4444",
  "Slope of Enlightenment":        "#34d399",
  "Plateau of Productivity":       "#a78bfa",
  "Unknown":                       "#6b7280",
};

/* ── Decorative empty-state Hype Cycle SVG ────────────── */
function PlaceholderHypeCycleSVG() {
  const zones = [
    { x: 0,   w: 118, color: "#60a5fa", label: "Innovation Trigger" },
    { x: 118, w: 110, color: "#f59e0b", label: "Peak of Inflated Expectations" },
    { x: 228, w: 78,  color: "#ef4444", label: "Trough of Disillusionment" },
    { x: 306, w: 97,  color: "#34d399", label: "Slope of Enlightenment" },
    { x: 403, w: 77,  color: "#a78bfa", label: "Plateau of Productivity" },
  ];

  // SVG path for the Hype Cycle bell curve, viewBox 480x240
  const curvePath =
    "M 14,210 " +
    "C 40,210 55,185 80,138 " +
    "C 100,100 115,38 155,22 " +
    "C 188,8 205,60 218,105 " +
    "C 232,150 240,188 268,200 " +
    "C 300,212 315,190 340,160 " +
    "C 370,124 395,100 466,88";

  return (
    <svg
      viewBox="0 0 480 240"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      style={{ width: "100%", maxWidth: 540, height: "auto" }}
    >
      <defs>
        <linearGradient id="ph-stroke" x1="0" y1="0" x2="100%" y2="0">
          <stop offset="0%"   stopColor="#60a5fa" />
          <stop offset="30%"  stopColor="#f59e0b" />
          <stop offset="52%"  stopColor="#ef4444" />
          <stop offset="72%"  stopColor="#34d399" />
          <stop offset="100%" stopColor="#a78bfa" />
        </linearGradient>
        <linearGradient id="ph-fill" x1="0" y1="0" x2="0" y2="100%">
          <stop offset="0%"   stopColor="#4f8ef7" stopOpacity="0.06" />
          <stop offset="100%" stopColor="#4f8ef7" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Zone backgrounds */}
      {zones.map((z) => (
        <rect
          key={z.label}
          x={z.x}
          y={0}
          width={z.w}
          height={220}
          fill={z.color}
          opacity={0.04}
        />
      ))}

      {/* Zone separators */}
      {zones.slice(1).map((z) => (
        <line
          key={z.x}
          x1={z.x}
          y1={0}
          x2={z.x}
          y2={220}
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
      ))}

      {/* Grid horizontals */}
      {[60, 120, 180].map((y) => (
        <line
          key={y}
          x1={14}
          y1={y}
          x2={476}
          y2={y}
          stroke="rgba(255,255,255,0.04)"
          strokeWidth={1}
        />
      ))}

      {/* Axes */}
      <line x1={14} y1={220} x2={476} y2={220} stroke="rgba(255,255,255,0.14)" strokeWidth={1.5} />
      <line x1={14} y1={10}  x2={14}  y2={220} stroke="rgba(255,255,255,0.14)" strokeWidth={1.5} />

      {/* Axis arrows */}
      <polygon points="474,217 480,220 474,223" fill="rgba(255,255,255,0.25)" />
      <polygon points="11,13 14,7 17,13" fill="rgba(255,255,255,0.25)" />

      {/* Axis labels */}
      <text x="244" y="238" textAnchor="middle" fill="rgba(255,255,255,0.22)" fontSize="10" fontFamily="Inter,sans-serif" letterSpacing="1">
        TIEMPO
      </text>
      <text x="6" y="120" textAnchor="middle" fill="rgba(255,255,255,0.22)" fontSize="10" fontFamily="Inter,sans-serif" letterSpacing="1"
        transform="rotate(-90, 6, 120)">
        EXPECTATIVAS
      </text>

      {/* Area fill under curve */}
      <path
        d={curvePath + " L 466,220 L 14,220 Z"}
        fill="url(#ph-fill)"
      />

      {/* Main curve */}
      <path
        d={curvePath}
        stroke="url(#ph-stroke)"
        strokeWidth={2.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Phase labels */}
      {zones.map((z) => (
        <text
          key={z.label + "-label"}
          x={z.x + z.w / 2}
          y={232}
          textAnchor="middle"
          fill={z.color}
          fontSize="7.5"
          fontFamily="Inter,sans-serif"
          fontWeight="600"
          opacity="0.6"
        >
          {z.label.split(" ").slice(-1)[0]}
        </text>
      ))}

      {/* Peak annotation */}
      <line x1={155} y1={22} x2={155} y2={40} stroke="#f59e0b" strokeWidth={1} strokeOpacity={0.4} strokeDasharray="3 2" />
      <circle cx={155} cy={22} r={5} fill="#f59e0b" opacity={0.85} />
      <circle cx={155} cy={22} r={2.5} fill="#fff" opacity={0.9} />

      {/* Trough annotation */}
      <circle cx={268} cy={200} r={4.5} fill="#ef4444" opacity={0.8} />
      <circle cx={268} cy={200} r={2} fill="#fff" opacity={0.9} />

      {/* Plateau annotation */}
      <circle cx={440} cy={90} r={4.5} fill="#a78bfa" opacity={0.8} />
      <circle cx={440} cy={90} r={2} fill="#fff" opacity={0.9} />
    </svg>
  );
}

/* ── Confidence gauge ─────────────────────────────────── */
function ConfidenceGauge({
  value,
  color,
}: {
  value: number;
  color: string;
}) {
  const pct = Math.round(value * 100);
  const circumference = 2 * Math.PI * 38;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div style={{ position: "relative", width: 96, height: 96 }}>
      <svg width={96} height={96} viewBox="0 0 96 96">
        {/* Track */}
        <circle
          cx={48}
          cy={48}
          r={38}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth={7}
        />
        {/* Progress arc */}
        <circle
          cx={48}
          cy={48}
          r={38}
          fill="none"
          stroke={color}
          strokeWidth={7}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 48 48)"
          style={{ transition: "stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)" }}
        />
      </svg>
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span
          style={{
            fontSize: 22,
            fontWeight: 800,
            letterSpacing: "-0.03em",
            color,
            lineHeight: 1,
          }}
        >
          {pct}
        </span>
        <span style={{ fontSize: 10, color: "var(--text-3)", fontWeight: 600 }}>%</span>
      </div>
    </div>
  );
}

/* ── Main page ────────────────────────────────────────── */
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
      setError(
        e instanceof Error ? e.message : "Error al conectar con el backend"
      );
      setRunning(false);
    }
  }

  const m = result?.hype_metrics;
  const phaseColor = m ? (PHASE_COLOR[m.phase] ?? "#6b7280") : "var(--accent)";
  const hasResults = !!(result && m);

  return (
    <div
      style={{
        display: "flex",
        height: "calc(100vh - var(--header-h))",
        overflow: "hidden",
      }}
    >

      {/* ════════════════════════════════════════
          LEFT PANEL — Configuration (380px)
      ════════════════════════════════════════ */}
      <div
        style={{
          width: 380,
          flexShrink: 0,
          display: "flex",
          flexDirection: "column",
          background: "var(--surface)",
          borderRight: "1px solid var(--border)",
          overflow: "hidden",
        }}
      >
        {/* Panel header */}
        <div
          style={{
            padding: "20px 24px",
            borderBottom: "1px solid var(--border)",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 6,
            }}
          >
            <Sparkles
              style={{ width: 15, height: 15, color: "var(--accent)" }}
            />
            <p
              style={{
                fontSize: 11,
                fontWeight: 700,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--accent)",
              }}
            >
              Configurar análisis
            </p>
          </div>
          <p
            style={{
              fontSize: 13,
              color: "var(--text-3)",
              lineHeight: 1.5,
            }}
          >
            Posicionamiento en el Hype Cycle de Gartner
          </p>
        </div>

        {/* Scrollable form */}
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "24px",
          }}
        >
          <QueryBuilder />
        </div>

        {/* Sticky bottom bar */}
        <div
          style={{
            padding: "16px 24px 20px",
            borderTop: "1px solid var(--border)",
            flexShrink: 0,
            background: "var(--surface)",
          }}
        >
          <button
            onClick={handleAnalyze}
            disabled={isRunning || !searchTerms.some((t) => t.value.trim())}
            className="btn btn-primary"
            style={{ width: "100%", justifyContent: "center", fontSize: 15, padding: "12px 20px" }}
          >
            {isRunning ? (
              <>
                <RefreshCw style={{ width: 16, height: 16 }} className="animate-spin" />
                Analizando...
              </>
            ) : (
              <>
                <Zap style={{ width: 16, height: 16 }} />
                Iniciar análisis
              </>
            )}
          </button>

          {progress && (
            <div style={{ marginTop: 12 }}>
              <AnalysisProgressBar progress={progress} />
            </div>
          )}

          {error && (
            <div
              style={{
                marginTop: 12,
                padding: "10px 14px",
                borderRadius: 10,
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.20)",
                fontSize: 13,
                color: "#fca5a5",
                lineHeight: 1.5,
              }}
            >
              {error}
            </div>
          )}
        </div>
      </div>

      {/* ════════════════════════════════════════
          RIGHT PANEL — Results / Empty state
      ════════════════════════════════════════ */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          background: "var(--bg)",
          position: "relative",
        }}
      >
        {hasResults ? (
          /* ── Results view ── */
          <div
            className="fade-in"
            style={{ padding: "32px 40px 64px" }}
          >

            {/* Phase hero banner */}
            <div
              style={{
                borderRadius: 20,
                padding: "32px 36px",
                background: `linear-gradient(135deg, ${phaseColor}12 0%, ${phaseColor}05 100%)`,
                border: `1px solid ${phaseColor}28`,
                marginBottom: 24,
                position: "relative",
                overflow: "hidden",
              }}
            >
              {/* Glow orb */}
              <div
                style={{
                  position: "absolute",
                  top: -40,
                  right: -40,
                  width: 200,
                  height: 200,
                  borderRadius: "50%",
                  background: `radial-gradient(circle, ${phaseColor}18 0%, transparent 70%)`,
                  pointerEvents: "none",
                }}
              />

              <div
                style={{
                  position: "relative",
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 32,
                  flexWrap: "wrap",
                }}
              >
                {/* Phase info */}
                <div style={{ flex: 1, minWidth: 240 }}>
                  {/* Breadcrumb */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      marginBottom: 12,
                    }}
                  >
                    <TrendingUp style={{ width: 13, height: 13, color: phaseColor }} />
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: phaseColor,
                      }}
                    >
                      Fase detectada
                    </span>
                  </div>

                  <h2
                    style={{
                      fontSize: 30,
                      fontWeight: 800,
                      letterSpacing: "-0.03em",
                      color: "var(--text-1)",
                      lineHeight: 1.15,
                      marginBottom: 8,
                    }}
                  >
                    {m!.phase}
                  </h2>

                  {result!.technology_name && (
                    <p
                      style={{
                        fontSize: 15,
                        color: "var(--text-2)",
                        marginBottom: 4,
                      }}
                    >
                      {result!.technology_name}
                    </p>
                  )}
                </div>

                {/* Confidence gauge */}
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: 8,
                  }}
                >
                  <ConfidenceGauge value={m!.confidence} color={phaseColor} />
                  <p
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      color: "var(--text-3)",
                    }}
                  >
                    Confianza
                  </p>
                </div>
              </div>

              {/* Metric cards row */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(4, 1fr)",
                  gap: 12,
                  marginTop: 24,
                  position: "relative",
                }}
              >
                {[
                  {
                    label: "Menciones totales",
                    value: m!.total_mentions.toLocaleString(),
                    color: phaseColor,
                    icon: BarChart3,
                  },
                  {
                    label: "Pico de menciones",
                    value: `${m!.peak_mentions.toLocaleString()}${m!.peak_year ? ` (${m!.peak_year})` : ""}`,
                    color: phaseColor,
                    icon: TrendingUp,
                  },
                  {
                    label: "Sentimiento",
                    value: m!.sentiment_avg.toFixed(2),
                    color: m!.sentiment_avg > 0 ? "#22c55e" : "#ef4444",
                    icon: Activity,
                  },
                  {
                    label: "Al Plateau",
                    value: m!.time_to_plateau,
                    color: "#a78bfa",
                    icon: ChevronRight,
                  },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    style={{
                      background: "rgba(255,255,255,0.04)",
                      border: "1px solid var(--border)",
                      borderRadius: 12,
                      padding: "14px 16px",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        marginBottom: 6,
                      }}
                    >
                      <stat.icon
                        style={{
                          width: 12,
                          height: 12,
                          color: stat.color,
                          opacity: 0.8,
                        }}
                      />
                      <p
                        style={{
                          fontSize: 10,
                          fontWeight: 700,
                          letterSpacing: "0.06em",
                          textTransform: "uppercase",
                          color: "var(--text-3)",
                        }}
                      >
                        {stat.label}
                      </p>
                    </div>
                    <p
                      style={{
                        fontSize: 16,
                        fontWeight: 700,
                        letterSpacing: "-0.02em",
                        color: stat.color,
                      }}
                    >
                      {stat.value}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Hype Cycle chart */}
            <div
              className="card"
              style={{ padding: "24px 28px", marginBottom: 20 }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  marginBottom: 20,
                }}
              >
                <TrendingUp
                  style={{ width: 16, height: 16, color: "var(--accent)" }}
                />
                <p
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    letterSpacing: "-0.01em",
                    color: "var(--text-1)",
                  }}
                >
                  Posición en el ciclo
                </p>
              </div>
              <HypeCycleChart
                queries={[result!]}
                highlightId={result!.query_id}
              />
            </div>

            {/* Timeline chart */}
            {result!.yearly_stats?.length > 0 && (
              <div
                className="card"
                style={{ padding: "24px 28px", marginBottom: 20 }}
              >
                <p
                  style={{
                    fontSize: 15,
                    fontWeight: 700,
                    letterSpacing: "-0.01em",
                    color: "var(--text-1)",
                    marginBottom: 20,
                  }}
                >
                  Menciones y sentimiento por año
                </p>
                <MentionsTimelineChart
                  data={result!.yearly_stats}
                  technologyName={result!.technology_name}
                />
              </div>
            )}

            {/* Meta footer */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "12px 16px",
                borderRadius: 10,
                background: "var(--surface)",
                border: "1px solid var(--border)",
                fontSize: 12,
                color: "var(--text-3)",
              }}
            >
              <span>
                ID:{" "}
                <code style={{ color: "var(--text-2)", fontFamily: "monospace" }}>
                  {result!.query_id}
                </code>
              </span>
              {result!.processing_time && (
                <span>Procesado en {result!.processing_time.toFixed(1)}s</span>
              )}
              <button
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-3)",
                  fontSize: 12,
                  fontFamily: "inherit",
                  transition: "color 0.15s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-1)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-3)"; }}
              >
                <Save style={{ width: 13, height: 13 }} />
                Exportar
              </button>
            </div>

          </div>
        ) : (
          /* ── Empty / loading state ── */
          <div
            style={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "48px 40px",
              gap: 0,
            }}
          >
            {/* Decorative SVG */}
            <div
              style={{
                width: "100%",
                maxWidth: 520,
                marginBottom: 32,
                padding: "28px 32px",
                background: "var(--surface)",
                border: "1px solid var(--border)",
                borderRadius: 20,
                boxShadow: "0 24px 64px rgba(0,0,0,0.2)",
              }}
            >
              <div style={{ marginBottom: 16 }}>
                <p
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "var(--text-3)",
                    marginBottom: 2,
                  }}
                >
                  Hype Cycle de Gartner
                </p>
                <p style={{ fontSize: 12, color: "var(--text-4)" }}>
                  Las 5 fases del ciclo tecnológico
                </p>
              </div>
              <PlaceholderHypeCycleSVG />
            </div>

            {/* Message */}
            <div
              style={{
                textAlign: "center",
                maxWidth: 400,
              }}
            >
              <h3
                style={{
                  fontSize: 20,
                  fontWeight: 700,
                  letterSpacing: "-0.02em",
                  color: "var(--text-1)",
                  marginBottom: 8,
                }}
              >
                {isRunning ? "Analizando tecnología..." : "Listo para analizar"}
              </h3>
              <p
                style={{
                  fontSize: 14,
                  color: "var(--text-2)",
                  lineHeight: 1.65,
                }}
              >
                {isRunning
                  ? "Buscando en fuentes web y calculando la posición en el Hype Cycle. Esto puede tomar un momento."
                  : "Configura los términos de búsqueda en el panel izquierdo y haz clic en «Iniciar análisis» para ver los resultados."}
              </p>
            </div>

            {/* Running progress bar */}
            {isRunning && (
              <div
                style={{
                  width: "100%",
                  maxWidth: 320,
                  marginTop: 24,
                  height: 4,
                  borderRadius: 99,
                  background: "var(--surface-3)",
                  overflow: "hidden",
                }}
              >
                <div
                  className="shimmer"
                  style={{
                    width: `${progress?.progress ?? 30}%`,
                    height: "100%",
                    borderRadius: 99,
                    background: "var(--accent)",
                    transition: "width 0.5s ease",
                  }}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { Activity, Play, Square, RotateCcw, TrendingUp, Calendar, BookOpen, Target, Sigma, ChevronDown, ChevronUp, Download, Copy, Check, Database, FileText, Upload, X, Filter, AlertTriangle, CheckCircle2, GitCompareArrows, Scale, Landmark } from "lucide-react";
import { streamPapersAnalysis, analyzeSCurveLocal, API_BASE_URL } from "@/lib/api";
import { SCurveChart } from "@/components/charts/SCurveChart";
import type {
  SCurveResult,
  SCurveProgress,
  SCurveModelFit,
  SCurveComparison,
  SCurvePatentsRequest,
  EPOStatus,
} from "@/types";

// ── Constants ─────────────────────────────────────────────────

const EXAMPLE_QUERIES = [
  {
    label: "Inteligencia Artificial en salud",
    query: 'TITLE-ABS-KEY("artificial intelligence") AND TITLE-ABS-KEY("healthcare" OR "medicine" OR "clinical")',
  },
  {
    label: "Blockchain en finanzas",
    query: 'TITLE-ABS-KEY("blockchain") AND TITLE-ABS-KEY("finance" OR "banking" OR "fintech")',
  },
  {
    label: "Energía solar fotovoltaica",
    query: 'TITLE-ABS-KEY("solar energy" OR "photovoltaic") AND TITLE-ABS-KEY("efficiency" OR "cells")',
  },
  {
    label: "Machine Learning en educación",
    query: 'TITLE-ABS-KEY("machine learning") AND TITLE-ABS-KEY("education" OR "learning" OR "teaching")',
  },
];

const PHASE_ICON: Record<string, string> = {
  Emergente: "🌱",
  Crecimiento: "📈",
  "Aceleración": "🚀",
  "Maduración": "🌿",
  "Saturación": "🏔️",
  // Fases del eje temporal, cuando el techo L no es estimable.
  "Pre-inflexión": "🌤️",
  "Post-inflexión": "🌇",
  "Sin datos": "❓",
};

const EPO_EXAMPLES = [
  { label: "Biodiésel", query: 'txt="biodiesel"' },
  { label: "Batería de estado sólido", query: 'txt="solid state battery"' },
  { label: "Hidrógeno verde", query: 'txt="green hydrogen"' },
  { label: "Captura de carbono", query: 'txt="carbon capture"' },
];

// ── SSE de patentes (EPO OPS) ─────────────────────────────────
//
// Mismo protocolo que `streamSCurveAnalysis` pero contra /analyze-patents y
// sin api_key en el cuerpo: las credenciales de OPS viven en el servidor.

function streamPatentsAnalysis(
  payload: SCurvePatentsRequest,
  onProgress: (evt: SCurveProgress) => void,
  onComplete: (result: SCurveResult | undefined) => void,
  onError: (message: string) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/s-curve/analyze-patents`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      if (!res.body) throw new Error("El servidor no devolvió un stream");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const data = frame
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("");
          if (!data) continue;

          let evt: SCurveProgress;
          try {
            evt = JSON.parse(data) as SCurveProgress;
          } catch {
            continue;
          }

          if (evt.status === "error") {
            controller.abort();
            onError(evt.message ?? "Error consultando EPO OPS");
            return;
          }
          onProgress(evt);
          if (evt.status === "completed") {
            controller.abort();
            onComplete(evt.result);
            return;
          }
        }
      }
      onComplete(undefined);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      onError(err instanceof Error ? err.message : "Error desconocido");
    }
  })();

  return () => controller.abort();
}

// ── Helpers ───────────────────────────────────────────────────

function fmt(n: number | null | undefined, digits = 0): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/**
 * Color del estadístico Durbin-Watson: verde cerca de 2 (residuos sin
 * estructura), ámbar por debajo de 1.5, rojo por debajo de 1 — ahí el R²
 * alto ya no significa buen ajuste, sólo serie creciente.
 */
function dwColor(dw: number | null): string {
  if (dw === null) return "var(--text-4)";
  if (dw < 1.0 || dw > 3.0) return "#ef4444";
  if (dw < 1.5 || dw > 2.5) return "#f59e0b";
  return "#10b981";
}

function MetricCard({
  icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  sub?: string;
  color?: string;
}) {
  return (
    <div
      className="card"
      style={{ padding: "18px 20px", display: "flex", alignItems: "flex-start", gap: 14 }}
    >
      <div
        style={{
          width: 38,
          height: 38,
          borderRadius: 10,
          background: color ? `${color}1a` : "rgba(79,142,247,0.12)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          color: color ?? "var(--accent)",
        }}
      >
        {icon}
      </div>
      <div style={{ minWidth: 0 }}>
        <p style={{ fontSize: 11, color: "var(--text-4)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 2 }}>
          {label}
        </p>
        <p style={{ fontSize: 20, fontWeight: 700, color: "var(--text-1)", lineHeight: 1 }}>
          {value}
        </p>
        {sub && (
          <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 3 }}>{sub}</p>
        )}
      </div>
    </div>
  );
}

function ProgressBar({ value, message }: { value: number; message: string }) {
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, color: "var(--text-3)" }}>{message}</span>
        <span style={{ fontSize: 12, color: "var(--text-3)", fontVariantNumeric: "tabular-nums" }}>
          {value}%
        </span>
      </div>
      <div
        style={{
          height: 6,
          borderRadius: 3,
          background: "rgba(255,255,255,0.06)",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            borderRadius: 3,
            width: `${value}%`,
            background: "linear-gradient(90deg, #4f8ef7, #8b5cf6)",
            transition: "width 0.4s ease",
          }}
        />
      </div>
    </div>
  );
}

// ── Query block ───────────────────────────────────────────────

function QueryBlock({
  query,
  yearFrom,
  yearTo,
}: {
  query: string;
  yearFrom: number;
  yearTo: number;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(query).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      style={{
        borderRadius: 12,
        border: "1px solid rgba(99,102,241,0.25)",
        background: "rgba(99,102,241,0.06)",
        overflow: "hidden",
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px",
          borderBottom: "1px solid rgba(99,102,241,0.15)",
          background: "rgba(99,102,241,0.08)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <Database style={{ width: 13, height: 13, color: "#818cf8" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#818cf8", letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Scopus Query
          </span>
          <span
            style={{
              fontSize: 10,
              padding: "1px 7px",
              borderRadius: 20,
              background: "rgba(99,102,241,0.15)",
              color: "#a5b4fc",
              fontWeight: 600,
              letterSpacing: "0.03em",
            }}
          >
            {yearFrom} – {yearTo}
          </span>
        </div>
        <button
          onClick={handleCopy}
          title="Copiar ecuación"
          style={{
            display: "flex",
            alignItems: "center",
            gap: 5,
            padding: "4px 10px",
            borderRadius: 6,
            border: "1px solid rgba(99,102,241,0.3)",
            background: copied ? "rgba(16,185,129,0.12)" : "transparent",
            color: copied ? "#6ee7b7" : "#818cf8",
            cursor: "pointer",
            fontSize: 11,
            fontWeight: 600,
            transition: "all 0.2s",
          }}
        >
          {copied ? (
            <><Check style={{ width: 11, height: 11 }} /> Copiado</>
          ) : (
            <><Copy style={{ width: 11, height: 11 }} /> Copiar</>
          )}
        </button>
      </div>

      {/* Query text */}
      <div style={{ padding: "12px 16px" }}>
        <code
          style={{
            display: "block",
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
            fontSize: 12,
            color: "#c7d2fe",
            lineHeight: 1.7,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {query}
        </code>
      </div>
    </div>
  );
}

// ── Aviso de autocorrelación ──────────────────────────────────

/**
 * El aviso que el R² no da.
 *
 * Un Durbin-Watson de 0.624 con R² de 0.9998 significa que el modelo va
 * sistemáticamente alto en un tramo y bajo en otro: el error no es ruido,
 * es sesgo de forma. Sin esta banda, la tarjeta muestra un ajuste
 * "excelente" que no lo es.
 */
function FitWarning({ text, tone = "warn" }: { text: string; tone?: "warn" | "info" }) {
  const color = tone === "warn" ? "#f59e0b" : "#6366f1";
  const [head, ...rest] = text.split(":");
  const body = rest.join(":").trim();
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 10,
        padding: "12px 14px",
        borderRadius: 10,
        border: `1px solid ${color}44`,
        background: `${color}0f`,
      }}
    >
      <AlertTriangle style={{ width: 15, height: 15, color, flexShrink: 0, marginTop: 1 }} />
      <p style={{ fontSize: 12.5, lineHeight: 1.55, color: "var(--text-2)" }}>
        <strong style={{ color }}>{head}{body ? ":" : ""}</strong>{" "}
        {body || null}
      </p>
    </div>
  );
}

// ── Tabla comparativa de modelos ──────────────────────────────

const MODEL_COLUMNS: Array<{
  key: string;
  label: string;
  title: string;
  align: "left" | "right";
  render: (m: SCurveModelFit) => React.ReactNode;
  color?: (m: SCurveModelFit) => string;
}> = [
  {
    key: "r2_adj",
    label: "R² aj.",
    title: "R² penalizado por los 3 parámetros. Sobre un acumulado siempre sale alto: no lo leas solo.",
    align: "right",
    render: (m) => (m.r2_adj === null ? "—" : m.r2_adj.toFixed(5)),
  },
  {
    key: "se",
    label: "EE",
    title: "Error estándar de la estimación, en unidades de la serie. Es el error real que el R² esconde.",
    align: "right",
    render: (m) => fmt(m.se, 3),
  },
  {
    key: "dw",
    label: "Durbin-Watson",
    title: "Autocorrelación de los residuos (0–4). 2 = sin estructura; muy por debajo = sesgo sistemático.",
    align: "right",
    render: (m) => (m.durbin_watson === null ? "—" : m.durbin_watson.toFixed(3)),
    color: (m) => dwColor(m.durbin_watson),
  },
  {
    key: "infl",
    label: "Inflexión",
    title: "Año de máxima velocidad de adopción según ese modelo.",
    align: "right",
    render: (m) => (m.inflection_year === null ? "—" : m.inflection_year.toFixed(1)),
  },
  {
    key: "L",
    label: "Saturación L",
    title: "Techo proyectado. Si el modelo lo topó contra el límite del ajuste, no es una estimación.",
    align: "right",
    render: (m) =>
      m.L_at_bound ? (
        <span title="L pegado al límite del ajuste: el techo no es estimable" style={{ color: "var(--text-4)" }}>
          no estimable
        </span>
      ) : (
        fmt(m.saturation_L)
      ),
  },
  {
    key: "sat",
    label: "% sat.",
    title: "Último acumulado observado sobre L.",
    align: "right",
    render: (m) =>
      m.L_at_bound || m.saturation_pct === null ? "—" : `${m.saturation_pct}%`,
  },
];

function ModelComparisonCard({ result }: { result: SCurveResult }) {
  const comp = result.model_comparison;
  if (!comp) return null;

  if (comp.note || comp.models.length === 0) {
    return (
      <div className="card" style={{ padding: 20 }}>
        <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)", marginBottom: 8 }}>
          Ajuste multi-modelo
        </p>
        <p style={{ fontSize: 12.5, color: "var(--text-3)", lineHeight: 1.6 }}>
          {comp.note ?? "No se pudo ajustar ningún modelo a esta serie."}
        </p>
      </div>
    );
  }

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, marginBottom: 4 }}>
        <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>
          Ajuste multi-modelo — {result.source_label ?? "serie"}
        </p>
        <span style={{ fontSize: 11, color: "var(--text-4)" }}>
          {comp.converged_count} de {comp.models.length} convergieron
        </span>
      </div>
      <p style={{ fontSize: 11.5, color: "var(--text-4)", lineHeight: 1.6, marginBottom: 14 }}>
        Los cinco modelos ajustan la misma serie. El R² apenas los separa; lo que
        cambia es el techo que extrapolan, y de ese techo sale la fase.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
          <thead>
            <tr>
              <th
                style={{
                  textAlign: "left",
                  padding: "8px 10px",
                  color: "var(--text-4)",
                  fontWeight: 500,
                  fontSize: 11.5,
                  borderBottom: "1px solid var(--border)",
                  whiteSpace: "nowrap",
                }}
              >
                Modelo
              </th>
              {MODEL_COLUMNS.map((c) => (
                <th
                  key={c.key}
                  title={c.title}
                  style={{
                    textAlign: c.align,
                    padding: "8px 10px",
                    color: "var(--text-4)",
                    fontWeight: 500,
                    fontSize: 11.5,
                    borderBottom: "1px solid var(--border)",
                    whiteSpace: "nowrap",
                    cursor: "help",
                  }}
                >
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {comp.models.map((m) => (
              <tr
                key={m.key}
                style={{
                  background: m.is_best ? "rgba(16,185,129,0.07)" : "transparent",
                  opacity: m.converged ? 1 : 0.45,
                }}
              >
                <td
                  style={{
                    padding: "9px 10px",
                    borderBottom: "1px solid var(--border)",
                    color: m.is_best ? "#10b981" : "var(--text-2)",
                    fontWeight: m.is_best ? 700 : 500,
                    whiteSpace: "nowrap",
                  }}
                >
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 7 }}>
                    {m.is_best && (
                      <CheckCircle2 style={{ width: 13, height: 13, flexShrink: 0 }} />
                    )}
                    {m.label}
                  </span>
                </td>
                {MODEL_COLUMNS.map((c) => (
                  <td
                    key={c.key}
                    style={{
                      textAlign: c.align,
                      padding: "9px 10px",
                      borderBottom: "1px solid var(--border)",
                      color: c.color ? c.color(m) : "var(--text-2)",
                      fontVariantNumeric: "tabular-nums",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {m.converged ? c.render(m) : "no converge"}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 14 }}>
        {comp.inflection_range && comp.inflection_range[0] !== comp.inflection_range[1] && (
          <p style={{ fontSize: 11.5, color: "var(--text-3)", lineHeight: 1.6 }}>
            Inflexión entre modelos:{" "}
            <strong style={{ color: "var(--text-2)" }}>
              {comp.inflection_range[0]}–{comp.inflection_range[1]}
            </strong>
            . Esa horquilla es la incertidumbre real sobre el año de inflexión —
            mucho más honesta que el ± de un solo modelo, que sólo mide ruido
            dentro de una forma funcional ya elegida.
          </p>
        )}
        {comp.warning && <FitWarning text={comp.warning} />}
      </div>
    </div>
  );
}

// ── Comparación papers ↔ patentes ─────────────────────────────

function ComparisonSideCard({ side }: { side: SCurveComparison["papers"] }) {
  const color = side.phase_color ?? "#6366f1";
  return (
    <div className="card" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {side.kind === "papers" ? (
          <BookOpen style={{ width: 15, height: 15, color: "#6366f1" }} />
        ) : (
          <Landmark style={{ width: 15, height: 15, color: "#10b981" }} />
        )}
        <span style={{ fontSize: 13.5, fontWeight: 700, color: "var(--text-1)" }}>
          {side.label}
        </span>
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {side.phase && (
          <span
            style={{
              fontSize: 11.5,
              fontWeight: 600,
              padding: "3px 10px",
              borderRadius: 20,
              background: `${color}1f`,
              color,
            }}
          >
            {side.phase}
            {side.saturation_pct !== null ? ` · ${side.saturation_pct}%` : ""}
          </span>
        )}
        {side.inflection_year !== null && (
          <span
            style={{
              fontSize: 11.5,
              fontWeight: 500,
              padding: "3px 10px",
              borderRadius: 20,
              background: "var(--surface-2)",
              color: "var(--text-2)",
              border: "1px solid var(--border)",
            }}
          >
            Inflexión {side.inflection_year}{" "}
            {side.inflection_passed ? "(superada)" : "(proyectada)"}
          </span>
        )}
      </div>

      <p style={{ fontSize: 11.5, color: "var(--text-3)", lineHeight: 1.65 }}>
        {fmt(side.records)} registros ({side.year_from}–{side.year_to})
        {side.peak_annual !== null && side.peak_annual_year !== null
          ? ` · pico anual ${fmt(side.peak_annual)} en ${side.peak_annual_year}`
          : ""}
        {side.best_model ? ` · mejor modelo: ${side.best_model}` : ""}
        {side.inflection_range && side.inflection_range[0] !== side.inflection_range[1]
          ? ` · inflexión entre modelos: ${side.inflection_range[0]}–${side.inflection_range[1]}`
          : ""}
      </p>

      {side.warning && <FitWarning text={side.warning} />}
    </div>
  );
}

/**
 * Desfase ciencia → tecnología.
 *
 * La lectura clásica es que la ciencia precede a la tecnología. Cuando el
 * dato dice lo contrario no es un fallo de cálculo: es el perfil de una
 * tecnología empujada por la industria, con la literatura llegando después
 * a explicar lo que ya se fabricaba.
 */
function ComparisonPanel({
  comparison,
  onClose,
}: {
  comparison: SCurveComparison;
  onClose: () => void;
}) {
  const lead =
    comparison.direction === "patents_lead"
      ? "#10b981"
      : comparison.direction === "papers_lead"
      ? "#6366f1"
      : "var(--text-3)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <ComparisonSideCard side={comparison.papers} />
        <ComparisonSideCard side={comparison.patents} />
      </div>

      <div
        className="card"
        style={{
          padding: "16px 20px",
          display: "flex",
          alignItems: "flex-start",
          gap: 12,
          border: `1px solid ${lead}33`,
          background: `${lead}0a`,
        }}
      >
        <GitCompareArrows style={{ width: 17, height: 17, color: lead, flexShrink: 0, marginTop: 1 }} />
        <div style={{ flex: 1 }}>
          <p style={{ fontSize: 13.5, color: "var(--text-2)", lineHeight: 1.6 }}>
            <strong style={{ color: "var(--text-1)" }}>
              Desfase ciencia → tecnología:
            </strong>{" "}
            {comparison.interpretation}
          </p>
          {comparison.lag_years !== null && comparison.direction === "patents_lead" && (
            <p style={{ fontSize: 11.5, color: "var(--text-4)", lineHeight: 1.6, marginTop: 6 }}>
              Las patentes van por delante: la tecnología se protegió antes de que
              la literatura la consolidara. Perfil de desarrollo empujado por la
              industria.
            </p>
          )}
          {comparison.lag_years !== null && comparison.direction === "papers_lead" && (
            <p style={{ fontSize: 11.5, color: "var(--text-4)", lineHeight: 1.6, marginTop: 6 }}>
              La ciencia va por delante: la transferencia a patentes todavía está
              en curso. Ese retraso es la ventana de oportunidad tecnológica.
            </p>
          )}
          {comparison.caveats.length > 0 && (
            <ul style={{ margin: "10px 0 0", paddingLeft: 18 }}>
              {comparison.caveats.map((c, i) => (
                <li
                  key={i}
                  style={{ fontSize: 11.5, color: "var(--text-4)", lineHeight: 1.6, marginBottom: 3 }}
                >
                  {c}
                </li>
              ))}
            </ul>
          )}
        </div>
        <button
          onClick={onClose}
          title="Cerrar comparación"
          style={{
            background: "transparent",
            border: "none",
            color: "var(--text-4)",
            cursor: "pointer",
            padding: 2,
            flexShrink: 0,
          }}
        >
          <X style={{ width: 14, height: 14 }} />
        </button>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────

// ── CSV row type ──────────────────────────────────────────────
interface CsvRow { count: number; year: number; docType: string; }

type TabKey = "scopus" | "epo" | "patents";

export default function SCurvePage() {
  // Tab
  const [activeTab, setActiveTab] = useState<TabKey>("scopus");

  // Scopus form state
  const [query, setQuery] = useState("");
  /* Si el servidor tiene SCOPUS_API_KEY no hay que pedirla: el campo pasa a
     ser opcional, para quien quiera gastar su propia cuota. */
  const [scopusOnServer, setScopusOnServer] = useState<boolean | null>(null);

  useEffect(() => {
    let vivo = true;
    fetch(`${API_BASE_URL}/api/s-curve/scopus/status`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => vivo && setScopusOnServer(Boolean(d?.configured)))
      .catch(() => vivo && setScopusOnServer(false));
    return () => {
      vivo = false;
    };
  }, []);
  const [yearFrom, setYearFrom] = useState(1990);
  const [yearTo, setYearTo] = useState(new Date().getFullYear());
  const [maxResults, setMaxResults] = useState<number | "">("");
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Patents (EPO OPS) state — sin api key: las credenciales son del servidor
  const [epoQuery, setEpoQuery] = useState("");
  const [epoYearFrom, setEpoYearFrom] = useState(2000);
  const [epoYearTo, setEpoYearTo] = useState(new Date().getFullYear());
  const [epoStatus, setEpoStatus] = useState<EPOStatus | null>(null);

  // Patents (CSV) state
  const [csvRows, setCsvRows] = useState<CsvRow[]>([]);
  const [csvDocTypes, setCsvDocTypes] = useState<string[]>([]);
  const [selectedTypes, setSelectedTypes] = useState<Set<string>>(new Set());
  const [patentTitle, setPatentTitle] = useState("");
  const [csvFileName, setCsvFileName] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Shared analysis state
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [result, setResult] = useState<SCurveResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Los dos últimos resultados por naturaleza de la fuente. Se guardan para
  // poder comparar papers contra patentes sin repetir ninguna descarga: una
  // corrida de EPO cuesta una petición por año y no conviene gastarla dos veces.
  const [papersResult, setPapersResult] = useState<SCurveResult | null>(null);
  const [patentsResult, setPatentsResult] = useState<SCurveResult | null>(null);
  const [comparison, setComparison] = useState<SCurveComparison | null>(null);
  const [comparing, setComparing] = useState(false);

  // Data table
  const [showTable, setShowTable] = useState(false);

  const cancelRef = useRef<(() => void) | null>(null);

  // Estado de EPO OPS: se consulta una vez al abrir la pestaña para no
  // dejar que el usuario escriba una consulta y descubra al final que las
  // credenciales del servidor no están puestas.
  useEffect(() => {
    if (activeTab !== "epo" || epoStatus) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/s-curve/epo/status`);
        const data = (await res.json()) as EPOStatus;
        if (!cancelled) setEpoStatus(data);
      } catch {
        if (!cancelled) {
          setEpoStatus({
            configured: false,
            authenticated: false,
            message: "No se pudo verificar el estado de EPO OPS.",
          });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeTab, epoStatus]);

  /** Archiva el resultado en el lado que le corresponde para la comparación. */
  const storeResult = useCallback((res: SCurveResult) => {
    setResult(res);
    if (res.source === "scopus") setPapersResult(res);
    else if (res.source === "epo" || res.source === "patents") setPatentsResult(res);
    setComparison(null);
  }, []);

  // ── CSV parser ──────────────────────────────────────────────
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setCsvFileName(file.name);

    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result as string;
      const lines = text.split(/\r?\n/).filter(Boolean);
      if (lines.length < 2) { setError("CSV vacío o sin datos."); return; }

      const rows: CsvRow[] = [];
      for (let i = 1; i < lines.length; i++) {
        // Handle quoted fields
        const cols = lines[i].match(/(".*?"|[^,]+)/g)?.map(c => c.replace(/^"|"$/g, "").trim()) ?? [];
        if (cols.length < 3) continue;
        const count = parseInt(cols[0], 10);
        const year = parseInt(cols[1], 10);
        const docType = cols[2];
        if (!isNaN(count) && !isNaN(year) && docType) {
          rows.push({ count, year, docType });
        }
      }

      const types = Array.from(new Set(rows.map(r => r.docType))).sort();
      const defaultSelected = new Set(
        types.filter(t => /patent/i.test(t) && !/amended|design|utility/i.test(t))
      );
      setCsvRows(rows);
      setCsvDocTypes(types);
      setSelectedTypes(defaultSelected.size > 0 ? defaultSelected : new Set(types));
      if (!patentTitle) setPatentTitle(file.name.replace(/\.[^.]+$/, ""));
      setError(null);
    };
    reader.readAsText(file);
  };

  const toggleType = (t: string) => {
    setSelectedTypes(prev => {
      const next = new Set(prev);
      if (next.has(t)) next.delete(t);
      else next.add(t);
      return next;
    });
  };

  // ── Patent analyze ──────────────────────────────────────────
  const handlePatentAnalyze = useCallback(async () => {
    if (csvRows.length === 0) { setError("Carga un archivo CSV primero."); return; }
    if (selectedTypes.size === 0) { setError("Selecciona al menos un tipo de documento."); return; }

    const counts: Record<number, number> = {};
    for (const row of csvRows) {
      if (selectedTypes.has(row.docType)) {
        counts[row.year] = (counts[row.year] ?? 0) + row.count;
      }
    }

    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(30);
    setProgressMsg("Procesando datos del CSV…");

    try {
      setProgress(60);
      setProgressMsg("Ajustando curva sigmoidal…");
      const res = await analyzeSCurveLocal(counts, patentTitle || "Análisis de patentes");
      // El CSV es una serie de patentes: se marca como tal para que entre en
      // la comparación con papers igual que la de EPO OPS.
      storeResult({ ...res, source: "patents", source_label: res.source_label ?? "patentes (CSV)" });
      setProgress(100);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setRunning(false);
    }
  }, [csvRows, selectedTypes, patentTitle, storeResult]);

  // ── Patentes vía EPO OPS ────────────────────────────────────
  const handleEpoAnalyze = useCallback(() => {
    if (!epoQuery.trim()) {
      setError("Escribe un término o una consulta CQL de EPO OPS.");
      return;
    }

    setError(null);
    setResult(null);
    setComparison(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Conectando con EPO OPS…");

    cancelRef.current = streamPatentsAnalysis(
      { query: epoQuery.trim(), year_from: epoYearFrom, year_to: epoYearTo },
      (evt: SCurveProgress) => {
        setProgress(evt.progress);
        setProgressMsg(evt.message);
      },
      (res) => {
        if (res) storeResult(res);
        setRunning(false);
        setProgress(100);
      },
      (err) => {
        setError(err);
        setRunning(false);
      }
    );
  }, [epoQuery, epoYearFrom, epoYearTo, storeResult]);

  // ── Desfase ciencia → tecnología ────────────────────────────
  const handleCompare = useCallback(async () => {
    if (!papersResult || !patentsResult) return;
    setComparing(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE_URL}/api/s-curve/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ papers: papersResult, patents: patentsResult }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      setComparison((await res.json()) as SCurveComparison);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "No se pudo comparar las series.");
    } finally {
      setComparing(false);
    }
  }, [papersResult, patentsResult]);

  const handleAnalyze = useCallback(() => {
    if (!query.trim()) { setError("La ecuación de búsqueda no puede estar vacía."); return; }
    if (scopusOnServer === false) {
      setError("El servidor no tiene configurada SCOPUS_API_KEY.");
      return;
    }

    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Iniciando…");

    // Conteo anual: una petición por año en vez de paginar el corpus entero.
    // maxResults deja de tener sentido aquí — no se descargan registros.
    const cancel = streamPapersAnalysis(
      {
        query: query.trim(),
        year_from: yearFrom,
        year_to: yearTo,
      },
      (evt: SCurveProgress) => {
        setProgress(evt.progress);
        setProgressMsg(evt.message);
      },
      (res) => {
        if (res) storeResult(res);
        else setResult(null);
        setRunning(false);
        setProgress(100);
      },
      (err) => {
        setError(err);
        setRunning(false);
      }
    );

    cancelRef.current = cancel;
  }, [query, scopusOnServer, yearFrom, yearTo, storeResult]);

  const handleStop = () => {
    cancelRef.current?.();
    setRunning(false);
    setProgressMsg("Análisis cancelado.");
  };

  const handleReset = () => {
    cancelRef.current?.();
    setRunning(false);
    setResult(null);
    setError(null);
    setProgress(0);
    setProgressMsg("");
    setComparison(null);
  };

  const downloadCSV = () => {
    if (!result) return;
    const header = "Año,Publicaciones Anuales,Acumulado\n";
    const rows = result.yearly_data
      .map((d) => `${d.year},${d.annual},${d.cumulative}`)
      .join("\n");
    const blob = new Blob([header + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `s_curve_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const hasResult = !!result;

  return (
    <div
      style={{
        display: "flex",
        height: "calc(100vh - var(--header-h))",
        overflow: "hidden",
      }}
    >
      {/* ── Left panel ─────────────────────────────────────── */}
      <aside
        style={{
          width: 380,
          flexShrink: 0,
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          overflowY: "auto",
          background: "rgba(6,9,15,0.6)",
        }}
      >
        {/* Header */}
        <div style={{ padding: "20px 24px 0", borderBottom: "1px solid var(--border)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div
              style={{
                width: 32, height: 32, borderRadius: 9,
                background: "linear-gradient(135deg,#6366f1,#8b5cf6)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              <Activity style={{ width: 16, height: 16, color: "#fff" }} />
            </div>
            <span style={{ fontSize: 15, fontWeight: 700, color: "var(--text-1)" }}>Curvas en S</span>
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 2, marginTop: 14, marginBottom: -1 }}>
            {([
              { key: "scopus", label: "Papers", icon: <Database style={{ width: 13, height: 13 }} /> },
              { key: "epo", label: "Patentes (EPO)", icon: <Landmark style={{ width: 13, height: 13 }} /> },
              { key: "patents", label: "CSV", icon: <FileText style={{ width: 13, height: 13 }} /> },
            ] as const).map(tab => (
              <button
                key={tab.key}
                onClick={() => { setActiveTab(tab.key); setResult(null); setError(null); }}
                style={{
                  display: "flex", alignItems: "center", gap: 6,
                  padding: "7px 14px",
                  borderRadius: "8px 8px 0 0",
                  border: "1px solid var(--border)",
                  borderBottom: activeTab === tab.key ? "1px solid var(--bg)" : "1px solid var(--border)",
                  background: activeTab === tab.key ? "var(--bg)" : "transparent",
                  color: activeTab === tab.key ? "var(--text-1)" : "var(--text-3)",
                  cursor: "pointer",
                  fontSize: 12,
                  fontWeight: activeTab === tab.key ? 600 : 500,
                  marginBottom: activeTab === tab.key ? -1 : 0,
                  transition: "color 0.15s",
                }}
              >
                {tab.icon}
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Form */}
        <div style={{ padding: 24, display: "flex", flexDirection: "column", gap: 18 }}>
        {activeTab === "scopus" ? (<>

          {/* Query */}
          <div>
            <label
              style={{ display: "block", fontSize: 12, fontWeight: 600, color: "var(--text-2)", marginBottom: 6 }}
            >
              Ecuación de búsqueda Scopus
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`TITLE-ABS-KEY("deep learning") AND TITLE-ABS-KEY("medicine")`}
              rows={5}
              disabled={running}
              style={{
                width: "100%",
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: 12,
                color: "var(--text-1)",
                fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                resize: "vertical",
                outline: "none",
                lineHeight: 1.6,
                transition: "border-color 0.15s",
              }}
              onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
              onBlur={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
            />
          </div>

          {/* Example queries */}
          <div>
            <p style={{ fontSize: 11, color: "var(--text-4)", marginBottom: 6, fontWeight: 500 }}>
              EJEMPLOS RÁPIDOS
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {EXAMPLE_QUERIES.map((eq) => (
                <button
                  key={eq.label}
                  onClick={() => setQuery(eq.query)}
                  disabled={running}
                  style={{
                    textAlign: "left",
                    padding: "7px 10px",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--text-3)",
                    fontSize: 12,
                    cursor: "pointer",
                    transition: "background 0.12s, color 0.12s, border-color 0.12s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "var(--surface-2)";
                    e.currentTarget.style.color = "var(--text-1)";
                    e.currentTarget.style.borderColor = "var(--border-2)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "transparent";
                    e.currentTarget.style.color = "var(--text-3)";
                    e.currentTarget.style.borderColor = "var(--border)";
                  }}
                >
                  {eq.label}
                </button>
              ))}
            </div>
          </div>

          {scopusOnServer === false && (
            <div
              style={{
                display: "flex",
                gap: 9,
                padding: "10px 12px",
                borderRadius: 9,
                background: "rgba(240,128,108,0.08)",
                border: "1px solid rgba(240,128,108,0.22)",
                fontSize: 12,
                color: "var(--text-2)",
                lineHeight: 1.5,
              }}
            >
              <AlertTriangle style={{ width: 15, height: 15, flexShrink: 0, color: "#f0806c", marginTop: 1 }} />
              <span>
                El servidor no tiene configurada <strong>SCOPUS_API_KEY</strong>.
                Avisa a quien administre el portal.
              </span>
            </div>
          )}

          {/* Advanced options toggle */}
          <div>
            <button
              onClick={() => setShowAdvanced((v) => !v)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                fontSize: 12,
                color: "var(--text-3)",
                background: "transparent",
                border: "none",
                cursor: "pointer",
                padding: 0,
                fontWeight: 500,
              }}
            >
              {showAdvanced ? <ChevronUp style={{ width: 14, height: 14 }} /> : <ChevronDown style={{ width: 14, height: 14 }} />}
              Opciones avanzadas
            </button>

            {showAdvanced && (
              <div
                style={{
                  marginTop: 12,
                  padding: 14,
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  background: "var(--surface-1)",
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                <div style={{ display: "flex", gap: 10 }}>
                  <div style={{ flex: 1 }}>
                    <label style={{ fontSize: 11, color: "var(--text-3)", display: "block", marginBottom: 4 }}>
                      Año desde
                    </label>
                    <input
                      type="number"
                      value={yearFrom}
                      min={1970}
                      max={yearTo}
                      onChange={(e) => setYearFrom(Number(e.target.value))}
                      disabled={running}
                      className="inp"
                      style={{ width: "100%" }}
                    />
                  </div>
                  <div style={{ flex: 1 }}>
                    <label style={{ fontSize: 11, color: "var(--text-3)", display: "block", marginBottom: 4 }}>
                      Año hasta
                    </label>
                    <input
                      type="number"
                      value={yearTo}
                      min={yearFrom}
                      max={new Date().getFullYear()}
                      onChange={(e) => setYearTo(Number(e.target.value))}
                      disabled={running}
                      className="inp"
                      style={{ width: "100%" }}
                    />
                  </div>
                </div>
                <div>
                  <label style={{ fontSize: 11, color: "var(--text-3)", display: "block", marginBottom: 4 }}>
                    Máx. resultados{" "}
                    <span style={{ color: "var(--text-4)" }}>(vacío = todos)</span>
                  </label>
                  <input
                    type="number"
                    value={maxResults}
                    min={25}
                    step={25}
                    placeholder="Sin límite"
                    onChange={(e) => setMaxResults(e.target.value === "" ? "" : Number(e.target.value))}
                    disabled={running}
                    className="inp"
                    style={{ width: "100%" }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#fca5a5",
                fontSize: 12,
                lineHeight: 1.5,
              }}
            >
              {error}
            </div>
          )}

          {/* Progress */}
          {running && <ProgressBar value={progress} message={progressMsg} />}

          {/* Actions */}
          <div style={{ display: "flex", gap: 8 }}>
            {!running ? (
              <button
                onClick={handleAnalyze}
                className="btn btn-primary"
                style={{ flex: 1, gap: 8 }}
              >
                <Play style={{ width: 14, height: 14 }} />
                Analizar
              </button>
            ) : (
              <button
                onClick={handleStop}
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                  padding: "10px 16px",
                  borderRadius: 10,
                  border: "1px solid rgba(239,68,68,0.4)",
                  background: "rgba(239,68,68,0.1)",
                  color: "#fca5a5",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                <Square style={{ width: 13, height: 13 }} />
                Detener
              </button>
            )}

            {hasResult && !running && (
              <button
                onClick={handleReset}
                title="Limpiar resultados"
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--text-3)",
                  cursor: "pointer",
                }}
              >
                <RotateCcw style={{ width: 14, height: 14 }} />
              </button>
            )}
          </div>
        </>) : activeTab === "epo" ? (<>

          {/* ── Patentes vía EPO OPS ──────────────────────── */}
          <div
            style={{
              display: "flex",
              alignItems: "flex-start",
              gap: 10,
              padding: "12px 14px",
              borderRadius: 10,
              border: "1px solid rgba(16,185,129,0.25)",
              background: "rgba(16,185,129,0.06)",
            }}
          >
            <Landmark style={{ width: 15, height: 15, color: "#10b981", flexShrink: 0, marginTop: 1 }} />
            <p style={{ fontSize: 11.5, color: "var(--text-3)", lineHeight: 1.6 }}>
              Automatizado vía <strong style={{ color: "var(--text-2)" }}>EPO OPS</strong> —
              patentes publicadas en la base mundial DOCDB. Las credenciales
              están en el servidor: no hace falta API key.
            </p>
          </div>

          {/* Estado del servicio */}
          {epoStatus && !epoStatus.authenticated && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#fca5a5",
                fontSize: 11.5,
                lineHeight: 1.55,
              }}
            >
              {epoStatus.message}
            </div>
          )}
          {epoStatus?.authenticated && (
            <p style={{ fontSize: 11, color: "var(--text-4)", display: "flex", alignItems: "center", gap: 6 }}>
              <CheckCircle2 style={{ width: 12, height: 12, color: "#10b981" }} />
              EPO OPS operativo
              {epoStatus.search_quota_per_min
                ? ` · ${epoStatus.search_quota_per_min} búsquedas/min disponibles`
                : ""}
            </p>
          )}

          <div>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "var(--text-2)", marginBottom: 6 }}>
              Consulta de patentes (CQL)
            </label>
            <textarea
              value={epoQuery}
              onChange={(e) => setEpoQuery(e.target.value)}
              placeholder={`txt="biodiesel"`}
              rows={3}
              disabled={running}
              style={{
                width: "100%",
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                padding: "10px 12px",
                fontSize: 12,
                color: "var(--text-1)",
                fontFamily: "var(--font-mono, monospace)",
                resize: "vertical",
                outline: "none",
              }}
            />
            <p style={{ fontSize: 10.5, color: "var(--text-4)", marginTop: 6, lineHeight: 1.55 }}>
              Un término suelto se envuelve en <code>txt=&quot;…&quot;</code> (título,
              resumen, solicitante e inventor). También admite CQL:{" "}
              <code>txt=&quot;biodiesel&quot; and pa=&quot;basf&quot;</code>.
            </p>
          </div>

          <div>
            <p style={{ fontSize: 10.5, fontWeight: 600, color: "var(--text-4)", textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 8 }}>
              Ejemplos rápidos
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {EPO_EXAMPLES.map((ex) => (
                <button
                  key={ex.label}
                  onClick={() => setEpoQuery(ex.query)}
                  disabled={running}
                  style={{
                    textAlign: "left",
                    padding: "8px 12px",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--text-3)",
                    fontSize: 11.5,
                    cursor: running ? "default" : "pointer",
                  }}
                >
                  {ex.label}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div>
              <label style={{ display: "block", fontSize: 11, color: "var(--text-3)", marginBottom: 5 }}>
                Desde
              </label>
              <input
                type="number"
                value={epoYearFrom}
                min={1900}
                max={epoYearTo}
                onChange={(e) => setEpoYearFrom(Number(e.target.value))}
                disabled={running}
                className="inp"
                style={{ width: "100%" }}
              />
            </div>
            <div>
              <label style={{ display: "block", fontSize: 11, color: "var(--text-3)", marginBottom: 5 }}>
                Hasta
              </label>
              <input
                type="number"
                value={epoYearTo}
                min={epoYearFrom}
                max={new Date().getFullYear()}
                onChange={(e) => setEpoYearTo(Number(e.target.value))}
                disabled={running}
                className="inp"
                style={{ width: "100%" }}
              />
            </div>
          </div>

          {/*
            OPS cobra una petición por año y limita las búsquedas por minuto,
            así que el coste del rango es visible antes de lanzarlo: un rango
            largo no falla, tarda.
          */}
          <p style={{ fontSize: 10.5, color: "var(--text-4)", lineHeight: 1.55 }}>
            {Math.max(epoYearTo - epoYearFrom + 1, 0)} años ={" "}
            {Math.max(epoYearTo - epoYearFrom + 1, 0)} peticiones a OPS. Con el
            servicio saturado (5 búsquedas/min) puede tardar varios minutos.
          </p>

          {/* Error */}
          {error && (
            <div
              style={{
                padding: "10px 12px",
                borderRadius: 8,
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "#fca5a5",
                fontSize: 12,
                lineHeight: 1.5,
              }}
            >
              {error}
            </div>
          )}

          {/* Progress */}
          {running && <ProgressBar value={progress} message={progressMsg} />}

          {/* Actions */}
          <div style={{ display: "flex", gap: 8 }}>
            {!running ? (
              <button
                onClick={handleEpoAnalyze}
                disabled={epoStatus ? !epoStatus.authenticated : false}
                className="btn btn-primary"
                style={{
                  flex: 1,
                  gap: 8,
                  opacity: epoStatus && !epoStatus.authenticated ? 0.5 : 1,
                }}
              >
                <Play style={{ width: 14, height: 14 }} />
                Analizar patentes
              </button>
            ) : (
              <button
                onClick={handleStop}
                style={{
                  flex: 1,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 8,
                  padding: "10px 16px",
                  borderRadius: 10,
                  border: "1px solid rgba(239,68,68,0.4)",
                  background: "rgba(239,68,68,0.1)",
                  color: "#fca5a5",
                  cursor: "pointer",
                  fontSize: 13,
                  fontWeight: 600,
                }}
              >
                <Square style={{ width: 13, height: 13 }} />
                Detener
              </button>
            )}
            {hasResult && !running && (
              <button
                onClick={handleReset}
                title="Limpiar resultados"
                style={{
                  padding: "10px 12px",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                  background: "transparent",
                  color: "var(--text-3)",
                  cursor: "pointer",
                }}
              >
                <RotateCcw style={{ width: 14, height: 14 }} />
              </button>
            )}
          </div>
        </>) : (

        /* ── Patents (CSV) tab ─────────────────────────── */
        <>
          {/* Title */}
          <div>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "var(--text-2)", marginBottom: 6 }}>
              Nombre / título del análisis
            </label>
            <input
              type="text"
              value={patentTitle}
              onChange={(e) => setPatentTitle(e.target.value)}
              placeholder="Ej: Patentes de emulsificación alimentaria"
              disabled={running}
              className="inp"
              style={{ width: "100%" }}
            />
          </div>

          {/* File upload */}
          <div>
            <label style={{ display: "block", fontSize: 12, fontWeight: 600, color: "var(--text-2)", marginBottom: 6 }}>
              Archivo CSV de Lens.org
            </label>
            <div
              onClick={() => fileInputRef.current?.click()}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "12px 14px",
                borderRadius: 10,
                border: csvFileName ? "1px solid rgba(99,102,241,0.4)" : "1px dashed var(--border)",
                background: csvFileName ? "rgba(99,102,241,0.06)" : "var(--surface-1)",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = "rgba(99,102,241,0.5)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = csvFileName ? "rgba(99,102,241,0.4)" : "var(--border)"; }}
            >
              <Upload style={{ width: 15, height: 15, color: csvFileName ? "#818cf8" : "var(--text-4)", flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                {csvFileName ? (
                  <>
                    <p style={{ fontSize: 12, fontWeight: 600, color: "#c7d2fe", margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{csvFileName}</p>
                    <p style={{ fontSize: 11, color: "var(--text-3)", margin: 0 }}>{csvRows.length} filas · {csvDocTypes.length} tipos</p>
                  </>
                ) : (
                  <p style={{ fontSize: 12, color: "var(--text-4)", margin: 0 }}>Haz clic para seleccionar CSV</p>
                )}
              </div>
              {csvFileName && (
                <button
                  onClick={(e) => { e.stopPropagation(); setCsvRows([]); setCsvDocTypes([]); setSelectedTypes(new Set()); setCsvFileName(""); }}
                  style={{ background: "transparent", border: "none", cursor: "pointer", color: "var(--text-3)", padding: 2 }}
                >
                  <X style={{ width: 13, height: 13 }} />
                </button>
              )}
            </div>
            <input ref={fileInputRef} type="file" accept=".csv" onChange={handleFileUpload} style={{ display: "none" }} />
            <p style={{ fontSize: 11, color: "var(--text-4)", marginTop: 5 }}>
              Exporta desde <span style={{ color: "#818cf8" }}>lens.org</span> → Analytics → By Year → Download CSV
            </p>
          </div>

          {/* Document type filter */}
          {csvDocTypes.length > 0 && (
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 8 }}>
                <Filter style={{ width: 12, height: 12, color: "var(--text-3)" }} />
                <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-2)" }}>
                  Tipos de documento
                </label>
                <span style={{ fontSize: 11, color: "var(--text-4)" }}>({selectedTypes.size} seleccionados)</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {csvDocTypes.map(t => {
                  const isSelected = selectedTypes.has(t);
                  const count = csvRows.filter(r => r.docType === t).reduce((s, r) => s + r.count, 0);
                  return (
                    <button
                      key={t}
                      onClick={() => toggleType(t)}
                      style={{
                        display: "flex", alignItems: "center", justifyContent: "space-between",
                        padding: "7px 10px", borderRadius: 8,
                        border: `1px solid ${isSelected ? "rgba(99,102,241,0.4)" : "var(--border)"}`,
                        background: isSelected ? "rgba(99,102,241,0.08)" : "transparent",
                        color: isSelected ? "#c7d2fe" : "var(--text-3)",
                        cursor: "pointer", fontSize: 12, textAlign: "left",
                        transition: "all 0.12s",
                      }}
                    >
                      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t}</span>
                      <span style={{ fontSize: 11, color: isSelected ? "#818cf8" : "var(--text-4)", marginLeft: 8, flexShrink: 0 }}>
                        {count.toLocaleString()}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Error */}
          {error && (
            <div style={{ padding: "10px 12px", borderRadius: 8, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", color: "#fca5a5", fontSize: 12, lineHeight: 1.5 }}>
              {error}
            </div>
          )}

          {/* Progress */}
          {running && <ProgressBar value={progress} message={progressMsg} />}

          {/* Actions */}
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handlePatentAnalyze}
              disabled={running || csvRows.length === 0}
              className="btn btn-primary"
              style={{ flex: 1, gap: 8, opacity: csvRows.length === 0 ? 0.5 : 1 }}
            >
              <Play style={{ width: 14, height: 14 }} />
              Analizar patentes
            </button>
            {hasResult && !running && (
              <button
                onClick={handleReset}
                title="Limpiar resultados"
                style={{ padding: "10px 12px", borderRadius: 10, border: "1px solid var(--border)", background: "transparent", color: "var(--text-3)", cursor: "pointer" }}
              >
                <RotateCcw style={{ width: 14, height: 14 }} />
              </button>
            )}
          </div>
        </>
        )}
        </div>
      </aside>

      {/* ── Right panel ────────────────────────────────────── */}
      <main style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {!hasResult && !running ? (
          /* Empty state */
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 20,
              padding: 48,
            }}
          >
            <EmptyStateSVG />
            <div style={{ textAlign: "center", maxWidth: 440 }}>
              <p style={{ fontSize: 18, fontWeight: 700, color: "var(--text-1)", marginBottom: 8 }}>
                Descubre el ciclo de vida tecnológico
              </p>
              <p style={{ fontSize: 13, color: "var(--text-3)", lineHeight: 1.7 }}>
                Ingresa una ecuación de búsqueda Scopus y tu API key para analizar cómo ha
                evolucionado la adopción de una tecnología a través de la literatura científica.
              </p>
            </div>
          </div>
        ) : running && !hasResult ? (
          /* Loading state */
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 24,
              padding: 48,
            }}
          >
            <div
              style={{
                width: 56,
                height: 56,
                borderRadius: "50%",
                background: "rgba(99,102,241,0.12)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Activity style={{ width: 24, height: 24, color: "#6366f1" }} className="spin" />
            </div>
            <div style={{ textAlign: "center", maxWidth: 380 }}>
              <p style={{ fontSize: 15, fontWeight: 600, color: "var(--text-1)", marginBottom: 6 }}>
                {progressMsg}
              </p>
              <div
                style={{
                  height: 4,
                  borderRadius: 2,
                  background: "rgba(255,255,255,0.06)",
                  overflow: "hidden",
                  width: 280,
                }}
              >
                <div
                  style={{
                    height: "100%",
                    borderRadius: 2,
                    width: `${progress}%`,
                    background: "linear-gradient(90deg,#6366f1,#8b5cf6)",
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
              <p style={{ fontSize: 12, color: "var(--text-4)", marginTop: 8 }}>
                {progress}% completado
              </p>
            </div>
          </div>
        ) : hasResult && result ? (
          /* Results */
          <div style={{ padding: 28, display: "flex", flexDirection: "column", gap: 24 }}>

            {/* Desfase ciencia → tecnología */}
            {comparison ? (
              <ComparisonPanel comparison={comparison} onClose={() => setComparison(null)} />
            ) : (
              papersResult && patentsResult && (
                <div
                  className="card"
                  style={{
                    padding: "14px 18px",
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                  }}
                >
                  <Scale style={{ width: 16, height: 16, color: "#8b5cf6", flexShrink: 0 }} />
                  <p style={{ flex: 1, fontSize: 12.5, color: "var(--text-3)", lineHeight: 1.55 }}>
                    Hay una serie de papers y otra de patentes en memoria. Compáralas
                    para medir cuántos años separan sus inflexiones.
                  </p>
                  <button
                    onClick={handleCompare}
                    disabled={comparing}
                    className="btn btn-primary"
                    style={{ gap: 8, flexShrink: 0, opacity: comparing ? 0.6 : 1 }}
                  >
                    <GitCompareArrows style={{ width: 14, height: 14 }} />
                    {comparing ? "Comparando…" : "Comparar papers ↔ patentes"}
                  </button>
                </div>
              )
            )}

            {/* Phase banner */}
            <div
              style={{
                padding: "20px 24px",
                borderRadius: 14,
                border: `1px solid ${result.phase.color}33`,
                background: `${result.phase.color}0d`,
                display: "flex",
                alignItems: "center",
                gap: 16,
              }}
            >
              <div
                style={{
                  width: 52,
                  height: 52,
                  borderRadius: 14,
                  background: `${result.phase.color}22`,
                  border: `1px solid ${result.phase.color}44`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 26,
                  flexShrink: 0,
                }}
              >
                {PHASE_ICON[result.phase.phase] ?? "📊"}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 4 }}>
                  <span
                    style={{
                      fontSize: 20,
                      fontWeight: 800,
                      color: result.phase.color,
                      letterSpacing: "-0.01em",
                    }}
                  >
                    {result.phase.phase}
                  </span>
                  {result.phase.saturation_pct !== null ? (
                    <span
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        padding: "2px 8px",
                        borderRadius: 20,
                        background: `${result.phase.color}22`,
                        color: result.phase.color,
                      }}
                    >
                      {result.phase.saturation_pct}% saturación
                    </span>
                  ) : (
                    <span
                      title="El techo de saturación no se puede estimar con los datos disponibles"
                      style={{
                        fontSize: 12,
                        fontWeight: 600,
                        padding: "2px 8px",
                        borderRadius: 20,
                        background: "var(--surface-3)",
                        color: "var(--text-3)",
                      }}
                    >
                      saturación no estimable
                    </span>
                  )}
                </div>
                <p style={{ fontSize: 13, color: "var(--text-3)", lineHeight: 1.5 }}>
                  {result.phase.description}
                </p>
              </div>
              {/* R² badge */}
              {result.r2 !== null && (
                <div
                  style={{
                    textAlign: "center",
                    padding: "10px 16px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid var(--border)",
                    flexShrink: 0,
                  }}
                >
                  <p style={{ fontSize: 11, color: "var(--text-4)", marginBottom: 2 }}>R² fit</p>
                  <p
                    style={{
                      fontSize: 20,
                      fontWeight: 800,
                      color: result.r2 > 0.9 ? "#10b981" : result.r2 > 0.7 ? "#f59e0b" : "#ef4444",
                    }}
                  >
                    {result.r2.toFixed(3)}
                  </p>
                </div>
              )}

              {/*
                El R² no viaja solo. Sobre un acumulado casi siempre supera
                0.99, y es el Durbin-Watson el que dice si ese número
                significa algo o sólo refleja que la serie crece.
              */}
              {result.sigmoid_params?.durbin_watson != null && (
                <div
                  title="Durbin-Watson sobre los residuos (0–4). Cerca de 2: sin estructura. Muy por debajo: el ajuste tiene sesgo sistemático que el R² no ve."
                  style={{
                    textAlign: "center",
                    padding: "10px 16px",
                    borderRadius: 10,
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid var(--border)",
                    flexShrink: 0,
                    cursor: "help",
                  }}
                >
                  <p style={{ fontSize: 11, color: "var(--text-4)", marginBottom: 2 }}>
                    Durbin-Watson
                  </p>
                  <p
                    style={{
                      fontSize: 20,
                      fontWeight: 800,
                      color: dwColor(result.sigmoid_params.durbin_watson),
                    }}
                  >
                    {result.sigmoid_params.durbin_watson.toFixed(3)}
                  </p>
                </div>
              )}
            </div>

            {/* Stats row */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(4, 1fr)",
                gap: 12,
              }}
            >
              <MetricCard
                icon={<BookOpen style={{ width: 16, height: 16 }} />}
                label="Publicaciones"
                value={result.total_publications.toLocaleString()}
                sub={`${result.year_from} – ${result.year_to}`}
              />
              <MetricCard
                icon={<Calendar style={{ width: 16, height: 16 }} />}
                label="Punto de inflexión"
                value={result.inflection_year ?? "—"}
                sub="Máximo crecimiento anual"
                color="#f59e0b"
              />
              <MetricCard
                icon={<Target style={{ width: 16, height: 16 }} />}
                label="Saturación proyectada"
                value={result.saturation_level ? result.saturation_level.toLocaleString() : "—"}
                sub="Nivel L del sigmoide"
                color={result.phase.color}
              />
              <MetricCard
                icon={<Sigma style={{ width: 16, height: 16 }} />}
                label="Tasa de crecimiento k"
                value={result.sigmoid_params ? result.sigmoid_params.k.toFixed(3) : "—"}
                sub="Parámetro sigmoidal"
                color="#8b5cf6"
              />
            </div>

            {/* Query block */}
            <QueryBlock query={result.query} yearFrom={result.year_from} yearTo={result.year_to} />

            {/* Main chart */}
            <div className="card" style={{ padding: 20 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: 16,
                }}
              >
                <div>
                  <p style={{ fontSize: 14, fontWeight: 700, color: "var(--text-1)" }}>
                    Curva S — Adopción tecnológica
                  </p>
                  <p style={{ fontSize: 12, color: "var(--text-3)", marginTop: 2 }}>
                    Publicaciones anuales (barras) + acumulado y ajuste sigmoidal (eje derecho)
                  </p>
                </div>
                <button
                  onClick={downloadCSV}
                  title="Descargar CSV"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "7px 12px",
                    borderRadius: 8,
                    border: "1px solid var(--border)",
                    background: "transparent",
                    color: "var(--text-3)",
                    cursor: "pointer",
                    fontSize: 12,
                    transition: "background 0.12s, color 0.12s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "var(--surface-2)";
                    e.currentTarget.style.color = "var(--text-1)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "transparent";
                    e.currentTarget.style.color = "var(--text-3)";
                  }}
                >
                  <Download style={{ width: 13, height: 13 }} />
                  CSV
                </button>
              </div>
              <SCurveChart result={result} height={400} />
            </div>

            {/* Sigmoid params card */}
            {result.sigmoid_params && (
              <div className="card" style={{ padding: 20 }}>
                <p
                  style={{
                    fontSize: 13,
                    fontWeight: 700,
                    color: "var(--text-1)",
                    marginBottom: 14,
                  }}
                >
                  Parámetros del modelo sigmoidal
                </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: 12,
                  }}
                >
                  {[
                    {
                      label: "L — Nivel de saturación",
                      value: result.sigmoid_params.L.toLocaleString(undefined, { maximumFractionDigits: 0 }),
                      desc: "Número máximo proyectado de publicaciones acumuladas",
                      color: result.phase.color,
                    },
                    {
                      label: "x₀ — Punto de inflexión",
                      value: result.sigmoid_params.x0.toFixed(1),
                      desc: "Año en que la tasa de adopción es máxima",
                      color: "#f59e0b",
                    },
                    {
                      label: "k — Tasa de crecimiento",
                      value: result.sigmoid_params.k.toFixed(4),
                      desc: "Velocidad de adopción (mayor = más rápida)",
                      color: "#8b5cf6",
                    },
                  ].map((p) => (
                    <div
                      key={p.label}
                      style={{
                        padding: "14px 16px",
                        borderRadius: 10,
                        background: "var(--surface-1)",
                        border: "1px solid var(--border)",
                      }}
                    >
                      <p
                        style={{
                          fontSize: 11,
                          color: "var(--text-4)",
                          textTransform: "uppercase",
                          letterSpacing: "0.05em",
                          marginBottom: 4,
                        }}
                      >
                        {p.label}
                      </p>
                      <p
                        style={{
                          fontSize: 24,
                          fontWeight: 800,
                          color: p.color,
                          lineHeight: 1,
                          marginBottom: 6,
                          fontVariantNumeric: "tabular-nums",
                        }}
                      >
                        {p.value}
                      </p>
                      <p style={{ fontSize: 11, color: "var(--text-3)", lineHeight: 1.5 }}>
                        {p.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Ajuste multi-modelo */}
            <ModelComparisonCard result={result} />

            {/* Data table toggle */}
            <div className="card" style={{ padding: 0, overflow: "hidden" }}>
              <button
                onClick={() => setShowTable((v) => !v)}
                style={{
                  width: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  padding: "16px 20px",
                  background: "transparent",
                  border: "none",
                  cursor: "pointer",
                  color: "var(--text-1)",
                }}
              >
                <span style={{ fontSize: 13, fontWeight: 700 }}>
                  Tabla de datos ({result.yearly_data.length} años)
                </span>
                {showTable ? (
                  <ChevronUp style={{ width: 15, height: 15, color: "var(--text-3)" }} />
                ) : (
                  <ChevronDown style={{ width: 15, height: 15, color: "var(--text-3)" }} />
                )}
              </button>

              {showTable && (
                <div style={{ overflowX: "auto", borderTop: "1px solid var(--border)" }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: 13,
                    }}
                  >
                    <thead>
                      <tr style={{ background: "var(--surface-1)" }}>
                        {["Año", "Publicaciones anuales", "Acumulado", "Δ% anual"].map((h) => (
                          <th
                            key={h}
                            style={{
                              padding: "10px 20px",
                              textAlign: "left",
                              fontSize: 11,
                              fontWeight: 600,
                              color: "var(--text-3)",
                              textTransform: "uppercase",
                              letterSpacing: "0.05em",
                              borderBottom: "1px solid var(--border)",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {result.yearly_data.map((row, idx) => {
                        const prev = idx > 0 ? result.yearly_data[idx - 1].annual : null;
                        const delta =
                          prev && prev > 0
                            ? (((row.annual - prev) / prev) * 100).toFixed(1)
                            : "—";
                        const isPos = prev && row.annual > prev;
                        const isNeg = prev && row.annual < prev;
                        return (
                          <tr
                            key={row.year}
                            style={{
                              borderBottom: "1px solid var(--border)",
                              transition: "background 0.1s",
                            }}
                            onMouseEnter={(e) =>
                              (e.currentTarget.style.background = "var(--surface-1)")
                            }
                            onMouseLeave={(e) =>
                              (e.currentTarget.style.background = "transparent")
                            }
                          >
                            <td
                              style={{
                                padding: "10px 20px",
                                fontWeight: 600,
                                color: "var(--text-1)",
                                fontVariantNumeric: "tabular-nums",
                              }}
                            >
                              {row.year}
                            </td>
                            <td
                              style={{
                                padding: "10px 20px",
                                color: "var(--text-2)",
                                fontVariantNumeric: "tabular-nums",
                              }}
                            >
                              {row.annual.toLocaleString()}
                            </td>
                            <td
                              style={{
                                padding: "10px 20px",
                                color: result.phase.color,
                                fontWeight: 600,
                                fontVariantNumeric: "tabular-nums",
                              }}
                            >
                              {row.cumulative.toLocaleString()}
                            </td>
                            <td
                              style={{
                                padding: "10px 20px",
                                color: isPos
                                  ? "#10b981"
                                  : isNeg
                                  ? "#ef4444"
                                  : "var(--text-3)",
                                fontVariantNumeric: "tabular-nums",
                              }}
                            >
                              {delta !== "—" ? `${Number(delta) > 0 ? "+" : ""}${delta}%` : "—"}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Footer meta */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "12px 16px",
                borderRadius: 10,
                background: "var(--surface-1)",
                border: "1px solid var(--border)",
              }}
            >
              <TrendingUp style={{ width: 13, height: 13, color: "var(--text-4)", flexShrink: 0 }} />
              <p style={{ fontSize: 11, color: "var(--text-4)", lineHeight: 1.4 }}>
                Fuente: Scopus (Elsevier) · Excluyendo patentes ·{" "}
                Analizado: {new Date(result.analyzed_at).toLocaleString("es-CO")} ·{" "}
                Consulta: <code style={{ fontFamily: "monospace", color: "var(--text-3)" }}>{result.query.slice(0, 80)}{result.query.length > 80 ? "…" : ""}</code>
              </p>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}

// ── Decorative empty state SVG ────────────────────────────────

function EmptyStateSVG() {
  return (
    <svg width="280" height="180" viewBox="0 0 280 180" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Grid lines */}
      {[40, 80, 120, 160].map((y) => (
        <line key={y} x1="20" y1={y} x2="260" y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {[60, 110, 160, 210].map((x) => (
        <line key={x} x1={x} y1="20" x2={x} y2="160" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}

      {/* Bars */}
      {[
        { x: 30, h: 10 }, { x: 55, h: 12 }, { x: 80, h: 18 }, { x: 105, h: 28 },
        { x: 130, h: 42 }, { x: 155, h: 58 }, { x: 180, h: 48 }, { x: 205, h: 40 },
        { x: 230, h: 34 },
      ].map((b, i) => (
        <rect
          key={i}
          x={b.x}
          y={160 - b.h}
          width={18}
          height={b.h}
          rx="3"
          fill="rgba(79,142,247,0.2)"
        />
      ))}

      {/* S-curve line */}
      <path
        d="M 30 155 C 60 150, 80 140, 100 125 C 120 110, 140 85, 155 65 C 170 45, 190 38, 220 35 C 240 33, 255 34, 265 34"
        stroke="url(#scGrad)"
        strokeWidth="2.5"
        fill="none"
        strokeLinecap="round"
      />

      {/* Dots on curve */}
      {[
        [30, 155], [75, 142], [120, 110], [155, 65], [200, 37], [248, 34],
      ].map(([cx, cy], i) => (
        <circle key={i} cx={cx} cy={cy} r="4" fill="#6366f1" opacity="0.8" />
      ))}

      {/* Inflection vertical dashed line */}
      <line x1="155" y1="30" x2="155" y2="160" stroke="rgba(251,191,36,0.4)" strokeWidth="1.5" strokeDasharray="4,4" />
      <text x="158" y="44" fill="#fbbf24" fontSize="9" opacity="0.7">x₀</text>

      {/* Gradient def */}
      <defs>
        <linearGradient id="scGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#8b5cf6" />
        </linearGradient>
      </defs>
    </svg>
  );
}

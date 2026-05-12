"use client";

import { useState, useRef, useCallback } from "react";
import { Activity, Play, Square, RotateCcw, TrendingUp, Calendar, BookOpen, Target, Sigma, ChevronDown, ChevronUp, Download, Copy, Check, Database, FileText, Upload, X, Filter } from "lucide-react";
import { streamSCurveAnalysis, analyzeSCurveLocal } from "@/lib/api";
import { SCurveChart } from "@/components/charts/SCurveChart";
import type { SCurveResult, SCurveProgress } from "@/types";

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
  "Sin datos": "❓",
};

// ── Helpers ───────────────────────────────────────────────────

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

// ── Main component ────────────────────────────────────────────

// ── CSV row type ──────────────────────────────────────────────
interface CsvRow { count: number; year: number; docType: string; }

export default function SCurvePage() {
  // Tab
  const [activeTab, setActiveTab] = useState<"scopus" | "patents">("scopus");

  // Scopus form state
  const [query, setQuery] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [yearFrom, setYearFrom] = useState(1990);
  const [yearTo, setYearTo] = useState(new Date().getFullYear());
  const [maxResults, setMaxResults] = useState<number | "">("");
  const [showAdvanced, setShowAdvanced] = useState(false);

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

  // Data table
  const [showTable, setShowTable] = useState(false);

  const cancelRef = useRef<(() => void) | null>(null);

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
      next.has(t) ? next.delete(t) : next.add(t);
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
      setResult(res);
      setProgress(100);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error desconocido");
    } finally {
      setRunning(false);
    }
  }, [csvRows, selectedTypes, patentTitle]);

  const handleAnalyze = useCallback(() => {
    if (!query.trim()) { setError("La ecuación de búsqueda no puede estar vacía."); return; }
    if (!apiKey.trim()) { setError("Ingresa tu API key de Scopus."); return; }

    setError(null);
    setResult(null);
    setRunning(true);
    setProgress(0);
    setProgressMsg("Iniciando…");

    const cancel = streamSCurveAnalysis(
      {
        query: query.trim(),
        api_key: apiKey.trim(),
        year_from: yearFrom,
        year_to: yearTo,
        max_results: maxResults === "" ? undefined : Number(maxResults),
      },
      (evt: SCurveProgress) => {
        setProgress(evt.progress);
        setProgressMsg(evt.message);
      },
      (res) => {
        setResult(res ?? null);
        setRunning(false);
        setProgress(100);
      },
      (err) => {
        setError(err);
        setRunning(false);
      }
    );

    cancelRef.current = cancel;
  }, [query, apiKey, yearFrom, yearTo, maxResults]);

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
              { key: "scopus", label: "Scopus", icon: <Database style={{ width: 13, height: 13 }} /> },
              { key: "patents", label: "Patentes (CSV)", icon: <FileText style={{ width: 13, height: 13 }} /> },
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

          {/* API Key */}
          <div>
            <label
              style={{ display: "block", fontSize: 12, fontWeight: 600, color: "var(--text-2)", marginBottom: 6 }}
            >
              API Key de Scopus
            </label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Elsevier API key…"
              disabled={running}
              className="inp"
              style={{ width: "100%" }}
            />
            <p style={{ fontSize: 11, color: "var(--text-4)", marginTop: 4 }}>
              Obtén tu key en{" "}
              <span style={{ color: "var(--accent)" }}>dev.elsevier.com</span>
            </p>
          </div>

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

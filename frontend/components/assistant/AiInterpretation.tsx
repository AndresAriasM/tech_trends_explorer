"use client";

/**
 * Interpretación con IA de un análisis ya corrido (curvas de papers y
 * patentes). Usa el informe prospectivo del asistente, anclado al corpus
 * metodológico con RAG: sólo cita fragmentos realmente recuperados.
 */

import { useState } from "react";
import { Bot, Loader2, Sparkles } from "lucide-react";

import { friendlyAiError, generateReport, type ForesightReport } from "@/lib/assistant";
import type { SCurveResult } from "@/types";

export function AiInterpretation({
  concept,
  papers,
  patents,
  extraContext,
}: {
  concept: string;
  papers?: SCurveResult | null;
  patents?: SCurveResult | null;
  extraContext?: string;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<ForesightReport | null>(null);

  const interpretar = async () => {
    setLoading(true);
    setError(null);
    try {
      setReport(
        await generateReport({
          concept: concept || "tecnología analizada",
          s_curve_papers: papers ?? null,
          s_curve_patents: patents ?? null,
          extra_context: extraContext,
        })
      );
    } catch (e) {
      setError(friendlyAiError(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="card" style={{ padding: 22, marginBottom: 22, border: "1px solid rgba(167,139,250,0.3)" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Bot style={{ width: 18, height: 18, color: "#a78bfa" }} />
          <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-1)" }}>Interpretación con IA</h3>
        </div>
        <button className="btn btn-primary" onClick={() => void interpretar()} disabled={loading}>
          {loading ? <Loader2 className="animate-spin" style={{ width: 15, height: 15 }} /> : <Sparkles style={{ width: 15, height: 15 }} />}
          {report ? "Volver a interpretar" : "Interpretar resultados"}
        </button>
      </div>

      {!report && !error && !loading && (
        <p style={{ fontSize: 12.5, color: "var(--text-3)", marginTop: 10 }}>
          Lectura prospectiva de estas curvas, anclada al corpus metodológico. Tarda unos segundos.
        </p>
      )}
      {error && <p style={{ fontSize: 12.5, color: "#fca5a5", marginTop: 10 }}>{error}</p>}

      {report && (
        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "#a78bfa" }}>
            Generado por IA · {report.ai_model}
            {report.corpus_grounded ? " · anclado al corpus" : " · sin respaldo del corpus"}
          </p>
          {report.sections.map((s, i) => (
            <div key={s.id ?? i}>
              <p style={{ fontSize: 13.5, fontWeight: 700, color: "var(--text-1)", marginBottom: 4 }}>{s.heading}</p>
              {s.text && <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.7 }}>{s.text}</p>}
              {s.items && (
                <ul style={{ paddingLeft: 18, color: "var(--text-2)", fontSize: 13, lineHeight: 1.7 }}>
                  {s.items.map((it, j) => (
                    <li key={j}>
                      {typeof it === "string" ? it : <><b style={{ color: "var(--text-1)" }}>{it.title}</b>{it.title ? ": " : ""}{it.text}</>}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
          {report.references?.length > 0 && (
            <p style={{ fontSize: 11.5, color: "var(--text-4)" }}>
              Referencias: {report.references.map((r) => `[${r.cite_key}]${r.title ? ` ${r.title}` : ""}`).join(" · ")}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

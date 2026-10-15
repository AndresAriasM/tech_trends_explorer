"use client";

/**
 * Traductor de ecuaciones. Cuando el taller trae una ecuación ya hecha (en
 * Scopus, Google Patents o EPO), la IA la lleva a los otros sistemas con
 * variantes —equivalente, amplia, acotada— y la persona elige cuál pegar.
 */

import { useState } from "react";
import { ArrowRightLeft, Check, Copy, Loader2, Languages } from "lucide-react";

import {
  SYSTEM_LABELS,
  friendlyAiError,
  translateQuery,
  type QuerySystem,
  type QueryTranslation,
} from "@/lib/assistant";

export function AiQueryTranslator({
  onUse,
  disabled = false,
  compact = false,
}: {
  /** Pega la opción elegida en el campo del sistema correspondiente. */
  onUse?: (system: QuerySystem, query: string, concept: string) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(!compact);
  const [query, setQuery] = useState("");
  const [source, setSource] = useState<QuerySystem | "">("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryTranslation | null>(null);
  const [used, setUsed] = useState<string | null>(null);

  const traducir = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setUsed(null);
    try {
      setResult(await translateQuery(query.trim(), source || undefined));
    } catch (e) {
      setError(friendlyAiError(e));
    } finally {
      setLoading(false);
    }
  };

  const copiar = async (q: string, key: string) => {
    try {
      await navigator.clipboard.writeText(q);
      setUsed(key);
    } catch {
      /* el navegador no dejó copiar: queda el texto seleccionable */
    }
  };

  if (!open) {
    return (
      <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => setOpen(true)} disabled={disabled}>
        <ArrowRightLeft style={{ width: 14, height: 14, color: "#a78bfa" }} /> Traducir una ecuación
      </button>
    );
  }

  return (
    <div style={{ padding: 16, borderRadius: 12, border: "1px solid rgba(167,139,250,0.3)", background: "rgba(167,139,250,0.05)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Languages style={{ width: 15, height: 15, color: "#a78bfa" }} />
        <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>Traductor de ecuaciones</p>
        <p style={{ fontSize: 11.5, color: "var(--text-3)" }}>Pega una ecuación de Scopus, Google Patents o EPO</p>
      </div>
      <textarea
        className="inp"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={'TITLE-ABS-KEY("food print*" OR "3D food") AND NOT TITLE-ABS-KEY(pharmaceutical)'}
        rows={3}
        disabled={disabled || loading}
        style={{ width: "100%", fontFamily: "var(--font-mono, monospace)", fontSize: 12.5 }}
      />
      <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap", alignItems: "center" }}>
        <select className="inp" value={source} onChange={(e) => setSource(e.target.value as QuerySystem | "")} disabled={disabled || loading}>
          <option value="">Viene de: detectar automáticamente</option>
          {(Object.keys(SYSTEM_LABELS) as QuerySystem[]).map((s) => (
            <option key={s} value={s}>Viene de: {SYSTEM_LABELS[s]}</option>
          ))}
        </select>
        <button className="btn btn-primary" onClick={() => void traducir()} disabled={disabled || loading || !query.trim()}>
          {loading ? <Loader2 className="animate-spin" style={{ width: 15, height: 15 }} /> : <ArrowRightLeft style={{ width: 15, height: 15 }} />}
          Traducir
        </button>
      </div>

      {error && <p style={{ fontSize: 12.5, color: "#fca5a5", marginTop: 10 }}>{error}</p>}

      {result && (
        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 14 }}>
          <p style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "#a78bfa" }}>
            Generado por IA · {result.ai_model} · origen: {SYSTEM_LABELS[result.origen]}
            {result.concepto ? ` · ${result.concepto}` : ""}
          </p>
          {result.advertencias.length > 0 && (
            <p style={{ fontSize: 12, color: "#fbbf24" }}>{result.advertencias.join(" · ")}</p>
          )}
          {(Object.keys(SYSTEM_LABELS) as QuerySystem[])
            .filter((s) => result.traducciones[s]?.length)
            .map((s) => (
              <div key={s}>
                <p style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text-1)", marginBottom: 6 }}>
                  {SYSTEM_LABELS[s]}
                  {s === result.origen && <span style={{ color: "var(--text-4)", fontWeight: 500 }}> · variantes del original</span>}
                </p>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {result.traducciones[s]!.map((o, i) => {
                    const key = `${s}-${i}`;
                    return (
                      <div key={key} style={{ padding: 10, borderRadius: 10, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "flex-start", flexWrap: "wrap" }}>
                          <span style={{ fontSize: 10.5, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "#a78bfa" }}>
                            {o.variante}
                          </span>
                          <div style={{ display: "flex", gap: 6 }}>
                            {onUse && (
                              <button
                                className="btn btn-primary"
                                style={{ fontSize: 11.5, padding: "3px 9px" }}
                                onClick={() => {
                                  onUse(s, o.query, result.concepto);
                                  setUsed(key);
                                }}
                              >
                                {used === key ? <Check style={{ width: 13, height: 13 }} /> : null}
                                {s === "scopus" ? "Usar en Scopus" : "Usar en patentes"}
                              </button>
                            )}
                            <button className="btn btn-ghost" style={{ fontSize: 11.5, padding: "3px 9px" }} onClick={() => void copiar(o.query, key)} title="Copiar">
                              <Copy style={{ width: 13, height: 13 }} />
                            </button>
                          </div>
                        </div>
                        <code style={{ display: "block", fontSize: 12, color: "var(--text-1)", marginTop: 6, whiteSpace: "pre-wrap", wordBreak: "break-word", userSelect: "all" }}>
                          {o.query}
                        </code>
                        {o.nota && <p style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 5 }}>{o.nota}</p>}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

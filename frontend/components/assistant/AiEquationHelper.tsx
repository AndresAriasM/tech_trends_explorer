"use client";

/**
 * Generador de ecuaciones con IA. Del concepto a las ecuaciones de Scopus,
 * EPO y Google Patents, con la justificación del modelo. Nada se aplica sin
 * que la persona pulse «Usar estas ecuaciones».
 */

import { useState } from "react";
import { Loader2, Sparkles, Wand2 } from "lucide-react";

import {
  friendlyAiError,
  generateStrategy,
  scopusToGooglePatents,
  type SearchStrategy,
} from "@/lib/assistant";

export interface AppliedEquations {
  concept: string;
  scopus: string;
  epo: string;
  google: string;
  hypeTerms: string[];
}

export function AiEquationHelper({
  onApply,
  disabled = false,
  compact = false,
}: {
  onApply?: (eq: AppliedEquations) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const [open, setOpen] = useState(!compact);
  const [concept, setConcept] = useState("");
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [strategy, setStrategy] = useState<SearchStrategy | null>(null);

  const generar = async () => {
    if (!concept.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setStrategy(await generateStrategy(concept.trim(), notes.trim()));
    } catch (e) {
      setError(friendlyAiError(e));
    } finally {
      setLoading(false);
    }
  };

  if (!open) {
    return (
      <button className="btn btn-ghost" style={{ fontSize: 12 }} onClick={() => setOpen(true)} disabled={disabled}>
        <Wand2 style={{ width: 14, height: 14, color: "#a78bfa" }} /> Generar ecuaciones con IA
      </button>
    );
  }

  const google = strategy ? scopusToGooglePatents(strategy.scopus_query) : "";

  return (
    <div style={{ padding: 16, borderRadius: 12, border: "1px solid rgba(167,139,250,0.3)", background: "rgba(167,139,250,0.05)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Wand2 style={{ width: 15, height: 15, color: "#a78bfa" }} />
        <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>Asistente de ecuaciones</p>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <input
          className="inp"
          placeholder="Concepto, p. ej. hidrógeno verde"
          value={concept}
          onChange={(e) => setConcept(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void generar()}
          style={{ flex: "1 1 220px" }}
          disabled={disabled || loading}
        />
        <input
          className="inp"
          placeholder="Contexto opcional (sector, alcance, exclusiones)"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          style={{ flex: "2 1 300px" }}
          disabled={disabled || loading}
        />
        <button className="btn btn-primary" onClick={() => void generar()} disabled={disabled || loading || !concept.trim()}>
          {loading ? <Loader2 className="animate-spin" style={{ width: 15, height: 15 }} /> : <Sparkles style={{ width: 15, height: 15 }} />}
          Generar
        </button>
      </div>

      {error && <p style={{ fontSize: 12.5, color: "#fca5a5", marginTop: 10 }}>{error}</p>}

      {strategy && (
        <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
          <p style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: "0.05em", textTransform: "uppercase", color: "#a78bfa" }}>
            Generado por IA · {strategy.ai_model} · revísalo antes de correr
          </p>
          {[
            ["Scopus", strategy.scopus_query],
            ["Google Patents", google],
            ["EPO OPS", strategy.epo_query],
            ["Hype Cycle", strategy.hype_cycle_terms.join(" · ")],
          ].map(([k, v]) => (
            <div key={k}>
              <p style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>{k}</p>
              <code style={{ display: "block", fontSize: 12, color: "var(--text-1)", background: "var(--surface-2)", padding: "8px 10px", borderRadius: 8, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                {v}
              </code>
            </div>
          ))}
          {strategy.rationale && (
            <p style={{ fontSize: 12.5, color: "var(--text-2)", lineHeight: 1.6 }}>{strategy.rationale}</p>
          )}
          {onApply && (
            <div>
              <button
                className="btn btn-primary"
                onClick={() =>
                  onApply({
                    concept: strategy.concept || concept,
                    scopus: strategy.scopus_query,
                    epo: strategy.epo_query,
                    google,
                    hypeTerms: strategy.hype_cycle_terms,
                  })
                }
              >
                Usar estas ecuaciones
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

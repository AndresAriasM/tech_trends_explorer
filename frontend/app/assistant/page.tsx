"use client";

/**
 * /assistant — el asistente de IA en un solo lugar.
 *
 * El asistente vive sobre todo DENTRO de los módulos (ecuaciones en la Matriz,
 * interpretación de resultados, análisis del tablero). Esta página reúne lo que
 * no depende de un análisis concreto: generar ecuaciones, consultar el corpus
 * metodológico y ver si el proveedor está conectado.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { BookOpen, Bot, CheckCircle2, Loader2, Search, XCircle } from "lucide-react";

import { AiEquationHelper } from "@/components/assistant/AiEquationHelper";
import {
  friendlyAiError,
  providerStatus,
  searchCorpus,
  type CorpusChunk,
  type ProviderStatus,
} from "@/lib/assistant";

export default function AssistantPage() {
  const [status, setStatus] = useState<ProviderStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [buscando, setBuscando] = useState(false);
  const [chunks, setChunks] = useState<CorpusChunk[] | null>(null);
  const [errorCorpus, setErrorCorpus] = useState<string | null>(null);

  useEffect(() => {
    providerStatus()
      .then(setStatus)
      .catch((e) => setStatusError(e instanceof Error ? e.message : "Sin respuesta del backend"));
  }, []);

  const buscar = async () => {
    if (!q.trim()) return;
    setBuscando(true);
    setErrorCorpus(null);
    try {
      setChunks((await searchCorpus(q.trim())).chunks);
    } catch (e) {
      setErrorCorpus(friendlyAiError(e));
    } finally {
      setBuscando(false);
    }
  };

  return (
    <div style={{ padding: "28px 40px 64px", maxWidth: 1100, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
      <header>
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <Bot style={{ width: 22, height: 22, color: "#a78bfa" }} />
          <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--text-1)" }}>Asistente IA</h1>
        </div>
        <p style={{ fontSize: 13.5, color: "var(--text-3)", marginTop: 8, maxWidth: 760, lineHeight: 1.6 }}>
          Genera ecuaciones de búsqueda, interpreta resultados y consulta el corpus metodológico. Todo lo que
          produce queda marcado como generado por IA. También está disponible dentro de{" "}
          <Link href="/patents-papers" style={{ color: "#a78bfa" }}>Matriz Pat./Papers</Link> y del{" "}
          <Link href="/matrix-board" style={{ color: "#a78bfa" }}>Tablero de la matriz</Link>.
        </p>
      </header>

      <section className="card" style={{ padding: 16, display: "flex", alignItems: "center", gap: 10, fontSize: 13 }}>
        {status ? (
          status.configured ? (
            <>
              <CheckCircle2 style={{ width: 16, height: 16, color: "#34d399" }} />
              <span style={{ color: "var(--text-2)" }}>
                Proveedor <b style={{ color: "var(--text-1)" }}>{status.provider}</b> · modelo{" "}
                <b style={{ color: "var(--text-1)" }}>{status.model}</b> · clave configurada. Si no hay créditos, el
                primer uso lo avisará.
              </span>
            </>
          ) : (
            <>
              <XCircle style={{ width: 16, height: 16, color: "#fca5a5" }} />
              <span style={{ color: "var(--text-2)" }}>Falta la clave del proveedor (OPENAI_API_KEY en el backend).</span>
            </>
          )
        ) : (
          <span style={{ color: "var(--text-3)" }}>{statusError ?? "Comprobando el proveedor…"}</span>
        )}
      </section>

      <AiEquationHelper />

      <section className="card" style={{ padding: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
          <BookOpen style={{ width: 15, height: 15, color: "#a78bfa" }} />
          <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)" }}>Corpus metodológico (papers de Zartha)</p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <input
            className="inp"
            placeholder="p. ej. ¿cómo se calcula el punto de inflexión de una curva en S?"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void buscar()}
            style={{ flex: 1 }}
          />
          <button className="btn btn-primary" onClick={() => void buscar()} disabled={buscando || !q.trim()}>
            {buscando ? <Loader2 className="animate-spin" style={{ width: 15, height: 15 }} /> : <Search style={{ width: 15, height: 15 }} />}
            Buscar
          </button>
        </div>
        {errorCorpus && <p style={{ fontSize: 12.5, color: "#fca5a5", marginTop: 10 }}>{errorCorpus}</p>}
        {chunks && (
          <div style={{ marginTop: 14, display: "flex", flexDirection: "column", gap: 10 }}>
            {chunks.length === 0 && <p style={{ fontSize: 13, color: "var(--text-3)" }}>El corpus no tiene fragmentos suficientemente parecidos.</p>}
            {chunks.map((c, i) => (
              <div key={i} style={{ padding: 12, borderRadius: 10, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
                <p style={{ fontSize: 12.5, fontWeight: 700, color: "var(--text-1)" }}>
                  [{c.cite_key}] {c.title}{c.year ? ` (${c.year})` : ""}
                  <span style={{ color: "var(--text-4)", fontWeight: 500 }}> · similitud {c.similarity.toFixed(2)}</span>
                </p>
                <p style={{ fontSize: 12.5, color: "var(--text-2)", marginTop: 6, lineHeight: 1.6 }}>
                  {c.content.length > 600 ? `${c.content.slice(0, 600)}…` : c.content}
                </p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

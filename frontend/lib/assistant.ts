/**
 * Cliente del asistente de IA (/api/assistant).
 *
 * El asistente es una capa transversal: cada módulo lo invoca para generar
 * ecuaciones de búsqueda o interpretar resultados. Todo lo que devuelve viene
 * marcado `is_ai_generated` y la UI lo muestra así.
 *
 * Proveedor: OpenAI (backend, `OPENAI_API_KEY` / `OPENAI_MODEL`). Sin créditos
 * el backend responde 502 con el mensaje del proveedor; `friendlyAiError` lo
 * traduce para la pantalla.
 */

import { API_BASE_URL } from "@/lib/api";
import type { SCurveResult } from "@/types";

export interface AiMarker {
  is_ai_generated: true;
  ai_model: string;
  ai_provider: string;
  generated_at: string;
}

export interface SearchStrategy extends AiMarker {
  concept: string;
  scopus_query: string;
  epo_query: string;
  hype_cycle_terms: string[];
  year_from: number;
  year_to: number;
  rationale: string;
}

export interface ReportSection {
  id?: string;
  heading: string;
  kind?: string;
  text?: string;
  items?: Array<string | { title: string; text: string }>;
}

export interface ForesightReport extends AiMarker {
  concept: string;
  sections: ReportSection[];
  references: Array<{ cite_key: string; title?: string; year?: number | null }>;
  corpus_grounded: boolean;
  citation_warnings: string[];
  markdown: string;
}

export interface CorpusChunk {
  cite_key: string;
  title: string;
  year: number | null;
  content: string;
  similarity: number;
  url?: string | null;
}

export interface ProviderStatus {
  provider: string;
  model: string;
  configured: boolean;
}

async function call<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${API_BASE_URL}/api/assistant${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j?.detail) detail = String(j.detail);
    } catch {
      /* sin cuerpo JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const providerStatus = () => call<ProviderStatus>("/status");

export const generateStrategy = (concept: string, notes = "", year_from?: number, year_to?: number) =>
  call<SearchStrategy>("/search-strategy", { concept, notes, year_from, year_to });

export const generateReport = (payload: {
  concept: string;
  s_curve_papers?: SCurveResult | null;
  s_curve_patents?: SCurveResult | null;
  extra_context?: string;
}) => call<ForesightReport>("/report", payload);

export type QuerySystem = "scopus" | "google_patents" | "epo";

export interface QueryTranslation extends AiMarker {
  query: string;
  origen: QuerySystem;
  concepto: string;
  traducciones: Partial<Record<QuerySystem, Array<{ variante: string; query: string; nota: string }>>>;
  advertencias: string[];
}

export const translateQuery = (query: string, source?: QuerySystem) =>
  call<QueryTranslation>("/translate-query", { query, source: source ?? null });

export const SYSTEM_LABELS: Record<QuerySystem, string> = {
  scopus: "Scopus",
  google_patents: "Google Patents",
  epo: "EPO OPS",
};

export const searchCorpus = (query: string, k = 6) =>
  call<{ count: number; chunks: CorpusChunk[] }>("/corpus/search", { query, k });

/** Mensaje legible para errores del proveedor (créditos, clave, red). */
export function friendlyAiError(e: unknown): string {
  const msg = e instanceof Error ? e.message : String(e);
  if (/credits|insufficient_quota|429/i.test(msg))
    return "El proveedor de IA no tiene créditos. Carga créditos en OpenAI y vuelve a intentar.";
  if (/OPENAI_API_KEY|api key|401/i.test(msg))
    return "Falta configurar la clave del proveedor de IA (OPENAI_API_KEY en el backend).";
  return `El asistente de IA falló: ${msg.slice(0, 220)}`;
}

/**
 * Ecuación de Google Patents a partir de una de Scopus. Google no tiene
 * TITLE-ABS-KEY ni operadores de proximidad: se quitan los campos y W/n o
 * PRE/n pasan a AND. Es una aproximación; revísala antes de correr.
 */
export function scopusToGooglePatents(scopus: string): string {
  return scopus
    .replace(/\b(TITLE-ABS-KEY|TITLE-ABS|TITLE|ABS|KEY|AUTHKEY)\s*\(/gi, "(")
    .replace(/\s+(W|PRE)\/\d+\s+/gi, " AND ")
    .replace(/\s+AND\s+NOT\s+/gi, " -")
    .replace(/\s{2,}/g, " ")
    .trim();
}

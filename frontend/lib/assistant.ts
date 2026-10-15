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

/** Campos de Scopus y su equivalente en Google Patents (TI = título, AB = resumen). */
const FIELD_MAP: Record<string, string[]> = {
  "TITLE-ABS-KEY": ["TI", "AB"],
  "TITLE-ABS": ["TI", "AB"],
  TITLE: ["TI"],
  ABS: ["AB"],
  KEY: ["TI", "AB"],
  AUTHKEY: ["TI", "AB"],
};

/**
 * Ecuación de Google Patents a partir de una de Scopus, respetando el campo.
 *
 * Google busca por defecto en TODO el texto de la patente; Scopus en título,
 * resumen y palabras clave. Sin traducir el campo, las series no se comparan:
 * para biodiésel en 2020, título+resumen da 639 patentes y texto completo 7 352.
 * Por eso TITLE-ABS-KEY(x) → (TI=(x) OR AB=(x)) y TITLE(x) → TI=(x).
 * Proximidad W/n y PRE/n pasan a AND, AND NOT a exclusión con guion, y el
 * comodín * se quita porque Google aplica stemming por su cuenta.
 */
export function scopusToGooglePatents(scopus: string): string {
  let out = "";
  let i = 0;
  const re = /\b(TITLE-ABS-KEY|TITLE-ABS|TITLE|ABS|KEY|AUTHKEY)\s*\(/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(scopus))) {
    out += scopus.slice(i, m.index);
    // Contenido entre el paréntesis de apertura y su cierre balanceado.
    let depth = 1;
    let j = m.index + m[0].length;
    for (; j < scopus.length && depth > 0; j++) {
      if (scopus[j] === "(") depth++;
      else if (scopus[j] === ")") depth--;
    }
    const inner = scopus.slice(m.index + m[0].length, j - 1);
    const fields = FIELD_MAP[m[1].toUpperCase()] ?? ["TI", "AB"];
    out += fields.length === 1 ? `${fields[0]}=(${inner})` : `(${fields.map((f) => `${f}=(${inner})`).join(" OR ")})`;
    i = j;
    re.lastIndex = j;
  }
  out += scopus.slice(i);
  return out
    // Google no trunca con *: aplica stemming propio (TI=(packaging) también
    // encuentra package/packaged; TI=(packag*) da MENOS resultados). Se quita.
    .replace(/(\w)\*/g, "$1")
    .replace(/\s+(W|PRE)\/\d+\s+/gi, " AND ")
    .replace(/\s+AND\s+NOT\s+/gi, " -")
    .replace(/\s{2,}/g, " ")
    .trim();
}

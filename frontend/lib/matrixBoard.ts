/**
 * Cliente del tablero de la matriz patentes/papers (/api/matrix-board).
 *
 * Una sesión agrupa las tecnologías que un taller corre en vivo desde
 * /patents-papers; /matrix-board las dibuja juntas y pide el análisis de IA.
 */

import { API_BASE_URL } from "@/lib/api";
import type { SCurveComparison, SCurveResult } from "@/types";

/** Fuente de la curva científica. Hoy sólo Scopus; OpenAlex y otras vendrán. */
export type PapersSource = "scopus" | "openalex" | "semantic_scholar" | "lens";

export const PAPERS_SOURCE_LABELS: Record<string, string> = {
  scopus: "Scopus",
  openalex: "OpenAlex",
  semantic_scholar: "Semantic Scholar",
  lens: "Lens",
};

export const PATENT_SOURCE_LABELS: Record<string, string> = {
  google: "Google Patents",
  epo: "EPO OPS",
};

export type Quadrant = "maturity" | "industrial" | "disruption" | "watch";

export interface MatrixSession {
  id: string;
  name: string;
  description: string;
  reference_year: number;
  created_at: string;
  ai_generated_at: string | null;
  n_runs?: number;
  ai_analysis?: MatrixAnalysis | null;
}

interface SeriesSummary {
  total: number | null;
  yearly_data: Array<{ year: number; annual: number; cumulative: number }>;
  peak_annual: number | null;
  peak_annual_year: number | null;
  phase: { phase?: string; saturation_pct?: number | null } | null;
  source_label?: string | null;
}

export interface MatrixRun {
  id: string;
  session_id: string;
  label: string;
  scopus_query: string;
  papers_source: PapersSource;
  patents_query: string;
  patent_source: "epo" | "google";
  year_from: number;
  year_to: number;
  papers: SeriesSummary;
  patents: SeriesSummary;
  papers_inflection: number | null;
  patents_inflection: number | null;
  papers_range: [number, number] | null;
  patents_range: [number, number] | null;
  quadrant: Quadrant | null;
  quadrant_label: string | null;
  robust: boolean | null;
  created_at: string;
}

export interface MatrixAnalysis {
  resumen: string;
  por_tecnologia: Array<{ label: string; cuadrante: string; lectura: string; alerta: string | null }>;
  por_cuadrante: Array<{ cuadrante: string; tecnologias: string[]; recomendacion: string }>;
  preguntas_para_el_panel: string[];
  citas: string[];
  fuentes: Array<{ cite_key: string; title: string; year: number | null; url: string | null }>;
  is_ai_generated: true;
  ai_model: string;
  generated_at: string;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}/api/matrix-board${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* sin cuerpo JSON */
    }
    throw new Error(detail);
  }
  return (res.status === 204 ? undefined : await res.json()) as T;
}

export const listSessions = () => call<MatrixSession[]>("/sessions");

export const createSession = (name: string, reference_year = 2025) =>
  call<MatrixSession>("/sessions", { method: "POST", body: JSON.stringify({ name, reference_year }) });

export const deleteSession = (id: string) => call<void>(`/sessions/${id}`, { method: "DELETE" });

export const getSession = (id: string) =>
  call<{ session: MatrixSession; runs: MatrixRun[] }>(`/sessions/${id}`);

export const saveRun = (
  sessionId: string,
  body: {
    label: string;
    scopus_query: string;
    papers_source: PapersSource;
    patents_query: string;
    patent_source: "epo" | "google";
    year_from: number;
    year_to: number;
    papers: SCurveResult;
    patents: SCurveResult;
    comparison: SCurveComparison | null;
  }
) =>
  call<Pick<MatrixRun, "id" | "papers_inflection" | "patents_inflection" | "quadrant" | "quadrant_label" | "robust">>(
    `/sessions/${sessionId}/runs`,
    { method: "POST", body: JSON.stringify(body) }
  );

export const renameRun = (id: string, label: string) =>
  call<{ id: string; label: string }>(`/runs/${id}`, { method: "PATCH", body: JSON.stringify({ label }) });

export const deleteRun = (id: string) => call<void>(`/runs/${id}`, { method: "DELETE" });

export const analyzeSession = (id: string) =>
  call<MatrixAnalysis>(`/sessions/${id}/analysis`, { method: "POST" });

export const QUADRANTS: Record<Quadrant, { label: string; color: string; tint: string; recomendacion: string }> = {
  watch: {
    label: "Vigilancia Tecnológica y Apuestas a Futuro",
    color: "#60a5fa",
    tint: "rgba(96,165,250,0.10)",
    recomendacion: "Vigilancia, investigación exploratoria, pilotos pequeños.",
  },
  disruption: {
    label: "Oportunidad de Disrupción",
    color: "#a78bfa",
    tint: "rgba(167,139,250,0.10)",
    recomendacion: "Transferencia tecnológica, prototipos, protección de PI.",
  },
  maturity: {
    label: "Madurez y Optimización",
    color: "#34d399",
    tint: "rgba(52,211,153,0.10)",
    recomendacion: "Eficiencia de costos, innovación incremental y de proceso.",
  },
  industrial: {
    label: "Explotación Industrial y Ciencia Aplicada",
    color: "#fb923c",
    tint: "rgba(251,146,60,0.10)",
    recomendacion: "Escalar lo patentado, investigación aplicada con la industria.",
  },
};

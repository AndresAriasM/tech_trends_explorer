/**
 * Cliente HTTP del módulo Delphi.
 *
 * Vive aquí y no en `lib/api.ts` porque el router del backend aún no está
 * registrado en `main.py`: mientras tanto, el módulo se mantiene aparte y no
 * contamina el cliente general.
 */

import type {
  DelphiParticipant,
  DelphiRound,
  DelphiStudy,
  DelphiVariable,
  SchemaCheck,
  SpearmanResult,
  StudyReport,
  VariableMap,
  VariableRoundStats,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/** Prefijo con el que se montará el router cuando se registre en main.py. */
export const DELPHI_PREFIX = "/api/delphi";

export class DelphiApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    /** 503 = la migración 001 no está aplicada; es un caso esperado. */
    readonly schemaMissing = false
  ) {
    super(message);
    this.name = "DelphiApiError";
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${DELPHI_PREFIX}${path}`, {
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
      ...init,
    });
  } catch {
    throw new DelphiApiError(
      `No se pudo contactar el backend en ${API_BASE}.`,
      0
    );
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* el cuerpo no era JSON */
    }
    throw new DelphiApiError(detail, res.status, res.status === 503);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/* ── Diagnóstico ─────────────────────────────────────────────── */

export const checkSchema = () => req<SchemaCheck>("/schema-check");

/* ── Estudios ────────────────────────────────────────────────── */

export const listStudies = () => req<DelphiStudy[]>("/studies");

export const getStudy = (id: string) =>
  req<
    DelphiStudy & {
      variables: DelphiVariable[];
      rounds: DelphiRound[];
      participants_summary: {
        n_invited: number;
        n_registered: number;
        n_dropped: number;
        mean_k: number | null;
      };
    }
  >(`/studies/${id}`);

export const createStudy = (payload: {
  title: string;
  description?: string;
  owner_id?: string;
  horizon_year?: number | null;
  scale_type?: string;
  dichotomy_threshold?: number;
  max_rounds?: number;
}) =>
  req<DelphiStudy>("/studies", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const updateStudy = (id: string, payload: Partial<DelphiStudy>) =>
  req<DelphiStudy>(`/studies/${id}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });

export const deleteStudy = (id: string) =>
  req<void>(`/studies/${id}`, { method: "DELETE" });

/* ── Panel ───────────────────────────────────────────────────── */

export const listParticipants = (id: string) =>
  req<DelphiParticipant[]>(`/studies/${id}/participants`);

/* ── Análisis (§9) ───────────────────────────────────────────── */

/** `threshold` reanaliza con otro umbral de dicotomía sin migrar nada (§5.3). */
export const getStudyStats = (id: string, threshold?: number) =>
  req<StudyReport>(
    `/studies/${id}/stats${threshold != null ? `?threshold=${threshold}` : ""}`
  );

export const getRoundStats = (id: string, roundNumber: number) =>
  req<{
    round_number: number;
    poblacion: "agregado";
    n_respondents: number;
    n_population: number;
    variables: VariableRoundStats[];
    by_family: { family: string; variables: VariableRoundStats[] }[];
    spearman: SpearmanResult;
  }>(`/studies/${id}/rounds/${roundNumber}/stats`);

export const getStudyMap = (id: string, threshold?: number) =>
  req<VariableMap>(
    `/studies/${id}/map${threshold != null ? `?threshold=${threshold}` : ""}`
  );

/**
 * HTTP client + SSE helpers for the Tech Trends Explorer backend.
 *
 * Reconstructed: the original file was never committed because the root
 * .gitignore ignores `lib/`. See docs/MIGRATION_ROADMAP.md.
 */

import type {
  AnalyzeRequest,
  EPOStatus,
  SCurveComparison,
  SCurvePatentsRequest,
  AnalyzeResponse,
  Category,
  HypeCycleQuery,
  SCurveProgress,
  SCurveRequest,
  SCurveResult,
  TaskProgress,
} from "@/types";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* body was not JSON — keep the status line */
    }
    throw new Error(detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/* ── Categories ─────────────────────────────────────────────── */

export function getCategories(): Promise<Category[]> {
  return request<Category[]>("/api/categories");
}

export function createCategory(payload: {
  name: string;
  description?: string;
  parent_category_id?: string;
}): Promise<Category> {
  return request<Category>("/api/categories", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateCategory(
  categoryId: string,
  payload: { name?: string; description?: string }
): Promise<Category> {
  return request<Category>(`/api/categories/${categoryId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteCategory(categoryId: string): Promise<void> {
  return request<void>(`/api/categories/${categoryId}`, { method: "DELETE" });
}

/* ── Hype Cycle ─────────────────────────────────────────────── */

export function getHypeCycleHistory(
  categoryId?: string
): Promise<HypeCycleQuery[]> {
  const qs = categoryId
    ? `?category_id=${encodeURIComponent(categoryId)}`
    : "";
  return request<HypeCycleQuery[]>(`/api/hype-cycle/history${qs}`);
}

export function getHypeCycleQuery(queryId: string): Promise<HypeCycleQuery> {
  return request<HypeCycleQuery>(`/api/hype-cycle/queries/${queryId}`);
}

export function startHypeCycleAnalysis(
  payload: AnalyzeRequest
): Promise<AnalyzeResponse> {
  return request<AnalyzeResponse>("/api/hype-cycle/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/**
 * Subscribe to a task's progress stream.
 * Returns a cleanup function that closes the connection.
 */
export function streamTaskProgress(
  taskId: string,
  onProgress: (p: TaskProgress) => void,
  onComplete: (result: HypeCycleQuery | undefined) => void,
  onError: (message: string) => void
): () => void {
  const source = new EventSource(`${API_BASE_URL}/api/tasks/${taskId}/stream`);
  let settled = false;

  source.onmessage = (event) => {
    let payload: TaskProgress;
    try {
      payload = JSON.parse(event.data) as TaskProgress;
    } catch {
      return;
    }

    onProgress(payload);

    if (payload.status === "completed") {
      settled = true;
      source.close();
      onComplete(payload.result);
    } else if (payload.status === "failed") {
      settled = true;
      source.close();
      onError(payload.error ?? payload.message ?? "El análisis falló");
    }
  };

  source.onerror = () => {
    if (settled) return;
    settled = true;
    source.close();
    onError("Se perdió la conexión con el servidor");
  };

  return () => {
    settled = true;
    source.close();
  };
}

/* ── S-Curve ────────────────────────────────────────────────── */

/**
 * Run a streaming S-curve analysis. The endpoint is a POST that replies with
 * text/event-stream, so EventSource (GET-only) cannot be used here.
 * Returns a cancel function that aborts the request.
 */
export function streamSCurveAnalysis(
  payload: SCurveRequest,
  onProgress: (evt: SCurveProgress) => void,
  onComplete: (result: SCurveResult | undefined) => void,
  onError: (message: string) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/s-curve/analyze`, {
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

        // SSE frames are separated by a blank line.
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
            onError(evt.message ?? "Error en el análisis");
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

/** Fit a sigmoid to pre-aggregated yearly counts (no Scopus call). */
export function analyzeSCurveLocal(
  yearlyCounts: Record<number, number>,
  title: string
): Promise<SCurveResult> {
  return request<SCurveResult>("/api/s-curve/analyze-local", {
    method: "POST",
    body: JSON.stringify({ yearly_counts: yearlyCounts, title }),
  });
}


/* ── Patentes (EPO OPS) y matriz papers ↔ patentes ─────────── */

/**
 * Lector SSE genérico: consume un endpoint que responde text/event-stream
 * con eventos {status, progress, message} y un último {status:"completed"}.
 *
 * EventSource no sirve porque estos endpoints son POST.
 */
function streamSCurveEndpoint(
  path: string,
  body: unknown,
  onProgress: (evt: SCurveProgress) => void,
  onComplete: (result: SCurveResult | undefined) => void,
  onError: (message: string) => void
): () => void {
  const controller = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${API_BASE_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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
            onError(evt.message ?? "Error en el análisis");
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

/** Curva de patentes vía EPO OPS. Las credenciales viven en el servidor. */
export function streamPatentsAnalysis(
  payload: SCurvePatentsRequest,
  onProgress: (evt: SCurveProgress) => void,
  onComplete: (result: SCurveResult | undefined) => void,
  onError: (message: string) => void
): () => void {
  return streamSCurveEndpoint(
    "/api/s-curve/analyze-patents",
    payload,
    onProgress,
    onComplete,
    onError
  );
}

/**
 * Curva de publicaciones vía Scopus, por conteo anual.
 *
 * Una petición por año leyendo sólo el total, en vez de paginar el corpus de
 * 25 en 25: para biodiésel son 27 peticiones y ~50 s frente a 2.518
 * peticiones y cerca de una hora. La clave es del servidor.
 *
 * Si algún día hace falta el detalle de cada publicación (países, títulos),
 * ese es `streamSCurveAnalysis`, que sí descarga los registros.
 */
export function streamPapersAnalysis(
  payload: { query: string; year_from?: number; year_to?: number },
  onProgress: (evt: SCurveProgress) => void,
  onComplete: (result: SCurveResult | undefined) => void,
  onError: (message: string) => void
): () => void {
  return streamSCurveEndpoint(
    "/api/s-curve/analyze-papers",
    payload,
    onProgress,
    onComplete,
    onError
  );
}

/** Desfase ciencia → tecnología entre dos análisis ya calculados. */
export function comparePapersPatents(
  papers: SCurveResult,
  patents: SCurveResult
): Promise<SCurveComparison> {
  return request<SCurveComparison>("/api/s-curve/compare", {
    method: "POST",
    body: JSON.stringify({ papers, patents }),
  });
}

/** ¿Están configuradas las credenciales de EPO OPS en el servidor? */
export function getEPOStatus(): Promise<EPOStatus> {
  return request<EPOStatus>("/api/s-curve/epo/status");
}

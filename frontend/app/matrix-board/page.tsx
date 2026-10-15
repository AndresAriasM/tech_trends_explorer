"use client";

/**
 * /matrix-board — tablero de la matriz patentes/papers de una sesión de taller.
 *
 * Las tecnologías se corren y guardan desde /patents-papers; aquí se ven
 * juntas con la convención de Zartha (X = inflexión de papers, Y = inflexión
 * de patentes, corte en el año de referencia) y se pide el análisis de IA.
 * La página se refresca sola cada pocos segundos para que, en el taller,
 * lo que se guarda en otra pestaña aparezca en la proyección.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Bot, LayoutGrid, Loader2, Plus, RefreshCw, Sparkles, Trash2 } from "lucide-react";

import {
  QUADRANTS,
  analyzeSession,
  createSession,
  deleteRun,
  getSession,
  listSessions,
  type MatrixAnalysis,
  type MatrixRun,
  type MatrixSession,
  type Quadrant,
} from "@/lib/matrixBoard";

const REFRESH_MS = 8000;

export default function MatrixBoardPage() {
  const [sessions, setSessions] = useState<MatrixSession[]>([]);
  const [sessionId, setSessionId] = useState<string>("");
  const [session, setSession] = useState<MatrixSession | null>(null);
  const [runs, setRuns] = useState<MatrixRun[]>([]);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<MatrixAnalysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);

  const loadSessions = useCallback(async () => {
    try {
      const list = await listSessions();
      setSessions(list);
      setSessionId((cur) => cur || list[0]?.id || "");
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudieron cargar las sesiones");
    }
  }, []);

  const loadSession = useCallback(async (id: string) => {
    if (!id) return;
    try {
      const data = await getSession(id);
      setSession(data.session);
      setRuns(data.runs);
      setAnalysis((cur) => cur ?? (data.session.ai_analysis as MatrixAnalysis | null) ?? null);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo cargar la sesión");
    }
  }, []);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    setAnalysis(null);
    void loadSession(sessionId);
    const t = setInterval(() => void loadSession(sessionId), REFRESH_MS);
    return () => clearInterval(t);
  }, [sessionId, loadSession]);

  const crear = async () => {
    if (!newName.trim()) return;
    try {
      const s = await createSession(newName.trim());
      setNewName("");
      await loadSessions();
      setSessionId(s.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "No se pudo crear la sesión");
    }
  };

  const quitar = async (id: string) => {
    await deleteRun(id);
    void loadSession(sessionId);
  };

  const analizar = async () => {
    if (!sessionId) return;
    setAnalyzing(true);
    setError(null);
    try {
      setAnalysis(await analyzeSession(sessionId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "El análisis de IA falló");
    } finally {
      setAnalyzing(false);
    }
  };

  const ref = session?.reference_year ?? 2025;

  return (
    <div style={{ padding: "28px 40px 64px", maxWidth: 1320, margin: "0 auto" }}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 20, flexWrap: "wrap", marginBottom: 22 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
            <LayoutGrid style={{ width: 22, height: 22, color: "#38bdf8" }} />
            <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--text-1)" }}>Tablero de la matriz</h1>
          </div>
          <p style={{ fontSize: 13.5, color: "var(--text-3)", marginTop: 8, maxWidth: 720, lineHeight: 1.6 }}>
            Todas las tecnologías de la sesión en la matriz patentes/papers. Cada punto es la mediana
            de la inflexión de cinco modelos de curva en S; las barras muestran el rango entre modelos.
            Se guardan desde <strong style={{ color: "var(--text-2)" }}>Matriz Pat./Papers</strong> y aparecen aquí solas.
          </p>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <select
            className="inp"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            style={{ minWidth: 240 }}
          >
            {sessions.length === 0 && <option value="">Sin sesiones</option>}
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name} · {s.n_runs ?? 0} tecnologías
              </option>
            ))}
          </select>
          <input
            className="inp"
            placeholder="Nueva sesión, p. ej. Taller de hoy"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && void crear()}
            style={{ width: 230 }}
          />
          <button className="btn btn-ghost" onClick={() => void crear()} disabled={!newName.trim()}>
            <Plus style={{ width: 15, height: 15 }} /> Crear
          </button>
          <button className="btn btn-ghost" onClick={() => void loadSession(sessionId)} title="Recargar">
            <RefreshCw style={{ width: 15, height: 15 }} />
          </button>
        </div>
      </header>

      {error && (
        <div className="card" style={{ padding: 14, marginBottom: 18, display: "flex", gap: 10, color: "#fca5a5", fontSize: 13 }}>
          <AlertTriangle style={{ width: 16, height: 16, flexShrink: 0 }} /> {error}
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 360px", gap: 20, alignItems: "start" }}>
        <section className="card" style={{ padding: 20 }}>
          {runs.length === 0 ? (
            <p style={{ padding: "80px 20px", textAlign: "center", color: "var(--text-3)", fontSize: 14, lineHeight: 1.7 }}>
              Esta sesión todavía no tiene tecnologías.<br />
              Ve a <strong style={{ color: "var(--text-2)" }}>Matriz Pat./Papers</strong>, analiza una tecnología y pulsa
              «Guardar en la sesión».
            </p>
          ) : (
            <BoardChart runs={runs} refYear={ref} />
          )}
        </section>

        <aside style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {(Object.keys(QUADRANTS) as Quadrant[]).map((q) => {
            const enQ = runs.filter((r) => r.quadrant === q);
            return (
              <div key={q} className="card" style={{ padding: "12px 14px", borderLeft: `3px solid ${QUADRANTS[q].color}` }}>
                <p style={{ fontSize: 12.5, fontWeight: 700, color: QUADRANTS[q].color }}>{QUADRANTS[q].label}</p>
                <p style={{ fontSize: 11.5, color: "var(--text-3)", margin: "3px 0 6px" }}>{QUADRANTS[q].recomendacion}</p>
                <p style={{ fontSize: 12.5, color: "var(--text-1)" }}>
                  {enQ.length ? enQ.map((r) => `T${runs.indexOf(r) + 1} ${r.label}`).join(" · ") : "—"}
                </p>
              </div>
            );
          })}
        </aside>
      </div>

      {runs.length > 0 && (
        <section className="card" style={{ padding: 20, marginTop: 20, overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5, fontVariantNumeric: "tabular-nums" }}>
            <thead>
              <tr style={{ color: "var(--text-3)", textAlign: "left" }}>
                {["", "Tecnología", "Papers", "Patentes", "Fuente", "Infl. papers", "Infl. patentes", "Cuadrante", ""].map((h, i) => (
                  <th key={i} style={{ padding: "6px 8px", fontWeight: 600, borderBottom: "1px solid var(--border-2)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {runs.map((r, i) => (
                <tr key={r.id} style={{ borderBottom: "1px solid var(--border)", color: "var(--text-2)" }}>
                  <td style={{ padding: "8px", fontWeight: 700, color: "var(--text-1)" }}>T{i + 1}</td>
                  <td style={{ padding: "8px", color: "var(--text-1)" }}>{r.label}</td>
                  <td style={{ padding: "8px" }}>{r.papers.total?.toLocaleString() ?? "—"}</td>
                  <td style={{ padding: "8px" }}>{r.patents.total?.toLocaleString() ?? "—"}</td>
                  <td style={{ padding: "8px" }}>{r.patent_source === "google" ? "Google Patents" : "EPO OPS"}</td>
                  <td style={{ padding: "8px" }}>{fmtInfl(r.papers_inflection, r.papers_range)}</td>
                  <td style={{ padding: "8px" }}>{fmtInfl(r.patents_inflection, r.patents_range)}</td>
                  <td style={{ padding: "8px", color: r.quadrant ? QUADRANTS[r.quadrant].color : undefined }}>
                    {r.quadrant ? QUADRANTS[r.quadrant].label : "Sin inflexión estimable"}
                    {r.robust === false && <span style={{ color: "var(--text-4)" }}> · depende del modelo</span>}
                  </td>
                  <td style={{ padding: "8px", textAlign: "right" }}>
                    <button className="btn btn-ghost" style={{ padding: "4px 8px" }} onClick={() => void quitar(r.id)} title="Quitar de la sesión">
                      <Trash2 style={{ width: 13, height: 13 }} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <section className="card" style={{ padding: 22, marginTop: 20, border: "1px solid rgba(167,139,250,0.3)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <Bot style={{ width: 18, height: 18, color: "#a78bfa" }} />
            <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-1)" }}>Análisis prospectivo con IA</h2>
          </div>
          <button className="btn btn-primary" onClick={() => void analizar()} disabled={analyzing || runs.length === 0}>
            {analyzing ? <Loader2 className="animate-spin" style={{ width: 15, height: 15 }} /> : <Sparkles style={{ width: 15, height: 15 }} />}
            {analysis ? "Regenerar análisis" : "Generar análisis"}
          </button>
        </div>

        {!analysis && !analyzing && (
          <p style={{ fontSize: 13, color: "var(--text-3)", marginTop: 12 }}>
            Lee las cifras de todas las tecnologías de la sesión y propone una lectura por cuadrante y preguntas para el panel.
          </p>
        )}

        {analysis && <AnalysisView a={analysis} />}
      </section>
    </div>
  );
}

function fmtInfl(v: number | null, r: [number, number] | null) {
  if (v === null) return "—";
  return r ? `${Math.round(v)} (${Math.round(r[0])}–${Math.round(r[1])})` : `${Math.round(v)}`;
}

/* ── Gráfica ─────────────────────────────────────────────────────────────── */

function BoardChart({ runs, refYear }: { runs: MatrixRun[]; refYear: number }) {
  const W = 900, H = 560, m = { l: 64, r: 24, t: 20, b: 52 };
  const placed = runs.filter((r) => r.papers_inflection !== null && r.patents_inflection !== null);

  const dom = useMemo(() => {
    const xs = placed.map((r) => r.papers_inflection as number);
    const ys = placed.map((r) => r.patents_inflection as number);
    const pad = (lo: number, hi: number): [number, number] => {
      const a = Math.floor(Math.min(lo, refYear - 10) / 5) * 5;
      const b = Math.ceil(Math.max(hi, refYear + 10) / 5) * 5;
      return [a, Math.min(b, 2100)];
    };
    return {
      x: pad(Math.min(...xs, refYear) - 3, Math.max(...xs, refYear) + 5),
      y: pad(Math.min(...ys, refYear) - 3, Math.max(...ys, refYear) + 5),
    };
  }, [placed, refYear]);

  const sx = (v: number) => m.l + ((Math.min(Math.max(v, dom.x[0]), dom.x[1]) - dom.x[0]) / (dom.x[1] - dom.x[0])) * (W - m.l - m.r);
  const sy = (v: number) => H - m.b - ((Math.min(Math.max(v, dom.y[0]), dom.y[1]) - dom.y[0]) / (dom.y[1] - dom.y[0])) * (H - m.t - m.b);
  const ticks = (d: [number, number]) => {
    const step = d[1] - d[0] > 60 ? 20 : d[1] - d[0] > 30 ? 10 : 5;
    const out: number[] = [];
    for (let v = Math.ceil(d[0] / step) * step; v <= d[1]; v += step) out.push(v);
    return out;
  };
  const cx = sx(refYear), cy = sy(refYear);
  const quad: Array<[Quadrant, number, number, number, number]> = [
    ["disruption", m.l, m.t, cx - m.l, cy - m.t],
    ["watch", cx, m.t, W - m.r - cx, cy - m.t],
    ["maturity", m.l, cy, cx - m.l, H - m.b - cy],
    ["industrial", cx, cy, W - m.r - cx, H - m.b - cy],
  ];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} role="img" aria-label="Matriz patentes/papers de la sesión">
      {quad.map(([q, x, y, w, h]) => (
        <rect key={q} x={x} y={y} width={Math.max(w, 0)} height={Math.max(h, 0)} fill={QUADRANTS[q].tint} />
      ))}
      {ticks(dom.x).map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} stroke="var(--border)" />
          <text x={sx(t)} y={H - m.b + 18} textAnchor="middle" fontSize="12" fill="var(--text-3)">{t}</text>
        </g>
      ))}
      {ticks(dom.y).map((t) => (
        <g key={`y${t}`}>
          <line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} stroke="var(--border)" />
          <text x={m.l - 8} y={sy(t) + 4} textAnchor="end" fontSize="12" fill="var(--text-3)">{t}</text>
        </g>
      ))}
      <line x1={cx} x2={cx} y1={m.t} y2={H - m.b} stroke="var(--text-2)" strokeDasharray="6 5" strokeWidth={1.5} />
      <line x1={m.l} x2={W - m.r} y1={cy} y2={cy} stroke="var(--text-2)" strokeDasharray="6 5" strokeWidth={1.5} />
      <text x={(m.l + W - m.r) / 2} y={H - 10} textAnchor="middle" fontSize="12.5" fill="var(--text-3)">
        Año de inflexión de papers (mediana de 5 modelos) →
      </text>
      <text x={16} y={(m.t + H - m.b) / 2} textAnchor="middle" fontSize="12.5" fill="var(--text-3)" transform={`rotate(-90 16 ${(m.t + H - m.b) / 2})`}>
        Año de inflexión de patentes →
      </text>
      <text x={W - m.r - 8} y={m.t + 18} textAnchor="end" fontSize="12" fontWeight="700" fill={QUADRANTS.watch.color}>Vigilancia · apuestas a futuro</text>
      <text x={m.l + 8} y={m.t + 18} fontSize="12" fontWeight="700" fill={QUADRANTS.disruption.color}>Disrupción</text>
      <text x={m.l + 8} y={H - m.b - 10} fontSize="12" fontWeight="700" fill={QUADRANTS.maturity.color}>Madurez</text>
      <text x={W - m.r - 8} y={H - m.b - 10} textAnchor="end" fontSize="12" fontWeight="700" fill={QUADRANTS.industrial.color}>Explotación industrial</text>

      {placed.map((r) => {
        const i = runs.indexOf(r);
        const x = sx(r.papers_inflection as number), y = sy(r.patents_inflection as number);
        const color = r.quadrant ? QUADRANTS[r.quadrant].color : "#94a3b8";
        const pr = r.papers_range, tr = r.patents_range;
        return (
          <g key={r.id}>
            <title>
              {`T${i + 1} ${r.label}\nPapers: ${fmtInfl(r.papers_inflection, pr)}\nPatentes: ${fmtInfl(r.patents_inflection, tr)}\n${r.quadrant_label ?? ""}${r.robust === false ? " (depende del modelo)" : ""}`}
            </title>
            {pr && <line x1={sx(pr[0])} x2={sx(pr[1])} y1={y} y2={y} stroke={color} strokeWidth={2} opacity={0.5} />}
            {tr && <line x1={x} x2={x} y1={sy(tr[0])} y2={sy(tr[1])} stroke={color} strokeWidth={2} opacity={0.5} />}
            <circle cx={x} cy={y} r={11} fill={color} stroke="var(--surface-2)" strokeWidth={3} />
            <text x={x} y={y + 4} textAnchor="middle" fontSize="10" fontWeight="800" fill="#0b1220">{i + 1}</text>
            <text x={x + 16} y={y - 10} fontSize="13" fontWeight="700" fill="var(--text-1)">{r.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── Análisis ────────────────────────────────────────────────────────────── */

function AnalysisView({ a }: { a: MatrixAnalysis }) {
  return (
    <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 16 }}>
      <p style={{ fontSize: 11, color: "#a78bfa", fontWeight: 600, letterSpacing: "0.04em", textTransform: "uppercase" }}>
        Generado por IA · {a.ai_model} · {new Date(a.generated_at).toLocaleString()} · no revisado por el panel
      </p>
      <p style={{ fontSize: 14.5, color: "var(--text-1)", lineHeight: 1.7, maxWidth: 900 }}>{a.resumen}</p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 12 }}>
        {a.por_tecnologia?.map((t) => (
          <div key={t.label} style={{ padding: 14, borderRadius: 10, background: "var(--surface-2)", border: "1px solid var(--border)" }}>
            <p style={{ fontSize: 13.5, fontWeight: 700, color: "var(--text-1)" }}>{t.label}</p>
            <p style={{ fontSize: 12.5, color: "var(--text-2)", marginTop: 6, lineHeight: 1.6 }}>{t.lectura}</p>
            {t.alerta && (
              <p style={{ fontSize: 12, color: "#fbbf24", marginTop: 6, display: "flex", gap: 6 }}>
                <AlertTriangle style={{ width: 13, height: 13, flexShrink: 0, marginTop: 2 }} /> {t.alerta}
              </p>
            )}
          </div>
        ))}
      </div>

      {a.preguntas_para_el_panel?.length > 0 && (
        <div>
          <p style={{ fontSize: 13, fontWeight: 700, color: "var(--text-1)", marginBottom: 6 }}>Preguntas para el panel</p>
          <ol style={{ paddingLeft: 20, color: "var(--text-2)", fontSize: 13, lineHeight: 1.8 }}>
            {a.preguntas_para_el_panel.map((q) => <li key={q}>{q}</li>)}
          </ol>
        </div>
      )}

      {a.fuentes?.length > 0 && (
        <p style={{ fontSize: 11.5, color: "var(--text-4)" }}>
          Fuentes del corpus: {a.fuentes.map((f) => `[${f.cite_key}] ${f.title}${f.year ? ` (${f.year})` : ""}`).join(" · ")}
        </p>
      )}
    </div>
  );
}

"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import {
  Maximize2, Minimize2, RefreshCw, TrendingUp, TrendingDown,
  Minus, ChevronDown, ChevronUp, Sparkles, AlertTriangle, Star,
  BarChart3, Brain, ArrowRight,
} from "lucide-react";
import type { WorkshopSynthesis, VariableSynthesis, RatingLabel } from "@/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/* ── Paleta de ratings ─────────────────────────────────────────────────────── */
const R = {
  Malo:      { text: "text-red-300",     bg: "bg-red-500/15",     bar: "bg-red-500",     border: "border-red-500/30",    glow: "shadow-red-500/20"     },
  Regular:   { text: "text-amber-300",   bg: "bg-amber-500/15",   bar: "bg-amber-400",   border: "border-amber-500/30",  glow: "shadow-amber-500/20"   },
  Bueno:     { text: "text-sky-300",     bg: "bg-sky-500/15",     bar: "bg-sky-400",     border: "border-sky-500/30",    glow: "shadow-sky-500/20"     },
  Excelente: { text: "text-emerald-300", bg: "bg-emerald-500/15", bar: "bg-emerald-400", border: "border-emerald-500/30",glow: "shadow-emerald-500/20" },
} as const;

const LABELS: RatingLabel[] = ["Malo", "Regular", "Bueno", "Excelente"];

const CAT_META: Record<string, { icon: string; color: string; accent: string }> = {
  "Económica / Financiera":       { icon: "💰", color: "text-amber-300",  accent: "bg-amber-500/10 border-amber-500/20"   },
  "Social / Geopolítica":         { icon: "🌎", color: "text-rose-300",   accent: "bg-rose-500/10 border-rose-500/20"     },
  "Digital / Tecnológica":        { icon: "⚡", color: "text-blue-300",   accent: "bg-blue-500/10 border-blue-500/20"     },
  "Institucional / Colaboración": { icon: "🤝", color: "text-purple-300", accent: "bg-purple-500/10 border-purple-500/20" },
  "Académica / Investigación":    { icon: "🎓", color: "text-teal-300",   accent: "bg-teal-500/10 border-teal-500/20"     },
  "Regulatoria / Normativa":      { icon: "📜", color: "text-orange-300", accent: "bg-orange-500/10 border-orange-500/20" },
};

function avgLabel(avg: number | null): RatingLabel | null {
  if (avg === null) return null;
  if (avg < 1.5) return "Malo";
  if (avg < 2.5) return "Regular";
  if (avg < 3.5) return "Bueno";
  return "Excelente";
}

function ratingName(n: number): RatingLabel {
  const map: Record<number, RatingLabel> = { 1: "Malo", 2: "Regular", 3: "Bueno", 4: "Excelente" };
  return map[n] ?? "Regular";
}

/* ── Score bar (1–4 scale) ──────────────────────────────────────────────────── */
function ScoreBar({ score }: { score: number | null }) {
  if (score === null) return <span className="text-xs text-[var(--text-4)]">—</span>;
  const pct   = ((score - 1) / 3) * 100;
  const label = avgLabel(score);
  const color = label ? R[label].bar : "bg-[var(--text-4)]";
  return (
    <div className="flex items-center gap-2 w-full">
      <div className="flex-1 h-1.5 bg-[var(--surface-3)] rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color} transition-all duration-500`}
             style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[11px] font-mono font-bold text-[var(--text-2)] w-7 text-right">
        {score.toFixed(1)}
      </span>
    </div>
  );
}

/* ── Stacked distribution bar ──────────────────────────────────────────────── */
function StackedBar({ dist, total }: { dist: Record<string, number>; total: number }) {
  return (
    <div className="flex rounded-full overflow-hidden h-2.5 w-full gap-[1px]">
      {LABELS.map((l) => {
        const count = dist[l] ?? 0;
        const pct   = total > 0 ? (count / total) * 100 : 0;
        return pct > 0 ? (
          <div key={l} title={`${l}: ${count} (${Math.round(pct)}%)`}
               className={`h-full ${R[l].bar} transition-all duration-500`}
               style={{ width: `${pct}%` }} />
        ) : null;
      })}
    </div>
  );
}

/* ── Rating chip ──────────────────────────────────────────────────────────────  */
function RatingChip({ label, faint = false, large = false }: {
  label: RatingLabel; faint?: boolean; large?: boolean;
}) {
  const s = R[label];
  return (
    <span className={`inline-block rounded font-bold border ${
      large ? "text-xs px-2.5 py-1" : "text-[10px] px-2 py-0.5"
    } ${faint ? `${s.bg} ${s.text} ${s.border} opacity-70` : `${s.bg} ${s.text} ${s.border}`}`}>
      {label}
    </span>
  );
}

/* ── Score pill (hero) ──────────────────────────────────────────────────────── */
function ScorePill({ label, score, caption, highlight = false }: {
  label: string; score: number | null; caption?: RatingLabel | null; highlight?: boolean;
}) {
  const style = caption ? R[caption] : null;
  return (
    <div className={`rounded-2xl border px-5 py-4 text-center min-w-[120px] ${
      highlight
        ? "border-[var(--accent-border)] bg-gradient-to-b from-[var(--accent-sub)] to-[var(--surface-2)]"
        : "border-[var(--border-2)] bg-[var(--surface-3)]"
    }`}>
      <p className="text-[10px] text-[var(--text-3)] mb-1 font-mono uppercase tracking-widest">{label}</p>
      {score !== null ? (
        <>
          <p className={`text-4xl font-black leading-none mb-1 ${style?.text ?? "text-[var(--text-1)]"}`}>
            {score.toFixed(1)}
          </p>
          <p className="text-[10px] text-[var(--text-4)] font-mono">/4.0</p>
          {caption && <p className={`text-xs font-bold mt-1.5 ${style?.text}`}>{caption}</p>}
        </>
      ) : (
        <p className="text-3xl font-black text-[var(--text-4)]">—</p>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════ */

export default function ResultsPage() {
  const { code } = useParams<{ code: string }>();
  const [synthesis, setSynthesis] = useState<WorkshopSynthesis | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [title, setTitle]         = useState("");
  const [loading, setLoading]     = useState(true);
  const [tab, setTab]             = useState<"overview" | "variables" | "ai">("overview");
  const [catFilter, setCatFilter] = useState("Todas");
  const [fullscreen, setFullscreen] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/workshop/join/${code}`)
      .then((r) => r.json())
      .then((d) => { setSessionId(d.session_id); setTitle(d.title); })
      .catch(() => {});
  }, [code]);

  const load = useCallback(async (showSpinner = false) => {
    if (!sessionId) return;
    if (showSpinner) setRefreshing(true);
    const res = await fetch(`${API}/api/workshop/${sessionId}/synthesis`);
    setSynthesis(await res.json());
    setLoading(false);
    if (showSpinner) setRefreshing(false);
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      load();
      const t = setInterval(() => load(), 20_000);
      return () => clearInterval(t);
    }
  }, [sessionId, load]);

  /* Fullscreen toggle */
  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
      setFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setFullscreen(false);
    }
  }
  useEffect(() => {
    const handler = () => setFullscreen(!!document.fullscreenElement);
    document.addEventListener("fullscreenchange", handler);
    return () => document.removeEventListener("fullscreenchange", handler);
  }, []);

  /* ── Loading / empty states ── */
  if (loading) return (
    <div className="flex items-center justify-center min-h-[70vh]">
      <div className="flex flex-col items-center gap-3">
        <div className="w-10 h-10 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
        <p className="text-sm text-[var(--text-3)]">Cargando análisis…</p>
      </div>
    </div>
  );

  if (!synthesis) return (
    <div className="flex items-center justify-center min-h-[70vh] text-[var(--text-3)]">
      No hay datos disponibles
    </div>
  );

  /* ── Derived data ── */
  const categories  = ["Todas", ...new Set(synthesis.variables.map((v) => v.category))];
  const filtered    = catFilter === "Todas" ? synthesis.variables : synthesis.variables.filter((v) => v.category === catFilter);
  const withData    = synthesis.variables.filter((v) => v.avg_current !== null);
  const ol          = synthesis.overall.label_current as RatingLabel | null;
  const of_         = synthesis.overall.label_2032 as RatingLabel | null;

  /* Insights */
  const bestCat = (() => {
    const byCat: Record<string, number[]> = {};
    for (const v of withData) {
      if (!byCat[v.category]) byCat[v.category] = [];
      byCat[v.category].push(v.avg_current!);
    }
    let best = { cat: "", score: 0 };
    for (const [cat, scores] of Object.entries(byCat)) {
      const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
      if (avg > best.score) { best = { cat, score: avg }; }
    }
    return best;
  })();

  const mostImproved = withData
    .filter((v) => v.avg_2032 !== null)
    .sort((a, b) => (b.avg_2032! - b.avg_current!) - (a.avg_2032! - a.avg_current!))[0] ?? null;

  const topDivergence = [...(synthesis.divergences ?? [])][0] ?? null;

  return (
    <div className="max-w-6xl mx-auto py-8 px-6 print:py-4 print:px-8">

      {/* ── HERO HEADER ────────────────────────────────────────────────── */}
      <div className="relative overflow-hidden rounded-2xl border border-[var(--border-2)]
                      bg-gradient-to-br from-[var(--surface)] via-[var(--surface-2)] to-[var(--surface)]
                      mb-8 p-8 print:rounded-none print:border-0">
        {/* Decoración fondo */}
        <div className="absolute inset-0 opacity-40 pointer-events-none"
             style={{ background: "radial-gradient(ellipse 70% 80% at 80% 50%, rgba(79,142,247,0.10) 0%, transparent 70%)" }} />
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-[var(--accent)]/30 to-transparent" />

        <div className="relative">
          {/* Breadcrumb */}
          <div className="flex items-center gap-2 mb-4">
            <span className="text-[10px] font-mono text-[var(--text-3)] tracking-widest uppercase">
              LACCEI · Prospectiva 2032
            </span>
            <span className="w-1 h-1 rounded-full bg-[var(--text-4)]" />
            <span className="text-[10px] font-mono text-[var(--text-3)]">Sala {code}</span>
            <span className="w-1 h-1 rounded-full bg-[var(--text-4)]" />
            <span className="text-[10px] font-mono text-[var(--text-3)]">
              {synthesis.n_experts} experto{synthesis.n_experts !== 1 ? "s" : ""} · {synthesis.n_variables} variables
            </span>
          </div>

          <div className="flex items-start justify-between gap-6 flex-wrap">
            <div className="flex-1 min-w-0">
              <h1 className="text-3xl font-black text-[var(--text-1)] leading-tight mb-3">{title}</h1>

              {/* Global dist bar */}
              {withData.length > 0 && (
                <div className="max-w-sm">
                  <p className="text-[10px] text-[var(--text-4)] font-mono uppercase tracking-wider mb-1.5">
                    Distribución general — estado actual
                  </p>
                  <StackedBar
                    dist={LABELS.reduce((acc, l) => {
                      acc[l] = withData.filter((v) => avgLabel(v.avg_current) === l).length;
                      return acc;
                    }, {} as Record<string, number>)}
                    total={withData.length}
                  />
                  <div className="flex gap-3 mt-1.5">
                    {LABELS.map((l) => {
                      const count = withData.filter((v) => avgLabel(v.avg_current) === l).length;
                      return count > 0 ? (
                        <span key={l} className={`text-[10px] font-medium ${R[l].text}`}>
                          {l} {count}
                        </span>
                      ) : null;
                    })}
                  </div>
                </div>
              )}
            </div>

            {/* KPI scores */}
            <div className="flex items-center gap-3 flex-shrink-0">
              <ScorePill
                label="Estado actual"
                score={synthesis.overall.avg_current}
                caption={ol}
              />
              <div className="flex flex-col items-center gap-1">
                <ArrowRight className="w-5 h-5 text-[var(--accent)]" />
                {synthesis.overall.avg_current !== null && synthesis.overall.avg_2032 !== null && (
                  <span className={`text-[10px] font-bold font-mono ${
                    synthesis.overall.avg_2032 > synthesis.overall.avg_current
                      ? "text-emerald-400" : synthesis.overall.avg_2032 < synthesis.overall.avg_current
                      ? "text-red-400" : "text-[var(--text-4)]"
                  }`}>
                    {synthesis.overall.avg_2032 > synthesis.overall.avg_current ? "+" : ""}
                    {(synthesis.overall.avg_2032 - synthesis.overall.avg_current).toFixed(2)}
                  </span>
                )}
              </div>
              <ScorePill
                label="Proyección 2032"
                score={synthesis.overall.avg_2032}
                caption={of_}
                highlight
              />
            </div>
          </div>
        </div>
      </div>

      {/* ── TABS + ACTIONS ──────────────────────────────────────────────── */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div className="flex gap-1 bg-[var(--surface-2)] rounded-xl p-1">
          {(["overview", "variables", "ai"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-1.5
                ${tab === t
                  ? "bg-[var(--accent)] text-white shadow-lg shadow-blue-500/20"
                  : "text-[var(--text-2)] hover:text-[var(--text-1)]"}`}>
              {t === "overview" && <BarChart3 className="w-3.5 h-3.5" />}
              {t === "ai" && <Brain className="w-3.5 h-3.5" />}
              {{ overview: "Resumen", variables: "Variables", ai: "Análisis IA" }[t]}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button onClick={() => load(true)} disabled={refreshing}
            className="p-2 rounded-lg text-[var(--text-3)] hover:text-[var(--text-2)] hover:bg-[var(--surface-2)] transition-all">
            <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
          </button>
          <button onClick={toggleFullscreen}
            className="p-2 rounded-lg text-[var(--text-3)] hover:text-[var(--text-2)] hover:bg-[var(--surface-2)] transition-all"
            title={fullscreen ? "Salir de pantalla completa" : "Pantalla completa (presentación)"}>
            {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* ════════════════════ TAB: RESUMEN ════════════════════ */}
      {tab === "overview" && (
        <div className="space-y-5">

          {/* Insight cards */}
          {withData.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* Mejor categoría */}
              {bestCat.cat && (
                <InsightCard
                  icon={<Star className="w-4 h-4 text-amber-400" />}
                  color="border-amber-500/20 bg-amber-500/5"
                  label="Mejor categoría"
                  value={CAT_META[bestCat.cat]?.icon ?? "●"}
                  sub={bestCat.cat}
                  score={bestCat.score}
                />
              )}

              {/* Más mejorada hacia 2032 */}
              {mostImproved && (mostImproved.avg_2032! - mostImproved.avg_current!) > 0 && (
                <InsightCard
                  icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
                  color="border-emerald-500/20 bg-emerald-500/5"
                  label="Mayor mejora a 2032"
                  value={`+${(mostImproved.avg_2032! - mostImproved.avg_current!).toFixed(2)}`}
                  sub={mostImproved.variable_name}
                />
              )}

              {/* Mayor disenso */}
              {topDivergence && (
                <InsightCard
                  icon={<AlertTriangle className="w-4 h-4 text-amber-400" />}
                  color="border-amber-500/20 bg-amber-500/5"
                  label="Mayor disenso"
                  value={`σ ${topDivergence.std_current?.toFixed(2)}`}
                  sub={topDivergence.variable_name}
                />
              )}
            </div>
          )}

          {/* Resumen por categoría */}
          <div className="card">
            <div className="flex items-center gap-2 mb-5">
              <BarChart3 className="w-4 h-4 text-[var(--text-3)]" />
              <h2 className="text-sm font-bold text-[var(--text-1)]">Calificación por categoría</h2>
            </div>
            <CategoryOverview variables={synthesis.variables} />
          </div>

          {/* Evolución */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <EvolutionCard
              icon={<TrendingUp className="w-5 h-5 text-emerald-400" />}
              label="Mejoran hacia 2032"
              color="text-emerald-400"
              variables={withData.filter((v) => (v.avg_2032 ?? 0) > (v.avg_current ?? 0))}
            />
            <EvolutionCard
              icon={<Minus className="w-5 h-5 text-sky-300" />}
              label="Se mantienen estables"
              color="text-sky-300"
              variables={withData.filter((v) => v.avg_2032 === v.avg_current)}
            />
            <EvolutionCard
              icon={<TrendingDown className="w-5 h-5 text-red-400" />}
              label="Empeoran hacia 2032"
              color="text-red-400"
              variables={withData.filter((v) => (v.avg_2032 ?? 0) < (v.avg_current ?? 0))}
            />
          </div>

          {/* Disensos */}
          {synthesis.divergences.length > 0 && (
            <div className="card border-l-4 border-amber-500/40">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                <h2 className="text-sm font-bold text-[var(--text-1)]">Mayores disensos en el panel</h2>
              </div>
              <p className="text-xs text-[var(--text-3)] mb-4">
                Variables donde los expertos mostraron mayor dispersión de opiniones
              </p>
              <div className="space-y-3">
                {synthesis.divergences.map((v) => <DivergedRow key={v.variable_id} variable={v} />)}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ════════════════════ TAB: VARIABLES ════════════════════ */}
      {tab === "variables" && (
        <div>
          <div className="flex flex-wrap gap-2 mb-5">
            {categories.map((cat) => {
              const meta = CAT_META[cat];
              return (
                <button key={cat} onClick={() => setCatFilter(cat)}
                  className={`text-xs px-3 py-1.5 rounded-full border font-medium transition-all
                    ${catFilter === cat
                      ? "bg-[var(--accent)] border-[var(--accent)] text-white"
                      : "bg-[var(--surface-2)] border-[var(--border)] text-[var(--text-2)] hover:border-[var(--border-2)]"}`}>
                  {cat === "Todas" ? "Todas" : `${meta?.icon ?? ""} ${cat}`}
                </button>
              );
            })}
          </div>
          <div className="space-y-3">
            {filtered.map((v) => <VariableRow key={v.variable_id} variable={v} />)}
          </div>
        </div>
      )}

      {/* ════════════════════ TAB: ANÁLISIS IA ════════════════════ */}
      {tab === "ai" && (
        <div className="space-y-4">
          {/* Intro */}
          <div className="card bg-gradient-to-br from-purple-500/10 to-[var(--surface)] border border-purple-500/20">
            <div className="flex items-start gap-3">
              <div className="w-9 h-9 rounded-xl bg-purple-500/20 border border-purple-500/30
                              flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-purple-300" />
              </div>
              <div>
                <p className="text-sm font-bold text-purple-200 mb-1">Experto Sintético IA — GPT-4o</p>
                <p className="text-xs text-purple-300/70 leading-relaxed">
                  GPT-4o evaluó cada variable usando evidencia de noticias recientes (SerpAPI) y
                  conocimiento sobre el ecosistema LACCEI. Las divergencias &gt;1 punto señalan
                  áreas de debate estratégico entre la percepción humana y el análisis de datos.
                </p>
              </div>
            </div>
          </div>

          {/* Tabla comparativa */}
          <div className="card overflow-hidden">
            <div className="flex items-center gap-2 mb-4">
              <Brain className="w-4 h-4 text-purple-300" />
              <h2 className="text-sm font-bold text-[var(--text-1)]">Panel humano vs. Experto IA</h2>
            </div>
            <ComparisonTable variables={synthesis.variables} />
          </div>

          {/* Justificaciones */}
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-4 h-4 text-purple-300" />
              <h2 className="text-sm font-bold text-[var(--text-1)]">Justificaciones del Experto IA</h2>
            </div>
            <div className="space-y-3 max-h-[680px] overflow-y-auto pr-1">
              {synthesis.variables.filter((v) => v.ai_vote).map((v) => (
                <AIJustificationCard key={v.variable_id} variable={v} />
              ))}
              {synthesis.variables.every((v) => !v.ai_vote) && (
                <div className="text-center py-10 text-[var(--text-3)]">
                  <Brain className="w-8 h-8 mx-auto mb-2 opacity-40" />
                  <p className="text-sm">Los votos IA aún no han sido generados.</p>
                  <p className="text-xs mt-1">Actívalos desde el hub de la sesión.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <div className="mt-10 pt-6 border-t border-[var(--border)] flex items-center justify-between
                      text-[10px] text-[var(--text-4)] flex-wrap gap-2 print:mt-4">
        <span>LACCEI · Tech Trends Explorer · UPB 2036</span>
        <span>Sala {code} · {synthesis.n_experts} expertos · {synthesis.n_variables} variables · 2026 → 2032</span>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════════════════
   SUB-COMPONENTES
══════════════════════════════════════════════════════════════════════════════ */

function InsightCard({ icon, color, label, value, sub, score }: {
  icon: React.ReactNode; color: string; label: string;
  value: string; sub: string; score?: number;
}) {
  return (
    <div className={`rounded-2xl border p-4 ${color}`}>
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <span className="text-[10px] font-mono text-[var(--text-3)] uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-2xl font-black text-[var(--text-1)] mb-0.5">{value}</div>
      {score !== undefined && (
        <div className="text-xs font-mono text-[var(--text-3)] mb-1">{score.toFixed(2)}/4.0</div>
      )}
      <p className="text-[11px] text-[var(--text-2)] leading-snug line-clamp-2">{sub}</p>
    </div>
  );
}

/* ── Category overview table ──────────────────────────────────────────────── */
function CategoryOverview({ variables }: { variables: VariableSynthesis[] }) {
  const byCategory: Record<string, VariableSynthesis[]> = {};
  for (const v of variables) {
    if (!byCategory[v.category]) byCategory[v.category] = [];
    byCategory[v.category].push(v);
  }

  return (
    <div className="space-y-3">
      {Object.entries(byCategory).map(([cat, vars]) => {
        const withD  = vars.filter((v) => v.avg_current !== null);
        const avgC   = withD.length ? withD.reduce((s, v) => s + v.avg_current!, 0) / withD.length : null;
        const avgF   = withD.length ? withD.reduce((s, v) => s + (v.avg_2032 ?? v.avg_current!), 0) / withD.length : null;
        const lc     = avgLabel(avgC);
        const lf     = avgLabel(avgF);
        const delta  = avgC !== null && avgF !== null ? avgF - avgC : null;
        const meta   = CAT_META[cat] ?? { icon: "●", color: "text-[var(--text-2)]", accent: "" };

        const totalDist: Record<string, number> = { Malo: 0, Regular: 0, Bueno: 0, Excelente: 0 };
        let totalVotes = 0;
        for (const v of withD) {
          for (const lbl of LABELS) {
            totalDist[lbl] = (totalDist[lbl] ?? 0) + (v.distribution_current[lbl] ?? 0);
            totalVotes += v.distribution_current[lbl] ?? 0;
          }
        }

        return (
          <div key={cat} className="rounded-xl bg-[var(--surface-2)] border border-[var(--border)] p-4
                                    hover:border-[var(--border-2)] transition-colors">
            <div className="flex items-start gap-4 flex-wrap">
              {/* Icon + name */}
              <div className="flex items-center gap-2.5 min-w-[200px] flex-1">
                <span className="text-xl w-7">{meta.icon}</span>
                <div>
                  <p className="text-sm font-bold text-[var(--text-1)]">{cat}</p>
                  <p className="text-[10px] text-[var(--text-3)]">{vars.length} variables</p>
                </div>
              </div>

              {/* Current score */}
              <div className="flex-1 min-w-[140px]">
                <p className="text-[10px] text-[var(--text-4)] font-mono mb-1.5">Estado actual</p>
                <div className="flex items-center gap-2 mb-1.5">
                  {lc ? <RatingChip label={lc} /> : <span className="text-xs text-[var(--text-4)]">—</span>}
                </div>
                <ScoreBar score={avgC} />
              </div>

              {/* Arrow + delta */}
              <div className="flex flex-col items-center justify-center px-1 pt-3">
                <ArrowRight className="w-4 h-4 text-[var(--text-4)]" />
                {delta !== null && (
                  <span className={`text-[10px] font-bold font-mono mt-0.5 ${
                    delta > 0 ? "text-emerald-400" : delta < 0 ? "text-red-400" : "text-[var(--text-4)]"
                  }`}>
                    {delta > 0 ? "+" : ""}{delta.toFixed(2)}
                  </span>
                )}
              </div>

              {/* 2032 score */}
              <div className="flex-1 min-w-[140px]">
                <p className="text-[10px] text-[var(--text-4)] font-mono mb-1.5">Proyección 2032</p>
                <div className="flex items-center gap-2 mb-1.5">
                  {lf ? <RatingChip label={lf} /> : <span className="text-xs text-[var(--text-4)]">—</span>}
                </div>
                <ScoreBar score={avgF} />
              </div>

              {/* Distribution */}
              {totalVotes > 0 && (
                <div className="flex-1 min-w-[120px]">
                  <p className="text-[10px] text-[var(--text-4)] font-mono mb-1.5">Distribución</p>
                  <StackedBar dist={totalDist} total={totalVotes} />
                  <div className="flex gap-2 mt-1">
                    {LABELS.filter(l => (totalDist[l] ?? 0) > 0).map(l => (
                      <span key={l} className={`text-[9px] font-medium ${R[l].text}`}>
                        {l[0]}{Math.round((totalDist[l] / totalVotes) * 100)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Evolution card ───────────────────────────────────────────────────────── */
function EvolutionCard({ icon, label, color, variables }: {
  icon: React.ReactNode; label: string; color: string; variables: VariableSynthesis[];
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="card">
      <div className="flex items-start justify-between mb-3">
        <div className={`text-5xl font-black ${color}`}>{variables.length}</div>
        {icon}
      </div>
      <p className="text-xs text-[var(--text-2)] font-medium mb-2">{label}</p>
      {variables.length > 0 && (
        <>
          <button onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 text-[10px] text-[var(--text-3)] hover:text-[var(--text-2)] transition-colors">
            {open ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {open ? "Ocultar" : "Ver variables"}
          </button>
          {open && (
            <ul className="mt-2 space-y-1">
              {variables.map((v) => (
                <li key={v.variable_id} className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--text-4)] flex-shrink-0" />
                  <span className="text-[10px] text-[var(--text-2)] leading-snug">{v.variable_name}</span>
                  {v.avg_current !== null && v.avg_2032 !== null && (
                    <span className={`text-[10px] font-mono font-bold ml-auto ${color}`}>
                      {v.avg_2032 > v.avg_current ? "+" : ""}{(v.avg_2032 - v.avg_current).toFixed(2)}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}

/* ── Diverged row ─────────────────────────────────────────────────────────── */
function DivergedRow({ variable: v }: { variable: VariableSynthesis }) {
  const total = Object.values(v.distribution_current).reduce((a, b) => a + b, 0);
  return (
    <div className="p-4 rounded-xl bg-[var(--surface-2)] border border-[var(--border)]">
      <div className="flex items-center justify-between gap-4 mb-3 flex-wrap">
        <div>
          <p className="text-sm font-semibold text-[var(--text-1)]">{v.variable_name}</p>
          <p className="text-xs text-[var(--text-3)]">{v.category}</p>
        </div>
        <div className="flex items-center gap-3 text-xs flex-shrink-0">
          <span className="font-mono font-bold text-amber-300">σ = {v.std_current?.toFixed(2)}</span>
          {v.consensus_score !== null && (
            <span className="text-[var(--text-3)]">{Math.round(v.consensus_score * 100)}% consenso</span>
          )}
        </div>
      </div>
      <div className="mb-2">
        <StackedBar dist={v.distribution_current} total={total} />
      </div>
      <div className="flex gap-4">
        {LABELS.map((lbl) => {
          const count = v.distribution_current[lbl] ?? 0;
          const pct   = total > 0 ? Math.round((count / total) * 100) : 0;
          return count > 0 ? (
            <span key={lbl} className={`text-[10px] font-medium ${R[lbl].text}`}>
              {lbl}: {count} ({pct}%)
            </span>
          ) : null;
        })}
      </div>
    </div>
  );
}

/* ── Variable row ─────────────────────────────────────────────────────────── */
function VariableRow({ variable: v }: { variable: VariableSynthesis }) {
  const [open, setOpen] = useState(false);
  const lc      = v.label_current as RatingLabel | null;
  const lf      = v.label_2032 as RatingLabel | null;
  const hasData = v.n_responses > 0;
  const delta   = v.avg_current !== null && v.avg_2032 !== null ? v.avg_2032 - v.avg_current : null;

  return (
    <div className="card">
      <button onClick={() => setOpen((o) => !o)} className="w-full text-left">
        <div className="flex items-center gap-3">
          <span className="text-sm font-black text-[var(--text-4)] w-6 text-right flex-shrink-0">
            {v.variable_id}
          </span>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-[var(--text-1)] truncate">{v.variable_name}</p>
            <p className="text-[10px] text-[var(--text-3)]">
              {v.category} · {v.n_responses} respuesta{v.n_responses !== 1 ? "s" : ""}
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {lc ? <RatingChip label={lc} /> : <span className="text-xs text-[var(--text-4)]">Sin datos</span>}
            {lc && lf && (
              <>
                <ArrowRight className="w-3 h-3 text-[var(--text-4)]" />
                <RatingChip label={lf} />
              </>
            )}
            {delta !== null && Math.abs(delta) > 0.01 && (
              <span className={`text-[10px] font-mono font-bold ml-1 ${
                delta > 0 ? "text-emerald-400" : "text-red-400"
              }`}>
                {delta > 0 ? "+" : ""}{delta.toFixed(2)}
              </span>
            )}
            {v.consensus_score !== null && hasData && (
              <div className="hidden sm:flex items-center gap-1 ml-1">
                <div className="w-10 h-1 bg-[var(--surface-3)] rounded-full overflow-hidden">
                  <div className={`h-full rounded-full ${
                    v.consensus_score > 0.7 ? "bg-emerald-500"
                    : v.consensus_score > 0.4 ? "bg-amber-500"
                    : "bg-red-500"
                  }`} style={{ width: `${v.consensus_score * 100}%` }} />
                </div>
              </div>
            )}
            {open ? <ChevronUp className="w-4 h-4 text-[var(--text-3)]" /> : <ChevronDown className="w-4 h-4 text-[var(--text-3)]" />}
          </div>
        </div>

        {/* Score bars (always visible) */}
        {hasData && (
          <div className="grid grid-cols-2 gap-3 mt-3 pl-9">
            <div>
              <p className="text-[9px] text-[var(--text-4)] font-mono mb-1">ACTUAL</p>
              <ScoreBar score={v.avg_current} />
            </div>
            <div>
              <p className="text-[9px] text-[var(--text-4)] font-mono mb-1">2032</p>
              <ScoreBar score={v.avg_2032} />
            </div>
          </div>
        )}
      </button>

      {/* Expanded detail */}
      {open && hasData && (
        <div className="mt-4 pt-4 border-t border-[var(--border)] pl-9 space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <DistChart title="Estado actual (2026)" dist={v.distribution_current} avg={v.avg_current} />
            <DistChart title="Proyección 2032" dist={v.distribution_2032} avg={v.avg_2032} />
          </div>

          {v.ai_vote && (
            <div className="rounded-xl bg-purple-500/10 border border-purple-500/20 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Sparkles className="w-3.5 h-3.5 text-purple-300" />
                <span className="text-xs font-bold text-purple-300">Experto Sintético IA</span>
                <RatingChip label={ratingName(v.ai_vote.rating_current)} faint />
                <ArrowRight className="w-3 h-3 text-purple-400/50" />
                <RatingChip label={ratingName(v.ai_vote.rating_2032)} faint />
                {(v.ai_divergence_current ?? 0) > 0.9 && (
                  <span className="ml-auto flex items-center gap-1 text-amber-400 text-[10px]">
                    <AlertTriangle className="w-3 h-3" /> Δ {v.ai_divergence_current?.toFixed(1)}
                  </span>
                )}
              </div>
              <p className="text-xs text-purple-200/70 leading-relaxed">{v.ai_vote.justification}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ── Distribution chart ───────────────────────────────────────────────────── */
function DistChart({ title, dist, avg }: {
  title: string; dist: Record<string, number>; avg: number | null;
}) {
  const total = Object.values(dist).reduce((a, b) => a + b, 0);
  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-[var(--text-2)] font-medium">{title}</p>
        {avg !== null && (
          <span className="text-[10px] font-mono font-bold text-[var(--text-3)]">{avg.toFixed(2)}/4.0</span>
        )}
      </div>
      <div className="space-y-1.5">
        {LABELS.map((lbl) => {
          const count = dist[lbl] ?? 0;
          const pct   = total > 0 ? (count / total) * 100 : 0;
          return (
            <div key={lbl} className="flex items-center gap-2">
              <span className={`text-[10px] font-bold w-16 ${R[lbl].text}`}>{lbl}</span>
              <div className="flex-1 h-2 bg-[var(--surface-3)] rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${R[lbl].bar} transition-all`}
                     style={{ width: `${pct}%` }} />
              </div>
              <span className="text-[10px] text-[var(--text-3)] w-7 text-right">
                {count > 0 ? `${count} (${Math.round(pct)}%)` : "0"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── AI Comparison table ──────────────────────────────────────────────────── */
function ComparisonTable({ variables }: { variables: VariableSynthesis[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className="text-left py-3 pr-3 text-[var(--text-3)] font-medium w-8">#</th>
            <th className="text-left py-3 pr-4 text-[var(--text-3)] font-medium">Variable</th>
            <th className="text-center py-3 px-2 text-[var(--text-3)] font-medium">Panel<br/>Actual</th>
            <th className="text-center py-3 px-2 text-[var(--text-3)] font-medium">IA<br/>Actual</th>
            <th className="text-center py-3 px-2 text-[var(--text-3)] font-medium">Panel<br/>2032</th>
            <th className="text-center py-3 px-2 text-[var(--text-3)] font-medium">IA<br/>2032</th>
            <th className="text-center py-3 px-2 text-[var(--text-3)] font-medium">Δ</th>
          </tr>
        </thead>
        <tbody>
          {variables.map((v) => {
            const lc  = v.label_current as RatingLabel | null;
            const lf  = v.label_2032 as RatingLabel | null;
            const aiC = v.ai_vote ? ratingName(v.ai_vote.rating_current) : null;
            const aiF = v.ai_vote ? ratingName(v.ai_vote.rating_2032)    : null;
            const div = v.ai_divergence_current ?? 0;
            const highDiv = div > 0.9;
            return (
              <tr key={v.variable_id}
                className={`border-b border-[var(--border)]/50 transition-colors
                  ${highDiv ? "bg-amber-500/5 hover:bg-amber-500/10" : "hover:bg-[var(--surface-2)]/50"}`}>
                <td className="py-2.5 pr-3 text-[var(--text-4)] font-mono">{v.variable_id}</td>
                <td className="py-2.5 pr-4">
                  <p className="font-medium text-[var(--text-1)] leading-snug">{v.variable_name}</p>
                  <p className="text-[var(--text-4)] text-[10px]">{v.category}</p>
                </td>
                <td className="py-2.5 px-2 text-center">
                  {lc ? <RatingChip label={lc} /> : <span className="text-[var(--text-4)]">—</span>}
                </td>
                <td className="py-2.5 px-2 text-center">
                  {aiC ? <RatingChip label={aiC} faint /> : <span className="text-[var(--text-4)]">—</span>}
                </td>
                <td className="py-2.5 px-2 text-center">
                  {lf ? <RatingChip label={lf} /> : <span className="text-[var(--text-4)]">—</span>}
                </td>
                <td className="py-2.5 px-2 text-center">
                  {aiF ? <RatingChip label={aiF} faint /> : <span className="text-[var(--text-4)]">—</span>}
                </td>
                <td className="py-2.5 px-2 text-center">
                  {div > 0 ? (
                    <span className={`font-bold font-mono ${highDiv ? "text-amber-400" : "text-[var(--text-3)]"}`}>
                      {highDiv && <AlertTriangle className="w-3 h-3 inline mr-0.5" />}
                      {div.toFixed(1)}
                    </span>
                  ) : (
                    <span className="text-[var(--text-4)]">—</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="text-[10px] text-[var(--text-4)] mt-3">
        Δ = diferencia entre promedio del panel y voto IA (escala 1–4). Valores &gt;1 indican divergencia significativa.
      </p>
    </div>
  );
}

/* ── AI justification card ─────────────────────────────────────────────────── */
function AIJustificationCard({ variable: v }: { variable: VariableSynthesis }) {
  const [expanded, setExpanded] = useState(false);
  const highDiv = (v.ai_divergence_current ?? 0) > 0.9;
  return (
    <div className={`rounded-xl border p-4 transition-colors ${
      highDiv
        ? "bg-amber-500/5 border-amber-500/20"
        : "bg-[var(--surface-2)] border-[var(--border)]"
    }`}>
      <div className="flex items-start justify-between gap-3 mb-2">
        <div>
          <p className="text-sm font-semibold text-[var(--text-1)]">{v.variable_name}</p>
          <p className="text-xs text-[var(--text-3)]">{v.category}</p>
        </div>
        {v.ai_vote && (
          <div className="flex items-center gap-2 flex-shrink-0">
            <RatingChip label={ratingName(v.ai_vote.rating_current)} faint />
            <ArrowRight className="w-3 h-3 text-[var(--text-4)]" />
            <RatingChip label={ratingName(v.ai_vote.rating_2032)} faint />
          </div>
        )}
      </div>

      {highDiv && (
        <div className="flex items-center gap-1.5 text-amber-300/80 text-[10px] mb-2">
          <AlertTriangle className="w-3 h-3 flex-shrink-0" />
          <span>
            Diverge {v.ai_divergence_current?.toFixed(1)} pts del panel humano — área de debate estratégico
          </span>
        </div>
      )}

      <p className={`text-xs text-[var(--text-2)] leading-relaxed ${!expanded && "line-clamp-3"}`}>
        {v.ai_vote?.justification}
      </p>
      {(v.ai_vote?.justification?.length ?? 0) > 200 && (
        <button onClick={() => setExpanded(e => !e)}
          className="text-[10px] text-[var(--accent)] hover:text-[var(--accent-hover)] mt-1.5 transition-colors">
          {expanded ? "Leer menos ▴" : "Leer más ▾"}
        </button>
      )}
    </div>
  );
}

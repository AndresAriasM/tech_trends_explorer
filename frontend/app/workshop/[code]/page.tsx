"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ExternalLink, Copy, Check, Upload, Sparkles, BarChart3,
  ChevronLeft, ChevronDown, ChevronUp, FileJson, CircleHelp,
} from "lucide-react";
import type { WorkshopSession } from "@/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const VARIABLES_PREVIEW = [
  { id: 1,  name: "Costos operativos - tecnológicos",          cat: "Económica" },
  { id: 2,  name: "Fondos en educación e innovación",          cat: "Económica" },
  { id: 3,  name: "Sostenibilidad financiera",                 cat: "Económica" },
  { id: 4,  name: "Pobreza digital en capacidad de cómputo",   cat: "Económica" },
  { id: 5,  name: "Inseguridad en países sedes",               cat: "Social" },
  { id: 6,  name: "Migración",                                 cat: "Social" },
  { id: 7,  name: "Geopolítica global",                        cat: "Social" },
  { id: 8,  name: "Problemas sociales de la región",           cat: "Social" },
  { id: 9,  name: "Brechas por uso de IA",                     cat: "Digital" },
  { id: 10, name: "Barreras lingüísticas",                     cat: "Digital" },
  { id: 11, name: "Colaboración entre países - regiones",      cat: "Institucional" },
  { id: 12, name: "Colaboración regional",                     cat: "Institucional" },
  { id: 13, name: "Noción de altruismo - voluntariado",        cat: "Institucional" },
  { id: 14, name: "Gobernanza institucional",                  cat: "Institucional" },
  { id: 15, name: "Prioridades de miembros LACCEI",            cat: "Institucional" },
  { id: 16, name: "Exigencias en investigación de profesores", cat: "Académica" },
  { id: 17, name: "Igualdad, equidad e inclusión",             cat: "Académica" },
  { id: 18, name: "Transformación mercado laboral",            cat: "Académica" },
  { id: 19, name: "Normativas internacionales sobre IA",       cat: "Regulatoria" },
  { id: 20, name: "Brechas en acceso a la educación",          cat: "Regulatoria" },
];

const CAT_COLORS: Record<string, string> = {
  "Económica":     "bg-amber-500/10 text-amber-300",
  "Social":        "bg-rose-500/10 text-rose-300",
  "Digital":       "bg-blue-500/10 text-blue-300",
  "Institucional": "bg-purple-500/10 text-purple-300",
  "Académica":     "bg-teal-500/10 text-teal-300",
  "Regulatoria":   "bg-orange-500/10 text-orange-300",
};

/* ─────────────────────────── Página principal ─────────────────────────────── */

export default function SessionHubPage() {
  const { code } = useParams<{ code: string }>();
  const router   = useRouter();

  const [session, setSession]           = useState<WorkshopSession | null>(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState("");
  const [aiLoading, setAiLoading]       = useState(false);
  const [aiDone, setAiDone]             = useState(false);
  const [showImport, setShowImport]     = useState(false);
  const [importJson, setImportJson]     = useState("");
  const [importing, setImporting]       = useState(false);
  const [importResult, setImportResult] = useState<{ experts: number; votes: number } | null>(null);
  const [copied, setCopied]             = useState(false);
  // gforms_url viene del backend en join/{code} — no está en WorkshopSession directamente
  const [gformsUrl, setGformsUrl]       = useState("");

  useEffect(() => {
    fetch(`${API}/api/workshop/join/${code}`)
      .then((r) => { if (!r.ok) throw new Error("Código no válido"); return r.json(); })
      .then((d)  => { setSession(d); setAiDone(d.ai_votes_ready); setGformsUrl(d.gforms_url ?? ""); })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [code]);

  async function triggerAI() {
    if (!session) return;
    setAiLoading(true);
    await fetch(`${API}/api/workshop/${session.session_id}/ai-vote`, { method: "POST" });
    setAiLoading(false);
    setAiDone(true);
  }

  async function handleImport() {
    if (!session || !importJson.trim()) return;
    setImporting(true);
    try {
      const parsed = JSON.parse(importJson);
      const body   = Array.isArray(parsed) ? { experts: parsed, clear_existing: true } : parsed;
      const res    = await fetch(`${API}/api/workshop/${session.session_id}/import`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
      });
      const data = await res.json();
      setImportResult({ experts: data.experts_imported, votes: data.votes_saved });
    } catch {
      alert("Error al importar. Verifica el formato JSON.");
    } finally {
      setImporting(false);
    }
  }

  function copyLink(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }

  /* ── States ── */
  if (loading) return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin" />
    </div>
  );

  if (error) return (
    <div className="max-w-md mx-auto py-20 text-center px-6">
      <p className="text-red-300 mb-4">{error}</p>
      <button onClick={() => router.push("/workshop")} className="btn btn-primary">← Volver</button>
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto py-10 px-6">

      {/* ── Header ── */}
      <div className="flex items-start justify-between gap-4 mb-8 flex-wrap">
        <div>
          <button onClick={() => router.push("/workshop")}
            className="flex items-center gap-1 text-xs text-[var(--text-3)] hover:text-[var(--text-2)] transition-colors mb-2">
            <ChevronLeft className="w-3.5 h-3.5" /> Sesiones
          </button>
          <h1 className="text-2xl font-bold text-[var(--text-1)]">{session?.title}</h1>
          {session?.description && (
            <p className="text-sm text-[var(--text-2)] mt-1">{session.description}</p>
          )}
          <div className="flex items-center gap-2 mt-2">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
              session?.status === "active"
                ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                : "bg-[var(--surface-3)] text-[var(--text-3)] border-[var(--border)]"
            }`}>
              {session?.status === "active" ? "Activa" : "Cerrada"}
            </span>
            {aiDone && (
              <span className="text-xs px-2 py-0.5 rounded-full font-medium border bg-purple-500/10 text-purple-300 border-purple-500/20">
                <Sparkles className="w-3 h-3 inline mr-1" />IA lista
              </span>
            )}
          </div>
        </div>

        {/* Código */}
        <div className="card text-center px-6 py-3 flex-shrink-0">
          <p className="text-[10px] text-[var(--text-3)] mb-1 font-mono tracking-widest">CÓDIGO DE SALA</p>
          <div className="text-3xl font-black tracking-widest gradient-text">{code}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">

        {/* ── Columna derecha: acciones ── */}
        <div className="lg:col-span-1 space-y-4">

          {/* Google Form */}
          <div className="card">
            <p className="section-label mb-3">Google Form</p>
            {gformsUrl ? (
              <>
                <p className="text-xs text-[var(--text-2)] mb-3 leading-relaxed">
                  Los expertos votan externamente. Comparte este enlace antes o durante el taller.
                </p>
                <a href={gformsUrl} target="_blank" rel="noopener noreferrer"
                  className="btn btn-primary w-full justify-center text-sm mb-2 flex items-center gap-2">
                  <ExternalLink className="w-3.5 h-3.5" /> Abrir formulario
                </a>
                <button onClick={() => copyLink(gformsUrl)}
                  className="btn w-full border border-[var(--border-2)] bg-[var(--surface-3)]
                             text-[var(--text-2)] text-xs flex items-center justify-center gap-2">
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? "Copiado" : "Copiar enlace"}
                </button>
              </>
            ) : (
              <p className="text-xs text-[var(--text-3)]">
                No hay URL configurada. Agrégala desde la lista de sesiones.
              </p>
            )}
          </div>

          {/* Importar votos */}
          <div className="card">
            <p className="section-label mb-3">Importar resultados</p>
            <p className="text-xs text-[var(--text-2)] mb-3 leading-relaxed">
              Una vez recolectados los votos en Google Forms, conviértelos a JSON e impórtalos aquí.
            </p>
            <button onClick={() => setShowImport(true)}
              className="btn w-full border border-[var(--accent-border)] bg-[var(--accent-sub)]
                         text-[var(--accent)] text-sm hover:bg-[var(--accent)]/15 transition-colors
                         flex items-center justify-center gap-2">
              <Upload className="w-3.5 h-3.5" /> Importar votos JSON
            </button>
            {importResult && (
              <div className="mt-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 px-3 py-2 text-xs text-emerald-300 flex items-center gap-2">
                <Check className="w-3.5 h-3.5 flex-shrink-0" />
                {importResult.experts} expertos · {importResult.votes} votos importados
              </div>
            )}
          </div>

          {/* Análisis IA */}
          <div className="card border border-purple-500/20 bg-purple-500/5">
            <p className="section-label mb-1" style={{ color: "rgb(192 132 252)" }}>
              <Sparkles className="w-3.5 h-3.5 inline mr-1.5" />Experto Sintético IA
            </p>
            <p className="text-xs text-[var(--text-2)] mb-3 leading-relaxed">
              GPT-4o evalúa las 20 variables con evidencia de SerpAPI y genera un voto fundamentado con justificación.
            </p>
            {aiDone ? (
              <div className="rounded-lg bg-purple-500/10 border border-purple-500/20 px-3 py-2 text-xs text-purple-300 text-center flex items-center justify-center gap-2">
                <Sparkles className="w-3.5 h-3.5" /> Votos IA generados
              </div>
            ) : (
              <button onClick={triggerAI} disabled={aiLoading}
                className="btn w-full bg-purple-600/20 border border-purple-500/30 text-purple-300
                           hover:bg-purple-600/30 transition-colors text-sm disabled:opacity-50 flex items-center justify-center gap-2">
                {aiLoading ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-purple-300/30 border-t-purple-300 rounded-full animate-spin" />
                    Generando…
                  </>
                ) : (
                  <><Sparkles className="w-3.5 h-3.5" /> Generar votos IA</>
                )}
              </button>
            )}
          </div>

          {/* Ver resultados */}
          <button onClick={() => router.push(`/workshop/${code}/results`)}
            className="btn btn-primary w-full py-3 text-sm flex items-center justify-center gap-2">
            <BarChart3 className="w-4 h-4" /> Ver dashboard de resultados
          </button>
        </div>

        {/* ── Columna izquierda: variables + guía Google Forms ── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Variables */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <p className="section-label">Variables evaluadas</p>
              <span className="text-xs text-[var(--text-3)]">20 variables · 2 dimensiones temporales</span>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-[380px] overflow-y-auto pr-1">
              {VARIABLES_PREVIEW.map((v) => (
                <div key={v.id}
                  className="flex items-start gap-2.5 p-2.5 rounded-lg bg-[var(--surface-3)] hover:bg-[var(--surface-4)] transition-colors">
                  <span className="text-xs font-black text-[var(--text-4)] flex-shrink-0 mt-0.5 w-5 text-right">
                    {v.id}
                  </span>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-[var(--text-1)] leading-snug">{v.name}</p>
                    <span className={`inline-block text-[10px] px-1.5 py-0.5 rounded mt-1 font-medium ${CAT_COLORS[v.cat] ?? ""}`}>
                      {v.cat}
                    </span>
                  </div>
                </div>
              ))}
            </div>
            <div className="border-t border-[var(--border)] mt-4 pt-3">
              <p className="text-xs text-[var(--text-3)] text-center">
                Escala: <span className="text-red-400">Malo</span> ·{" "}
                <span className="text-amber-400">Regular</span> ·{" "}
                <span className="text-sky-400">Bueno</span> ·{" "}
                <span className="text-emerald-400">Excelente</span>
                {" "}· Cada variable se califica en 2026 y proyección 2032
              </p>
            </div>
          </div>

          {/* Guía Google Forms */}
          <GFormsGuide />
        </div>
      </div>

      {/* ── Modal importar ── */}
      {showImport && (
        <ImportModal
          onClose={() => setShowImport(false)}
          onImport={handleImport}
          json={importJson}
          setJson={setImportJson}
          loading={importing}
          result={importResult}
        />
      )}
    </div>
  );
}

/* ── Guía Google Forms ─────────────────────────────────────────────────────── */
function GFormsGuide() {
  const [open, setOpen] = useState(false);

  return (
    <div className="card border border-blue-500/20 bg-blue-500/5">
      <button onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CircleHelp className="w-4 h-4 text-blue-400" />
          <p className="text-sm font-semibold text-blue-300">Cómo configurar Google Forms para importar</p>
        </div>
        {open ? <ChevronUp className="w-4 h-4 text-[var(--text-3)]" /> : <ChevronDown className="w-4 h-4 text-[var(--text-3)]" />}
      </button>

      {open && (
        <div className="mt-4 space-y-4 text-xs text-[var(--text-2)]">

          {/* Pasos */}
          <div className="space-y-3">
            {[
              {
                step: "01",
                title: "Crea el formulario",
                desc: "En Google Forms, agrega una pregunta de opción múltiple por cada variable × 2 dimensiones (40 preguntas en total).",
                example: `"Costos operativos — Estado actual (2026)"\nOpciones: 1-Malo / 2-Regular / 3-Bueno / 4-Excelente`,
                tip: "Tip: Ordénalas como Variable 1 Actual, Variable 1 2032, Variable 2 Actual…"
              },
              {
                step: "02",
                title: "Recolecta las respuestas",
                desc: "Comparte el formulario con los expertos. Puede ser presencial (proyectado) o enviado por correo.",
              },
              {
                step: "03",
                title: "Exporta a Google Sheets",
                desc: 'En Google Forms → "Respuestas" → ícono de Google Sheets. Obtendrás una hoja con una fila por experto.',
              },
              {
                step: "04",
                title: "Convierte a JSON",
                desc: "Usa este patrón en Google Apps Script o manualmente:",
                code: `[\n  {\n    "expert_name": "Dr. García",\n    "votes": [\n      {"variable_id": 1, "rating_current": 3, "rating_2032": 4},\n      {"variable_id": 2, "rating_current": 2, "rating_2032": 3}\n      // ... 20 variables\n    ]\n  }\n]`,
              },
              {
                step: "05",
                title: "Importa aquí",
                desc: 'Pega el JSON en el botón "Importar votos JSON" de esta página.',
              },
            ].map((item) => (
              <div key={item.step} className="flex gap-3">
                <div className="w-7 h-7 rounded-full bg-blue-500/15 border border-blue-500/25
                                flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-[9px] font-black font-mono text-blue-400">{item.step}</span>
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-[var(--text-1)] mb-0.5">{item.title}</p>
                  <p className="leading-relaxed">{item.desc}</p>
                  {item.example && (
                    <pre className="mt-1.5 bg-[var(--surface-2)] rounded px-2.5 py-2 text-[10px] text-[var(--text-2)] whitespace-pre-wrap">
                      {item.example}
                    </pre>
                  )}
                  {item.code && (
                    <pre className="mt-1.5 bg-[var(--surface-2)] rounded px-2.5 py-2 text-[10px] text-[var(--text-2)] overflow-x-auto">
                      {item.code}
                    </pre>
                  )}
                  {item.tip && (
                    <p className="mt-1 text-blue-400/70 text-[10px]">{item.tip}</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Escala de ratings */}
          <div className="rounded-xl bg-[var(--surface-2)] border border-[var(--border)] p-3">
            <p className="font-semibold text-[var(--text-1)] mb-2">Escala numérica para el JSON</p>
            <div className="grid grid-cols-4 gap-2 text-center">
              {[["1", "Malo", "text-red-400"],["2", "Regular", "text-amber-400"],["3", "Bueno", "text-sky-400"],["4", "Excelente", "text-emerald-400"]].map(([n, l, c]) => (
                <div key={n} className="rounded-lg bg-[var(--surface-3)] py-2">
                  <div className={`text-xl font-black ${c}`}>{n}</div>
                  <div className={`text-[10px] font-medium ${c}`}>{l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Import Modal ──────────────────────────────────────────────────────────── */
function ImportModal({ onClose, onImport, json, setJson, loading, result }: {
  onClose:  () => void;
  onImport: () => void;
  json:     string;
  setJson:  (v: string) => void;
  loading:  boolean;
  result:   { experts: number; votes: number } | null;
}) {
  const EXAMPLE = JSON.stringify([
    {
      expert_name: "María García",
      votes: [
        { variable_id: 1,  rating_current: 2, rating_2032: 3 },
        { variable_id: 2,  rating_current: 3, rating_2032: 4 },
        { variable_id: 3,  rating_current: 2, rating_2032: 3 },
        { variable_id: 4,  rating_current: 1, rating_2032: 2 },
        // ... hasta variable_id 20
      ],
    },
  ], null, 2);

  function loadExample() {
    setJson(EXAMPLE);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="bg-[var(--surface)] border border-[var(--border-2)] rounded-2xl w-full max-w-2xl
                      max-h-[90vh] overflow-y-auto shadow-2xl">
        <div className="p-6">
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-[var(--accent-sub)] border border-[var(--accent-border)]
                              flex items-center justify-center">
                <FileJson className="w-4 h-4 text-[var(--accent)]" />
              </div>
              <div>
                <h2 className="text-base font-bold text-[var(--text-1)]">Importar resultados</h2>
                <p className="text-xs text-[var(--text-3)]">Pega el JSON exportado de Google Forms</p>
              </div>
            </div>
            <button onClick={onClose} className="text-[var(--text-3)] hover:text-[var(--text-1)] transition-colors">✕</button>
          </div>

          {/* Instrucción formato */}
          <div className="rounded-xl bg-[var(--accent-sub)] border border-[var(--accent-border)] p-4 mb-4">
            <p className="text-xs font-semibold text-[var(--accent)] mb-2">Formato esperado</p>
            <p className="text-xs text-[var(--text-2)] leading-relaxed">
              Array JSON: cada objeto = un experto, con sus 20 votos. Rating 1-4 (1=Malo, 4=Excelente).
              Necesitas <code className="bg-[var(--surface-3)] px-1 rounded">rating_current</code> y{" "}
              <code className="bg-[var(--surface-3)] px-1 rounded">rating_2032</code> por variable.
            </p>
          </div>

          {/* Textarea */}
          <textarea
            value={json}
            onChange={(e) => setJson(e.target.value)}
            placeholder='[{"expert_name": "Nombre Apellido", "votes": [{"variable_id": 1, "rating_current": 3, "rating_2032": 4}, ...]}]'
            rows={12}
            className="w-full bg-[var(--surface-2)] border border-[var(--border-2)] rounded-xl px-4 py-3
                       text-[var(--text-1)] placeholder:text-[var(--text-4)] text-xs font-mono resize-none
                       focus:outline-none focus:border-[var(--accent)] mb-3"
          />

          {result && (
            <div className="mb-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 px-4 py-3
                            text-emerald-300 text-sm flex items-center gap-2">
              <Check className="w-4 h-4 flex-shrink-0" />
              {result.experts} expertos · {result.votes} votos importados correctamente
            </div>
          )}

          <div className="flex gap-3">
            <button onClick={onClose}
              className="btn bg-[var(--surface-3)] border border-[var(--border-2)] text-[var(--text-2)]">
              Cancelar
            </button>
            <button onClick={loadExample}
              className="btn bg-[var(--surface-3)] border border-[var(--border-2)] text-[var(--text-3)] text-xs">
              Cargar ejemplo
            </button>
            <button onClick={onImport} disabled={loading || !json.trim()}
              className="btn btn-primary flex-1 disabled:opacity-50 flex items-center justify-center gap-2">
              {loading ? (
                <>
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Importando…
                </>
              ) : (
                <><Upload className="w-4 h-4" /> Importar votos</>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

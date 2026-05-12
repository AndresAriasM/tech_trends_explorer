"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import type { WorkshopSession } from "@/types";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function WorkshopPage() {
  const router = useRouter();
  const [sessions, setSessions]       = useState<WorkshopSession[]>([]);
  const [view, setView]               = useState<"list" | "create">("list");
  const [title, setTitle]             = useState("");
  const [description, setDescription] = useState("");
  const [gformsUrl, setGformsUrl]     = useState("");
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState("");

  useEffect(() => { loadSessions(); }, []);

  async function loadSessions() {
    try {
      const res = await fetch(`${API}/api/workshop/sessions`);
      setSessions(await res.json());
    } catch { /* silencioso */ }
  }

  async function handleCreate(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!title.trim()) return;
    setLoading(true); setError("");
    try {
      const res = await fetch(`${API}/api/workshop/sessions`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ title, description, gforms_url: gformsUrl }),
      });
      if (!res.ok) throw new Error(await res.text());
      const s: WorkshopSession = await res.json();
      router.push(`/workshop/${s.code}`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Error al crear la sesión");
    } finally { setLoading(false); }
  }

  return (
    <div className="max-w-4xl mx-auto py-10 px-6">

      {/* ── Hero ── */}
      <div className="mb-10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-blue-500
                          flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl font-bold text-[var(--text-1)] leading-tight">
              Taller Prospectivo LACCEI 2032
            </h1>
            <p className="text-sm text-[var(--text-3)]">
              Gestión de sesiones · Importación Google Forms · Análisis IA
            </p>
          </div>
        </div>

        {/* Flujo de trabajo */}
        <div className="grid grid-cols-4 gap-2 mb-2">
          {[
            { step: "01", label: "Crear sesión",      icon: "📋", color: "border-blue-500/30 bg-blue-500/5" },
            { step: "02", label: "Google Forms",       icon: "📝", color: "border-violet-500/30 bg-violet-500/5" },
            { step: "03", label: "Importar votos",     icon: "⬆", color: "border-teal-500/30 bg-teal-500/5" },
            { step: "04", label: "Análisis IA",        icon: "✦", color: "border-purple-500/30 bg-purple-500/5" },
          ].map((s) => (
            <div key={s.step} className={`rounded-xl border px-3 py-3 ${s.color}`}>
              <div className="text-lg mb-1">{s.icon}</div>
              <div className="text-[10px] text-[var(--text-3)] font-mono mb-0.5">{s.step}</div>
              <div className="text-xs font-semibold text-[var(--text-2)]">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Tabs ── */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex gap-1 bg-[var(--surface-2)] rounded-xl p-1">
          {(["list", "create"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors
                ${view === v ? "bg-[var(--accent)] text-white" : "text-[var(--text-2)] hover:text-[var(--text-1)]"}`}>
              {{ list: "Sesiones", create: "+ Nueva sesión" }[v]}
            </button>
          ))}
        </div>
        {view === "list" && (
          <button onClick={loadSessions}
            className="text-xs text-[var(--text-3)] hover:text-[var(--text-2)] transition-colors">
            ↻ Actualizar
          </button>
        )}
      </div>

      {/* ── Vista: Lista de sesiones ── */}
      {view === "list" && (
        <div className="space-y-3">
          {sessions.length === 0 ? (
            <div className="card text-center py-16">
              <div className="text-4xl mb-3">📋</div>
              <p className="text-[var(--text-2)] font-medium mb-1">No hay sesiones aún</p>
              <p className="text-sm text-[var(--text-3)] mb-4">Crea la primera sesión del taller</p>
              <button onClick={() => setView("create")} className="btn btn-primary">
                Crear sesión →
              </button>
            </div>
          ) : (
            sessions.map((s) => <SessionCard key={s.session_id} session={s} />)
          )}
        </div>
      )}

      {/* ── Vista: Crear sesión ── */}
      {view === "create" && (
        <div className="card max-w-xl">
          <h2 className="text-base font-bold text-[var(--text-1)] mb-5">Nueva sesión de taller</h2>
          <form onSubmit={handleCreate} className="space-y-4">

            <Field label="Nombre del taller" required>
              <input type="text" value={title} onChange={(e) => setTitle(e.target.value)}
                placeholder="Ej: LACCEI Annual Conference 2026 — Prospectiva"
                className="input-field" required />
            </Field>

            <Field label="Descripción" hint="Opcional">
              <textarea value={description} onChange={(e) => setDescription(e.target.value)}
                placeholder="Contexto, objetivo y participantes del taller..."
                rows={3} className="input-field resize-none" />
            </Field>

            <Field label="URL del Google Form" hint="Opcional — se mostrará como acceso directo">
              <input type="url" value={gformsUrl} onChange={(e) => setGformsUrl(e.target.value)}
                placeholder="https://forms.gle/..."
                className="input-field" />
              <p className="text-xs text-[var(--text-3)] mt-1.5">
                La votación de los expertos se realiza externamente a través de este formulario
              </p>
            </Field>

            {error && (
              <div className="rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3 text-red-300 text-sm">
                {error}
              </div>
            )}

            <div className="flex gap-3 pt-1">
              <button type="button" onClick={() => setView("list")}
                className="btn bg-[var(--surface-3)] border border-[var(--border-2)] text-[var(--text-2)]">
                Cancelar
              </button>
              <button type="submit" disabled={loading || !title.trim()}
                className="btn btn-primary flex-1 disabled:opacity-50">
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Creando…
                  </span>
                ) : "Crear sesión →"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

function SessionCard({ session }: { session: WorkshopSession }) {
  const router = useRouter();
  const date = new Date(session.created_at).toLocaleDateString("es-CO", {
    day: "numeric", month: "short", year: "numeric",
  });

  return (
    <div className="card card-hover cursor-pointer group" onClick={() => router.push(`/workshop/${session.code}`)}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className={`text-xs px-2 py-0.5 rounded-full font-medium border
              ${session.status === "active"
                ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                : "bg-[var(--surface-3)] text-[var(--text-3)] border-[var(--border)]"}`}>
              {session.status === "active" ? "Activa" : "Cerrada"}
            </span>
            {session.ai_votes_ready && (
              <span className="text-xs px-2 py-0.5 rounded-full font-medium border
                               bg-purple-500/10 text-purple-300 border-purple-500/20">
                ✦ IA lista
              </span>
            )}
          </div>
          <h3 className="text-sm font-bold text-[var(--text-1)] truncate">{session.title}</h3>
          {session.description && (
            <p className="text-xs text-[var(--text-3)] mt-0.5 truncate">{session.description}</p>
          )}
        </div>
        <div className="text-right flex-shrink-0">
          <div className="font-mono text-lg font-black text-[var(--text-2)] group-hover:text-[var(--accent)] transition-colors">
            {session.code}
          </div>
          <div className="text-xs text-[var(--text-3)]">{date}</div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, hint, required, children }: {
  label: string; hint?: string; required?: boolean; children: React.ReactNode;
}) {
  return (
    <div>
      <label className="flex items-center gap-1.5 text-sm text-[var(--text-2)] mb-1.5">
        {label}
        {required && <span className="text-[var(--accent)]">*</span>}
        {hint && <span className="text-[var(--text-4)] text-xs">— {hint}</span>}
      </label>
      {children}
    </div>
  );
}

"use client";

/**
 * /delphi — lista de estudios Delphi.
 *
 * Reconstruye la pantalla perdida («Mis estudios», tarjetas con rondas,
 * preguntas y panelistas) sobre el modelo nuevo: el estudio declara su
 * horizonte, su umbral de dicotomía y su criterio de parada **antes** de
 * abrir la ronda 1, que es lo que el ejercicio original no hacía.
 */

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  BarChart3,
  Loader2,
  Plus,
  RefreshCw,
  Trash2,
  Users,
} from "lucide-react";

import {
  DelphiApiError,
  createStudy,
  deleteStudy,
  listStudies,
} from "@/components/delphi/api";
import { Notice } from "@/components/delphi/primitives";
import type { DelphiStudy } from "@/components/delphi/types";

const STATUS_BADGE: Record<DelphiStudy["status"], string> = {
  draft: "badge-gray",
  active: "badge-green",
  closed: "badge-blue",
};

const STATUS_LABEL: Record<DelphiStudy["status"], string> = {
  draft: "Borrador",
  active: "Activo",
  closed: "Cerrado",
};

export default function DelphiStudiesPage() {
  const [creating, setCreating] = useState(false);
  const [actionError, setActionError] = useState<string>("");

  const {
    data: studies = [],
    isFetching,
    error: queryError,
    refetch,
  } = useQuery({
    queryKey: ["delphi", "studies"],
    queryFn: listStudies,
    retry: false,
  });

  const error =
    queryError instanceof DelphiApiError
      ? queryError
      : queryError
        ? new DelphiApiError(String(queryError), 0)
        : actionError
          ? new DelphiApiError(actionError, 0)
          : null;
  const loading = isFetching && studies.length === 0;

  async function handleRemove(s: DelphiStudy) {
    if (
      !window.confirm(
        `Eliminar «${s.title}» y todas sus rondas, variables y respuestas?`
      )
    )
      return;
    try {
      setActionError("");
      await deleteStudy(s.id);
      await refetch();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Error al eliminar");
    }
  }

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-8">
        <h1 className="page-title">Panel Delphi</h1>
        <p className="mt-1.5 max-w-3xl text-[14px] text-[var(--text-2)]">
          Estudios por rondas: define las variables, invita al panel por token,
          conserva su identidad entre rondas y mide si la opinión converge.
        </p>
      </header>

      {error && (
        <div className="mb-6">
          <Notice
            tone={error.schemaMissing ? "warn" : "error"}
            title={
              error.schemaMissing
                ? "El esquema del módulo todavía no está aplicado"
                : "No se pudo cargar la lista"
            }
          >
            {error.schemaMissing ? (
              <>
                Falta correr{" "}
                <code className="rounded bg-[var(--surface-3)] px-1.5 py-0.5 text-[12px]">
                  backend/migrations/001_delphi_prospecta.sql
                </code>{" "}
                y registrar el router en <code>main.py</code>. Hasta entonces
                esta pantalla no tiene de dónde leer.
              </>
            ) : (
              error.message
            )}
          </Notice>
        </div>
      )}

      <div className="mb-5 flex items-center justify-between gap-4">
        <h2 className="section-title">Mis estudios</h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => void refetch()}
            className="btn btn-ghost"
            disabled={isFetching}
          >
            <RefreshCw size={14} className={isFetching ? "spin" : undefined} />
            Actualizar
          </button>
          <button
            onClick={() => setCreating((v) => !v)}
            className="btn btn-primary"
          >
            <Plus size={15} />
            Nuevo estudio Delphi
          </button>
        </div>
      </div>

      {creating && (
        <CreateStudyForm
          onCancel={() => setCreating(false)}
          onCreated={async () => {
            setCreating(false);
            await refetch();
          }}
        />
      )}

      {loading ? (
        <div className="grid gap-4 md:grid-cols-2">
          {[0, 1].map((i) => (
            <div key={i} className="card h-[186px] p-5">
              <div className="shimmer h-5 w-2/3" />
              <div className="shimmer mt-3 h-3 w-full" />
              <div className="shimmer mt-2 h-3 w-4/5" />
            </div>
          ))}
        </div>
      ) : studies.length === 0 && !error ? (
        <div className="card px-6 py-16 text-center">
          <Users size={30} className="mx-auto mb-3 text-[var(--text-4)]" />
          <p className="text-[15px] font-medium text-[var(--text-2)]">
            Todavía no hay estudios
          </p>
          <p className="mx-auto mt-1 max-w-md text-[13px] text-[var(--text-3)]">
            Un estudio Delphi declara su horizonte, su umbral de dicotomía y su
            criterio de parada antes de abrir la primera ronda.
          </p>
          <button
            onClick={() => setCreating(true)}
            className="btn btn-primary mx-auto mt-5"
          >
            <Plus size={15} />
            Crear el primero
          </button>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2">
          {studies.map((s) => (
            <StudyCard key={s.id} study={s} onRemove={() => handleRemove(s)} />
          ))}
        </div>
      )}
    </div>
  );
}

function StudyCard({
  study,
  onRemove,
}: {
  study: DelphiStudy;
  onRemove: () => void;
}) {
  const meta = [
    `${study.n_rounds ?? 0} ronda${study.n_rounds === 1 ? "" : "s"}`,
    `${study.n_variables ?? 0} variables`,
    `${study.n_invited ?? 0} panelistas`,
  ].join(" · ");

  return (
    <article className="card card-hover flex flex-col p-5">
      <div className="mb-2 flex items-start justify-between gap-3">
        <h3 className="text-[16px] font-bold leading-snug text-[var(--text-1)]">
          {study.title}
        </h3>
        <span className={`badge ${STATUS_BADGE[study.status]} shrink-0`}>
          {STATUS_LABEL[study.status]}
        </span>
      </div>

      <p className="line-clamp-2 text-[13px] leading-relaxed text-[var(--text-3)]">
        {study.description || "Sin descripción."}
      </p>

      <p className="mt-3 text-[12px] text-[var(--text-3)]">{meta}</p>

      {/* Los parámetros que el ejercicio original tenía enterrados en un
          literal del notebook, aquí a la vista. */}
      <dl className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-[var(--text-3)]">
        {study.horizon_year && (
          <div className="flex gap-1.5">
            <dt>Horizonte</dt>
            <dd className="font-semibold text-[var(--text-2)]">
              {study.horizon_year}
            </dd>
          </div>
        )}
        <div className="flex gap-1.5" title="El parámetro más consecuente del análisis (§2)">
          <dt>Umbral</dt>
          <dd className="font-semibold text-[var(--text-2)]">
            ≥ {study.dichotomy_threshold}
          </dd>
        </div>
        <div className="flex gap-1.5">
          <dt>Máx. rondas</dt>
          <dd className="font-semibold text-[var(--text-2)]">
            {study.max_rounds}
          </dd>
        </div>
      </dl>

      <div className="mt-4 flex items-center gap-2 border-t border-[var(--border)] pt-4">
        <Link href={`/delphi/${study.id}`} className="btn btn-primary">
          <BarChart3 size={14} />
          Estadísticas
        </Link>
        <button
          onClick={onRemove}
          className="btn btn-ghost ml-auto px-3"
          aria-label={`Eliminar ${study.title}`}
          title="Eliminar estudio"
        >
          <Trash2 size={14} />
        </button>
      </div>
    </article>
  );
}

function CreateStudyForm({
  onCancel,
  onCreated,
}: {
  onCancel: () => void;
  onCreated: () => void | Promise<void>;
}) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [horizon, setHorizon] = useState<string>("2036");
  const [threshold, setThreshold] = useState(4);
  const [maxRounds, setMaxRounds] = useState(3);
  const [ownerId, setOwnerId] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (title.trim().length < 3) return;
    setBusy(true);
    setErr("");
    try {
      await createStudy({
        title: title.trim(),
        description: description.trim(),
        horizon_year: horizon ? Number(horizon) : null,
        dichotomy_threshold: threshold,
        max_rounds: maxRounds,
        ...(ownerId.trim() ? { owner_id: ownerId.trim() } : {}),
      });
      await onCreated();
    } catch (e2) {
      setErr(e2 instanceof Error ? e2.message : "Error al crear el estudio");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="card mb-5 space-y-4 p-5">
      <div>
        <label className="label-sm mb-1.5 block">Título</label>
        <input
          className="inp"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Empleos del futuro en tecnología — horizonte 2036"
          autoFocus
        />
      </div>
      <div>
        <label className="label-sm mb-1.5 block">Descripción</label>
        <textarea
          className="inp"
          rows={2}
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Qué pregunta responde el panel y a quién se invita."
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <div>
          <label className="label-sm mb-1.5 block">Horizonte</label>
          <input
            className="inp"
            type="number"
            min={2000}
            max={2100}
            value={horizon}
            onChange={(e) => setHorizon(e.target.value)}
          />
        </div>
        <div>
          <label
            className="label-sm mb-1.5 block"
            title="Con qué valor de la escala se considera «prioritario». Cambiarlo cambia el mapa entero, así que es configuración del estudio y no un literal (§2)."
          >
            Umbral de dicotomía
          </label>
          <input
            className="inp"
            type="number"
            min={0}
            max={10}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
        </div>
        <div>
          <label className="label-sm mb-1.5 block">Máximo de rondas</label>
          <input
            className="inp"
            type="number"
            min={1}
            max={10}
            value={maxRounds}
            onChange={(e) => setMaxRounds(Number(e.target.value))}
          />
        </div>
      </div>

      <div>
        <label className="label-sm mb-1.5 block">
          owner_id (opcional si sólo hay un perfil)
        </label>
        <input
          className="inp"
          value={ownerId}
          onChange={(e) => setOwnerId(e.target.value)}
          placeholder="uuid del perfil dueño"
        />
      </div>

      <p className="flex items-start gap-2 text-[12px] leading-relaxed text-[var(--text-3)]">
        <AlertTriangle size={14} className="mt-0.5 shrink-0" />
        El criterio de parada por defecto —estabilidad ≥ 85 % o RIQ ≤ 1— queda
        declarado desde ya, antes de abrir la ronda 1. Es lo que impide
        elegirlo al ver los resultados.
      </p>

      {err && <Notice tone="error">{err}</Notice>}

      <div className="flex gap-2">
        <button
          type="submit"
          className="btn btn-primary"
          disabled={busy || title.trim().length < 3}
        >
          {busy && <Loader2 size={14} className="spin" />}
          Crear estudio
        </button>
        <button type="button" onClick={onCancel} className="btn btn-ghost">
          Cancelar
        </button>
      </div>
    </form>
  );
}

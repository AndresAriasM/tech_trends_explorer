"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import {
  Plus,
  Trash2,
  FolderOpen,
  TrendingUp,
  X,
  History as HistoryIcon,
} from "lucide-react";
import {
  getCategories,
  createCategory,
  deleteCategory,
  getHypeCycleHistory,
} from "@/lib/api";
import type { Category, HypeCycleQuery, HypePhase } from "@/types";

const PHASE_COLORS: Record<HypePhase, string> = {
  "Innovation Trigger": "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment": "#ef4444",
  "Slope of Enlightenment": "#34d399",
  "Plateau of Productivity": "#818cf8",
  Unknown: "#9ca3af",
};

const PHASE_SHORT: Record<HypePhase, string> = {
  "Innovation Trigger": "Lanzamiento",
  "Peak of Inflated Expectations": "Pico",
  "Trough of Disillusionment": "Valle",
  "Slope of Enlightenment": "Rampa",
  "Plateau of Productivity": "Meseta",
  Unknown: "N/D",
};

export default function CategoriesPage() {
  const queryClient = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [error, setError] = useState("");

  const { data: categories = [], isLoading } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  const createMutation = useMutation({
    mutationFn: createCategory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["categories"] });
      setNewName("");
      setNewDesc("");
      setShowCreate(false);
      setError("");
    },
    onError: () => setError("Error al crear la categoría"),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteCategory,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["categories"] }),
  });

  function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    createMutation.mutate({ name: newName, description: newDesc });
  }

  return (
    <div style={{ padding: "32px 48px", maxWidth: 1400, margin: "0 auto" }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 16,
          flexWrap: "wrap",
          marginBottom: 32,
        }}
      >
        <div>
          <h1
            className="heading-glow"
            style={{ fontSize: 30, fontWeight: 800, letterSpacing: "-0.02em", marginBottom: 6 }}
          >
            Categorías
          </h1>
          <p style={{ fontSize: 14, color: "var(--text-2)", maxWidth: 640 }}>
            Agrupa análisis relacionados y visualiza su Hype Cycle consolidado
            con un solo clic.
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
          <Plus style={{ width: 14, height: 14 }} />
          Nueva categoría
        </button>
      </div>

      {/* Create modal */}
      {showCreate && (
        <div
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.65)",
            backdropFilter: "blur(6px)",
            zIndex: 50,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 16,
          }}
          onClick={() => setShowCreate(false)}
        >
          <div
            className="card"
            style={{ padding: 28, width: "100%", maxWidth: 480 }}
            onClick={(e) => e.stopPropagation()}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 20,
              }}
            >
              <p style={{ fontSize: 18, fontWeight: 700, color: "var(--text-1)" }}>
                Nueva categoría
              </p>
              <button
                onClick={() => setShowCreate(false)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-3)",
                  cursor: "pointer",
                  padding: 4,
                  display: "flex",
                }}
                aria-label="Cerrar"
              >
                <X style={{ width: 18, height: 18 }} />
              </button>
            </div>
            <form onSubmit={handleCreate}>
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  color: "var(--text-3)",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  marginBottom: 6,
                }}
              >
                Nombre
              </label>
              <input
                type="text"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                placeholder="Ej. LACCEI 2026 - Mora paper"
                autoFocus
                className="inp"
                style={{ width: "100%", marginBottom: 16 }}
              />
              <label
                style={{
                  display: "block",
                  fontSize: 11,
                  color: "var(--text-3)",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  marginBottom: 6,
                }}
              >
                Descripción (opcional)
              </label>
              <textarea
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="¿Para qué es esta categoría?"
                rows={3}
                className="inp"
                style={{ width: "100%", marginBottom: 20, resize: "vertical" }}
              />
              {error && (
                <p style={{ fontSize: 12, color: "#ef4444", marginBottom: 12 }}>{error}</p>
              )}
              <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
                <button
                  type="button"
                  className="btn btn-ghost"
                  onClick={() => setShowCreate(false)}
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={createMutation.isPending || !newName.trim()}
                >
                  {createMutation.isPending ? "Creando…" : "Crear"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Grid / empty state */}
      {isLoading ? (
        <p style={{ color: "var(--text-3)", textAlign: "center", padding: 80 }}>
          Cargando categorías…
        </p>
      ) : categories.length === 0 ? (
        <div className="card" style={{ padding: 80, textAlign: "center" }}>
          <FolderOpen
            style={{
              width: 40,
              height: 40,
              color: "var(--text-4)",
              margin: "0 auto 16px",
            }}
          />
          <p style={{ fontSize: 15, color: "var(--text-2)", marginBottom: 8 }}>
            Aún no tienes categorías
          </p>
          <p
            style={{
              fontSize: 13,
              color: "var(--text-3)",
              marginBottom: 20,
              maxWidth: 360,
              margin: "0 auto 20px",
            }}
          >
            Crea una para agrupar tus análisis y graficarlos en un solo Hype Cycle.
          </p>
          <button className="btn btn-primary" onClick={() => setShowCreate(true)}>
            <Plus style={{ width: 14, height: 14 }} />
            Crear la primera categoría
          </button>
        </div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
            gap: 16,
          }}
        >
          {categories.map((cat) => (
            <CategoryCard
              key={cat.category_id}
              category={cat}
              onDelete={() => deleteMutation.mutate(cat.category_id)}
              canDelete={cat.category_id !== "default"}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Category Card ────────────────────────────────────────────

function CategoryCard({
  category,
  onDelete,
  canDelete,
}: {
  category: Category;
  onDelete: () => void;
  canDelete: boolean;
}) {
  const { data: queries = [] } = useQuery<HypeCycleQuery[]>({
    queryKey: ["hype-cycle-history", category.category_id],
    queryFn: () => getHypeCycleHistory(category.category_id),
  });

  const phaseCounts: Partial<Record<HypePhase, number>> = {};
  queries.forEach((q) => {
    const p = q.hype_metrics.phase;
    phaseCounts[p] = (phaseCounts[p] ?? 0) + 1;
  });
  const totalMentions = queries.reduce(
    (acc, q) => acc + q.hype_metrics.total_mentions,
    0
  );
  const isEmpty = queries.length === 0;

  return (
    <div
      className="card card-hover"
      style={{
        padding: 20,
        display: "flex",
        flexDirection: "column",
        minHeight: 220,
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 12,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            minWidth: 0,
            flex: 1,
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 8,
              background: "var(--accent-sub)",
              border: "1px solid var(--accent-border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
          >
            <FolderOpen style={{ width: 15, height: 15, color: "var(--accent)" }} />
          </div>
          <div style={{ minWidth: 0 }}>
            <p
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "var(--text-1)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
              }}
              title={category.name}
            >
              {category.name}
            </p>
            <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 2 }}>
              {queries.length} {queries.length === 1 ? "análisis" : "análisis"}
              {totalMentions > 0 && ` · ${totalMentions.toLocaleString()} menciones`}
            </p>
          </div>
        </div>
        {canDelete && (
          <button
            onClick={() => {
              if (confirm(`¿Eliminar la categoría "${category.name}"?`)) onDelete();
            }}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-3)",
              cursor: "pointer",
              padding: 6,
              borderRadius: 6,
              transition: "color 0.15s, background 0.15s",
              flexShrink: 0,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "#ef4444";
              e.currentTarget.style.background = "rgba(239,68,68,0.08)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--text-3)";
              e.currentTarget.style.background = "transparent";
            }}
            aria-label="Eliminar categoría"
          >
            <Trash2 style={{ width: 14, height: 14 }} />
          </button>
        )}
      </div>

      {/* Description */}
      {category.description ? (
        <p
          style={{
            fontSize: 12.5,
            color: "var(--text-2)",
            marginBottom: 14,
            lineHeight: 1.5,
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
          }}
        >
          {category.description}
        </p>
      ) : (
        <p
          style={{
            fontSize: 12,
            color: "var(--text-4)",
            fontStyle: "italic",
            marginBottom: 14,
          }}
        >
          Sin descripción
        </p>
      )}

      {/* Phase distribution */}
      {!isEmpty ? (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 4,
            marginBottom: 16,
          }}
        >
          {Object.entries(phaseCounts).map(([phase, count]) => (
            <span
              key={phase}
              title={phase}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 5,
                fontSize: 10.5,
                padding: "3px 9px",
                borderRadius: 999,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border)",
                color: "var(--text-2)",
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: PHASE_COLORS[phase as HypePhase],
                }}
              />
              {PHASE_SHORT[phase as HypePhase]} · {count}
            </span>
          ))}
        </div>
      ) : (
        <p
          style={{
            fontSize: 11,
            color: "var(--text-4)",
            fontStyle: "italic",
            marginBottom: 16,
          }}
        >
          Sin análisis aún
        </p>
      )}

      {/* CTAs */}
      <div style={{ marginTop: "auto", display: "flex", gap: 8 }}>
        <Link
          href={
            isEmpty
              ? "#"
              : `/hype-cycle/compare?category=${category.category_id}`
          }
          className="btn btn-primary"
          style={{
            flex: 1,
            justifyContent: "center",
            fontSize: 12.5,
            pointerEvents: isEmpty ? "none" : "auto",
            opacity: isEmpty ? 0.35 : 1,
          }}
          title={
            isEmpty
              ? "Ejecuta análisis en esta categoría primero"
              : "Ver todos los análisis sobre una sola curva"
          }
        >
          <TrendingUp style={{ width: 13, height: 13 }} />
          Ver Hype Cycle
        </Link>
        <Link
          href={`/history?category=${category.category_id}`}
          className="btn btn-ghost"
          style={{ fontSize: 12.5 }}
          title="Ver análisis individuales"
        >
          <HistoryIcon style={{ width: 13, height: 13 }} />
        </Link>
      </div>
    </div>
  );
}

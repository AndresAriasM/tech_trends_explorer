"use client";

import { useState } from "react";
import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import {
  Plus, Trash2, Copy, Check, Terminal,
  Lightbulb, ChevronDown, RotateCcw, FolderOpen,
} from "lucide-react";
import { useAnalysisStore } from "@/store/analysisStore";
import { getCategories } from "@/lib/api";
import type { SearchTerm } from "@/types";
// cn is unused but kept for future use
// import { cn } from "@/lib/utils";

// ── Operator metadata ─────────────────────────────────────────

const OPERATOR_META = {
  AND: { label: "AND", color: "#60a5fa", bg: "rgba(96,165,250,0.10)", border: "rgba(96,165,250,0.25)", desc: "Ambos términos deben aparecer" },
  OR:  { label: "OR",  color: "#34d399", bg: "rgba(52,211,153,0.10)", border: "rgba(52,211,153,0.25)", desc: "Cualquiera de los términos" },
  NOT: { label: "NOT", color: "#f87171", bg: "rgba(248,113,113,0.10)", border: "rgba(248,113,113,0.25)", desc: "Excluye este término" },
} as const;

const OPERATOR_CYCLE: SearchTerm["operator"][] = ["AND", "OR", "NOT"];

// ── Query string builder ─────────────────────────────────────

function buildQuery(terms: SearchTerm[]): string {
  return terms
    .filter((t) => t.value.trim())
    .map((t, i) => {
      const val = t.exact_match ? `"${t.value.trim()}"` : t.value.trim();
      return i === 0 ? val : `${t.operator} ${val}`;
    })
    .join(" ");
}

// Render query with coloured spans (returns segments for rendering)
function parseQuerySegments(terms: SearchTerm[]): Array<{ text: string; type: "term" | "operator" | "quote" }> {
  const segments: Array<{ text: string; type: "term" | "operator" | "quote" }> = [];
  const filled = terms.filter((t) => t.value.trim());
  filled.forEach((t, i) => {
    if (i > 0) {
      segments.push({ text: ` ${t.operator} `, type: "operator" });
    }
    if (t.exact_match) {
      segments.push({ text: '"', type: "quote" });
      segments.push({ text: t.value.trim(), type: "term" });
      segments.push({ text: '"', type: "quote" });
    } else {
      segments.push({ text: t.value.trim(), type: "term" });
    }
  });
  return segments;
}

// ── Template presets ─────────────────────────────────────────

interface Template {
  label: string;
  terms: SearchTerm[];
}

const TEMPLATES: Template[] = [
  {
    label: "Tecnología sola",
    terms: [{ value: "", operator: "AND", exact_match: true }],
  },
  {
    label: "Tech + IA",
    terms: [
      { value: "", operator: "AND", exact_match: true },
      { value: "artificial intelligence", operator: "AND", exact_match: true },
    ],
  },
  {
    label: "Tech + adopción",
    terms: [
      { value: "", operator: "AND", exact_match: true },
      { value: "adoption", operator: "AND", exact_match: false },
      { value: "market", operator: "OR", exact_match: false },
    ],
  },
  {
    label: "Sin patentes",
    terms: [
      { value: "", operator: "AND", exact_match: true },
      { value: "patent", operator: "NOT", exact_match: false },
    ],
  },
  {
    label: "Broad search",
    terms: [
      { value: "", operator: "AND", exact_match: true },
      { value: "", operator: "OR", exact_match: true },
    ],
  },
];

// ── Main component ────────────────────────────────────────────

export function QueryBuilder() {
  const {
    searchTerms, technologyName, technologyDescription, minYear, autoSave, categoryId,
    addTerm, updateTerm, removeTerm, setSearchTerms,
    setTechnologyName, setTechnologyDescription, setMinYear, setAutoSave, setCategoryId,
  } = useAnalysisStore();

  const { data: categories = [] } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  const [copied, setCopied] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);

  const queryString = buildQuery(searchTerms);
  const segments = parseQuerySegments(searchTerms);
  const filledCount = searchTerms.filter((t) => t.value.trim()).length;

  function handleCopy() {
    if (!queryString) return;
    navigator.clipboard.writeText(queryString).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  function cycleOperator(index: number) {
    const term = searchTerms[index];
    const idx = OPERATOR_CYCLE.indexOf(term.operator);
    const next = OPERATOR_CYCLE[(idx + 1) % OPERATOR_CYCLE.length];
    updateTerm(index, { ...term, operator: next });
  }

  function applyTemplate(tpl: Template) {
    setSearchTerms(tpl.terms.map((t) => ({ ...t })));
    setShowTemplates(false);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>

      {/* ── Technology metadata ─────────────────────────────── */}
      <Section label="Tecnología">
        <input
          type="text"
          value={technologyName}
          onChange={(e) => setTechnologyName(e.target.value)}
          placeholder="Ej: Generative AI"
          className="inp"
          style={{ fontSize: 14 }}
        />
      </Section>

      <div style={{ height: 1, background: "var(--border)", margin: "0 0 18px" }} />

      {/* ── Templates ────────────────────────────────────────── */}
      <div style={{ marginBottom: 20 }}>
        <button
          onClick={() => setShowTemplates((v) => !v)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            color: showTemplates ? "var(--accent)" : "var(--text-3)",
            background: "none",
            border: "none",
            cursor: "pointer",
            padding: 0,
            transition: "color 0.15s",
            marginBottom: showTemplates ? 10 : 0,
          }}
        >
          <Lightbulb style={{ width: 12, height: 12 }} />
          Plantillas rápidas
          <ChevronDown
            style={{
              width: 12,
              height: 12,
              transition: "transform 0.2s",
              transform: showTemplates ? "rotate(180deg)" : "rotate(0deg)",
            }}
          />
        </button>

        {showTemplates && (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              padding: "12px",
              background: "var(--surface-2)",
              borderRadius: 10,
              border: "1px solid var(--border)",
            }}
          >
            {TEMPLATES.map((tpl) => (
              <button
                key={tpl.label}
                onClick={() => applyTemplate(tpl)}
                style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: "5px 10px",
                  borderRadius: 7,
                  background: "var(--surface-3)",
                  border: "1px solid var(--border)",
                  color: "var(--text-2)",
                  cursor: "pointer",
                  transition: "all 0.15s",
                  letterSpacing: "-0.01em",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "var(--accent-sub)";
                  e.currentTarget.style.borderColor = "var(--accent-border)";
                  e.currentTarget.style.color = "var(--accent)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "var(--surface-3)";
                  e.currentTarget.style.borderColor = "var(--border)";
                  e.currentTarget.style.color = "var(--text-2)";
                }}
              >
                {tpl.label}
              </button>
            ))}
            <p style={{ width: "100%", fontSize: 10, color: "var(--text-3)", marginTop: 4, lineHeight: 1.5 }}>
              El primer término vacío será reemplazado al escribir la tecnología.
            </p>
          </div>
        )}
      </div>

      {/* ── Boolean term builder ─────────────────────────────── */}
      <div style={{ marginBottom: 4 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 12,
          }}
        >
          <p
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--text-3)",
            }}
          >
            Ecuación de búsqueda
            {filledCount > 0 && (
              <span
                style={{
                  marginLeft: 8,
                  fontSize: 10,
                  fontWeight: 700,
                  padding: "2px 7px",
                  borderRadius: 99,
                  background: "var(--accent-sub)",
                  border: "1px solid var(--accent-border)",
                  color: "var(--accent)",
                  letterSpacing: "0.02em",
                }}
              >
                {filledCount} {filledCount === 1 ? "término" : "términos"}
              </span>
            )}
          </p>
          <button
            onClick={() => addTerm({ value: "", operator: "AND", exact_match: true })}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              fontSize: 11,
              fontWeight: 600,
              color: "var(--accent)",
              background: "var(--accent-sub)",
              border: "1px solid var(--accent-border)",
              borderRadius: 7,
              padding: "5px 10px",
              cursor: "pointer",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(79,142,247,0.16)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "var(--accent-sub)";
            }}
          >
            <Plus style={{ width: 12, height: 12 }} />
            Agregar término
          </button>
        </div>

        {/* Term cards */}
        <div style={{ display: "flex", flexDirection: "column", gap: 0 }}>
          {searchTerms.map((term, i) => (
            <div key={i}>
              <TermCard
                term={term}
                index={i}
                isFirst={i === 0}
                isOnly={searchTerms.length === 1}
                onChange={(t) => updateTerm(i, t)}
                onRemove={() => removeTerm(i)}
              />

              {/* Operator connector between terms */}
              {i < searchTerms.length - 1 && (
                <OperatorConnector
                  operator={searchTerms[i + 1].operator}
                  onClick={() => cycleOperator(i + 1)}
                />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Query preview ─────────────────────────────────────── */}
      {filledCount > 0 && (
        <div
          style={{
            marginTop: 20,
            borderRadius: 12,
            overflow: "hidden",
            border: "1px solid var(--accent-border)",
            background: "rgba(6,9,15,0.6)",
          }}
        >
          {/* Header */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "10px 14px",
              borderBottom: "1px solid rgba(79,142,247,0.12)",
              background: "rgba(79,142,247,0.06)",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Terminal style={{ width: 12, height: 12, color: "var(--accent)" }} />
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  color: "var(--accent)",
                }}
              >
                Query preview
              </span>
              <span
                style={{
                  fontSize: 10,
                  color: "var(--text-3)",
                  marginLeft: 4,
                }}
              >
                · {queryString.length} chars
              </span>
            </div>
            <button
              onClick={handleCopy}
              title="Copiar query"
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontSize: 11,
                fontWeight: 600,
                color: copied ? "#34d399" : "var(--text-3)",
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: "3px 6px",
                borderRadius: 5,
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => {
                if (!copied) e.currentTarget.style.color = "var(--text-1)";
              }}
              onMouseLeave={(e) => {
                if (!copied) e.currentTarget.style.color = "var(--text-3)";
              }}
            >
              {copied
                ? <><Check style={{ width: 11, height: 11 }} /> Copiado</>
                : <><Copy style={{ width: 11, height: 11 }} /> Copiar</>
              }
            </button>
          </div>

          {/* Query string with syntax highlighting */}
          <div
            style={{
              padding: "14px 16px",
              fontFamily: '"SF Mono", "Fira Code", "Cascadia Code", monospace',
              fontSize: 12.5,
              lineHeight: 1.7,
              wordBreak: "break-all",
              minHeight: 48,
            }}
          >
            {segments.map((seg, i) => (
              <span
                key={i}
                style={{
                  color:
                    seg.type === "operator"
                      ? (() => {
                          const op = seg.text.trim() as keyof typeof OPERATOR_META;
                          return OPERATOR_META[op]?.color ?? "var(--text-2)";
                        })()
                      : seg.type === "quote"
                      ? "rgba(167,139,250,0.8)"
                      : "var(--text-1)",
                  fontWeight: seg.type === "operator" ? 700 : 400,
                }}
              >
                {seg.text}
              </span>
            ))}
          </div>

          {/* Footer note */}
          <div
            style={{
              padding: "8px 14px",
              borderTop: "1px solid rgba(79,142,247,0.08)",
              fontSize: 10,
              color: "var(--text-3)",
              letterSpacing: "0.02em",
            }}
          >
            Compatible con SerpAPI · Google Search · Scopus
          </div>
        </div>
      )}

      <div style={{ height: 1, background: "var(--border)", margin: "20px 0 18px" }} />

      {/* ── Search parameters ─────────────────────────────────── */}
      <Section label="Rango temporal">
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 5, letterSpacing: "0.04em" }}>
              Desde
            </p>
            <input
              type="number"
              value={minYear}
              onChange={(e) => setMinYear(Number(e.target.value))}
              min={2000}
              max={new Date().getFullYear()}
              className="inp"
              style={{ fontSize: 14 }}
            />
          </div>
          <div
            style={{
              width: 24,
              height: 1,
              background: "var(--border)",
              flexShrink: 0,
              marginTop: 18,
            }}
          />
          <div style={{ flex: 1 }}>
            <p style={{ fontSize: 10, color: "var(--text-3)", marginBottom: 5, letterSpacing: "0.04em" }}>
              Hasta
            </p>
            <input
              type="number"
              value={new Date().getFullYear()}
              disabled
              className="inp"
              style={{ fontSize: 14, opacity: 0.45 }}
            />
          </div>
        </div>
        <p style={{ fontSize: 10, color: "var(--text-4)", marginTop: 6, lineHeight: 1.5 }}>
          Se analizarán artículos y noticias desde {minYear} hasta hoy.
        </p>
      </Section>

      <div style={{ height: 1, background: "var(--border)", margin: "0 0 18px" }} />

      {/* ── Description ──────────────────────────────────────── */}
      <Section label="Descripción (opcional)">
        <textarea
          value={technologyDescription}
          onChange={(e) => setTechnologyDescription(e.target.value)}
          rows={2}
          placeholder="Contexto adicional sobre la tecnología…"
          className="inp resize-none"
          style={{ fontSize: 13, lineHeight: 1.6 }}
        />
      </Section>

      <div style={{ height: 1, background: "var(--border)", margin: "0 0 18px" }} />

      {/* ── Category selector ────────────────────────────────── */}
      <Section
        label="Categoría"
        hint="Agrupa análisis para comparar su Hype Cycle"
      >
        <div style={{ display: "flex", gap: 8 }}>
          <div style={{ position: "relative", flex: 1 }}>
            <FolderOpen
              style={{
                position: "absolute",
                left: 12,
                top: "50%",
                transform: "translateY(-50%)",
                width: 14,
                height: 14,
                color: "var(--text-3)",
                pointerEvents: "none",
              }}
            />
            <select
              value={categoryId}
              onChange={(e) => setCategoryId(e.target.value)}
              className="inp"
              style={{
                width: "100%",
                fontSize: 13,
                paddingLeft: 34,
                cursor: "pointer",
                appearance: "none",
              }}
            >
              <option value="default">Sin categoría (default)</option>
              {categories
                .filter((c) => c.category_id !== "default")
                .map((c) => (
                  <option key={c.category_id} value={c.category_id}>
                    {c.name}
                  </option>
                ))}
            </select>
            <ChevronDown
              style={{
                position: "absolute",
                right: 12,
                top: "50%",
                transform: "translateY(-50%)",
                width: 14,
                height: 14,
                color: "var(--text-3)",
                pointerEvents: "none",
              }}
            />
          </div>
          <Link
            href="/categories"
            className="btn btn-ghost"
            style={{ fontSize: 12, whiteSpace: "nowrap", flexShrink: 0 }}
            title="Gestionar categorías"
          >
            <Plus style={{ width: 13, height: 13 }} />
            Nueva
          </Link>
        </div>
      </Section>

      {/* ── Auto-save toggle ─────────────────────────────────── */}
      <label
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          padding: "12px 14px",
          borderRadius: 10,
          background: "var(--surface-2)",
          border: "1px solid var(--border)",
        }}
      >
        <div>
          <p style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)", marginBottom: 2 }}>
            Guardar automáticamente
          </p>
          <p style={{ fontSize: 11, color: "var(--text-3)" }}>
            Almacenar resultado en DynamoDB
          </p>
        </div>
        <Toggle value={autoSave} onChange={setAutoSave} />
      </label>

    </div>
  );
}

// ── Term card ─────────────────────────────────────────────────

interface TermCardProps {
  term: SearchTerm;
  index: number;
  isFirst: boolean;
  isOnly: boolean;
  onChange: (t: SearchTerm) => void;
  onRemove: () => void;
}

function TermCard({ term, index, isFirst, isOnly, onChange, onRemove }: TermCardProps) {
  const isEmpty = !term.value.trim();

  return (
    <div
      style={{
        borderRadius: 12,
        background: "var(--surface-2)",
        border: `1px solid ${isEmpty ? "var(--border)" : "var(--border-2)"}`,
        padding: "12px 14px",
        transition: "border-color 0.15s",
      }}
    >
      {/* Header row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 10,
        }}
      >
        {/* Index badge */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 7,
          }}
        >
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.04em",
              width: 20,
              height: 20,
              borderRadius: "50%",
              background: isEmpty ? "var(--surface-3)" : "var(--accent-sub)",
              border: `1px solid ${isEmpty ? "var(--border)" : "var(--accent-border)"}`,
              color: isEmpty ? "var(--text-3)" : "var(--accent)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
              transition: "all 0.15s",
            }}
          >
            {index + 1}
          </span>
          {isFirst && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: "var(--text-3)",
                letterSpacing: "0.04em",
              }}
            >
              término principal
            </span>
          )}
        </div>

        {/* Exact match toggle + remove */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {/* Exact match pill */}
          <button
            onClick={() => onChange({ ...term, exact_match: !term.exact_match })}
            title={term.exact_match ? 'Búsqueda exacta: "término"' : "Búsqueda aproximada: término"}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 10,
              fontWeight: 700,
              padding: "3px 8px",
              borderRadius: 99,
              cursor: "pointer",
              transition: "all 0.15s",
              background: term.exact_match
                ? "rgba(167,139,250,0.12)"
                : "var(--surface-3)",
              border: term.exact_match
                ? "1px solid rgba(167,139,250,0.28)"
                : "1px solid var(--border)",
              color: term.exact_match ? "#c084fc" : "var(--text-3)",
              fontFamily: term.exact_match ? '"SF Mono", monospace' : "inherit",
              letterSpacing: term.exact_match ? "0" : "0.04em",
            }}
          >
            {term.exact_match ? (
              <><span style={{ opacity: 0.7 }}>"</span>exacto<span style={{ opacity: 0.7 }}>"</span></>
            ) : (
              "~ aproximado"
            )}
          </button>

          {/* Remove */}
          <button
            onClick={onRemove}
            disabled={isOnly}
            title="Eliminar término"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 24,
              height: 24,
              borderRadius: 7,
              background: "none",
              border: "none",
              cursor: isOnly ? "not-allowed" : "pointer",
              color: "var(--text-4)",
              transition: "all 0.15s",
              opacity: isOnly ? 0.3 : 1,
            }}
            onMouseEnter={(e) => {
              if (!isOnly) {
                e.currentTarget.style.background = "rgba(248,113,113,0.10)";
                e.currentTarget.style.color = "#f87171";
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "none";
              e.currentTarget.style.color = "var(--text-4)";
            }}
          >
            <Trash2 style={{ width: 13, height: 13 }} />
          </button>
        </div>
      </div>

      {/* Input */}
      <input
        type="text"
        value={term.value}
        onChange={(e) => onChange({ ...term, value: e.target.value })}
        placeholder={
          index === 0
            ? "Ej: Generative AI, Quantum Computing…"
            : `Término ${index + 1}…`
        }
        className="inp"
        style={{
          fontSize: 14,
          fontWeight: term.value ? 500 : 400,
          letterSpacing: term.exact_match && term.value ? "-0.01em" : "0",
        }}
        autoFocus={index > 0 && !term.value}
      />

      {/* Value preview */}
      {term.value.trim() && (
        <div
          style={{
            marginTop: 8,
            fontSize: 11,
            color: "var(--text-3)",
            fontFamily: '"SF Mono", monospace',
            lineHeight: 1.4,
          }}
        >
          <span style={{ color: "var(--text-4)" }}>→ </span>
          {term.exact_match ? (
            <>
              <span style={{ color: "rgba(167,139,250,0.8)" }}>"</span>
              <span style={{ color: "var(--text-2)" }}>{term.value.trim()}</span>
              <span style={{ color: "rgba(167,139,250,0.8)" }}>"</span>
            </>
          ) : (
            <span style={{ color: "var(--text-2)" }}>{term.value.trim()}</span>
          )}
        </div>
      )}
    </div>
  );
}

// ── Operator connector ────────────────────────────────────────

function OperatorConnector({
  operator,
  onClick,
}: {
  operator: SearchTerm["operator"];
  onClick: () => void;
}) {
  const meta = OPERATOR_META[operator];

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 0,
        padding: "6px 0",
      }}
    >
      {/* Left line */}
      <div
        style={{
          flex: 1,
          height: 1,
          background: `linear-gradient(to right, var(--border), ${meta.color}40)`,
        }}
      />

      {/* Operator pill — click to cycle */}
      <button
        onClick={onClick}
        title={`${meta.desc}. Clic para cambiar.`}
        style={{
          fontSize: 11,
          fontWeight: 800,
          letterSpacing: "0.08em",
          padding: "4px 12px",
          borderRadius: 99,
          background: meta.bg,
          border: `1px solid ${meta.border}`,
          color: meta.color,
          cursor: "pointer",
          transition: "all 0.15s",
          display: "flex",
          alignItems: "center",
          gap: 5,
          flexShrink: 0,
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.transform = "scale(1.05)";
          e.currentTarget.style.boxShadow = `0 0 12px ${meta.color}30`;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.transform = "scale(1)";
          e.currentTarget.style.boxShadow = "none";
        }}
      >
        <RotateCcw style={{ width: 9, height: 9, opacity: 0.6 }} />
        {meta.label}
      </button>

      {/* Right line */}
      <div
        style={{
          flex: 1,
          height: 1,
          background: `linear-gradient(to left, var(--border), ${meta.color}40)`,
        }}
      />
    </div>
  );
}

// ── Reusable helpers ──────────────────────────────────────────

function Section({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 18 }}>
      <p
        style={{
          fontSize: 10,
          fontWeight: 700,
          letterSpacing: "0.10em",
          textTransform: "uppercase",
          color: "var(--text-3)",
          marginBottom: hint ? 2 : 8,
        }}
      >
        {label}
      </p>
      {hint && (
        <p
          style={{
            fontSize: 11,
            color: "var(--text-3)",
            marginBottom: 8,
            lineHeight: 1.4,
          }}
        >
          {hint}
        </p>
      )}
      {children}
    </div>
  );
}

function Toggle({ value, onChange }: { value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      role="switch"
      aria-checked={value}
      onClick={() => onChange(!value)}
      style={{
        position: "relative",
        width: 36,
        height: 20,
        borderRadius: 99,
        background: value ? "var(--accent)" : "var(--surface-4)",
        border: "none",
        cursor: "pointer",
        transition: "background 0.2s",
        flexShrink: 0,
        boxShadow: value ? "0 0 12px var(--accent-glow)" : "none",
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 2,
          left: value ? 18 : 2,
          width: 16,
          height: 16,
          borderRadius: "50%",
          background: "#fff",
          transition: "left 0.2s cubic-bezier(0.4,0,0.2,1)",
          boxShadow: "0 1px 4px rgba(0,0,0,0.3)",
        }}
      />
    </button>
  );
}

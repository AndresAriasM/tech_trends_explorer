"use client";

import { Plus, Trash2, ChevronDown, Terminal } from "lucide-react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { SearchTerm } from "@/types";
import { cn } from "@/lib/utils";

function buildQueryPreview(terms: SearchTerm[]): string {
  return terms
    .filter((t) => t.value.trim())
    .map((t, i) => {
      const val = t.exact_match ? `"${t.value}"` : t.value;
      return i === 0 ? val : `${t.operator} ${val}`;
    })
    .join(" ");
}

export function QueryBuilder() {
  const {
    searchTerms, technologyName, technologyDescription, minYear, autoSave,
    addTerm, updateTerm, removeTerm,
    setTechnologyName, setTechnologyDescription, setMinYear, setAutoSave,
  } = useAnalysisStore();

  const preview = buildQueryPreview(searchTerms);

  return (
    <div className="space-y-5">

      {/* Technology name */}
      <Field label="Tecnología">
        <input
          type="text"
          value={technologyName}
          onChange={(e) => setTechnologyName(e.target.value)}
          placeholder="Ej: Generative AI"
          className="inp"
        />
      </Field>

      {/* Description */}
      <Field label="Descripción (opcional)">
        <textarea
          value={technologyDescription}
          onChange={(e) => setTechnologyDescription(e.target.value)}
          rows={2}
          placeholder="Breve descripción de la tecnología..."
          className="inp resize-none"
        />
      </Field>

      {/* Año mínimo */}
      <Field label="Año mínimo">
        <input
          type="number"
          value={minYear}
          onChange={(e) => setMinYear(Number(e.target.value))}
          min={2000}
          max={new Date().getFullYear()}
          className="inp"
        />
      </Field>

      {/* Terms */}
      <div>
        <div className="flex items-center justify-between mb-2.5">
          <p className="text-xs font-semibold" style={{ color: "var(--t2)" }}>
            Términos de búsqueda
          </p>
          <button
            onClick={() => addTerm({ value: "", operator: "AND", exact_match: true })}
            className="flex items-center gap-1 text-xs font-medium transition-colors hover:opacity-80"
            style={{ color: "var(--accent)" }}
          >
            <Plus className="w-3.5 h-3.5" /> Agregar
          </button>
        </div>
        <div className="space-y-2">
          {searchTerms.map((term, i) => (
            <TermRow
              key={i}
              term={term}
              index={i}
              isFirst={i === 0}
              onChange={(t) => updateTerm(i, t)}
              onRemove={() => removeTerm(i)}
            />
          ))}
        </div>
      </div>

      {/* Query preview */}
      {preview && (
        <div
          className="rounded-lg p-3 space-y-1.5"
          style={{ background: "rgba(59,124,255,.06)", border: "1px solid rgba(59,124,255,.14)" }}
        >
          <div className="flex items-center gap-1.5">
            <Terminal className="w-3 h-3" style={{ color: "var(--accent)" }} />
            <p className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: "var(--accent)" }}>
              Query
            </p>
          </div>
          <code className="text-xs break-all leading-relaxed" style={{ color: "#86efac" }}>
            {preview}
          </code>
        </div>
      )}

      {/* Auto-save toggle */}
      <label className="flex items-center gap-2.5 cursor-pointer group">
        <div
          className={cn(
            "relative w-8 h-4 rounded-full transition-colors",
            autoSave ? "bg-blue-600" : "bg-white/10"
          )}
          onClick={() => setAutoSave(!autoSave)}
        >
          <div
            className={cn(
              "absolute top-0.5 w-3 h-3 rounded-full bg-white transition-transform",
              autoSave ? "translate-x-4" : "translate-x-0.5"
            )}
          />
        </div>
        <span className="text-xs" style={{ color: "var(--t2)" }}>
          Guardar automáticamente en DynamoDB
        </span>
      </label>
    </div>
  );
}

/* ── Helpers ── */

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--t2)" }}>{label}</p>
      {children}
    </div>
  );
}

interface TermRowProps {
  term: SearchTerm; index: number; isFirst: boolean;
  onChange: (t: SearchTerm) => void; onRemove: () => void;
}

function TermRow({ term, index, isFirst, onChange, onRemove }: TermRowProps) {
  return (
    <div className="flex items-center gap-2">
      {/* Operator */}
      {!isFirst ? (
        <div className="relative shrink-0">
          <select
            value={term.operator}
            onChange={(e) => onChange({ ...term, operator: e.target.value as SearchTerm["operator"] })}
            className="appearance-none text-xs font-bold pl-2.5 pr-6 py-2 rounded-lg cursor-pointer"
            style={{
              background: "var(--s3)",
              border: "1px solid var(--border)",
              color: term.operator === "NOT" ? "#fca5a5" : term.operator === "OR" ? "#86efac" : "#93c5fd",
            }}
          >
            <option value="AND">AND</option>
            <option value="OR">OR</option>
            <option value="NOT">NOT</option>
          </select>
          <ChevronDown className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 pointer-events-none" style={{ color: "var(--t3)" }} />
        </div>
      ) : (
        <div className="w-[58px] shrink-0 text-center">
          <span className="text-[10px] font-bold px-2 py-1 rounded" style={{ color: "var(--t3)", background: "var(--s2)" }}>
            #{index + 1}
          </span>
        </div>
      )}

      {/* Input */}
      <input
        type="text"
        value={term.value}
        onChange={(e) => onChange({ ...term, value: e.target.value })}
        placeholder={`Término ${index + 1}…`}
        className="inp flex-1"
        style={{ fontSize: 13 }}
      />

      {/* Exact match */}
      <label className="flex items-center gap-1 text-[11px] cursor-pointer shrink-0 whitespace-nowrap" style={{ color: "var(--t3)" }}>
        <input
          type="checkbox"
          checked={term.exact_match}
          onChange={(e) => onChange({ ...term, exact_match: e.target.checked })}
          className="accent-blue-500 w-3 h-3"
        />
        Exacto
      </label>

      {/* Remove */}
      <button
        onClick={onRemove}
        disabled={index === 0}
        className="p-1.5 rounded-lg transition-colors disabled:opacity-15"
        style={{ color: "var(--t3)" }}
        onMouseEnter={(e) => (e.currentTarget.style.color = "#fca5a5")}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--t3)")}
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

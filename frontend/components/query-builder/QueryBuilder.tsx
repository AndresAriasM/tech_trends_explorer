"use client";

import { Plus, Trash2, ChevronDown } from "lucide-react";
import { useAnalysisStore } from "@/store/analysisStore";
import type { SearchTerm } from "@/types";

// Preview of the final search query string — mirrors master analysis.py build_query()
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
    searchTerms,
    technologyName,
    technologyDescription,
    minYear,
    autoSave,
    addTerm,
    updateTerm,
    removeTerm,
    setTechnologyName,
    setTechnologyDescription,
    setMinYear,
    setAutoSave,
  } = useAnalysisStore();

  const preview = buildQueryPreview(searchTerms);

  return (
    <div className="space-y-6">
      {/* Technology info */}
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Nombre de la tecnología
          </label>
          <input
            type="text"
            value={technologyName}
            onChange={(e) => setTechnologyName(e.target.value)}
            placeholder="Ej: Artificial Intelligence"
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Año mínimo
          </label>
          <input
            type="number"
            value={minYear}
            onChange={(e) => setMinYear(Number(e.target.value))}
            min={2000}
            max={new Date().getFullYear()}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-300 mb-1">
          Descripción (opcional)
        </label>
        <textarea
          value={technologyDescription}
          onChange={(e) => setTechnologyDescription(e.target.value)}
          rows={2}
          placeholder="Descripción breve de la tecnología..."
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
        />
      </div>

      {/* Search terms */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-300">
            Términos de búsqueda
          </label>
          <button
            onClick={() =>
              addTerm({ value: "", operator: "AND", exact_match: true })
            }
            className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            Agregar término
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
        <div className="bg-gray-800 rounded-lg p-3">
          <p className="text-xs text-gray-500 mb-1">Vista previa de búsqueda</p>
          <code className="text-xs text-green-400 break-all">{preview}</code>
        </div>
      )}

      {/* Auto-save toggle */}
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={autoSave}
          onChange={(e) => setAutoSave(e.target.checked)}
          className="w-4 h-4 rounded accent-blue-500"
        />
        <span className="text-sm text-gray-400">
          Guardar automáticamente en DynamoDB
        </span>
      </label>
    </div>
  );
}

// ── Single term row ──────────────────────────────────────────

interface TermRowProps {
  term: SearchTerm;
  index: number;
  isFirst: boolean;
  onChange: (t: SearchTerm) => void;
  onRemove: () => void;
}

function TermRow({ term, index, isFirst, onChange, onRemove }: TermRowProps) {
  return (
    <div className="flex items-center gap-2">
      {/* Operator — hidden for first term */}
      {!isFirst ? (
        <div className="relative">
          <select
            value={term.operator}
            onChange={(e) =>
              onChange({ ...term, operator: e.target.value as SearchTerm["operator"] })
            }
            className="appearance-none bg-gray-700 border border-gray-600 rounded-lg pl-3 pr-7 py-2 text-xs text-white focus:outline-none focus:border-blue-500 cursor-pointer"
          >
            <option value="AND">AND</option>
            <option value="OR">OR</option>
            <option value="NOT">NOT</option>
          </select>
          <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 w-3 h-3 text-gray-400 pointer-events-none" />
        </div>
      ) : (
        <div className="w-[72px]" />
      )}

      {/* Term input */}
      <input
        type="text"
        value={term.value}
        onChange={(e) => onChange({ ...term, value: e.target.value })}
        placeholder={`Término ${index + 1}`}
        className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
      />

      {/* Exact match */}
      <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer whitespace-nowrap">
        <input
          type="checkbox"
          checked={term.exact_match}
          onChange={(e) => onChange({ ...term, exact_match: e.target.checked })}
          className="accent-blue-500"
        />
        Exacto
      </label>

      {/* Remove */}
      <button
        onClick={onRemove}
        disabled={index === 0}
        className="text-gray-600 hover:text-red-400 disabled:opacity-20 transition-colors"
      >
        <Trash2 className="w-4 h-4" />
      </button>
    </div>
  );
}

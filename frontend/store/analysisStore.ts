import { create } from "zustand";
import type { HypeCycleQuery, TaskProgress, SearchTerm } from "@/types";

interface AnalysisStore {
  // Current analysis state
  taskId: string | null;
  progress: TaskProgress | null;
  result: HypeCycleQuery | null;
  isRunning: boolean;
  error: string | null;

  // Selected history item
  selectedQuery: HypeCycleQuery | null;

  // Query builder draft
  searchTerms: SearchTerm[];
  technologyName: string;
  technologyDescription: string;
  categoryId: string;
  minYear: number;
  autoSave: boolean;

  // Actions
  setTaskId: (id: string) => void;
  setProgress: (p: TaskProgress) => void;
  setResult: (r: HypeCycleQuery) => void;
  setRunning: (v: boolean) => void;
  setError: (e: string | null) => void;
  setSelectedQuery: (q: HypeCycleQuery | null) => void;
  resetAnalysis: () => void;

  // Query builder actions
  addTerm: (term: SearchTerm) => void;
  updateTerm: (index: number, term: SearchTerm) => void;
  removeTerm: (index: number) => void;
  setTechnologyName: (v: string) => void;
  setTechnologyDescription: (v: string) => void;
  setCategoryId: (v: string) => void;
  setMinYear: (v: number) => void;
  setAutoSave: (v: boolean) => void;
}

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  taskId: null,
  progress: null,
  result: null,
  isRunning: false,
  error: null,
  selectedQuery: null,

  searchTerms: [{ value: "", operator: "AND", exact_match: true }],
  technologyName: "",
  technologyDescription: "",
  categoryId: "default",
  minYear: 2015,
  autoSave: true,

  setTaskId: (id) => set({ taskId: id }),
  setProgress: (p) => set({ progress: p }),
  setResult: (r) => set({ result: r }),
  setRunning: (v) => set({ isRunning: v }),
  setError: (e) => set({ error: e }),
  setSelectedQuery: (q) => set({ selectedQuery: q }),
  resetAnalysis: () =>
    set({ taskId: null, progress: null, result: null, isRunning: false, error: null }),

  addTerm: (term) =>
    set((s) => ({ searchTerms: [...s.searchTerms, term] })),
  updateTerm: (index, term) =>
    set((s) => {
      const terms = [...s.searchTerms];
      terms[index] = term;
      return { searchTerms: terms };
    }),
  removeTerm: (index) =>
    set((s) => ({
      searchTerms: s.searchTerms.filter((_, i) => i !== index),
    })),
  setTechnologyName: (v) => set({ technologyName: v }),
  setTechnologyDescription: (v) => set({ technologyDescription: v }),
  setCategoryId: (v) => set({ categoryId: v }),
  setMinYear: (v) => set({ minYear: v }),
  setAutoSave: (v) => set({ autoSave: v }),
}));

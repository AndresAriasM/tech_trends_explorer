"use client";

import { Loader2, CheckCircle, XCircle } from "lucide-react";
import type { TaskProgress } from "@/types";

interface Props {
  progress: TaskProgress;
}

export function AnalysisProgressBar({ progress }: Props) {
  const { status, progress: pct, message } = progress;

  const isError = status === "failed";
  const isDone = status === "completed";

  return (
    <div className="bg-gray-800 rounded-xl p-4 space-y-3">
      {/* Header */}
      <div className="flex items-center gap-2">
        {isDone ? (
          <CheckCircle className="w-4 h-4 text-green-400 shrink-0" />
        ) : isError ? (
          <XCircle className="w-4 h-4 text-red-400 shrink-0" />
        ) : (
          <Loader2 className="w-4 h-4 text-blue-400 animate-spin shrink-0" />
        )}
        <span
          className={`text-sm font-medium ${
            isDone
              ? "text-green-400"
              : isError
              ? "text-red-400"
              : "text-blue-400"
          }`}
        >
          {isDone ? "Análisis completado" : isError ? "Error en el análisis" : "Analizando..."}
        </span>
        <span className="ml-auto text-xs text-gray-500">{pct}%</span>
      </div>

      {/* Bar */}
      <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${
            isDone ? "bg-green-500" : isError ? "bg-red-500" : "bg-blue-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Message */}
      <p className="text-xs text-gray-400">{message}</p>
    </div>
  );
}

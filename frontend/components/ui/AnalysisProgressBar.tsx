"use client";

import { Loader2, CheckCircle2, XCircle } from "lucide-react";
import type { TaskProgress } from "@/types";

interface Props { progress: TaskProgress; }

export function AnalysisProgressBar({ progress }: Props) {
  const { status, progress: pct, message } = progress;
  const isDone  = status === "completed";
  const isError = status === "failed";

  const color  = isDone ? "#22c55e" : isError ? "#ef4444" : "var(--accent)";
  const bgGlow = isDone ? "rgba(34,197,94,.1)" : isError ? "rgba(239,68,68,.1)" : "var(--accent-sub)";

  return (
    <div
      className="rounded-xl p-3.5 space-y-2.5"
      style={{ background: bgGlow, border: `1px solid ${isDone ? "rgba(34,197,94,.2)" : isError ? "rgba(239,68,68,.2)" : "rgba(59,124,255,.15)"}` }}
    >
      <div className="flex items-center gap-2">
        {isDone
          ? <CheckCircle2 className="w-3.5 h-3.5 shrink-0" style={{ color }} />
          : isError
          ? <XCircle className="w-3.5 h-3.5 shrink-0" style={{ color }} />
          : <Loader2 className="w-3.5 h-3.5 shrink-0 animate-spin" style={{ color }} />
        }
        <span className="text-xs font-semibold" style={{ color }}>
          {isDone ? "Completado" : isError ? "Error" : "Procesando..."}
        </span>
        <span className="ml-auto text-xs tabular-nums" style={{ color: "var(--t3)" }}>{pct}%</span>
      </div>

      {/* Track */}
      <div className="w-full h-1.5 rounded-full overflow-hidden" style={{ background: "var(--s3)" }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>

      <p className="text-[11px] leading-relaxed" style={{ color: "var(--t2)" }}>{message}</p>
    </div>
  );
}

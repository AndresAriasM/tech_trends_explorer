"use client";

import Link from "next/link";
import {
  TrendingUp, ArrowRight, Zap, History, FolderOpen,
  BarChart3, Activity, Clock, CheckCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";

const RECENT = [
  { tech: "Generative AI",       phase: "Peak of Inflated Expectations", confidence: 91, ago: "hace 2h" },
  { tech: "Quantum Computing",   phase: "Innovation Trigger",            confidence: 78, ago: "hace 5h" },
  { tech: "Digital Twins",       phase: "Slope of Enlightenment",        confidence: 85, ago: "ayer" },
  { tech: "Blockchain",          phase: "Trough of Disillusionment",     confidence: 82, ago: "hace 2d" },
];

const PHASE_COLOR: Record<string, string> = {
  "Innovation Trigger":            "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment":     "#ef4444",
  "Slope of Enlightenment":        "#34d399",
  "Plateau of Productivity":       "#818cf8",
};

export default function DashboardPage() {
  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6 fade-in">

      {/* Hero */}
      <div
        className="relative overflow-hidden rounded-2xl p-8"
        style={{
          background: "linear-gradient(135deg,rgba(59,124,255,.1) 0%,rgba(124,92,255,.07) 60%,rgba(59,219,255,.05) 100%)",
          border: "1px solid rgba(59,124,255,.2)",
        }}
      >
        <div
          className="absolute -top-20 -right-20 w-72 h-72 rounded-full pointer-events-none"
          style={{ background: "radial-gradient(circle,rgba(59,124,255,.12) 0%,transparent 70%)" }}
        />
        <div className="relative flex items-start justify-between gap-6 flex-wrap">
          <div className="space-y-4">
            <div className="flex items-center gap-2">
              <div className="pulse" />
              <span className="text-xs font-semibold" style={{ color: "var(--accent)" }}>Sistema activo</span>
            </div>
            <h1 className="text-3xl font-bold" style={{ color: "var(--t1)" }}>
              Tech Trends <span className="gradient-text">Explorer</span>
            </h1>
            <p className="text-sm max-w-lg" style={{ color: "var(--t2)", lineHeight: 1.75 }}>
              Plataforma de análisis basada en el{" "}
              <strong style={{ color: "var(--t1)" }}>Hype Cycle de Gartner</strong>.
              Analiza el posicionamiento de tecnologías usando fuentes web, académicas y de mercado.
            </p>
            <div className="flex items-center gap-3">
              <Link href="/hype-cycle" className="btn btn-primary">
                <Zap className="w-4 h-4" /> Nuevo análisis
              </Link>
              <Link href="/history" className="btn btn-ghost">
                <History className="w-4 h-4" /> Historial
              </Link>
            </div>
          </div>

          {/* Stat pills */}
          <div className="flex flex-col gap-2 shrink-0">
            {[
              { label: "Análisis guardados", value: "—", icon: CheckCircle, color: "#22c55e" },
              { label: "Categorías",         value: "—", icon: FolderOpen,  color: "#3b7cff" },
              { label: "Último análisis",    value: "—", icon: Clock,       color: "#f59e0b" },
            ].map((s) => (
              <div
                key={s.label}
                className="flex items-center gap-3 px-4 py-2.5 rounded-xl"
                style={{ background: "rgba(255,255,255,.04)", border: "1px solid var(--border)" }}
              >
                <s.icon className="w-3.5 h-3.5 shrink-0" style={{ color: s.color }} />
                <div>
                  <p className="text-[10px]" style={{ color: "var(--t3)" }}>{s.label}</p>
                  <p className="text-sm font-bold" style={{ color: "var(--t1)" }}>{s.value}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Module cards */}
      <div>
        <p className="section-label mb-3">Módulos</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <ModuleCard
            icon={TrendingUp} title="Hype Cycle" color="#3b7cff"
            badge="Activo" badgeClass="badge-blue"
            description="Determina la fase de una tecnología en el ciclo de Gartner."
            actions={[
              { href: "/hype-cycle", label: "Analizar",    icon: Zap },
              { href: "/history",    label: "Historial",   icon: History },
              { href: "/categories", label: "Categorías",  icon: FolderOpen },
            ]}
          />
          <ModuleCard
            icon={Activity} title="Curvas S" color="#a78bfa"
            badge="En desarrollo" badgeClass="badge-purple"
            description="Analiza la adopción tecnológica con publicaciones académicas de Scopus."
            actions={[{ href: "/s-curve", label: "Próximamente", icon: BarChart3 }]}
            disabled
          />
          <ModuleCard
            icon={TrendingUp} title="Tendencias" color="#22c55e"
            badge="En desarrollo" badgeClass="badge-gray"
            description="Mide volumen y sentimiento de menciones en fuentes web y noticias."
            actions={[]}
            disabled
          />
        </div>
      </div>

      {/* Recent activity */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="section-label">Actividad reciente</p>
          <Link href="/history" className="text-xs flex items-center gap-1" style={{ color: "var(--accent)" }}>
            Ver todo <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
        <div className="rounded-xl overflow-hidden" style={{ border: "1px solid var(--border)", background: "var(--s1)" }}>
          {RECENT.map((item, i) => (
            <div
              key={i}
              className="flex items-center gap-4 px-5 py-3.5 hover:bg-white/[.018] transition-colors"
              style={{ borderBottom: i < RECENT.length - 1 ? "1px solid var(--border)" : "none" }}
            >
              <div className="w-2 h-2 rounded-full shrink-0" style={{ background: PHASE_COLOR[item.phase] ?? "#9ca3af" }} />
              <p className="text-sm font-medium flex-1" style={{ color: "var(--t1)" }}>{item.tech}</p>
              <p className="text-xs hidden sm:block" style={{ color: "var(--t2)" }}>{item.phase}</p>
              <div className="flex items-center gap-2 shrink-0">
                <div className="w-16 h-1.5 rounded-full overflow-hidden" style={{ background: "var(--s3)" }}>
                  <div className="h-full rounded-full" style={{ width: `${item.confidence}%`, background: PHASE_COLOR[item.phase] ?? "#3b7cff" }} />
                </div>
                <span className="text-xs w-8 text-right" style={{ color: "var(--t3)" }}>{item.confidence}%</span>
              </div>
              <p className="text-xs shrink-0" style={{ color: "var(--t3)" }}>{item.ago}</p>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

interface Action { href: string; label: string; icon: React.ElementType; }
interface ModuleCardProps {
  icon: React.ElementType; title: string; description: string;
  badge: string; badgeClass: string; color: string;
  disabled?: boolean; actions: Action[];
}

function ModuleCard({ icon: Icon, title, description, badge, badgeClass, color, disabled, actions }: ModuleCardProps) {
  return (
    <div className={cn("card card-hover flex flex-col p-5", disabled && "opacity-50")}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center justify-center w-9 h-9 rounded-xl" style={{ background: `${color}18`, border: `1px solid ${color}28` }}>
          <Icon className="w-4 h-4" style={{ color }} />
        </div>
        <span className={cn("badge", badgeClass)}>{badge}</span>
      </div>
      <h3 className="text-sm font-bold mb-1.5" style={{ color: "var(--t1)" }}>{title}</h3>
      <p className="text-xs leading-relaxed flex-1" style={{ color: "var(--t2)" }}>{description}</p>
      {!disabled && actions.length > 0 && (
        <div className="mt-4 pt-4 space-y-0.5" style={{ borderTop: "1px solid var(--border)" }}>
          {actions.map(({ href, label, icon: AIcon }) => (
            <Link key={href} href={href}
              className="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs font-medium transition-colors hover:bg-white/5"
              style={{ color: "var(--t2)" }}
            >
              <AIcon className="w-3.5 h-3.5 shrink-0" style={{ color }} />
              <span className="flex-1">{label}</span>
              <ArrowRight className="w-3 h-3 opacity-25" />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

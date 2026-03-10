"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  TrendingUp, History, FolderOpen, BarChart3,
  Search, ChevronRight, PanelLeft, Zap, Activity,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV = [
  {
    group: "Hype Cycle",
    icon: TrendingUp,
    color: "#3b7cff",
    badge: "Activo",
    badgeClass: "badge-blue",
    items: [
      { href: "/hype-cycle", label: "Análisis",    icon: Zap },
      { href: "/history",    label: "Historial",   icon: History },
      { href: "/categories", label: "Categorías",  icon: FolderOpen },
    ],
  },
  {
    group: "Curvas S",
    icon: Activity,
    color: "#a78bfa",
    badge: "Pronto",
    badgeClass: "badge-purple",
    items: [
      { href: "/s-curve", label: "Análisis Scopus", icon: BarChart3 },
    ],
  },
  {
    group: "Tendencias",
    icon: Search,
    color: "#22c55e",
    badge: "Pronto",
    badgeClass: "badge-gray",
    items: [
      { href: "/trends", label: "Búsqueda web", icon: Search },
    ],
  },
];

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: Props) {
  const pathname = usePathname();

  return (
    <aside
      className="sidebar flex flex-col shrink-0"
      style={{
        width: collapsed ? 60 : "var(--sidebar-w)",
        transition: "width .2s cubic-bezier(.4,0,.2,1)",
        minHeight: "100vh",
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-3 px-3 py-3 shrink-0"
        style={{ height: "var(--header-h)", borderBottom: "1px solid var(--border)" }}
      >
        <div
          className="flex items-center justify-center shrink-0 rounded-lg"
          style={{
            width: 32, height: 32,
            background: "linear-gradient(135deg,#3b7cff 0%,#7c5cff 100%)",
            boxShadow: "0 0 14px rgba(59,124,255,.4)",
          }}
        >
          <TrendingUp className="w-4 h-4 text-white" />
        </div>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <p className="text-sm font-bold leading-none" style={{ color: "var(--t1)" }}>Tech Trends</p>
            <p className="text-[10px] mt-0.5" style={{ color: "var(--t3)" }}>Explorer · UPB 2036</p>
          </div>
        )}
        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg transition-colors hover:bg-white/5 shrink-0"
          style={{ color: "var(--t3)" }}
          title={collapsed ? "Expandir" : "Colapsar"}
        >
          <PanelLeft className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-3 overflow-y-auto overflow-x-hidden space-y-1">
        {NAV.map((group, gi) => {
          const GroupIcon = group.icon;
          const isDisabledGroup = group.badge === "Pronto";

          return (
            <div key={group.group}>
              {/* Group label */}
              {!collapsed ? (
                <div className="flex items-center gap-2 px-4 py-1.5">
                  <GroupIcon className="w-3.5 h-3.5 shrink-0" style={{ color: group.color }} />
                  <span className="section-label flex-1 m-0 p-0">{group.group}</span>
                  <span className={cn("badge", group.badgeClass)}>{group.badge}</span>
                </div>
              ) : (
                gi > 0 && (
                  <div className="my-2 mx-3" style={{ height: 1, background: "var(--border)" }} />
                )
              )}

              {/* Items */}
              <div className={cn("space-y-0.5", collapsed ? "px-1.5" : "px-2")}>
                {group.items.map(({ href, label, icon: Icon }) => {
                  const active = pathname === href || (href !== "/" && pathname.startsWith(href));
                  return (
                    <Link
                      key={href}
                      href={isDisabledGroup ? "#" : href}
                      title={collapsed ? label : undefined}
                      onClick={(e) => isDisabledGroup && e.preventDefault()}
                      className={cn(
                        "flex items-center gap-2.5 py-2 rounded-lg text-[13px] font-medium transition-all duration-150",
                        collapsed ? "justify-center px-0" : "px-3",
                        active ? "text-blue-400" : "text-[color:var(--t2)] hover:text-[color:var(--t1)]",
                        isDisabledGroup && "opacity-35 cursor-not-allowed pointer-events-none",
                      )}
                      style={{
                        background: active ? "var(--accent-sub)" : "transparent",
                        border: `1px solid ${active ? "rgba(59,124,255,.18)" : "transparent"}`,
                      }}
                    >
                      <Icon
                        className="w-3.5 h-3.5 shrink-0"
                        style={{ color: active ? "var(--accent)" : undefined }}
                      />
                      {!collapsed && (
                        <>
                          <span className="flex-1">{label}</span>
                          {active && <ChevronRight className="w-3 h-3 opacity-40" />}
                        </>
                      )}
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>

      {/* User stub — auth-ready */}
      <div className="px-2 py-2 shrink-0" style={{ borderTop: "1px solid var(--border)" }}>
        <div
          className={cn(
            "flex items-center gap-2.5 p-2 rounded-lg cursor-pointer transition-colors hover:bg-white/5",
            collapsed && "justify-center",
          )}
        >
          <div
            className="w-7 h-7 rounded-full shrink-0 flex items-center justify-center text-[11px] font-bold"
            style={{ background: "linear-gradient(135deg,#3b7cff,#7c5cff)", color: "#fff" }}
          >
            A
          </div>
          {!collapsed && (
            <>
              <div className="min-w-0 flex-1">
                <p className="text-xs font-semibold truncate" style={{ color: "var(--t1)" }}>Andrés Arias</p>
                <p className="text-[10px] truncate" style={{ color: "var(--t3)" }}>UPB · Data Scientist</p>
              </div>
              <div className="pulse shrink-0" title="Conectado" />
            </>
          )}
        </div>
      </div>
    </aside>
  );
}

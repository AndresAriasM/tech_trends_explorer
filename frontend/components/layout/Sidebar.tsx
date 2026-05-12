"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  TrendingUp, History, FolderOpen,
  Search, ChevronRight, PanelLeft, Zap, Activity, Users,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_GROUPS = [
  {
    group: "HYPE CYCLE",
    badge: "Activo",
    badgeClass: "badge-blue",
    items: [
      { href: "/hype-cycle", label: "Análisis",   icon: Zap },
      { href: "/history",    label: "Historial",  icon: History },
      { href: "/categories", label: "Categorías", icon: FolderOpen },
    ],
    disabled: false,
  },
  {
    group: "CURVAS S",
    badge: "Activo",
    badgeClass: "badge-blue",
    items: [
      { href: "/s-curve", label: "Análisis S", icon: Activity },
    ],
    disabled: false,
  },
  {
    group: "PROSPECTIVA",
    badge: "Nuevo",
    badgeClass: "badge-blue",
    items: [
      { href: "/workshop", label: "Taller LACCEI", icon: Users },
    ],
    disabled: false,
  },
  {
    group: "PRÓXIMAMENTE",
    badge: null,
    badgeClass: null,
    items: [
      { href: "/trends", label: "Búsqueda Web", icon: Search },
    ],
    disabled: true,
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
      className="sidebar flex flex-col"
      style={{ width: collapsed ? 60 : 280 }}
    >
      {/* ── Logo area (64px tall) ── */}
      <div
        className="flex items-center gap-3 shrink-0 px-4"
        style={{
          height: 64,
          borderBottom: "1px solid var(--border)",
        }}
      >
        {/* Icon */}
        <div
          className="flex items-center justify-center shrink-0 rounded-xl"
          style={{
            width: 36,
            height: 36,
            background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
            boxShadow: "0 0 16px rgba(79,142,247,0.4)",
            flexShrink: 0,
          }}
        >
          <TrendingUp style={{ width: 18, height: 18, color: "#fff" }} />
        </div>

        {/* Wordmark */}
        {!collapsed && (
          <div className="flex-1 min-w-0">
            <p
              className="font-bold leading-none truncate"
              style={{
                fontSize: 13.5,
                color: "var(--text-1)",
                letterSpacing: "-0.025em",
                fontFeatureSettings: '"cv02", "cv11"',
              }}
            >
              Tech Trends Explorer
            </p>
            <p
              className="mt-0.5 truncate"
              style={{ fontSize: 10, color: "var(--text-3)", letterSpacing: "0.08em", textTransform: "uppercase" }}
            >
              UPB · 2036
            </p>
          </div>
        )}

        {/* Collapse toggle */}
        <button
          onClick={onToggle}
          title={collapsed ? "Expandir sidebar" : "Colapsar sidebar"}
          style={{
            padding: "7px",
            borderRadius: 8,
            color: "var(--text-3)",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            flexShrink: 0,
            transition: "background 0.15s, color 0.15s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.06)";
            e.currentTarget.style.color = "var(--text-2)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "var(--text-3)";
          }}
        >
          <PanelLeft style={{ width: 15, height: 15 }} />
        </button>
      </div>

      {/* ── Navigation ── */}
      <nav
        className="flex-1 overflow-y-auto overflow-x-hidden"
        style={{ padding: "16px 0" }}
      >
        {NAV_GROUPS.map((group, gi) => (
          <div key={group.group} style={{ marginBottom: gi < NAV_GROUPS.length - 1 ? 20 : 0 }}>
            {/* Group label */}
            {!collapsed ? (
              <div
                className="flex items-center gap-2 px-5 mb-1"
                style={{ marginBottom: 6 }}
              >
                <span className="label-sm" style={{ color: "var(--text-3)" }}>
                  {group.group}
                </span>
                {group.badge && (
                  <span className={cn("badge", group.badgeClass ?? "")}>
                    {group.badge}
                  </span>
                )}
              </div>
            ) : (
              gi > 0 && (
                <div
                  style={{
                    height: 1,
                    background: "var(--border)",
                    margin: "12px 12px",
                  }}
                />
              )
            )}

            {/* Items */}
            <div style={{ padding: "0 8px" }}>
              {group.items.map(({ href, label, icon: Icon }) => {
                const active = pathname === href || (href !== "/" && pathname.startsWith(href));

                return (
                  <Link
                    key={href}
                    href={group.disabled ? "#" : href}
                    title={collapsed ? label : undefined}
                    onClick={(e) => group.disabled && e.preventDefault()}
                    className={cn(
                      "flex items-center gap-3 rounded-xl transition-all duration-150",
                      collapsed ? "justify-center px-0" : "px-3",
                      group.disabled && "pointer-events-none",
                    )}
                    style={{
                      height: 44,
                      fontSize: 14,
                      fontWeight: active ? 600 : 500,
                      color: active
                        ? "var(--accent)"
                        : group.disabled
                        ? "var(--text-4)"
                        : "var(--text-2)",
                      background: active
                        ? "rgba(79,142,247,0.10)"
                        : "transparent",
                      borderLeft: active && !collapsed
                        ? "2px solid var(--accent)"
                        : "2px solid transparent",
                      marginBottom: 2,
                      textDecoration: "none",
                      transition: "all 0.15s",
                    }}
                    onMouseEnter={(e) => {
                      if (!active && !group.disabled) {
                        e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                        e.currentTarget.style.color = "var(--text-1)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (!active && !group.disabled) {
                        e.currentTarget.style.background = "transparent";
                        e.currentTarget.style.color = group.disabled
                          ? "var(--text-4)"
                          : "var(--text-2)";
                      }
                    }}
                  >
                    <Icon
                      style={{
                        width: 16,
                        height: 16,
                        flexShrink: 0,
                        color: active ? "var(--accent)" : "inherit",
                      }}
                    />
                    {!collapsed && (
                      <>
                        <span style={{ flex: 1 }}>{label}</span>
                        {active && (
                          <ChevronRight
                            style={{ width: 13, height: 13, opacity: 0.5 }}
                          />
                        )}
                      </>
                    )}
                  </Link>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* ── User area (72px) ── */}
      <div
        className="shrink-0"
        style={{
          borderTop: "1px solid var(--border)",
          padding: "12px 8px",
          height: 72,
          display: "flex",
          alignItems: "center",
        }}
      >
        <div
          className={cn(
            "flex items-center gap-3 rounded-xl cursor-pointer w-full",
            collapsed ? "justify-center" : "px-3",
          )}
          style={{
            height: 48,
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.05)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
          }}
        >
          {/* Avatar */}
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 13,
              fontWeight: 700,
              color: "#fff",
              flexShrink: 0,
            }}
          >
            A
          </div>

          {!collapsed && (
            <>
              <div style={{ flex: 1, minWidth: 0 }}>
                <p
                  className="truncate"
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: "var(--text-1)",
                    lineHeight: 1.2,
                  }}
                >
                  Andrés Arias
                </p>
                <p
                  className="truncate"
                  style={{
                    fontSize: 11,
                    color: "var(--text-3)",
                    marginTop: 2,
                  }}
                >
                  Data Scientist · UPB
                </p>
              </div>
              {/* Online dot */}
              <div
                className="pulse-dot shrink-0"
                title="En línea"
              />
            </>
          )}
        </div>
      </div>
    </aside>
  );
}

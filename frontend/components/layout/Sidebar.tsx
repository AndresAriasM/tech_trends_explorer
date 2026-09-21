"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronRight, PanelLeft } from "lucide-react";
import { cn } from "@/lib/utils";
import { NAV_SECTIONS, type NavItem } from "@/lib/modules";

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

function isActive(pathname: string, href: string) {
  return pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
}

export function Sidebar({ collapsed, onToggle }: Props) {
  const pathname = usePathname();

  return (
    <aside
      className="sidebar flex flex-col"
      style={{ width: collapsed ? 60 : 280 }}
    >
      {/* ── Marca ── */}
      <div
        className="flex items-center gap-3 shrink-0 px-4"
        style={{ height: 64, borderBottom: "1px solid var(--border)" }}
      >
        <Link
          href="/"
          className="flex items-center gap-3 min-w-0 flex-1"
          style={{ textDecoration: "none" }}
          aria-label="Prospecta — inicio"
        >
          <Image
            src="/prospecta-mark.png"
            alt=""
            width={34}
            height={34}
            priority
            style={{ width: 34, height: 34, objectFit: "contain", flexShrink: 0 }}
          />
          {!collapsed && (
            <div className="min-w-0">
              <p
                className="font-bold leading-none truncate"
                style={{
                  fontSize: 15,
                  color: "var(--text-1)",
                  letterSpacing: "-0.02em",
                }}
              >
                Prospecta
              </p>
              <p
                className="mt-1 truncate"
                style={{
                  fontSize: 9.5,
                  color: "var(--text-3)",
                  letterSpacing: "0.09em",
                  textTransform: "uppercase",
                }}
              >
                Prospectiva · UPB
              </p>
            </div>
          )}
        </Link>

        <button
          onClick={onToggle}
          title={collapsed ? "Expandir menú" : "Colapsar menú"}
          aria-label={collapsed ? "Expandir menú" : "Colapsar menú"}
          aria-expanded={!collapsed}
          style={{
            padding: 7,
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

      {/* ── Navegación ── */}
      <nav
        className="flex-1 overflow-y-auto overflow-x-hidden"
        style={{ padding: "14px 0" }}
        aria-label="Módulos"
      >
        {NAV_SECTIONS.map((section, si) => (
          <div key={section.id} style={{ marginBottom: 16 }}>
            {!collapsed ? (
              <div className="px-5" style={{ marginBottom: 4 }}>
                <span className="label-sm" style={{ color: "var(--text-3)" }}>
                  {section.label}
                </span>
              </div>
            ) : (
              si > 0 && (
                <div
                  style={{
                    height: 1,
                    background: "var(--border)",
                    margin: "10px 12px",
                  }}
                />
              )
            )}

            <div style={{ padding: "0 8px" }}>
              {section.items.map((item) => (
                <NavLink
                  key={item.href}
                  item={item}
                  collapsed={collapsed}
                  active={isActive(pathname, item.href)}
                />
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* ── Usuario ── */}
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
            collapsed ? "justify-center" : "px-3"
          )}
          style={{ height: 48, transition: "background 0.15s" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.05)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
          }}
        >
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #2f6fe0 0%, #6aa9ff 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 13,
              fontWeight: 700,
              color: "#fff",
              flexShrink: 0,
            }}
          >
            AA
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
                  style={{ fontSize: 11, color: "var(--text-3)", marginTop: 2 }}
                >
                  Data Scientist · UPB
                </p>
              </div>
              <div className="pulse-dot shrink-0" title="En línea" />
            </>
          )}
        </div>
      </div>
    </aside>
  );
}

/* ── Ítem de navegación ─────────────────────────────────────── */

function NavLink({
  item,
  collapsed,
  active,
}: {
  item: NavItem;
  collapsed: boolean;
  active: boolean;
}) {
  const { href, label, icon: Icon, disabled } = item;

  const restingColor = disabled ? "var(--text-4)" : "var(--text-2)";
  const title = collapsed
    ? disabled
      ? `${label} — próximamente`
      : label
    : undefined;

  return (
    <Link
      href={disabled ? "#" : href}
      title={title}
      aria-disabled={disabled || undefined}
      aria-current={active ? "page" : undefined}
      tabIndex={disabled ? -1 : undefined}
      onClick={(e) => disabled && e.preventDefault()}
      className={cn(
        "flex items-center gap-3 rounded-xl transition-all duration-150",
        collapsed ? "justify-center px-0" : "px-3",
        disabled && "cursor-default"
      )}
      style={{
        height: 40,
        fontSize: 13.5,
        fontWeight: active ? 600 : 500,
        color: active ? "var(--accent)" : restingColor,
        background: active ? "rgba(79,142,247,0.10)" : "transparent",
        borderLeft:
          active && !collapsed
            ? "2px solid var(--accent)"
            : "2px solid transparent",
        marginBottom: 2,
        textDecoration: "none",
        transition: "all 0.15s",
      }}
      onMouseEnter={(e) => {
        if (!active && !disabled) {
          e.currentTarget.style.background = "rgba(255,255,255,0.04)";
          e.currentTarget.style.color = "var(--text-1)";
        }
      }}
      onMouseLeave={(e) => {
        if (!active && !disabled) {
          e.currentTarget.style.background = "transparent";
          e.currentTarget.style.color = restingColor;
        }
      }}
    >
      <Icon
        style={{
          width: 16,
          height: 16,
          flexShrink: 0,
          color: active ? "var(--accent)" : "inherit",
          opacity: disabled ? 0.55 : 1,
        }}
      />
      {!collapsed && (
        <>
          <span style={{ flex: 1, minWidth: 0 }} className="truncate">
            {label}
          </span>
          {disabled ? (
            <span
              style={{
                fontSize: 9,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: "var(--text-4)",
                border: "1px solid var(--border)",
                borderRadius: 5,
                padding: "1px 5px",
                flexShrink: 0,
              }}
            >
              Pronto
            </span>
          ) : (
            active && (
              <ChevronRight style={{ width: 13, height: 13, opacity: 0.5 }} />
            )
          )}
        </>
      )}
    </Link>
  );
}

"use client";

import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronRight, PanelLeft, Settings2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { ROLE_LABELS, getRole, getUser, onAuthChange, type Role, type SessionUser } from "@/lib/auth";
import {
  HOME_ITEM,
  NAV_SECTIONS,
  type NavItem,
  type SectionBadge,
} from "@/lib/modules";

interface Props {
  collapsed: boolean;
  onToggle: () => void;
}

function isActive(pathname: string, href: string) {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

const BADGE_TONE: Record<SectionBadge["tone"], { fg: string; bg: string; bd: string }> = {
  green:  { fg: "#22c55e", bg: "rgba(34,197,94,0.10)",   bd: "rgba(34,197,94,0.24)" },
  purple: { fg: "#a78bfa", bg: "rgba(167,139,250,0.10)", bd: "rgba(167,139,250,0.24)" },
  gray:   { fg: "var(--text-3)", bg: "rgba(255,255,255,0.04)", bd: "var(--border)" },
};

export function Sidebar({ collapsed, onToggle }: Props) {
  const pathname = usePathname();

  const [user, setUser] = useState<SessionUser | null>(null);
  const [role, setRole] = useState<Role | null>(null);
  useEffect(() => {
    const sync = () => {
      setUser(getUser());
      setRole(getRole());
    };
    sync();
    return onAuthChange(sync);
  }, []);

  /* La lista de módulos es más larga que la ventana en pantallas cortas:
     su scroll es propio y se avisa con un degradado arriba/abajo. */
  const listRef = useRef<HTMLDivElement>(null);
  const [shade, setShade] = useState({ top: false, bottom: false });

  const syncShade = useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    setShade({
      top: scrollTop > 4,
      bottom: scrollTop + clientHeight < scrollHeight - 4,
    });
  }, []);

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    const raf = requestAnimationFrame(syncShade);
    const ro = new ResizeObserver(syncShade);
    ro.observe(el);
    window.addEventListener("resize", syncShade);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("resize", syncShade);
    };
  }, [syncShade, collapsed]);

  /* Si la lista desborda, el módulo en curso puede quedar fuera de vista:
     se trae al área visible al cambiar de ruta. */
  useEffect(() => {
    const el = listRef.current?.querySelector('[aria-current="page"]');
    el?.scrollIntoView({ block: "nearest" });
  }, [pathname]);

  return (
    <aside
      className="sidebar flex flex-col"
      style={{ width: collapsed ? "var(--sidebar-collapsed-w)" : "var(--sidebar-w)" }}
    >
      {/* ── Marca (misma altura que la barra superior) ── */}
      <div
        className="flex items-center shrink-0"
        style={{
          height: "var(--header-h)",
          padding: collapsed ? "0 12px" : "0 12px 0 14px",
          gap: 10,
          borderBottom: "1px solid var(--border)",
        }}
      >
        <Link
          href="/"
          className="flex items-center min-w-0 flex-1"
          style={{ textDecoration: "none", gap: 10 }}
          aria-label="Prospecta — inicio"
        >
          <Image
            src="/prospecta-mark.png"
            alt=""
            width={32}
            height={32}
            priority
            style={{ width: 32, height: 32, objectFit: "contain", flexShrink: 0 }}
          />
          {!collapsed && (
            <div className="min-w-0">
              <p
                className="font-bold leading-none truncate"
                style={{ fontSize: 15, color: "var(--text-1)", letterSpacing: "-0.02em" }}
              >
                Prospecta
              </p>
              <p
                className="truncate"
                style={{
                  fontSize: 8.5,
                  marginTop: 3,
                  color: "var(--text-3)",
                  letterSpacing: "0.10em",
                  textTransform: "uppercase",
                  lineHeight: 1,
                }}
              >
                Portal de prospectiva · UPB
              </p>
            </div>
          )}
        </Link>

        {!collapsed && (
          <button
            onClick={onToggle}
            className="icon-btn shrink-0"
            title="Colapsar menú"
            aria-label="Colapsar menú"
            aria-expanded
            style={{ width: 28, height: 28 }}
          >
            <PanelLeft style={{ width: 15, height: 15 }} />
          </button>
        )}
      </div>

      {collapsed && (
        <button
          onClick={onToggle}
          className="icon-btn shrink-0"
          title="Expandir menú"
          aria-label="Expandir menú"
          aria-expanded={false}
          style={{ width: 28, height: 28, margin: "10px auto 0" }}
        >
          <PanelLeft style={{ width: 15, height: 15 }} />
        </button>
      )}

      {/* ── Navegación ── */}
      <div className="relative flex-1 min-h-0">
        <div className={cn("nav-fade nav-fade-top", shade.top && "is-on")} />
        <div className={cn("nav-fade nav-fade-bottom", shade.bottom && "is-on")} />

        <nav
          ref={listRef}
          onScroll={syncShade}
          className="nav-scroll h-full"
          style={{
            padding: `12px var(--nav-gutter) 14px`,
            display: "flex",
            flexDirection: "column",
            gap: "var(--nav-section-gap)",
          }}
          aria-label="Módulos"
        >
          {/* Inicio, suelto por encima de las secciones */}
          <NavLink
            item={HOME_ITEM}
            collapsed={collapsed}
            active={isActive(pathname, HOME_ITEM.href)}
          />

          {NAV_SECTIONS.map((section) => (
            <div key={section.id}>
              {collapsed ? (
                <div
                  aria-hidden
                  style={{
                    height: 1,
                    background: "var(--border)",
                    margin: `0 4px var(--nav-label-gap)`,
                  }}
                />
              ) : (
                <div
                  className="flex items-center justify-between"
                  style={{
                    padding: "0 10px",
                    marginBottom: "var(--nav-label-gap)",
                    gap: 8,
                    minHeight: 16,
                  }}
                >
                  <span
                    style={{
                      fontSize: 9.5,
                      fontWeight: 700,
                      letterSpacing: "0.10em",
                      textTransform: "uppercase",
                      color: "var(--text-3)",
                      lineHeight: 1.25,
                    }}
                  >
                    {section.label}
                  </span>
                  {section.badge && <SectionPill badge={section.badge} />}
                </div>
              )}

              <div style={{ display: "flex", flexDirection: "column", gap: "var(--nav-item-gap)" }}>
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
      </div>

      {/* ── Usuario ── */}
      <div
        className="shrink-0 flex items-center"
        style={{
          height: "var(--nav-footer-h)",
          borderTop: "1px solid var(--border)",
          padding: `0 ${collapsed ? "8px" : "10px"}`,
        }}
      >
        <Link
          href="/cuenta"
          title="Mi cuenta"
          className={cn(
            "flex items-center rounded-xl w-full",
            collapsed && "justify-center"
          )}
          style={{
            gap: 10,
            height: 46,
            padding: collapsed ? 0 : "0 8px",
            textDecoration: "none",
            background:
              pathname === "/cuenta" ? "rgba(79,142,247,0.10)" : "transparent",
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => {
            if (pathname !== "/cuenta")
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
          }}
          onMouseLeave={(e) => {
            if (pathname !== "/cuenta")
              e.currentTarget.style.background = "transparent";
          }}
        >
          <span
            style={{
              width: 30,
              height: 30,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 12,
              fontWeight: 700,
              color: "#fff",
              flexShrink: 0,
            }}
          >
            {(user?.email?.[0] ?? "?").toUpperCase()}
          </span>

          {!collapsed && (
            <>
              <span style={{ flex: 1, minWidth: 0 }}>
                <span
                  className="truncate"
                  style={{
                    display: "block",
                    fontSize: 12.5,
                    fontWeight: 600,
                    color: "var(--text-1)",
                    lineHeight: 1.25,
                  }}
                >
                  {user?.email?.split("@")[0] ?? "Sin sesión"}
                </span>
                <span
                  className="truncate"
                  style={{
                    display: "block",
                    fontSize: 10.5,
                    color: "var(--text-3)",
                    marginTop: 1,
                  }}
                >
                  {role ? ROLE_LABELS[role] : "—"}
                </span>
              </span>
              <Settings2
                style={{ width: 14, height: 14, color: "var(--text-3)", flexShrink: 0 }}
              />
            </>
          )}
        </Link>
      </div>

    </aside>
  );
}

/* ── Distintivo de sección ──────────────────────────────────── */

function SectionPill({ badge }: { badge: SectionBadge }) {
  const tone = BADGE_TONE[badge.tone];
  return (
    <span
      style={{
        flexShrink: 0,
        fontSize: 8.5,
        fontWeight: 700,
        letterSpacing: "0.06em",
        textTransform: "uppercase",
        lineHeight: 1,
        padding: "3px 6px",
        borderRadius: 99,
        color: tone.fg,
        background: tone.bg,
        border: `1px solid ${tone.bd}`,
      }}
    >
      {badge.text}
    </span>
  );
}

/* ── Ítem de navegación ─────────────────────────────────────── */

function NavLink({
  item,
  collapsed,
  active,
}: {
  item: Pick<NavItem, "href" | "label" | "icon" | "disabled">;
  collapsed: boolean;
  active: boolean;
}) {
  const { href, label, icon: Icon, disabled } = item;

  const restingColor = disabled ? "var(--text-4)" : "var(--text-2)";
  const title = collapsed
    ? disabled
      ? `${label} — próximamente`
      : label
    : disabled
      ? `${label} — próximamente`
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
        "relative flex items-center transition-colors duration-150",
        collapsed ? "justify-center" : "",
        disabled && "cursor-default"
      )}
      style={{
        height: "var(--nav-item-h)",
        gap: 10,
        padding: collapsed ? 0 : "0 10px",
        borderRadius: 9,
        fontSize: 13.5,
        fontWeight: active ? 600 : 500,
        color: active ? "var(--text-1)" : restingColor,
        background: active ? "rgba(79,142,247,0.12)" : "transparent",
        textDecoration: "none",
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
      {/* Marca de posición: barra al borde del panel, no dentro del ítem,
          para que no desplace el texto ni cambie la altura de la fila. */}
      {active && (
        <span
          aria-hidden
          style={{
            position: "absolute",
            left: "calc(-1 * var(--nav-gutter))",
            top: 6,
            bottom: 6,
            width: 3,
            borderRadius: "0 3px 3px 0",
            background: "var(--accent)",
          }}
        />
      )}

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
                fontSize: 8.5,
                fontWeight: 600,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                color: "var(--text-4)",
                border: "1px solid var(--border)",
                borderRadius: 5,
                padding: "2px 5px",
                lineHeight: 1,
                flexShrink: 0,
              }}
            >
              Pronto
            </span>
          ) : (
            active && (
              <ChevronRight
                style={{ width: 13, height: 13, opacity: 0.5, flexShrink: 0 }}
              />
            )
          )}
        </>
      )}
    </Link>
  );
}

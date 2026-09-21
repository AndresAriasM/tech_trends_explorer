"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Bell, ChevronDown, ChevronRight, Home, Search } from "lucide-react";
import {
  getModuleByPath,
  getRouteByPath,
  getSectionCrumb,
} from "@/lib/modules";

const NOTIFICATIONS = 3;

const USER = { short: "Andrés A.", initial: "A" };

/** Atajo del buscador. Se muestra el símbolo de macOS, como en el diseño. */
const SEARCH_SHORTCUT = "⌘K";

/** Miga de pan derivada del registro de módulos, no de un mapa paralelo. */
function useCrumb(pathname: string): { section: string; page: string } {
  if (pathname === "/") return { section: "Portal", page: "Inicio" };

  const mod = getModuleByPath(pathname);
  if (!mod) return { section: "Portal", page: "" };

  const route = getRouteByPath(pathname) ?? mod.routes[0];
  // Rutas anidadas (p. ej. /workshop/ABC123) cuelgan del nombre de su módulo.
  const page =
    route && pathname !== route.href ? mod.name : (route?.label ?? mod.name);

  return { section: getSectionCrumb(route.section), page };
}

export function TopBar() {
  const pathname = usePathname();
  const crumb = useCrumb(pathname);

  return (
    <header
      className="shrink-0 flex items-center"
      style={{
        height: "var(--header-h)",
        gap: 24,
        padding: "0 20px",
        borderBottom: "1px solid var(--border)",
        background: "rgba(6,9,15,0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        position: "sticky",
        top: 0,
        zIndex: 40,
      }}
    >
      {/* ── Miga de pan ── */}
      <nav
        className="flex items-center min-w-0"
        style={{ gap: 7, flex: "0 1 auto" }}
        aria-label="Ruta"
      >
        <Link
          href="/"
          className="icon-btn"
          style={{ width: 26, height: 26, flexShrink: 0 }}
          aria-label="Inicio"
          title="Inicio"
        >
          <Home style={{ width: 15, height: 15 }} />
        </Link>
        <ChevronRight style={{ width: 13, height: 13, color: "var(--text-4)", flexShrink: 0 }} />
        <span
          className="truncate"
          style={{ fontSize: 13, fontWeight: 500, color: "var(--text-3)" }}
        >
          {crumb.section}
        </span>
        {crumb.page && (
          <>
            <ChevronRight
              style={{ width: 13, height: 13, color: "var(--text-4)", flexShrink: 0 }}
            />
            <span
              className="truncate"
              style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)" }}
            >
              {crumb.page}
            </span>
          </>
        )}
      </nav>

      {/* ── Buscador central ── */}
      <div className="flex justify-end flex-1 min-w-0">
        <button
          type="button"
          className="topbar-search"
          title="Buscar en el portal"
          aria-label="Buscar en el portal"
        >
          <Search style={{ width: 14, height: 14, flexShrink: 0 }} />
          <span className="truncate" style={{ flex: 1 }}>
            Buscar análisis, tecnologías…
          </span>
          <span className="kbd" aria-hidden>
            {SEARCH_SHORTCUT}
          </span>
        </button>
      </div>

      {/* ── Acciones ── */}
      <div className="flex items-center shrink-0" style={{ gap: 10 }}>
        <button
          className="icon-btn"
          style={{ width: 32, height: 32, position: "relative" }}
          title={`${NOTIFICATIONS} notificaciones`}
          aria-label={`${NOTIFICATIONS} notificaciones`}
        >
          <Bell style={{ width: 16, height: 16 }} />
          {NOTIFICATIONS > 0 && (
            <span
              style={{
                position: "absolute",
                top: 2,
                right: 1,
                minWidth: 14,
                height: 14,
                padding: "0 3px",
                borderRadius: 99,
                background: "var(--accent)",
                border: "1.5px solid var(--bg)",
                color: "#fff",
                fontSize: 8.5,
                fontWeight: 700,
                lineHeight: "11px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {NOTIFICATIONS > 9 ? "9+" : NOTIFICATIONS}
            </span>
          )}
        </button>

        <button
          className="flex items-center"
          style={{
            gap: 8,
            height: 34,
            padding: "0 8px 0 5px",
            borderRadius: 99,
            border: "1px solid var(--border)",
            background: "transparent",
            cursor: "pointer",
            fontFamily: "inherit",
            transition: "background 0.15s, border-color 0.15s",
          }}
          title="Cuenta"
          aria-label="Menú de usuario"
          aria-haspopup="menu"
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--surface-2)";
            e.currentTarget.style.borderColor = "var(--border-2)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.borderColor = "var(--border)";
          }}
        >
          <span
            style={{
              width: 24,
              height: 24,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 11,
              fontWeight: 700,
              color: "#fff",
              flexShrink: 0,
            }}
          >
            {USER.initial}
          </span>
          <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-2)" }}>
            {USER.short}
          </span>
          <ChevronDown style={{ width: 14, height: 14, color: "var(--text-3)" }} />
        </button>
      </div>
    </header>
  );
}

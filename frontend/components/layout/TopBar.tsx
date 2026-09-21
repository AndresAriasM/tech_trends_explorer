"use client";

import { usePathname } from "next/navigation";
import { Bell, Search, ChevronRight } from "lucide-react";
import { getModuleByPath } from "@/lib/modules";

/** Miga de pan derivada del registro de módulos, no de un mapa paralelo. */
function useCrumb(pathname: string): { module: string; page: string } {
  if (pathname === "/") return { module: "Prospecta", page: "Inicio" };

  const mod = getModuleByPath(pathname);
  if (!mod) return { module: "Prospecta", page: "" };

  const route = mod.routes.find(
    (r) => pathname === r.href || pathname.startsWith(`${r.href}/`)
  );
  // Rutas anidadas (p. ej. /workshop/ABC123) cuelgan del label de su módulo.
  const page =
    route && pathname !== route.href ? mod.name : route?.label ?? mod.name;
  return { module: mod.name, page };
}

export function TopBar() {
  const pathname = usePathname();
  const crumb = useCrumb(pathname);

  return (
    <header
      className="glass shrink-0 flex items-center px-8 gap-4"
      style={{
        height: "var(--header-h)",
        borderBottom: "1px solid var(--border)",
        background: "rgba(6,9,15,0.85)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        position: "sticky",
        top: 0,
        zIndex: 40,
      }}
    >
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <span
          style={{
            fontSize: 13,
            fontWeight: 500,
            color: "var(--text-3)",
          }}
        >
          {crumb.module}
        </span>
        <ChevronRight style={{ width: 13, height: 13, color: "var(--text-4)" }} />
        <span
          style={{
            fontSize: 13,
            fontWeight: 600,
            color: "var(--text-1)",
          }}
        >
          {crumb.page}
        </span>
      </div>

      {/* Right actions */}
      <div className="flex items-center gap-1">
        {/* Search */}
        <button
          title="Buscar"
          style={{
            padding: "8px",
            borderRadius: 9,
            color: "var(--text-3)",
            background: "transparent",
            border: "none",
            cursor: "pointer",
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
          <Search style={{ width: 16, height: 16 }} />
        </button>

        {/* Notifications */}
        <button
          title="Notificaciones"
          style={{
            padding: "8px",
            borderRadius: 9,
            color: "var(--text-3)",
            background: "transparent",
            border: "none",
            cursor: "pointer",
            position: "relative",
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
          <Bell style={{ width: 16, height: 16 }} />
          {/* Notification dot */}
          <span
            style={{
              position: "absolute",
              top: 7,
              right: 7,
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "var(--accent)",
              border: "1.5px solid var(--bg)",
            }}
          />
        </button>

        {/* Divider */}
        <div
          style={{
            width: 1,
            height: 20,
            background: "var(--border)",
            margin: "0 6px",
          }}
        />

        {/* User pill */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 12px",
            borderRadius: 10,
            border: "1px solid var(--border)",
            cursor: "pointer",
            transition: "background 0.15s, border-color 0.15s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--surface-2)";
            e.currentTarget.style.borderColor = "var(--border-2)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.borderColor = "var(--border)";
          }}
        >
          <div
            style={{
              width: 22,
              height: 22,
              borderRadius: "50%",
              background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 10,
              fontWeight: 700,
              color: "#fff",
              flexShrink: 0,
            }}
          >
            A
          </div>
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--text-2)",
            }}
          >
            Andrés A.
          </span>
        </div>
      </div>
    </header>
  );
}

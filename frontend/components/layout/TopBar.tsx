"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Home, LogOut, Search, UserRound } from "lucide-react";
import {
  ROLE_LABELS,
  getUser,
  getRole,
  onAuthChange,
  signOut,
  type Role,
  type SessionUser,
} from "@/lib/auth";
import {
  getModuleByPath,
  getRouteByPath,
  getSectionCrumb,
} from "@/lib/modules";


/** Iniciales a partir del correo, mientras no haya nombre en el perfil. */
function inicial(email?: string | null) {
  return (email?.trim()?.[0] ?? "?").toUpperCase();
}

function nombreCorto(email?: string | null) {
  if (!email) return "Cuenta";
  return email.split("@")[0];
}

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

  const [user, setUser] = useState<SessionUser | null>(null);
  const [role, setRole] = useState<Role | null>(null);
  const [open, setOpen] = useState(false);
  const [saliendo, setSaliendo] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const sync = () => {
      setUser(getUser());
      setRole(getRole());
    };
    sync();
    return onAuthChange(sync);
  }, []);

  // Cerrar el menú al hacer clic fuera o con Escape.
  useEffect(() => {
    if (!open) return;
    const fuera = (e: MouseEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", fuera);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", fuera);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const salir = useCallback(async () => {
    setSaliendo(true);
    try {
      await signOut();
    } finally {
      window.location.href = "/login";
    }
  }, []);

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
          disabled
          title="La búsqueda global aún no está disponible"
          aria-label="Buscar en el portal (aún no disponible)"
        >
          <Search style={{ width: 14, height: 14, flexShrink: 0 }} />
          <span className="truncate" style={{ flex: 1 }}>
            Buscar… (pronto)
          </span>
          <span className="kbd" aria-hidden>
            {SEARCH_SHORTCUT}
          </span>
        </button>
      </div>

      {/* ── Acciones ── */}
      <div className="flex items-center shrink-0" style={{ gap: 10 }}>

        <div ref={menuRef} style={{ position: "relative" }}>
          <button
            onClick={() => setOpen((v) => !v)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "5px 10px 5px 5px",
              borderRadius: 99,
              border: "1px solid var(--border)",
              background: open ? "var(--surface-2)" : "transparent",
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "background 0.15s, border-color 0.15s",
            }}
            title="Cuenta"
            aria-label="Menú de usuario"
            aria-haspopup="menu"
            aria-expanded={open}
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
              {inicial(user?.email)}
            </span>
            <span style={{ fontSize: 13, fontWeight: 500, color: "var(--text-2)" }}>
              {nombreCorto(user?.email)}
            </span>
            <ChevronDown style={{ width: 14, height: 14, color: "var(--text-3)" }} />
          </button>

          {open && (
            <div
              role="menu"
              style={{
                position: "absolute",
                top: "calc(100% + 8px)",
                right: 0,
                minWidth: 236,
                background: "var(--surface)",
                border: "1px solid var(--border-2)",
                borderRadius: 12,
                boxShadow: "0 12px 32px rgba(0,0,0,0.45)",
                padding: 6,
                zIndex: 60,
              }}
            >
              <div style={{ padding: "8px 10px 10px" }}>
                <p
                  className="truncate"
                  style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)" }}
                >
                  {user?.email ?? "Sin sesión"}
                </p>
                {role && (
                  <span
                    style={{
                      display: "inline-block",
                      marginTop: 6,
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.04em",
                      textTransform: "uppercase",
                      color: "var(--accent)",
                      background: "rgba(79,142,247,0.12)",
                      border: "1px solid rgba(79,142,247,0.25)",
                      borderRadius: 6,
                      padding: "2px 7px",
                    }}
                  >
                    {ROLE_LABELS[role]}
                  </span>
                )}
              </div>

              <div style={{ height: 1, background: "var(--border)", margin: "2px 0 4px" }} />

              <Link
                href="/cuenta"
                role="menuitem"
                onClick={() => setOpen(false)}
                className="menu-item"
              >
                <UserRound style={{ width: 15, height: 15 }} />
                Mi cuenta
              </Link>

              <button
                role="menuitem"
                onClick={salir}
                disabled={saliendo}
                className="menu-item"
                style={{ color: "#f0806c", width: "100%" }}
              >
                <LogOut style={{ width: 15, height: 15 }} />
                {saliendo ? "Cerrando sesión…" : "Cerrar sesión"}
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

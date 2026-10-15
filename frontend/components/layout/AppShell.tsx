"use client";

import { useState } from "react";
import { usePathname } from "next/navigation";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

/**
 * Rutas que se pintan sin el armazón de la aplicación.
 *
 * El layout raíz envuelve todo en AppShell, así que sin esto la pantalla de
 * acceso mostraba el menú y el nombre del usuario a quien todavía no ha
 * entrado. Lo limpio sería un route group (app), pero eso obliga a mover
 * todas las páginas; esto resuelve lo mismo sin tocarlas.
 */
const BARE_ROUTES = ["/login", "/invite"];

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const toggle = () => setCollapsed((v) => !v);

  const isBare = BARE_ROUTES.some(
    (r) => pathname === r || pathname.startsWith(`${r}/`)
  );

  if (isBare) {
    return (
      <div
        style={{
          background: "var(--bg)",
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {children}
      </div>
    );
  }

  return (
    <div
      className="flex"
      style={{ background: "var(--bg)", height: "100vh", overflow: "hidden" }}
    >
      {/* El panel lateral es fijo y tiene su propio scroll interno. */}
      <Sidebar collapsed={collapsed} onToggle={toggle} />

      {/* El contenido se desplaza por el ancho del panel. */}
      <div
        className="flex flex-col flex-1 min-w-0"
        style={{
          marginLeft: collapsed
            ? "var(--sidebar-collapsed-w)"
            : "var(--sidebar-w)",
          transition: "margin-left 0.22s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        <TopBar />

        {/* Un único contenedor con scroll: antes lo tenían main y la ventana
            a la vez, y salían dos barras que se estorbaban. */}
        <main
          className="flex-1 min-w-0 overflow-y-auto overflow-x-hidden fade-in"
          style={{ height: "calc(100vh - var(--header-h))" }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}

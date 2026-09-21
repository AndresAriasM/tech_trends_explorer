"use client";

import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const toggle = () => setCollapsed((v) => !v);

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

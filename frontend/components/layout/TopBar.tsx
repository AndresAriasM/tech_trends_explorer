"use client";

import { usePathname } from "next/navigation";
import { Bell, Settings } from "lucide-react";

const TITLES: Record<string, { label: string; sub: string }> = {
  "/":            { label: "Dashboard",     sub: "Resumen general" },
  "/hype-cycle":  { label: "Hype Cycle",    sub: "Análisis de posición en el ciclo de Gartner" },
  "/history":     { label: "Historial",     sub: "Consultas guardadas en DynamoDB" },
  "/categories":  { label: "Categorías",    sub: "Organización de tecnologías" },
  "/s-curve":     { label: "Curvas S",      sub: "Adopción tecnológica vía Scopus" },
  "/trends":      { label: "Tendencias",    sub: "Búsqueda de tendencias web" },
};

export function TopBar() {
  const pathname = usePathname();
  const info = TITLES[pathname] ?? { label: "Tech Trends Explorer", sub: "" };

  return (
    <header
      className="flex items-center gap-4 px-6 shrink-0"
      style={{
        height: "var(--header-h)",
        borderBottom: "1px solid var(--border)",
        background: "rgba(7,9,15,0.8)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 40,
      }}
    >
      {/* Title */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-bold" style={{ color: "var(--t1)" }}>
            {info.label}
          </h1>
          {info.sub && (
            <>
              <span style={{ color: "var(--border)", fontSize: 14 }}>·</span>
              <p className="text-xs truncate" style={{ color: "var(--t2)" }}>
                {info.sub}
              </p>
            </>
          )}
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        {/* Notification stub */}
        <button
          className="relative p-2 rounded-lg transition-colors hover:bg-white/5"
          style={{ color: "var(--t3)" }}
          title="Notificaciones"
        >
          <Bell className="w-4 h-4" />
          <span
            className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full"
            style={{ background: "var(--accent)" }}
          />
        </button>

        {/* Settings stub */}
        <button
          className="p-2 rounded-lg transition-colors hover:bg-white/5"
          style={{ color: "var(--t3)" }}
          title="Configuración"
        >
          <Settings className="w-4 h-4" />
        </button>

        {/* Auth: user avatar (future login/logout) */}
        <div
          className="ml-2 flex items-center gap-2 px-3 py-1.5 rounded-lg cursor-pointer transition-colors hover:bg-white/5"
          style={{ border: "1px solid var(--border)" }}
        >
          <div
            className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-bold"
            style={{ background: "linear-gradient(135deg,#3b7cff,#7c5cff)", color: "#fff" }}
          >
            A
          </div>
          <span className="text-xs font-medium" style={{ color: "var(--t2)" }}>
            Andrés A.
          </span>
        </div>
      </div>
    </header>
  );
}

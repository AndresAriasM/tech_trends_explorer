"use client";

import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  const sidebarWidth = collapsed ? 60 : 280;

  return (
    <div
      className="flex min-h-screen"
      style={{ background: "var(--bg)" }}
    >
      {/* Fixed sidebar */}
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((v) => !v)} />

      {/* Content area pushed by sidebar width */}
      <div
        className="flex flex-col flex-1 min-w-0"
        style={{
          marginLeft: sidebarWidth,
          transition: "margin-left 0.22s cubic-bezier(0.4,0,0.2,1)",
        }}
      >
        {/* Sticky topbar */}
        <TopBar />

        {/* Scrollable main */}
        <main
          className="flex-1 overflow-auto fade-in"
          style={{ minHeight: `calc(100vh - var(--header-h))` }}
        >
          {children}
        </main>
      </div>
    </div>
  );
}

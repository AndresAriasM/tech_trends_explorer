"use client";

import { useState } from "react";
import { Sidebar } from "./Sidebar";
import { TopBar } from "./TopBar";

export function AppShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div
      className="flex min-h-screen"
      style={{ background: "var(--s0)" }}
    >
      {/* Sidebar */}
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((v) => !v)} />

      {/* Main area */}
      <div
        className="flex flex-col flex-1 min-w-0 transition-all duration-200"
        style={{
          marginLeft: 0,
        }}
      >
        <TopBar />
        <main className="flex-1 overflow-auto fade-in">
          {children}
        </main>
      </div>
    </div>
  );
}

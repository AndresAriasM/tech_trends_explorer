"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { TrendingUp, History, FolderOpen, Home } from "lucide-react";

const nav = [
  { href: "/", label: "Inicio", icon: Home },
  { href: "/hype-cycle", label: "Hype Cycle", icon: TrendingUp },
  { href: "/history", label: "Historial", icon: History },
  { href: "/categories", label: "Categorías", icon: FolderOpen },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 min-h-screen bg-gray-900 border-r border-gray-800 flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-lg font-bold text-white leading-tight">
          Tech Trends
          <span className="block text-blue-400 text-sm font-normal">
            Explorer
          </span>
        </h1>
        <p className="text-xs text-gray-500 mt-1">UPB · UPB2036</p>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-1">
        {nav.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-blue-600 text-white"
                  : "text-gray-400 hover:bg-gray-800 hover:text-white"
              }`}
            >
              <Icon className="w-4 h-4 shrink-0" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-800">
        <p className="text-xs text-gray-600 text-center">v2.2 · Next.js</p>
      </div>
    </aside>
  );
}

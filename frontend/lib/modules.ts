/**
 * Registro de módulos de la plataforma de prospectiva.
 *
 * Fuente única de verdad: la navegación, los estados de madurez y la relación
 * de cada módulo con el modelo de dominio se declaran aquí, no en el Sidebar.
 *
 * Ver docs/ARQUITECTURA_MODULOS.md para el modelo conceptual.
 */

import {
  Activity,
  Bot,
  Compass,
  FolderOpen,
  History,
  Map as MapIcon,
  Network,
  Radar,
  Users,
  Zap,
  type LucideIcon,
} from "lucide-react";

/** Madurez del módulo dentro de la reconstrucción. */
export type ModuleStatus = "active" | "in-progress" | "planned";

/**
 * Rol del módulo en el modelo de dominio:
 *  - evidence:      produce Evidencia a partir de datos (analítico)
 *  - judgment:      produce Juicio a partir de personas o IA (participativo)
 *  - synthesis:     combina Evidencia y Juicio en una lectura de futuro
 *  - cross-cutting: capa transversal, no es una sección con vida propia
 */
export type ModuleKind = "evidence" | "judgment" | "synthesis" | "cross-cutting";

/** Secciones de la navegación, en orden de aparición. */
export const SECTIONS = [
  { id: "analysis", label: "Análisis" },
  { id: "panel", label: "Panel de expertos" },
  { id: "synthesis", label: "Síntesis" },
  { id: "library", label: "Biblioteca" },
  { id: "assistant", label: "Asistente" },
] as const;

export type SectionId = (typeof SECTIONS)[number]["id"];

export interface ModuleRoute {
  href: string;
  /** Debe ser legible por sí solo: la navegación es una lista plana. */
  label: string;
  icon: LucideIcon;
  section: SectionId;
}

export interface ProspectiveModule {
  id: string;
  name: string;
  /** Una línea: qué pregunta responde este módulo. */
  purpose: string;
  status: ModuleStatus;
  kind: ModuleKind;
  /** Prefijo de las rutas del backend, si ya existen. */
  apiPrefix?: string;
  /** Color de acento de la tarjeta en el panel de inicio. */
  accent: string;
  /** Tres capacidades que resumen el módulo, para la tarjeta. */
  highlights: string[];
  routes: ModuleRoute[];
}

export const MODULES: ProspectiveModule[] = [
  {
    id: "hype-cycle",
    name: "Hype Cycle",
    purpose:
      "¿En qué punto del ciclo de expectativas de Gartner está una tecnología?",
    status: "active",
    kind: "evidence",
    apiPrefix: "/api/hype-cycle",
    accent: "#4f8ef7",
    highlights: [
      "Análisis en tiempo real",
      "Confianza por fase",
      "Histórico consultable",
    ],
    routes: [
      { href: "/hype-cycle", label: "Hype Cycle", icon: Zap, section: "analysis" },
      { href: "/history", label: "Historial", icon: History, section: "library" },
      {
        href: "/categories",
        label: "Categorías",
        icon: FolderOpen,
        section: "library",
      },
    ],
  },
  {
    id: "s-curve",
    name: "Curvas en S",
    purpose:
      "¿En qué fase de adopción está una tecnología según su curva de difusión?",
    status: "active",
    kind: "evidence",
    apiPrefix: "/api/s-curve",
    accent: "#a78bfa",
    highlights: ["Datos de Scopus", "Ajuste logístico", "Fase de adopción"],
    routes: [
      { href: "/s-curve", label: "Curvas en S", icon: Activity, section: "analysis" },
    ],
  },
  {
    id: "surveillance",
    name: "Vigilancia",
    purpose: "¿Qué señales nuevas aparecen en el entorno y con qué intensidad?",
    status: "planned",
    kind: "evidence",
    accent: "#22c55e",
    highlights: ["Señales emergentes", "Volumen y sentimiento", "Fuentes web y noticias"],
    routes: [
      { href: "/surveillance", label: "Vigilancia", icon: Radar, section: "analysis" },
    ],
  },
  {
    id: "delphi",
    name: "Delphi",
    purpose:
      "¿Qué opina un panel de expertos, y converge esa opinión entre rondas?",
    status: "planned",
    kind: "judgment",
    accent: "#f59e0b",
    highlights: ["Rondas iterativas", "Devolución anónima", "Medición de consenso"],
    routes: [{ href: "/delphi", label: "Delphi", icon: Users, section: "panel" }],
  },
  {
    id: "workshop",
    name: "Taller LACCEI",
    purpose:
      "Instrumento puntual de valoración por expertos para el evento LACCEI.",
    status: "active",
    kind: "judgment",
    apiPrefix: "/api/workshop",
    accent: "#38bdf8",
    highlights: ["Panel de expertos", "Votos sintéticos de IA", "Síntesis por variable"],
    routes: [
      { href: "/workshop", label: "Taller LACCEI", icon: Users, section: "panel" },
    ],
  },
  {
    id: "micmac",
    name: "MICMAC",
    purpose:
      "¿Qué variables mueven el sistema y cuáles son dependientes? (análisis estructural)",
    status: "planned",
    kind: "synthesis",
    accent: "#f472b6",
    highlights: ["Matriz de influencias", "Plano influencia-dependencia", "Variables clave"],
    routes: [
      { href: "/micmac", label: "MICMAC", icon: Network, section: "synthesis" },
    ],
  },
  {
    id: "scenarios",
    name: "Escenarios",
    purpose: "¿Qué futuros alternativos son plausibles y qué los diferencia?",
    status: "planned",
    kind: "synthesis",
    accent: "#2dd4bf",
    highlights: ["Ejes de incertidumbre", "Narrativas de futuro", "Probabilidad e impacto"],
    routes: [
      { href: "/scenarios", label: "Escenarios", icon: Compass, section: "synthesis" },
    ],
  },
  {
    id: "roadmap",
    name: "Roadmapping",
    purpose: "¿Qué hitos conectan la situación actual con el futuro elegido?",
    status: "planned",
    kind: "synthesis",
    accent: "#fb923c",
    highlights: ["Capas mercado-producto", "Hitos temporales", "Rutas tecnológicas"],
    routes: [
      { href: "/roadmap", label: "Hoja de ruta", icon: MapIcon, section: "synthesis" },
    ],
  },
  {
    id: "assistant",
    name: "Asistente IA",
    purpose:
      "Capa transversal: interpreta resultados y asiste en cada módulo. No es una sección aislada.",
    status: "planned",
    kind: "cross-cutting",
    accent: "#8b5cf6",
    highlights: [
      "Interpreta resultados",
      "Asiste en cada módulo",
      "Juicios trazables",
    ],
    routes: [
      { href: "/assistant", label: "Asistente IA", icon: Bot, section: "assistant" },
    ],
  },
];

/* ── Derivados para la navegación ──────────────────────────── */

export interface NavItem extends ModuleRoute {
  moduleId: string;
  status: ModuleStatus;
  /** Los módulos aún no construidos no son navegables. */
  disabled: boolean;
}

export interface NavSection {
  id: SectionId;
  label: string;
  items: NavItem[];
}

const ALL_ITEMS: NavItem[] = MODULES.flatMap((m) =>
  m.routes.map((r) => ({
    ...r,
    moduleId: m.id,
    status: m.status,
    disabled: m.status === "planned",
  }))
);

/** Secciones con al menos un ítem, en el orden de SECTIONS. */
export const NAV_SECTIONS: NavSection[] = SECTIONS.map((s) => ({
  id: s.id,
  label: s.label,
  items: ALL_ITEMS.filter((i) => i.section === s.id),
})).filter((s) => s.items.length > 0);

export function getModule(id: string): ProspectiveModule | undefined {
  return MODULES.find((m) => m.id === id);
}

/** Módulo al que pertenece una ruta, para títulos y migas de pan. */
export function getModuleByPath(pathname: string): ProspectiveModule | undefined {
  return MODULES.find((m) =>
    m.routes.some(
      (r) => pathname === r.href || (r.href !== "/" && pathname.startsWith(r.href))
    )
  );
}

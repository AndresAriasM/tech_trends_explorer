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
  Home,
  LayoutGrid,
  Map as MapIcon,
  MessagesSquare,
  Network,
  Radar,
  Radio,
  ShieldCheck,
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

/** Distintivo de la sección en el encabezado de la navegación. */
export type SectionBadge = { text: string; tone: "green" | "gray" | "purple" };

/** Secciones de la navegación, en orden de aparición. */
export const SECTIONS = [
  {
    id: "surveillance",
    label: "Vigilancia & Tendencias",
    /** Nombre corto para la miga de pan. */
    crumb: "Tendencias",
    badge: { text: "Activo", tone: "green" } as SectionBadge,
  },
  {
    id: "futures",
    label: "Escenarios & Futuros",
    crumb: "Futuros",
    badge: { text: "Próximamente", tone: "gray" } as SectionBadge,
  },
  {
    id: "participatory",
    label: "Prospectiva Participativa",
    crumb: "Participativa",
    badge: { text: "Nuevo", tone: "purple" } as SectionBadge,
  },
  { id: "library", label: "Biblioteca", crumb: "Biblioteca", badge: undefined },
  { id: "manage", label: "Gestión", crumb: "Gestión", badge: undefined },
] as const;

export type SectionId = (typeof SECTIONS)[number]["id"];

export interface ModuleRoute {
  href: string;
  /** Debe ser legible por sí solo: la navegación es una lista plana. */
  label: string;
  icon: LucideIcon;
  section: SectionId;
  /** El módulo existe en el registro pero no ocupa sitio en el menú lateral. */
  hiddenInNav?: boolean;
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
      {
        href: "/hype-cycle",
        label: "Hype Cycle",
        icon: Zap,
        section: "surveillance",
      },
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
      { href: "/s-curve", label: "Curvas S", icon: Activity, section: "surveillance" },
    ],
  },
  {
    id: "patents-papers",
    name: "Matriz Pat./Papers",
    purpose:
      "¿Cómo se relacionan la producción científica y la actividad de patentes de una tecnología?",
    status: "active",
    apiPrefix: "/api/s-curve",
    kind: "evidence",
    accent: "#38bdf8",
    highlights: [
      "Scopus frente a EPO OPS",
      "Desfase ciencia → tecnología",
      "Inflexión por mejor modelo",
    ],
    routes: [
      {
        href: "/patents-papers",
        label: "Matriz Pat./Papers",
        icon: LayoutGrid,
        section: "surveillance",
      },
    ],
  },
  {
    id: "surveillance",
    name: "Búsqueda Web",
    purpose: "¿Qué señales nuevas aparecen en el entorno y con qué intensidad?",
    status: "planned",
    kind: "evidence",
    accent: "#22c55e",
    highlights: [
      "Señales emergentes",
      "Volumen y sentimiento",
      "Fuentes web y noticias",
    ],
    routes: [
      {
        href: "/surveillance",
        label: "Búsqueda Web",
        icon: Radar,
        section: "surveillance",
      },
    ],
  },
  {
    id: "scenarios",
    name: "Escenarios",
    purpose: "¿Qué futuros alternativos son plausibles y qué los diferencia?",
    status: "planned",
    kind: "synthesis",
    accent: "#2dd4bf",
    highlights: [
      "Ejes de incertidumbre",
      "Narrativas de futuro",
      "Probabilidad e impacto",
    ],
    routes: [
      { href: "/scenarios", label: "Escenarios", icon: Compass, section: "futures" },
    ],
  },
  {
    id: "roadmap",
    name: "Roadmaps",
    purpose: "¿Qué hitos conectan la situación actual con el futuro elegido?",
    status: "planned",
    kind: "synthesis",
    accent: "#fb923c",
    highlights: [
      "Capas mercado-producto",
      "Hitos temporales",
      "Rutas tecnológicas",
    ],
    routes: [
      { href: "/roadmap", label: "Roadmaps", icon: MapIcon, section: "futures" },
    ],
  },
  {
    id: "weak-signals",
    name: "Señales débiles",
    purpose:
      "¿Qué indicios tempranos, todavía marginales, podrían cambiar el juego?",
    status: "planned",
    kind: "evidence",
    accent: "#eab308",
    highlights: [
      "Detección temprana",
      "Wild cards",
      "Seguimiento del horizonte",
    ],
    routes: [
      {
        href: "/weak-signals",
        label: "Señales débiles",
        icon: Radio,
        section: "futures",
      },
    ],
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
    highlights: [
      "Panel de expertos",
      "Votos sintéticos de IA",
      "Síntesis por variable",
    ],
    routes: [
      {
        href: "/workshop",
        label: "Taller LACCEI",
        icon: Users,
        section: "participatory",
      },
    ],
  },
  {
    id: "delphi",
    name: "Panel Delphi",
    purpose:
      "¿Qué opina un panel de expertos, y converge esa opinión entre rondas?",
    status: "active",
    kind: "judgment",
    apiPrefix: "/api/delphi",
    accent: "#f59e0b",
    highlights: [
      "Rondas iterativas",
      "Devolución anónima",
      "Medición de consenso",
    ],
    routes: [
      {
        href: "/delphi",
        label: "Panel Delphi",
        icon: MessagesSquare,
        section: "participatory",
      },
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
    highlights: [
      "Matriz de influencias",
      "Plano influencia-dependencia",
      "Variables clave",
    ],
    routes: [
      {
        href: "/micmac",
        label: "MICMAC",
        icon: Network,
        section: "futures",
        // Vive dentro del trabajo de escenarios; no ocupa sitio propio en el menú.
        hiddenInNav: true,
      },
    ],
  },
  {
    id: "admin",
    name: "Administración",
    purpose: "¿Quién entra al portal y con qué rol?",
    status: "active",
    kind: "cross-cutting",
    apiPrefix: "/api/auth",
    accent: "#64748b",
    highlights: [
      "Usuarios y roles",
      "Roles reservados por correo",
      "Acceso restringido a administradores",
    ],
    routes: [
      {
        href: "/admin",
        label: "Administración",
        icon: ShieldCheck,
        section: "manage",
      },
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
      {
        href: "/assistant",
        label: "Asistente IA",
        icon: Bot,
        section: "surveillance",
        // Transversal: aparece dentro de cada módulo, no como entrada de menú.
        hiddenInNav: true,
      },
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
  crumb: string;
  badge?: SectionBadge;
  items: NavItem[];
}

/** Sólo los módulos terminados son navegables; el resto evitaría un 404. */
function isNavigable(status: ModuleStatus): boolean {
  return status === "active";
}

const ALL_ITEMS: NavItem[] = MODULES.flatMap((m) =>
  m.routes.map((r) => ({
    ...r,
    moduleId: m.id,
    status: m.status,
    disabled: !isNavigable(m.status),
  }))
);

/** Entrada suelta que encabeza la navegación, por encima de las secciones. */
export const HOME_ITEM = {
  href: "/",
  label: "Inicio",
  icon: Home,
  disabled: false,
} as const;

/** Secciones con al menos un ítem visible, en el orden de SECTIONS. */
export const NAV_SECTIONS: NavSection[] = SECTIONS.map((s) => ({
  id: s.id,
  label: s.label,
  crumb: s.crumb,
  badge: s.badge,
  items: ALL_ITEMS.filter((i) => i.section === s.id && !i.hiddenInNav),
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

/** Ruta concreta que corresponde a un pathname, con su sección. */
export function getRouteByPath(pathname: string): ModuleRoute | undefined {
  for (const m of MODULES) {
    for (const r of m.routes) {
      if (pathname === r.href || (r.href !== "/" && pathname.startsWith(`${r.href}/`)))
        return r;
    }
  }
  return undefined;
}

/** Nombre corto de una sección, para la miga de pan. */
export function getSectionCrumb(section: SectionId): string {
  return SECTIONS.find((s) => s.id === section)?.crumb ?? "Portal";
}

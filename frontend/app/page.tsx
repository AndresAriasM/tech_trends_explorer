"use client";

import Link from "next/link";
import {
  TrendingUp, ArrowRight, Zap, History, FolderOpen,
  BarChart3, Activity, Clock, CheckCircle, Database,
} from "lucide-react";
import { cn } from "@/lib/utils";

/* ── Phase dot colours ───────────────────────────────────── */
const PHASE_COLOR: Record<string, string> = {
  "Innovation Trigger":            "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment":     "#ef4444",
  "Slope of Enlightenment":        "#34d399",
  "Plateau of Productivity":       "#818cf8",
};

/* ── Decorative mini Hype Cycle SVG ─────────────────────── */
function DecorativeHypeCycle() {
  // Hype Cycle bell curve approximation as cubic bezier path
  // Viewbox 360x180. X axis = time, Y axis = expectations (inverted: 0=top, 180=bottom)
  const curvePath =
    "M 10,160 " +
    "C 40,160 50,140 70,100 " +   // rise to trigger
    "C 85,75 95,30 120,20 " +     // peak
    "C 145,10 155,45 165,80 " +   // descent
    "C 175,110 185,140 200,148 " + // trough
    "C 220,156 230,145 250,120 " + // slope begins
    "C 270,95 290,80 350,72";     // plateau

  const dots = [
    { label: "Generative AI",  cx: 120, cy: 20,  color: "#f59e0b", phase: "Peak" },
    { label: "Quantum",        cx: 70,  cy: 100, color: "#60a5fa", phase: "Trigger" },
    { label: "Digital Twins",  cx: 250, cy: 120, color: "#34d399", phase: "Slope" },
    { label: "Blockchain",     cx: 200, cy: 148, color: "#ef4444", phase: "Trough" },
  ];

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        maxWidth: 400,
      }}
    >
      {/* Subtle glow backdrop */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: "radial-gradient(ellipse 70% 50% at 50% 50%, rgba(79,142,247,0.06) 0%, transparent 70%)",
          borderRadius: 24,
          pointerEvents: "none",
        }}
      />

      <svg
        viewBox="0 0 360 200"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        style={{ width: "100%", height: "auto", display: "block" }}
      >
        {/* Phase zone shading */}
        {/* Innovation Trigger zone */}
        <rect x="10" y="0" width="85" height="200" fill="rgba(96,165,250,0.03)" rx="0" />
        {/* Peak zone */}
        <rect x="95" y="0" width="80" height="200" fill="rgba(245,158,11,0.03)" rx="0" />
        {/* Trough zone */}
        <rect x="175" y="0" width="55" height="200" fill="rgba(239,68,68,0.03)" rx="0" />
        {/* Slope zone */}
        <rect x="230" y="0" width="70" height="200" fill="rgba(52,211,153,0.03)" rx="0" />
        {/* Plateau zone */}
        <rect x="300" y="0" width="55" height="200" fill="rgba(167,139,250,0.04)" rx="0" />

        {/* Grid lines */}
        {[40, 80, 120, 160].map((y) => (
          <line
            key={y}
            x1="10" y1={y} x2="350" y2={y}
            stroke="rgba(255,255,255,0.04)" strokeWidth="1"
          />
        ))}

        {/* Axes */}
        <line x1="10" y1="175" x2="355" y2="175" stroke="rgba(255,255,255,0.12)" strokeWidth="1" />
        <line x1="10" y1="10"  x2="10"  y2="175" stroke="rgba(255,255,255,0.12)" strokeWidth="1" />

        {/* Axis arrows */}
        <polygon points="355,172 360,175 355,178" fill="rgba(255,255,255,0.20)" />
        <polygon points="7,14 10,8 13,14" fill="rgba(255,255,255,0.20)" />

        {/* Axis labels */}
        <text x="180" y="195" textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="Inter,sans-serif">
          Tiempo
        </text>
        <text x="4" y="110" textAnchor="middle" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="Inter,sans-serif"
          transform="rotate(-90, 4, 110)">
          Expectativas
        </text>

        {/* Glow under curve */}
        <path
          d={curvePath + " L 350,175 L 10,175 Z"}
          fill="url(#curveGradient)"
          opacity="0.4"
        />

        {/* Main curve */}
        <path
          d={curvePath}
          stroke="url(#curveStroke)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Phase labels */}
        <text x="52" y="190" textAnchor="middle" fill="rgba(96,165,250,0.55)" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="600">Trigger</text>
        <text x="135" y="190" textAnchor="middle" fill="rgba(245,158,11,0.55)" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="600">Peak</text>
        <text x="202" y="190" textAnchor="middle" fill="rgba(239,68,68,0.55)" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="600">Trough</text>
        <text x="265" y="190" textAnchor="middle" fill="rgba(52,211,153,0.55)" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="600">Slope</text>
        <text x="327" y="190" textAnchor="middle" fill="rgba(167,139,250,0.55)" fontSize="7" fontFamily="Inter,sans-serif" fontWeight="600">Plateau</text>

        {/* Technology dots */}
        {dots.map((dot) => (
          <g key={dot.label}>
            {/* Pulsing halo */}
            <circle cx={dot.cx} cy={dot.cy} r="10" fill={dot.color} opacity="0.10">
              <animate
                attributeName="r"
                values="8;14;8"
                dur="3s"
                repeatCount="indefinite"
              />
              <animate
                attributeName="opacity"
                values="0.15;0.04;0.15"
                dur="3s"
                repeatCount="indefinite"
              />
            </circle>
            {/* Dot */}
            <circle cx={dot.cx} cy={dot.cy} r="5" fill={dot.color} opacity="0.9" />
            <circle cx={dot.cx} cy={dot.cy} r="2.5" fill="#fff" opacity="0.8" />

            {/* Label */}
            <text
              x={dot.cx}
              y={dot.cy - 12}
              textAnchor="middle"
              fill={dot.color}
              fontSize="8"
              fontFamily="Inter,sans-serif"
              fontWeight="600"
              opacity="0.85"
            >
              {dot.label}
            </text>
          </g>
        ))}

        {/* Gradient definitions */}
        <defs>
          <linearGradient id="curveStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#60a5fa" />
            <stop offset="33%"  stopColor="#f59e0b" />
            <stop offset="55%"  stopColor="#ef4444" />
            <stop offset="72%"  stopColor="#34d399" />
            <stop offset="100%" stopColor="#a78bfa" />
          </linearGradient>
          <linearGradient id="curveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%"   stopColor="#4f8ef7" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#4f8ef7" stopOpacity="0" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <div className="fade-in" style={{ padding: "0 0 64px" }}>

      {/* ── Hero Section ────────────────────────────────────── */}
      <section
        className="mesh-bg"
        style={{
          padding: "80px 48px",
          borderBottom: "1px solid var(--border)",
        }}
      >
        <div
          style={{
            maxWidth: 1120,
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            gap: 64,
          }}
        >
          {/* Left: copy */}
          <div style={{ flex: 1, minWidth: 0 }}>
            {/* Status pill */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                padding: "5px 12px",
                borderRadius: 99,
                background: "rgba(79,142,247,0.08)",
                border: "1px solid rgba(79,142,247,0.20)",
                marginBottom: 28,
              }}
            >
              <div className="pulse-dot" style={{ width: 6, height: 6 }} />
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 600,
                  color: "var(--accent)",
                  letterSpacing: "0.03em",
                }}
              >
                Sistema activo — UPB 2036
              </span>
            </div>

            {/* Headline */}
            <h1
              style={{
                fontSize: "clamp(36px, 4vw, 52px)",
                fontWeight: 800,
                letterSpacing: "-0.03em",
                lineHeight: 1.1,
                color: "var(--text-1)",
                marginBottom: 20,
              }}
            >
              Tech Trends{" "}
              <span className="gradient-text">Explorer</span>
            </h1>

            {/* Subtitle */}
            <p
              style={{
                fontSize: 17,
                lineHeight: 1.7,
                color: "var(--text-2)",
                maxWidth: 480,
                marginBottom: 36,
              }}
            >
              Plataforma de análisis basada en el{" "}
              <strong style={{ color: "var(--text-1)", fontWeight: 600 }}>
                Hype Cycle de Gartner
              </strong>
              . Determina el posicionamiento de tecnologías usando fuentes web,
              académicas y de mercado.
            </p>

            {/* CTAs */}
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Link href="/hype-cycle" className="btn btn-primary" style={{ fontSize: 15, padding: "12px 24px" }}>
                <Zap style={{ width: 17, height: 17 }} />
                Nuevo análisis
              </Link>
              <Link href="/history" className="btn btn-ghost" style={{ fontSize: 15, padding: "12px 24px" }}>
                <History style={{ width: 17, height: 17 }} />
                Ver historial
              </Link>
            </div>
          </div>

          {/* Right: animated SVG */}
          <div
            style={{
              flexShrink: 0,
              width: 380,
              padding: "24px 28px",
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 20,
              boxShadow: "0 24px 64px rgba(0,0,0,0.3), 0 0 0 1px rgba(79,142,247,0.06)",
            }}
          >
            <div style={{ marginBottom: 12 }}>
              <p className="label-sm" style={{ marginBottom: 4 }}>Hype Cycle de Gartner</p>
              <p style={{ fontSize: 12, color: "var(--text-3)" }}>
                Posición actual de tecnologías clave
              </p>
            </div>
            <DecorativeHypeCycle />
          </div>
        </div>
      </section>

      {/* ── Stats Row ────────────────────────────────────────── */}
      <section
        style={{
          maxWidth: 1120,
          margin: "0 auto",
          padding: "48px 48px 0",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 20,
          }}
        >
          {[
            {
              label: "Análisis guardados",
              value: "—",
              icon: Database,
              color: "#22c55e",
              trend: null,
            },
            {
              label: "Categorías",
              value: "—",
              icon: FolderOpen,
              color: "var(--accent)",
              trend: null,
            },
            {
              label: "Última ejecución",
              value: "—",
              icon: Clock,
              color: "#f59e0b",
              trend: null,
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="card"
              style={{
                padding: "24px 28px",
                display: "flex",
                alignItems: "center",
                gap: 18,
              }}
            >
              {/* Icon circle */}
              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 12,
                  background: `${stat.color}14`,
                  border: `1px solid ${stat.color}28`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                }}
              >
                <stat.icon style={{ width: 20, height: 20, color: stat.color }} />
              </div>

              <div>
                <p
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: "var(--text-3)",
                    marginBottom: 4,
                  }}
                >
                  {stat.label}
                </p>
                <p
                  style={{
                    fontSize: 28,
                    fontWeight: 800,
                    letterSpacing: "-0.03em",
                    color: "var(--text-1)",
                    lineHeight: 1,
                  }}
                >
                  {stat.value}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Module Cards ─────────────────────────────────────── */}
      <section
        style={{
          maxWidth: 1120,
          margin: "0 auto",
          padding: "48px 48px 0",
        }}
      >
        <p className="section-label" style={{ marginBottom: 20 }}>Módulos disponibles</p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(3, 1fr)",
            gap: 20,
          }}
        >
          <ModuleCard
            icon={TrendingUp}
            title="Hype Cycle"
            color="#4f8ef7"
            badge="Activo"
            badgeClass="badge-blue"
            description="Determina la fase de una tecnología en el ciclo de Gartner usando fuentes web, noticias y datos de mercado."
            features={["Análisis en tiempo real", "Confianza por fase", "Histórico DynamoDB"]}
            actions={[
              { href: "/hype-cycle", label: "Iniciar análisis", icon: Zap, primary: true },
              { href: "/history",    label: "Ver historial",    icon: History, primary: false },
              { href: "/categories", label: "Categorías",       icon: FolderOpen, primary: false },
            ]}
            glow
          />
          <ModuleCard
            icon={Activity}
            title="Curvas S"
            color="#a78bfa"
            badge="En desarrollo"
            badgeClass="badge-purple"
            description="Analiza la adopción tecnológica a través del tiempo con publicaciones académicas de Scopus."
            features={["Datos de Scopus", "Curvas logísticas", "Predicción temporal"]}
            actions={[{ href: "/s-curve", label: "Próximamente", icon: BarChart3, primary: false }]}
            disabled
          />
          <ModuleCard
            icon={TrendingUp}
            title="Tendencias Web"
            color="#22c55e"
            badge="En desarrollo"
            badgeClass="badge-gray"
            description="Mide el volumen y sentimiento de menciones en fuentes web, redes sociales y noticias."
            features={["Análisis de sentimiento", "Volumen de menciones", "Tendencias temporales"]}
            actions={[]}
            disabled
          />
        </div>
      </section>

      {/* ── Recent Activity ──────────────────────────────────── */}
      <section
        style={{
          maxWidth: 1120,
          margin: "0 auto",
          padding: "48px 48px 0",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 20,
          }}
        >
          <p className="section-label">Actividad reciente</p>
          <Link
            href="/history"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 13,
              fontWeight: 500,
              color: "var(--accent)",
              textDecoration: "none",
              transition: "opacity 0.15s",
            }}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = "0.75"; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = "1"; }}
          >
            Ver todo <ArrowRight style={{ width: 13, height: 13 }} />
          </Link>
        </div>

        {/* Empty state */}
        <div
          className="card"
          style={{
            padding: "64px 32px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 16,
            textAlign: "center",
          }}
        >
          {/* Icon */}
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 16,
              background: "var(--accent-sub)",
              border: "1px solid var(--accent-border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <TrendingUp style={{ width: 26, height: 26, color: "var(--accent)" }} />
          </div>
          <div>
            <p
              style={{
                fontSize: 16,
                fontWeight: 700,
                color: "var(--text-1)",
                letterSpacing: "-0.01em",
                marginBottom: 6,
              }}
            >
              Ejecuta tu primer análisis
            </p>
            <p
              style={{
                fontSize: 14,
                color: "var(--text-2)",
                lineHeight: 1.6,
                maxWidth: 360,
              }}
            >
              Tus análisis aparecerán aquí. Comienza analizando una tecnología con el módulo Hype Cycle.
            </p>
          </div>
          <Link
            href="/hype-cycle"
            className="btn btn-primary"
            style={{ marginTop: 4 }}
          >
            <Zap style={{ width: 15, height: 15 }} />
            Iniciar análisis
          </Link>
        </div>
      </section>

    </div>
  );
}

/* ── Module Card component ──────────────────────────────── */

interface Action {
  href: string;
  label: string;
  icon: React.ElementType;
  primary: boolean;
}

interface ModuleCardProps {
  icon: React.ElementType;
  title: string;
  description: string;
  badge: string;
  badgeClass: string;
  color: string;
  disabled?: boolean;
  glow?: boolean;
  actions: Action[];
  features: string[];
}

function ModuleCard({
  icon: Icon,
  title,
  description,
  badge,
  badgeClass,
  color,
  disabled,
  glow,
  actions,
  features,
}: ModuleCardProps) {
  return (
    <div
      className={cn("card card-hover flex flex-col", disabled && "opacity-55")}
      style={{
        padding: "28px",
        boxShadow: glow
          ? `0 0 0 1px ${color}20, 0 8px 40px rgba(0,0,0,0.2), 0 0 32px ${color}0a`
          : undefined,
        borderColor: glow ? `${color}25` : undefined,
      }}
    >
      {/* Icon + badge row */}
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          marginBottom: 20,
        }}
      >
        <div
          style={{
            width: 48,
            height: 48,
            borderRadius: 14,
            background: `${color}14`,
            border: `1px solid ${color}28`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <Icon style={{ width: 22, height: 22, color }} />
        </div>
        <span className={cn("badge", badgeClass)}>{badge}</span>
      </div>

      {/* Title */}
      <h3
        style={{
          fontSize: 18,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: "var(--text-1)",
          marginBottom: 10,
        }}
      >
        {title}
      </h3>

      {/* Description */}
      <p
        style={{
          fontSize: 14,
          lineHeight: 1.65,
          color: "var(--text-2)",
          flex: 1,
          marginBottom: 20,
        }}
      >
        {description}
      </p>

      {/* Feature list */}
      <ul
        style={{
          listStyle: "none",
          marginBottom: 24,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {features.map((f) => (
          <li
            key={f}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              fontSize: 13,
              color: "var(--text-2)",
            }}
          >
            <CheckCircle
              style={{
                width: 13,
                height: 13,
                color,
                flexShrink: 0,
                opacity: 0.8,
              }}
            />
            {f}
          </li>
        ))}
      </ul>

      {/* Actions */}
      {!disabled && actions.length > 0 && (
        <div
          style={{
            borderTop: "1px solid var(--border)",
            paddingTop: 20,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          {actions.map(({ href, label, icon: AIcon, primary }) =>
            primary ? (
              <Link
                key={href}
                href={href}
                className="btn btn-primary"
                style={{ justifyContent: "center", fontSize: 13 }}
              >
                <AIcon style={{ width: 14, height: 14 }} />
                {label}
              </Link>
            ) : (
              <Link
                key={href}
                href={href}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  padding: "8px 10px",
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 500,
                  color: "var(--text-2)",
                  textDecoration: "none",
                  transition: "background 0.15s, color 0.15s",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                  e.currentTarget.style.color = "var(--text-1)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "transparent";
                  e.currentTarget.style.color = "var(--text-2)";
                }}
              >
                <AIcon style={{ width: 13, height: 13, color, flexShrink: 0 }} />
                <span style={{ flex: 1 }}>{label}</span>
                <ArrowRight style={{ width: 12, height: 12, opacity: 0.3 }} />
              </Link>
            )
          )}
        </div>
      )}
    </div>
  );
}

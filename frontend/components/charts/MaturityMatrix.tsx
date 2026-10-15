"use client";

import { AlertTriangle } from "lucide-react";
import type { SCurveMaturityMatrix } from "@/types";

/**
 * Plano madurez científica × madurez tecnológica.
 *
 * Cada eje es la posición de una serie frente a SU propia inflexión, no su
 * volumen: así una tecnología con 60.000 papers y otra con 3.000 se sitúan
 * por la forma de su curva y no por el tamaño del campo.
 *
 * SVG a mano en vez de Plotly: son cuatro celdas, dos ejes y un punto, y
 * cargar una librería de gráficos para esto no se paga.
 */

const LADO = 340;
const MARGEN = 34; // espacio para los rótulos de los ejes
const CENTRO = MARGEN + (LADO - 2 * MARGEN) / 2;

/** [-1,1] → coordenada del SVG. El eje Y va invertido: arriba es más maduro. */
function ejeX(v: number) {
  return MARGEN + ((v + 1) / 2) * (LADO - 2 * MARGEN);
}
function ejeY(v: number) {
  return LADO - MARGEN - ((v + 1) / 2) * (LADO - 2 * MARGEN);
}

const CELDAS = [
  { id: "ciencia_por_delante", texto: "Ciencia\npor delante", x: 0.5, y: -0.5 },
  { id: "consolidada", texto: "Consolidada", x: 0.5, y: 0.5 },
  { id: "incipiente", texto: "Incipiente", x: -0.5, y: -0.5 },
  { id: "industria_por_delante", texto: "Industria\npor delante", x: -0.5, y: 0.5 },
];

export function MaturityMatrix({ matrix }: { matrix: SCurveMaturityMatrix }) {
  if (!matrix.available) {
    return (
      <div
        style={{
          display: "flex",
          gap: 9,
          padding: "12px 14px",
          borderRadius: 9,
          background: "rgba(245,158,11,0.07)",
          border: "1px solid rgba(245,158,11,0.2)",
          fontSize: 12.5,
          color: "var(--text-2)",
          lineHeight: 1.5,
        }}
      >
        <AlertTriangle style={{ width: 15, height: 15, flexShrink: 0, color: "#f59e0b", marginTop: 1 }} />
        <span>{matrix.reason}</span>
      </div>
    );
  }

  const px = ejeX(matrix.x ?? 0);
  const py = ejeY(matrix.y ?? 0);
  const color = matrix.color ?? "#4f8ef7";

  return (
    <div style={{ display: "flex", gap: 24, flexWrap: "wrap", alignItems: "flex-start" }}>
      <svg
        viewBox={`0 0 ${LADO} ${LADO}`}
        style={{ width: "100%", maxWidth: LADO, height: "auto", flexShrink: 0 }}
        role="img"
        aria-label={`Matriz de madurez: ${matrix.label}`}
      >
        {/* Fondo de los cuatro cuadrantes */}
        {CELDAS.map((c) => {
          const activo = c.id === matrix.quadrant;
          const x0 = c.x < 0 ? MARGEN : CENTRO;
          const y0 = c.y < 0 ? CENTRO : MARGEN;
          const lado = (LADO - 2 * MARGEN) / 2;
          return (
            <rect
              key={c.id}
              x={x0}
              y={y0}
              width={lado}
              height={lado}
              fill={activo ? `${color}1f` : "rgba(255,255,255,0.015)"}
              stroke={activo ? `${color}55` : "var(--border)"}
              strokeWidth={activo ? 1.5 : 1}
            />
          );
        })}

        {/* Rótulos de cada cuadrante */}
        {CELDAS.map((c) => {
          const activo = c.id === matrix.quadrant;
          const lineas = c.texto.split("\n");
          return (
            <text
              key={`t-${c.id}`}
              x={ejeX(c.x)}
              y={ejeY(c.y) - (lineas.length - 1) * 6}
              textAnchor="middle"
              fontSize="10"
              fontWeight={activo ? 700 : 500}
              fill={activo ? color : "var(--text-4)"}
              fontFamily="Inter, sans-serif"
            >
              {lineas.map((l, i) => (
                <tspan key={i} x={ejeX(c.x)} dy={i === 0 ? 0 : 12}>
                  {l}
                </tspan>
              ))}
            </text>
          );
        })}

        {/* Ejes en el cero: la inflexión de cada serie */}
        <line x1={CENTRO} y1={MARGEN} x2={CENTRO} y2={LADO - MARGEN} stroke="var(--border-2)" strokeWidth="1" strokeDasharray="3 3" />
        <line x1={MARGEN} y1={CENTRO} x2={LADO - MARGEN} y2={CENTRO} stroke="var(--border-2)" strokeWidth="1" strokeDasharray="3 3" />

        {/* Rótulos de los ejes */}
        <text x={LADO / 2} y={LADO - 8} textAnchor="middle" fontSize="9.5" fill="var(--text-3)" fontFamily="Inter, sans-serif">
          Madurez científica →
        </text>
        <text
          x={11}
          y={LADO / 2}
          textAnchor="middle"
          fontSize="9.5"
          fill="var(--text-3)"
          fontFamily="Inter, sans-serif"
          transform={`rotate(-90, 11, ${LADO / 2})`}
        >
          Madurez tecnológica →
        </text>

        {/* Posición de la tecnología */}
        <circle cx={px} cy={py} r="13" fill={color} opacity="0.16" />
        <circle
          cx={px}
          cy={py}
          r="6.5"
          fill={color}
          stroke="var(--bg)"
          strokeWidth="2"
          strokeDasharray={matrix.provisional ? "3 2" : undefined}
        />
      </svg>

      <div style={{ flex: "1 1 240px", minWidth: 220 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: "50%", background: color, flexShrink: 0 }} />
          <span style={{ fontSize: 16, fontWeight: 700, color: "var(--text-1)" }}>
            {matrix.label}
          </span>
          {matrix.provisional && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: "#f59e0b",
                border: "1px solid rgba(245,158,11,0.3)",
                borderRadius: 5,
                padding: "1px 6px",
              }}
            >
              Provisional
            </span>
          )}
        </div>

        <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.6 }}>
          {matrix.description}
        </p>

        {matrix.provisional && (
          <p style={{ fontSize: 11.5, color: "var(--text-4)", marginTop: 12, lineHeight: 1.5 }}>
            Alguna de las dos inflexiones está proyectada fuera de los datos
            observados, así que la posición es una previsión y no una medida.
          </p>
        )}
      </div>
    </div>
  );
}

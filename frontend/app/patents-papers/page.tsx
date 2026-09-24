"use client";

import { useCallback, useRef, useState } from "react";
import {
  AlertTriangle,
  BookOpen,
  FileText,
  GitCompareArrows,
  Landmark,
  Loader2,
  Play,
  RotateCcw,
  Square,
} from "lucide-react";
import {
  comparePapersPatents,
  streamPapersAnalysis,
  streamPatentsAnalysis,
} from "@/lib/api";
import { SCurveChart } from "@/components/charts/SCurveChart";
import { MaturityMatrix } from "@/components/charts/MaturityMatrix";
import type {
  SCurveComparison,
  SCurveComparisonSide,
  SCurveProgress,
  SCurveResult,
} from "@/types";

const AHORA = new Date().getFullYear();

/* Ecuaciones de ejemplo, con el volumen real medido contra cada API en el
   rango 2000-2026. El grupo es el cuadrante que se espera encontrar, no una
   medición: depende de dónde caiga la inflexión de cada serie. */
const EJEMPLOS: Array<{
  concepto: string;
  scopus: string;
  epo: string;
  papers: number;
  patentes: number;
  grupo: string;
}> = [
  // Campo maduro por ambos lados
  { concepto: "Biodiésel", scopus: 'TITLE-ABS-KEY("biodiesel")', epo: 'txt="biodiesel"', papers: 62931, patentes: 43760, grupo: "Consolidada" },
  { concepto: "CRISPR", scopus: 'TITLE-ABS-KEY("CRISPR")', epo: 'txt="CRISPR"', papers: 101336, patentes: 73984, grupo: "Consolidada" },

  // Más patentes que papers: la industria va por delante
  { concepto: "Vehículo autónomo", scopus: 'TITLE-ABS-KEY("autonomous vehicle")', epo: 'txt="autonomous vehicle"', papers: 71684, patentes: 115728, grupo: "Industria por delante" },
  { concepto: "Computación cuántica", scopus: 'TITLE-ABS-KEY("quantum computing")', epo: 'txt="quantum computing"', papers: 35608, patentes: 92349, grupo: "Industria por delante" },
  { concepto: "Gemelos digitales", scopus: 'TITLE-ABS-KEY("digital twin")', epo: 'txt="digital twin"', papers: 58349, patentes: 72599, grupo: "Industria por delante" },
  { concepto: "Baterías de estado sólido", scopus: 'TITLE-ABS-KEY("solid state battery")', epo: 'txt="solid state battery"', papers: 22880, patentes: 32706, grupo: "Industria por delante" },

  // Mucha publicación, poca patente: sigue en laboratorio
  { concepto: "Perovskita fotovoltaica", scopus: 'TITLE-ABS-KEY("perovskite solar cell")', epo: 'txt="perovskite solar cell"', papers: 46334, patentes: 15372, grupo: "Ciencia por delante" },
  { concepto: "Captura de carbono", scopus: 'TITLE-ABS-KEY("carbon capture")', epo: 'txt="carbon capture"', papers: 49924, patentes: 22008, grupo: "Ciencia por delante" },

  // Emergentes: pocas de las dos, y las más rápidas de probar
  { concepto: "Hidrógeno verde", scopus: 'TITLE-ABS-KEY("green hydrogen")', epo: 'txt="green hydrogen"', papers: 15045, patentes: 11475, grupo: "Incipiente" },
  { concepto: "Envases biodegradables", scopus: 'TITLE-ABS-KEY("biodegradable packaging")', epo: 'txt="biodegradable packaging"', papers: 2553, patentes: 2791, grupo: "Incipiente" },
];

const GRUPOS = [
  "Consolidada",
  "Industria por delante",
  "Ciencia por delante",
  "Incipiente",
] as const;

type EstadoLado = {
  corriendo: boolean;
  progreso: number;
  mensaje: string;
  resultado: SCurveResult | null;
  error: string | null;
};

const LADO_INICIAL: EstadoLado = {
  corriendo: false,
  progreso: 0,
  mensaje: "",
  resultado: null,
  error: null,
};

export default function PatentsPapersPage() {
  const [scopusQ, setScopusQ] = useState("");
  const [epoQ, setEpoQ] = useState("");
  const [yearFrom, setYearFrom] = useState(2000);
  const [yearTo, setYearTo] = useState(AHORA);

  const [papers, setPapers] = useState<EstadoLado>(LADO_INICIAL);
  const [patents, setPatents] = useState<EstadoLado>(LADO_INICIAL);
  const [comparacion, setComparacion] = useState<SCurveComparison | null>(null);
  const [comparando, setComparando] = useState(false);
  const [errorComparar, setErrorComparar] = useState<string | null>(null);

  const cancelarRef = useRef<Array<() => void>>([]);

  const corriendo = papers.corriendo || patents.corriendo;
  const listos = papers.resultado && patents.resultado;

  /** Lanza ambos análisis a la vez y compara cuando los dos terminen. */
  const analizar = useCallback(() => {
    if (!scopusQ.trim() || !epoQ.trim()) return;

    setComparacion(null);
    setErrorComparar(null);
    setPapers({ ...LADO_INICIAL, corriendo: true, mensaje: "Consultando Scopus…" });
    setPatents({ ...LADO_INICIAL, corriendo: true, mensaje: "Consultando EPO OPS…" });

    const recogidos: { papers?: SCurveResult; patents?: SCurveResult } = {};

    // Cuando los dos lados han terminado bien, se pide el desfase.
    const intentarComparar = async () => {
      if (!recogidos.papers || !recogidos.patents) return;
      setComparando(true);
      try {
        setComparacion(await comparePapersPatents(recogidos.papers, recogidos.patents));
      } catch (e: unknown) {
        setErrorComparar(
          e instanceof Error ? e.message : "No se pudo calcular el desfase."
        );
      } finally {
        setComparando(false);
      }
    };

    const cancelPapers = streamPapersAnalysis(
      { query: scopusQ.trim(), year_from: yearFrom, year_to: yearTo },
      (evt: SCurveProgress) =>
        setPapers((p) => ({ ...p, progreso: evt.progress, mensaje: evt.message })),
      (res) => {
        setPapers((p) => ({ ...p, corriendo: false, progreso: 100, resultado: res ?? null }));
        if (res) {
          recogidos.papers = res;
          void intentarComparar();
        }
      },
      (msg) => setPapers((p) => ({ ...p, corriendo: false, error: msg }))
    );

    const cancelPatents = streamPatentsAnalysis(
      { query: epoQ.trim(), year_from: yearFrom, year_to: yearTo },
      (evt: SCurveProgress) =>
        setPatents((p) => ({ ...p, progreso: evt.progress, mensaje: evt.message })),
      (res) => {
        setPatents((p) => ({ ...p, corriendo: false, progreso: 100, resultado: res ?? null }));
        if (res) {
          recogidos.patents = res;
          void intentarComparar();
        }
      },
      (msg) => setPatents((p) => ({ ...p, corriendo: false, error: msg }))
    );

    cancelarRef.current = [cancelPapers, cancelPatents];
  }, [scopusQ, epoQ, yearFrom, yearTo]);

  const detener = useCallback(() => {
    cancelarRef.current.forEach((c) => c());
    cancelarRef.current = [];
    setPapers((p) => ({ ...p, corriendo: false }));
    setPatents((p) => ({ ...p, corriendo: false }));
  }, []);

  const reiniciar = useCallback(() => {
    detener();
    setPapers(LADO_INICIAL);
    setPatents(LADO_INICIAL);
    setComparacion(null);
    setErrorComparar(null);
  }, [detener]);

  return (
    <div style={{ padding: "28px 40px 64px", maxWidth: 1240, margin: "0 auto" }}>
      <header style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
          <GitCompareArrows style={{ width: 22, height: 22, color: "#38bdf8" }} />
          <h1 style={{ fontSize: 24, fontWeight: 700, color: "var(--text-1)" }}>
            Matriz Patentes / Papers
          </h1>
        </div>
        <p style={{ fontSize: 13.5, color: "var(--text-3)", marginTop: 8, maxWidth: 760, lineHeight: 1.6 }}>
          Compara la curva de producción científica con la de actividad de
          patentes para la misma tecnología. La distancia entre sus dos puntos
          de inflexión mide el <strong style={{ color: "var(--text-2)" }}>desfase
          ciencia&nbsp;→&nbsp;tecnología</strong>: cuánto tarda lo que se publica
          en convertirse en lo que se protege.
        </p>
      </header>

      {/* ── Consulta ── */}
      <section className="card" style={{ padding: 22, marginBottom: 22 }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: 18,
          }}
        >
          <Campo
            icono={<BookOpen style={{ width: 14, height: 14, color: "#a78bfa" }} />}
            etiqueta="Ecuación Scopus (papers)"
            ayuda="Sintaxis de Elsevier"
          >
            <textarea
              className="inp"
              style={{ width: "100%", minHeight: 74, fontFamily: "var(--font-mono, monospace)", fontSize: 12.5 }}
              value={scopusQ}
              onChange={(e) => setScopusQ(e.target.value)}
              disabled={corriendo}
              placeholder={'TITLE-ABS-KEY("biodiesel")'}
            />
          </Campo>

          <Campo
            icono={<Landmark style={{ width: 14, height: 14, color: "#22c55e" }} />}
            etiqueta="Ecuación EPO OPS (patentes)"
            ayuda="Sintaxis CQL"
          >
            <textarea
              className="inp"
              style={{ width: "100%", minHeight: 74, fontFamily: "var(--font-mono, monospace)", fontSize: 12.5 }}
              value={epoQ}
              onChange={(e) => setEpoQ(e.target.value)}
              disabled={corriendo}
              placeholder={'txt="biodiesel"'}
            />
          </Campo>
        </div>

        <div style={{ marginTop: 18 }}>
          <p className="label-sm" style={{ color: "var(--text-3)", marginBottom: 4 }}>
            Ejemplos
          </p>
          <p style={{ fontSize: 11, color: "var(--text-4)", marginBottom: 12 }}>
            Volumen real medido en 2000–2026. El grupo es el cuadrante
            esperable, no una medición.
          </p>

          {GRUPOS.map((g) => (
            <div key={g} style={{ marginBottom: 10 }}>
              <p
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: "var(--text-4)",
                  marginBottom: 6,
                }}
              >
                {g}
              </p>
              <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                {EJEMPLOS.filter((e) => e.grupo === g).map((ej) => (
                  <button
                    key={ej.concepto}
                    className="btn btn-ghost"
                    style={{ fontSize: 12, padding: "6px 11px", flexDirection: "column", alignItems: "flex-start", gap: 1 }}
                    disabled={corriendo}
                    title={`${ej.papers.toLocaleString()} papers · ${ej.patentes.toLocaleString()} patentes`}
                    onClick={() => {
                      setScopusQ(ej.scopus);
                      setEpoQ(ej.epo);
                    }}
                  >
                    <span>{ej.concepto}</span>
                    <span style={{ fontSize: 9.5, color: "var(--text-4)", fontWeight: 500 }}>
                      {(ej.papers / 1000).toFixed(0)}k · {(ej.patentes / 1000).toFixed(0)}k
                    </span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            display: "flex",
            gap: 14,
            alignItems: "flex-end",
            marginTop: 18,
            flexWrap: "wrap",
          }}
        >
          <Campo etiqueta="Desde">
            <input
              className="inp"
              type="number"
              style={{ width: 110 }}
              value={yearFrom}
              min={1950}
              max={AHORA}
              disabled={corriendo}
              onChange={(e) => setYearFrom(Number(e.target.value))}
            />
          </Campo>
          <Campo etiqueta="Hasta">
            <input
              className="inp"
              type="number"
              style={{ width: 110 }}
              value={yearTo}
              min={1950}
              max={AHORA}
              disabled={corriendo}
              onChange={(e) => setYearTo(Number(e.target.value))}
            />
          </Campo>

          <div style={{ flex: 1 }} />

          {corriendo ? (
            <button className="btn btn-ghost" onClick={detener}>
              <Square style={{ width: 15, height: 15 }} />
              Detener
            </button>
          ) : (
            <>
              {(papers.resultado || patents.resultado) && (
                <button className="btn btn-ghost" onClick={reiniciar}>
                  <RotateCcw style={{ width: 15, height: 15 }} />
                  Limpiar
                </button>
              )}
              <button
                className="btn btn-primary"
                onClick={analizar}
                disabled={!scopusQ.trim() || !epoQ.trim()}
              >
                <Play style={{ width: 15, height: 15 }} />
                Analizar ambas fuentes
              </button>
            </>
          )}
        </div>

        <p style={{ fontSize: 11.5, color: "var(--text-4)", marginTop: 12, lineHeight: 1.5 }}>
          Las credenciales de Scopus y EPO están en el servidor. EPO consulta un
          año por petición: {Math.max(0, yearTo - yearFrom + 1)} años ≈{" "}
          {Math.max(0, yearTo - yearFrom + 1)} peticiones, y puede tardar varios
          minutos si el servicio está saturado.
        </p>
      </section>

      {/* ── Progreso ── */}
      {(corriendo || papers.error || patents.error) && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: 16,
            marginBottom: 22,
          }}
        >
          <Progreso titulo="Papers · Scopus" estado={papers} color="#a78bfa" />
          <Progreso titulo="Patentes · EPO OPS" estado={patents} color="#22c55e" />
        </div>
      )}

      {/* ── Desfase ── */}
      {comparando && (
        <div className="card" style={{ padding: 20, marginBottom: 22, display: "flex", gap: 10 }}>
          <Loader2 className="animate-spin" style={{ width: 16, height: 16, color: "var(--text-3)" }} />
          <span style={{ fontSize: 13, color: "var(--text-3)" }}>
            Calculando el desfase entre ambas series…
          </span>
        </div>
      )}

      {errorComparar && (
        <Aviso texto={errorComparar} />
      )}

      {comparacion?.matrix && (
        <section className="card" style={{ padding: 24, marginBottom: 22 }}>
          <p className="label-sm" style={{ color: "var(--text-3)", marginBottom: 16 }}>
            Matriz de madurez · ciencia × tecnología
          </p>
          <MaturityMatrix matrix={comparacion.matrix} />
        </section>
      )}

      {comparacion && (
        <section className="card" style={{ padding: 24, marginBottom: 22 }}>
          <p className="label-sm" style={{ color: "var(--text-3)", marginBottom: 10 }}>
            Desfase ciencia → tecnología
          </p>

          <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
            {comparacion.lag_years !== null ? (
              <>
                <span style={{ fontSize: 38, fontWeight: 700, color: "var(--text-1)", lineHeight: 1 }}>
                  {Math.abs(comparacion.lag_years)}
                </span>
                <span style={{ fontSize: 14, color: "var(--text-2)" }}>
                  años · {DIRECCION[comparacion.direction]}
                </span>
              </>
            ) : (
              <span style={{ fontSize: 15, color: "var(--text-2)" }}>
                No estimable
              </span>
            )}
          </div>

          <p style={{ fontSize: 13.5, color: "var(--text-2)", marginTop: 12, lineHeight: 1.6 }}>
            {comparacion.interpretation}
          </p>

          {comparacion.caveats.length > 0 && (
            <div style={{ marginTop: 16, display: "flex", flexDirection: "column", gap: 8 }}>
              {comparacion.caveats.map((c, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    gap: 9,
                    padding: "9px 12px",
                    borderRadius: 8,
                    background: "rgba(245,158,11,0.07)",
                    border: "1px solid rgba(245,158,11,0.2)",
                    fontSize: 12.5,
                    color: "var(--text-2)",
                    lineHeight: 1.5,
                  }}
                >
                  <AlertTriangle style={{ width: 14, height: 14, flexShrink: 0, color: "#f59e0b", marginTop: 1 }} />
                  <span>{c}</span>
                </div>
              ))}
            </div>
          )}

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 16,
              marginTop: 22,
            }}
          >
            <LadoResumen lado={comparacion.papers} acento="#a78bfa" />
            <LadoResumen lado={comparacion.patents} acento="#22c55e" />
          </div>
        </section>
      )}

      {/* ── Curvas ── */}
      {listos && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(440px, 1fr))",
            gap: 20,
          }}
        >
          {papers.resultado && (
            <div className="card" style={{ padding: 20 }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-1)", marginBottom: 14 }}>
                Publicaciones · Scopus
              </h3>
              <SCurveChart result={papers.resultado} />
            </div>
          )}
          {patents.resultado && (
            <div className="card" style={{ padding: 20 }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, color: "var(--text-1)", marginBottom: 14 }}>
                Patentes · EPO OPS
              </h3>
              <SCurveChart result={patents.resultado} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const DIRECCION: Record<SCurveComparison["direction"], string> = {
  patents_lead: "las patentes van por delante",
  papers_lead: "los papers van por delante",
  simultaneous: "avanzan a la par",
  unknown: "sin dirección estimable",
};

function LadoResumen({ lado, acento }: { lado: SCurveComparisonSide; acento: string }) {
  return (
    <div
      style={{
        padding: 16,
        borderRadius: 11,
        border: "1px solid var(--border)",
        background: "var(--surface-2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
        {lado.kind === "papers" ? (
          <BookOpen style={{ width: 14, height: 14, color: acento }} />
        ) : (
          <FileText style={{ width: 14, height: 14, color: acento }} />
        )}
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)" }}>
          {lado.label}
        </span>
        {lado.phase && (
          <span
            style={{
              marginLeft: "auto",
              fontSize: 11,
              fontWeight: 600,
              padding: "2px 8px",
              borderRadius: 20,
              background: `${lado.phase_color ?? acento}22`,
              color: lado.phase_color ?? acento,
            }}
          >
            {lado.phase}
          </span>
        )}
      </div>

      <Dato etiqueta="Registros" valor={lado.records?.toLocaleString() ?? "—"} />
      <Dato etiqueta="Periodo" valor={lado.year_from && lado.year_to ? `${lado.year_from}–${lado.year_to}` : "—"} />
      <Dato etiqueta="Mejor modelo" valor={lado.best_model ?? "—"} />
      <Dato
        etiqueta="Inflexión"
        valor={
          lado.inflection_year
            ? `${lado.inflection_year}${lado.inflection_passed ? " (superada)" : " (proyectada)"}`
            : "no estimable"
        }
      />
      {lado.inflection_range && (
        <Dato
          etiqueta="Rango entre modelos"
          valor={`${lado.inflection_range[0]}–${lado.inflection_range[1]}`}
        />
      )}
      <Dato
        etiqueta="Saturación"
        valor={lado.saturation_pct !== null ? `${lado.saturation_pct}%` : "no estimable"}
      />
      {lado.durbin_watson !== null && (
        <Dato etiqueta="Durbin-Watson" valor={String(lado.durbin_watson)} />
      )}

      {lado.warning && (
        <p
          style={{
            fontSize: 11.5,
            color: "#f59e0b",
            marginTop: 10,
            lineHeight: 1.45,
          }}
        >
          {lado.warning}
        </p>
      )}
    </div>
  );
}

function Dato({ etiqueta, valor }: { etiqueta: string; valor: string }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        gap: 12,
        padding: "5px 0",
        fontSize: 12.5,
      }}
    >
      <span style={{ color: "var(--text-3)" }}>{etiqueta}</span>
      <span style={{ color: "var(--text-1)", fontWeight: 500, textAlign: "right" }}>
        {valor}
      </span>
    </div>
  );
}

function Progreso({
  titulo,
  estado,
  color,
}: {
  titulo: string;
  estado: EstadoLado;
  color: string;
}) {
  return (
    <div className="card" style={{ padding: 18 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: "var(--text-1)" }}>
          {titulo}
        </span>
        {estado.corriendo && (
          <Loader2
            className="animate-spin"
            style={{ width: 13, height: 13, color: "var(--text-3)", marginLeft: "auto" }}
          />
        )}
      </div>

      {estado.error ? (
        <p style={{ fontSize: 12.5, color: "#f0806c", lineHeight: 1.5 }}>{estado.error}</p>
      ) : (
        <>
          <div
            style={{
              height: 5,
              borderRadius: 99,
              background: "var(--surface-3)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${estado.progreso}%`,
                background: color,
                transition: "width 0.3s",
              }}
            />
          </div>
          <p style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 8 }}>
            {estado.mensaje || "En espera…"}
          </p>
        </>
      )}
    </div>
  );
}

function Aviso({ texto }: { texto: string }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        padding: "11px 14px",
        borderRadius: 9,
        background: "rgba(240,128,108,0.08)",
        border: "1px solid rgba(240,128,108,0.22)",
        color: "#f0806c",
        fontSize: 13,
        marginBottom: 22,
      }}
    >
      <AlertTriangle style={{ width: 15, height: 15, flexShrink: 0, marginTop: 1 }} />
      <span>{texto}</span>
    </div>
  );
}

function Campo({
  etiqueta,
  ayuda,
  icono,
  children,
}: {
  etiqueta: string;
  ayuda?: string;
  icono?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div>
      <label
        style={{
          display: "flex",
          alignItems: "center",
          gap: 7,
          fontSize: 12,
          fontWeight: 600,
          color: "var(--text-2)",
          marginBottom: 6,
        }}
      >
        {icono}
        {etiqueta}
        {ayuda && (
          <span style={{ fontWeight: 500, color: "var(--text-4)" }}>· {ayuda}</span>
        )}
      </label>
      {children}
    </div>
  );
}

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Maximize2, X, Search, ExternalLink, ChevronsLeftRight } from "lucide-react";
import type { HypeCycleQuery, HypePhase } from "@/types";
import { HypeCycleChart } from "./HypeCycleChart";

const PHASE_COLORS: Record<HypePhase, string> = {
  "Innovation Trigger": "#60a5fa",
  "Peak of Inflated Expectations": "#f59e0b",
  "Trough of Disillusionment": "#ef4444",
  "Slope of Enlightenment": "#34d399",
  "Plateau of Productivity": "#818cf8",
  Unknown: "#9ca3af",
};

const PHASE_SHORT_ES: Record<HypePhase, string> = {
  "Innovation Trigger": "Lanz.",
  "Peak of Inflated Expectations": "Pico",
  "Trough of Disillusionment": "Valle",
  "Slope of Enlightenment": "Rampa",
  "Plateau of Productivity": "Meseta",
  Unknown: "—",
};

interface Props {
  queries: HypeCycleQuery[];
  highlightId?: string;
  /** Title shown in the fullscreen header (typically the category name). */
  title?: string;
}

/**
 * Tracks the live pixel size of an element. Returns 0/0 until the first
 * ResizeObserver tick lands.
 */
function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState<{ w: number; h: number }>({ w: 0, h: 0 });
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      const cr = entry.contentRect;
      setSize({ w: cr.width, h: cr.height });
    });
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

const ASIDE_DEFAULT = (n: number) => (n > 18 ? 460 : 320);
const ASIDE_MIN = 240;

export function HypeCycleViewer({ queries, highlightId, title }: Props) {
  const [fullscreen, setFullscreen] = useState(false);
  const [hoverId, setHoverId] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [mounted, setMounted] = useState(false);
  const [asideWidth, setAsideWidth] = useState(() => ASIDE_DEFAULT(queries.length));
  const [resizing, setResizing] = useState(false);

  // Portal target must only be referenced after hydration (SSR safety).
  useEffect(() => {
    setMounted(true);
  }, []);

  // Reset the panel width whenever fullscreen toggles so each session starts
  // from a sensible default — without forgetting user resizes mid-session.
  useEffect(() => {
    if (fullscreen) setAsideWidth(ASIDE_DEFAULT(queries.length));
  }, [fullscreen, queries.length]);

  // Drag-to-resize handler. Runs while `resizing` is true; updates width based
  // on cursor X relative to the right edge of the viewport.
  useEffect(() => {
    if (!resizing) return;
    const onMove = (e: MouseEvent) => {
      const maxW = Math.max(ASIDE_MIN, window.innerWidth - 380);
      const w = Math.min(maxW, Math.max(ASIDE_MIN, window.innerWidth - e.clientX));
      setAsideWidth(w);
    };
    const onUp = () => setResizing(false);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    const prevCursor = document.body.style.cursor;
    const prevSelect = document.body.style.userSelect;
    document.body.style.cursor = "ew-resize";
    document.body.style.userSelect = "none";
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      document.body.style.cursor = prevCursor;
      document.body.style.userSelect = prevSelect;
    };
  }, [resizing]);

  // Assign short numeric codes (1..N) ordered by position in the cycle (x axis).
  // The number is rendered *inside* each bubble — short enough to fit even at
  // dense sizes. The legend uses the same number so the mapping is obvious.
  const { labelMap, ordered } = useMemo(() => {
    const sorted = [...queries].sort((a, b) => {
      const dx = a.hype_metrics.hype_cycle_position_x - b.hype_metrics.hype_cycle_position_x;
      if (dx !== 0) return dx;
      return (a.technology_name ?? "").localeCompare(b.technology_name ?? "");
    });
    const map: Record<string, string> = {};
    sorted.forEach((q, i) => {
      map[q.query_id] = String(i + 1);
    });
    return { labelMap: map, ordered: sorted };
  }, [queries]);

  const effectiveHighlight = hoverId ?? highlightId;

  // Lock body scroll and bind ESC while the fullscreen overlay is open
  useEffect(() => {
    if (!fullscreen) return;
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setFullscreen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = prevOverflow;
      window.removeEventListener("keydown", onKey);
    };
  }, [fullscreen]);

  // Measure chart containers so Plotly receives an explicit pixel height
  // (percentage height is unreliable with react-plotly.js).
  const [embeddedRef, embeddedSize] = useElementSize<HTMLDivElement>();
  const [fullChartRef, fullChartSize] = useElementSize<HTMLDivElement>();

  const filtered = useMemo(() => {
    const f = filter.trim().toLowerCase();
    if (!f) return ordered;
    return ordered.filter((q) => {
      const name = (q.technology_name || q.search_query || "").toLowerCase();
      const code = labelMap[q.query_id] ?? "";
      return name.includes(f) || code.includes(f);
    });
  }, [ordered, filter, labelMap]);

  /* ── Compact legend (used in both modes) ─────────────── */
  function renderLegend(opts: { columns: 1 | 2; wrap?: boolean }) {
    const { columns, wrap = false } = opts;
    return (
      <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0 }}>
        <div
          style={{
            padding: "10px 12px",
            borderBottom: "1px solid var(--border)",
            flexShrink: 0,
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <div style={{ position: "relative", flex: 1 }}>
            <Search
              style={{
                position: "absolute",
                left: 8,
                top: "50%",
                transform: "translateY(-50%)",
                width: 12,
                height: 12,
                color: "var(--text-4, #64748b)",
                pointerEvents: "none",
              }}
            />
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder={`Filtrar ${ordered.length}…`}
              style={{
                width: "100%",
                padding: "6px 8px 6px 26px",
                fontSize: 11.5,
                background: "rgba(255,255,255,0.03)",
                border: "1px solid var(--border)",
                borderRadius: 7,
                color: "var(--text-1)",
                outline: "none",
                fontFamily: "inherit",
              }}
            />
          </div>
          <span style={{ fontSize: 10.5, color: "var(--text-3)", fontVariantNumeric: "tabular-nums" }}>
            {filtered.length}/{ordered.length}
          </span>
        </div>

        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "6px 8px",
            display: "grid",
            gridTemplateColumns: columns === 2 ? "1fr 1fr" : "1fr",
            gap: 3,
            alignContent: "start",
          }}
        >
          {filtered.length === 0 ? (
            <p
              style={{
                gridColumn: "1 / -1",
                padding: 24,
                fontSize: 11,
                color: "var(--text-3)",
                textAlign: "center",
              }}
            >
              Sin coincidencias
            </p>
          ) : (
            filtered.map((q) => {
              const code = labelMap[q.query_id];
              const color = PHASE_COLORS[q.hype_metrics.phase];
              const isActive = effectiveHighlight === q.query_id;
              return (
                <button
                  key={q.query_id}
                  type="button"
                  onMouseEnter={() => setHoverId(q.query_id)}
                  onMouseLeave={() => setHoverId(null)}
                  title={`${q.technology_name || q.search_query} · ${PHASE_SHORT_ES[q.hype_metrics.phase]} · ${(q.hype_metrics.confidence * 100).toFixed(0)}%`}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "22px 1fr auto",
                    alignItems: "center",
                    gap: 8,
                    width: "100%",
                    padding: "4px 7px",
                    borderRadius: 6,
                    border: `1px solid ${isActive ? color : "transparent"}`,
                    background: isActive ? `${color}1f` : "transparent",
                    cursor: "pointer",
                    textAlign: "left",
                    fontFamily: "inherit",
                    color: "var(--text-1)",
                    transition: "background 0.1s, border-color 0.1s",
                    minHeight: 24,
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: 22,
                      height: 18,
                      borderRadius: 5,
                      background: color,
                      color: "#0b0f1a",
                      fontSize: 10.5,
                      fontWeight: 700,
                      letterSpacing: "-0.02em",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {code}
                  </span>
                  <span
                    style={{
                      fontSize: 11.5,
                      fontWeight: 500,
                      color: "var(--text-1)",
                      whiteSpace: wrap ? "normal" : "nowrap",
                      overflow: "hidden",
                      textOverflow: wrap ? "clip" : "ellipsis",
                      lineHeight: 1.25,
                      minWidth: 0,
                      wordBreak: wrap ? "break-word" : "normal",
                    }}
                  >
                    {q.technology_name || q.search_query}
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      color: "var(--text-3)",
                      fontVariantNumeric: "tabular-nums",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {(q.hype_metrics.confidence * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })
          )}
        </div>
      </div>
    );
  }

  /* ── Fullscreen overlay (rendered via Portal to escape stacking/transform ancestors) ── */
  const fullH = fullChartSize.h > 80 ? fullChartSize.h - 8 : 480;
  const fullscreenOverlay = (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 99999,
        background: "var(--bg, #06090f)",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 20px",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface)",
          flexShrink: 0,
        }}
      >
        <div>
          <p
            style={{
              fontSize: 10,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: "var(--accent)",
              marginBottom: 2,
            }}
          >
            Hype Cycle · {queries.length} tecnologías
          </p>
          <h2
            style={{
              fontSize: 16,
              fontWeight: 700,
              color: "var(--text-1)",
              letterSpacing: "-0.01em",
            }}
          >
            {title ?? "Posición consolidada"}
          </h2>
        </div>
        <button
          type="button"
          onClick={() => setFullscreen(false)}
          className="btn btn-ghost"
          style={{ padding: "6px 12px", fontSize: 12 }}
          aria-label="Cerrar pantalla completa"
        >
          <X style={{ width: 13, height: 13 }} />
          Cerrar (Esc)
        </button>
      </header>

      <div style={{ flex: 1, display: "flex", minHeight: 0 }}>
        <div
          ref={fullChartRef}
          style={{
            flex: 1,
            minWidth: 0,
            padding: "8px 8px 4px 12px",
            overflow: "hidden",
          }}
        >
          <HypeCycleChart
            queries={queries}
            highlightId={effectiveHighlight}
            labelMap={labelMap}
            showLegend={false}
            height={fullH}
          />
        </div>
        {/* Drag handle — resize the legend panel to read full names */}
        <div
          onMouseDown={(e) => {
            e.preventDefault();
            setResizing(true);
          }}
          onDoubleClick={() => setAsideWidth(ASIDE_DEFAULT(queries.length))}
          title="Arrastrar para redimensionar · Doble click para restablecer"
          style={{
            width: 8,
            flexShrink: 0,
            cursor: "ew-resize",
            background: resizing ? "var(--accent)" : "var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            position: "relative",
            transition: "background 0.12s",
          }}
          onMouseEnter={(e) => {
            if (!resizing) e.currentTarget.style.background = "rgba(79,142,247,0.55)";
          }}
          onMouseLeave={(e) => {
            if (!resizing) e.currentTarget.style.background = "var(--border)";
          }}
        >
          <ChevronsLeftRight
            style={{
              width: 14,
              height: 14,
              color: resizing ? "#0b0f1a" : "rgba(226,232,240,0.9)",
              pointerEvents: "none",
            }}
          />
        </div>
        <aside
          style={{
            width: asideWidth,
            flexShrink: 0,
            background: "var(--surface)",
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
          }}
        >
          {renderLegend({
            columns: asideWidth >= 520 && queries.length > 12 ? 2 : 1,
            wrap: asideWidth >= 360,
          })}
        </aside>
      </div>
    </div>
  );

  /* ── Embedded view ───────────────────────────────────── */
  // Scale embedded chart height with how many techs we have, but cap it so the
  // whole page (stat cards + chart + legend + table) stays in one viewport.
  const embedHeight = queries.length > 18 ? 520 : queries.length > 8 ? 480 : 440;
  const legendWidth = queries.length > 18 ? 280 : 260;

  function openInWindow() {
    if (typeof window === "undefined") return;
    // Open the current page in a sized popup. The page itself stays the same;
    // the user can then click "Expandir" in that window to get a clean view.
    window.open(
      window.location.href,
      "hype-cycle-aux",
      "width=1600,height=900,noopener,noreferrer"
    );
  }

  return (
    <div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "flex-end",
          gap: 6,
          marginBottom: 6,
        }}
      >
        <button
          type="button"
          onClick={openInWindow}
          className="btn btn-ghost"
          style={{ padding: "6px 12px", fontSize: 12 }}
          aria-label="Abrir en ventana auxiliar"
          title="Abrir en una ventana auxiliar del navegador"
        >
          <ExternalLink style={{ width: 13, height: 13 }} />
          Ventana aux.
        </button>
        <button
          type="button"
          onClick={() => setFullscreen(true)}
          className="btn btn-ghost"
          style={{ padding: "6px 12px", fontSize: 12 }}
          aria-label="Expandir a pantalla completa"
        >
          <Maximize2 style={{ width: 13, height: 13 }} />
          Expandir
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: `minmax(0, 1fr) ${legendWidth}px`,
          gap: 14,
          alignItems: "stretch",
        }}
      >
        <div
          ref={embeddedRef}
          style={{ minWidth: 0, height: embedHeight, overflow: "hidden" }}
        >
          <HypeCycleChart
            queries={queries}
            highlightId={effectiveHighlight}
            labelMap={labelMap}
            showLegend={false}
            height={embeddedSize.h > 80 ? embeddedSize.h : embedHeight}
          />
        </div>
        <aside
          style={{
            border: "1px solid var(--border)",
            borderRadius: 10,
            background: "rgba(255,255,255,0.02)",
            height: embedHeight,
            display: "flex",
            flexDirection: "column",
            minHeight: 0,
            overflow: "hidden",
          }}
        >
          {renderLegend({ columns: 1 })}
        </aside>
      </div>

      {fullscreen && mounted && createPortal(fullscreenOverlay, document.body)}
    </div>
  );
}

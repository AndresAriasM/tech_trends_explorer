"use client";

import { useState } from "react";
import { ExternalLink, ChevronDown, ChevronUp } from "lucide-react";
import type { NewsItem } from "@/types";

const SENTIMENT_CONFIG = (s: number) => {
  if (s >= 0.05) return { label: "Positivo", color: "#22c55e", bg: "rgba(34,197,94,0.10)" };
  if (s <= -0.05) return { label: "Negativo", color: "#ef4444", bg: "rgba(239,68,68,0.10)" };
  return { label: "Neutral", color: "#f59e0b", bg: "rgba(245,158,11,0.10)" };
};

interface Props {
  items: NewsItem[];
}

export function NewsResultsList({ items }: Props) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 10;
  const pages = Math.ceil(items.length / PAGE_SIZE);
  const visible = items.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);

  function toggle(i: number) {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(i) ? next.delete(i) : next.add(i);
      return next;
    });
  }

  return (
    <div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {visible.map((item, idx) => {
          const globalIdx = page * PAGE_SIZE + idx;
          const isOpen = expanded.has(globalIdx);
          const sent = SENTIMENT_CONFIG(item.sentiment);

          return (
            <div
              key={globalIdx}
              style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid var(--border)",
                borderRadius: 10,
                overflow: "hidden",
                transition: "border-color 0.15s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--border-2)")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "var(--border)")}
            >
              {/* Header row */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "10px 14px",
                  cursor: "pointer",
                }}
                onClick={() => toggle(globalIdx)}
              >
                {/* Index */}
                <span style={{
                  fontSize: 10,
                  fontWeight: 700,
                  color: "var(--text-4)",
                  minWidth: 22,
                  textAlign: "right",
                }}>
                  {globalIdx + 1}
                </span>

                {/* Sentiment pill */}
                <span style={{
                  fontSize: 10,
                  fontWeight: 700,
                  letterSpacing: "0.04em",
                  padding: "2px 7px",
                  borderRadius: 99,
                  background: sent.bg,
                  color: sent.color,
                  flexShrink: 0,
                }}>
                  {item.sentiment.toFixed(2)}
                </span>

                {/* Title */}
                <span style={{
                  flex: 1,
                  fontSize: 13,
                  fontWeight: 500,
                  color: "var(--text-1)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}>
                  {item.title}
                </span>

                {/* Year */}
                {item.year && (
                  <span style={{ fontSize: 11, color: "var(--text-3)", flexShrink: 0 }}>
                    {item.year}
                  </span>
                )}

                {/* Expand icon */}
                {isOpen
                  ? <ChevronUp style={{ width: 14, height: 14, color: "var(--text-3)", flexShrink: 0 }} />
                  : <ChevronDown style={{ width: 14, height: 14, color: "var(--text-3)", flexShrink: 0 }} />
                }
              </div>

              {/* Expanded body */}
              {isOpen && (
                <div style={{
                  padding: "0 14px 14px 46px",
                  borderTop: "1px solid var(--border)",
                  display: "grid",
                  gridTemplateColumns: "1fr auto",
                  gap: 12,
                }}>
                  <div>
                    <p style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.6, margin: "10px 0 8px" }}>
                      {item.snippet}
                    </p>

                    {/* Meta row */}
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 10, fontSize: 12 }}>
                      {item.source && (
                        <span style={{ color: "var(--text-3)" }}>
                          📰 <span style={{ color: "var(--text-2)" }}>{item.source}</span>
                        </span>
                      )}
                      {item.country && (
                        <span style={{ color: "var(--text-3)" }}>
                          🌍 <span style={{ color: "var(--text-2)" }}>{item.country}</span>
                        </span>
                      )}
                      {item.date && (
                        <span style={{ color: "var(--text-3)" }}>
                          📅 <span style={{ color: "var(--text-2)" }}>{item.date}</span>
                        </span>
                      )}
                      {item.keywords.length > 0 && (
                        <span style={{ color: "var(--text-3)" }}>
                          🏷️{" "}
                          {item.keywords.map((kw, ki) => (
                            <span key={ki} style={{
                              display: "inline-block",
                              marginRight: 4,
                              padding: "1px 6px",
                              borderRadius: 4,
                              background: "rgba(96,165,250,0.10)",
                              color: "#93c5fd",
                              fontSize: 11,
                            }}>
                              {kw}
                            </span>
                          ))}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Link */}
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 4,
                      marginTop: 10,
                      fontSize: 12,
                      color: "var(--accent)",
                      textDecoration: "none",
                      flexShrink: 0,
                      alignSelf: "flex-start",
                    }}
                  >
                    <ExternalLink style={{ width: 12, height: 12 }} />
                    Abrir
                  </a>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {pages > 1 && (
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
          marginTop: 16,
        }}>
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => p - 1)}
            style={{
              padding: "5px 12px",
              borderRadius: 7,
              border: "1px solid var(--border)",
              background: "transparent",
              color: page === 0 ? "var(--text-4)" : "var(--text-2)",
              cursor: page === 0 ? "default" : "pointer",
              fontSize: 12,
              fontFamily: "inherit",
            }}
          >
            ← Anterior
          </button>
          <span style={{ fontSize: 12, color: "var(--text-3)" }}>
            {page + 1} / {pages}
          </span>
          <button
            disabled={page === pages - 1}
            onClick={() => setPage((p) => p + 1)}
            style={{
              padding: "5px 12px",
              borderRadius: 7,
              border: "1px solid var(--border)",
              background: "transparent",
              color: page === pages - 1 ? "var(--text-4)" : "var(--text-2)",
              cursor: page === pages - 1 ? "default" : "pointer",
              fontSize: 12,
              fontFamily: "inherit",
            }}
          >
            Siguiente →
          </button>
        </div>
      )}
    </div>
  );
}

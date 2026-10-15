"use client";

import dynamic from "next/dynamic";
import type { CountryStat } from "@/types";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// Country name → ISO-3 code mapping
const COUNTRY_ISO3: Record<string, string> = {
  USA: "USA", "United States": "USA", UK: "GBR", "United Kingdom": "GBR",
  China: "CHN", Japan: "JPN", Germany: "DEU", France: "FRA", India: "IND",
  Italy: "ITA", Canada: "CAN", "South Korea": "KOR", Russia: "RUS",
  Brazil: "BRA", Australia: "AUS", Spain: "ESP", Mexico: "MEX",
  Indonesia: "IDN", Netherlands: "NLD", Turkey: "TUR", "Saudi Arabia": "SAU",
  Switzerland: "CHE", Sweden: "SWE", Poland: "POL", Belgium: "BEL",
  Norway: "NOR", Austria: "AUT", UAE: "ARE", Iran: "IRN", Ireland: "IRL",
  Denmark: "DNK", Colombia: "COL", "South Africa": "ZAF", Egypt: "EGY",
  Chile: "CHL", Argentina: "ARG", Finland: "FIN", "Czech Republic": "CZE",
  Portugal: "PRT", Greece: "GRC", Vietnam: "VNM", "New Zealand": "NZL",
  Thailand: "THA", Algeria: "DZA", Qatar: "QAT", Peru: "PER",
  Romania: "ROU", Hungary: "HUN", Ukraine: "UKR", Philippines: "PHL",
  Singapore: "SGP", "Hong Kong": "HKG", Taiwan: "TWN", Pakistan: "PAK",
  Venezuela: "VEN", Croatia: "HRV", "North Korea": "PRK",
  Morocco: "MAR", Bangladesh: "BGD", Kazakhstan: "KAZ",
};

interface Props {
  data: CountryStat[];
}

export function GeoMap({ data }: Props) {
  const mapped = data
    .map((d) => ({ ...d, iso: COUNTRY_ISO3[d.country] }))
    .filter((d) => d.iso);

  if (mapped.length === 0) {
    return (
      <div style={{
        height: 280,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--text-4)",
        fontSize: 13,
      }}>
        Sin datos geográficos detectados
      </div>
    );
  }

  const maxMentions = Math.max(...mapped.map((d) => d.mentions));

  const trace = {
    type: "scattergeo" as const,
    locations: mapped.map((d) => d.iso),
    locationmode: "ISO-3" as const,
    text: mapped.map(
      (d) => `${d.country}<br>Menciones: ${d.mentions}<br>Sentimiento: ${d.sentiment_avg.toFixed(2)}`
    ),
    mode: "markers" as const,
    marker: {
      size: mapped.map((d) => 8 + (d.mentions / maxMentions) * 28),
      color: mapped.map((d) => d.sentiment_avg),
      colorscale: [
        [0, "#ef4444"],
        [0.5, "#f59e0b"],
        [1, "#22c55e"],
      ] as [number, string][],
      cmin: -1,
      cmax: 1,
      colorbar: {
        title: "Sent.",
        thickness: 10,
        len: 0.6,
        tickfont: { color: "#9ca3af", size: 10 },
        titlefont: { color: "#9ca3af", size: 10 },
      },
      line: { color: "rgba(255,255,255,0.15)", width: 1 },
      opacity: 0.85,
    },
    hovertemplate: "%{text}<extra></extra>",
  };

  const layout = {
    paper_bgcolor: "rgba(0,0,0,0)",
    geo: {
      bgcolor: "rgba(0,0,0,0)",
      showland: true,
      landcolor: "#1f2937",
      showocean: true,
      oceancolor: "#0f172a",
      showcountries: true,
      countrycolor: "#374151",
      showframe: false,
      projection: { type: "natural earth" as const },
    },
    margin: { t: 0, b: 0, l: 0, r: 0 },
  };

  return (
    <Plot
      data={[trace]}
      layout={layout as object}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: "100%", height: "360px" }}
    />
  );
}

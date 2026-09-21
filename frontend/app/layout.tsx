import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/layout/AppShell";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  metadataBase: new URL("http://localhost:3000"),
  title: {
    default: "Prospecta — Plataforma de prospectiva",
    template: "%s · Prospecta",
  },
  description:
    "Plataforma de prospectiva tecnológica: Hype Cycle, curvas en S, Delphi, " +
    "vigilancia, análisis estructural y escenarios. Universidad Pontificia Bolivariana.",
  applicationName: "Prospecta",
  openGraph: {
    title: "Prospecta — Plataforma de prospectiva",
    description:
      "Hype Cycle, curvas en S, Delphi, vigilancia y escenarios sobre un " +
      "modelo de dominio compartido.",
    siteName: "Prospecta",
    locale: "es_CO",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#06090f",
  colorScheme: "dark",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="es" dir="ltr" suppressHydrationWarning className={inter.variable}>
      <body>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}

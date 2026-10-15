"use client";

/**
 * frontend/app/login/page.tsx
 *
 * Pantalla de acceso.
 *
 * Se pinta como capa fija sobre toda la ventana (position: fixed, inset: 0)
 * en lugar de como una página más. El motivo es concreto: app/layout.tsx
 * envuelve *todo* en AppShell, así que sin esto el login aparecería con el
 * menú de módulos al lado — invitando a pulsar en una aplicación a la que
 * todavía no se ha entrado. La capa tapa el armazón sin tocarlo.
 */

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { AlertCircle, ArrowRight, CheckCircle2, Loader2 } from "lucide-react";

import {
  SUPABASE_ANON_KEY,
  SUPABASE_URL,
  requestPasswordReset,
  signIn,
  signUp,
} from "@/lib/auth";

type Mode = "login" | "signup";

export default function LoginPage() {
  const router = useRouter();

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [institution, setInstitution] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const configured = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);
  const isSignup = mode === "signup";
  const canSubmit =
    configured && !loading && email.trim().length > 3 && password.length >= 6;

  function switchMode(target: Mode) {
    setMode(target);
    setError(null);
    setNotice(null);
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;

    setLoading(true);
    setError(null);
    setNotice(null);

    try {
      if (isSignup) {
        const session = await signUp({ email, password, fullName, institution });
        if (!session) {
          // Sin sesión = el proyecto exige confirmar el correo. Decirlo, en
          // lugar de redirigir a una aplicación que devolverá 401.
          setNotice(
            "Cuenta creada. Revisa tu correo y confirma la dirección para entrar."
          );
          setMode("login");
          setPassword("");
          return;
        }
      } else {
        await signIn(email, password);
      }
      router.replace(redirectTarget());
      router.refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo completar el acceso.");
    } finally {
      setLoading(false);
    }
  }

  async function handleRecover() {
    if (!email.trim()) {
      setError("Escribe tu correo y volvemos a intentarlo.");
      return;
    }
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      await requestPasswordReset(email);
      setNotice("Si ese correo tiene cuenta, recibirás un enlace para restablecerla.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo enviar el enlace.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      className="mesh-bg"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        overflowY: "auto",
      }}
    >
      <div
        className="card fade-in"
        style={{
          width: "100%",
          maxWidth: 420,
          padding: "36px 32px 28px",
          background: "rgba(12, 17, 32, 0.82)",
          backdropFilter: "blur(20px)",
          WebkitBackdropFilter: "blur(20px)",
          boxShadow: "0 24px 80px rgba(0,0,0,0.55)",
        }}
      >
        {/* ── Marca ── */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            marginBottom: 28,
          }}
        >
          <Image
            src="/prospecta-mark.png"
            alt=""
            width={40}
            height={40}
            priority
            style={{ width: 40, height: 40, objectFit: "contain" }}
          />
          <div>
            <p
              className="gradient-text"
              style={{
                fontSize: 21,
                fontWeight: 800,
                letterSpacing: "-0.03em",
                lineHeight: 1,
              }}
            >
              Prospecta
            </p>
            <p
              style={{
                fontSize: 9.5,
                color: "var(--text-3)",
                letterSpacing: "0.09em",
                textTransform: "uppercase",
                marginTop: 4,
              }}
            >
              Prospectiva · UPB
            </p>
          </div>
        </div>

        <h1
          className="section-title"
          style={{ fontSize: 22, marginBottom: 6 }}
        >
          {isSignup ? "Crear cuenta" : "Entrar"}
        </h1>
        <p
          style={{
            fontSize: 13,
            color: "var(--text-2)",
            marginBottom: 24,
            lineHeight: 1.5,
          }}
        >
          {isSignup
            ? "El rol se asigna al darte de alta según la invitación del administrador."
            : "Accede con tu correo institucional para ver tus estudios y rondas."}
        </p>

        {!configured && (
          <Banner
            tone="error"
            icon={<AlertCircle style={{ width: 15, height: 15, flexShrink: 0 }} />}
          >
            Falta configurar <code>NEXT_PUBLIC_SUPABASE_URL</code> y{" "}
            <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code> en{" "}
            <code>frontend/.env.local</code>.
          </Banner>
        )}

        <form onSubmit={handleSubmit} noValidate>
          {isSignup && (
            <>
              <Field label="Nombre completo" htmlFor="full_name">
                <input
                  id="full_name"
                  className="inp"
                  type="text"
                  autoComplete="name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Andrés Arias"
                  disabled={loading}
                />
              </Field>
              <Field label="Institución" htmlFor="institution">
                <input
                  id="institution"
                  className="inp"
                  type="text"
                  autoComplete="organization"
                  value={institution}
                  onChange={(e) => setInstitution(e.target.value)}
                  placeholder="Universidad Pontificia Bolivariana"
                  disabled={loading}
                />
              </Field>
            </>
          )}

          <Field label="Correo" htmlFor="email">
            <input
              id="email"
              className="inp"
              type="email"
              required
              autoComplete="email"
              autoFocus={!isSignup}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="nombre@upb.edu.co"
              disabled={loading}
            />
          </Field>

          <Field
            label="Contraseña"
            htmlFor="password"
            hint={isSignup ? "Mínimo 6 caracteres" : undefined}
          >
            <input
              id="password"
              className="inp"
              type="password"
              required
              minLength={6}
              autoComplete={isSignup ? "new-password" : "current-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              disabled={loading}
            />
          </Field>

          <div aria-live="polite">
            {error && (
              <Banner
                tone="error"
                icon={<AlertCircle style={{ width: 15, height: 15, flexShrink: 0 }} />}
              >
                {error}
              </Banner>
            )}
            {notice && (
              <Banner
                tone="ok"
                icon={<CheckCircle2 style={{ width: 15, height: 15, flexShrink: 0 }} />}
              >
                {notice}
              </Banner>
            )}
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={!canSubmit}
            style={{ width: "100%", justifyContent: "center", marginTop: 6 }}
          >
            {loading ? (
              <>
                <Loader2 className="spin" style={{ width: 15, height: 15 }} />
                {isSignup ? "Creando cuenta…" : "Entrando…"}
              </>
            ) : (
              <>
                {isSignup ? "Crear cuenta" : "Entrar"}
                <ArrowRight style={{ width: 15, height: 15 }} />
              </>
            )}
          </button>
        </form>

        {!isSignup && (
          <button
            type="button"
            onClick={handleRecover}
            disabled={loading}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-3)",
              fontSize: 12,
              cursor: loading ? "not-allowed" : "pointer",
              padding: "12px 0 0",
              fontFamily: "inherit",
            }}
          >
            ¿Olvidaste tu contraseña?
          </button>
        )}

        <div className="divider" style={{ margin: "22px 0 16px" }} />

        <p style={{ fontSize: 12.5, color: "var(--text-3)", textAlign: "center" }}>
          {isSignup ? "¿Ya tienes cuenta?" : "¿Primera vez en Prospecta?"}{" "}
          <button
            type="button"
            onClick={() => switchMode(isSignup ? "login" : "signup")}
            style={{
              background: "none",
              border: "none",
              color: "var(--accent)",
              fontWeight: 600,
              fontSize: 12.5,
              cursor: "pointer",
              padding: 0,
              fontFamily: "inherit",
            }}
          >
            {isSignup ? "Entrar" : "Crear cuenta"}
          </button>
        </p>
      </div>
    </div>
  );
}

/* ── Piezas locales ─────────────────────────────────────────────────────── */

/**
 * A dónde volver tras entrar. Se lee de `window` en el momento de navegar —
 * no con useSearchParams— para no tener que envolver la página en un
 * <Suspense>, y se exige que sea una ruta interna: un `next` absoluto
 * convertiría el login en un redirector abierto.
 */
function redirectTarget(): string {
  if (typeof window === "undefined") return "/";
  const target = new URLSearchParams(window.location.search).get("next");
  return target && target.startsWith("/") && !target.startsWith("//")
    ? target
    : "/";
}

function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label
        htmlFor={htmlFor}
        className="label-sm"
        style={{ display: "block", marginBottom: 6, color: "var(--text-3)" }}
      >
        {label}
      </label>
      {children}
      {hint && (
        <p style={{ fontSize: 11, color: "var(--text-3)", marginTop: 5 }}>{hint}</p>
      )}
    </div>
  );
}

function Banner({
  tone,
  icon,
  children,
}: {
  tone: "error" | "ok";
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  const palette =
    tone === "error"
      ? { bg: "rgba(239, 68, 68, 0.10)", border: "rgba(239, 68, 68, 0.28)", fg: "#fca5a5" }
      : { bg: "rgba(34, 197, 94, 0.10)", border: "rgba(34, 197, 94, 0.28)", fg: "#86efac" };

  return (
    <div
      role={tone === "error" ? "alert" : "status"}
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 8,
        background: palette.bg,
        border: `1px solid ${palette.border}`,
        color: palette.fg,
        borderRadius: 10,
        padding: "10px 12px",
        fontSize: 12.5,
        lineHeight: 1.45,
        marginBottom: 14,
      }}
    >
      {icon}
      <span>{children}</span>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import { KeyRound, Loader2, LogOut, ShieldCheck, UserRound } from "lucide-react";
import {
  ROLE_LABELS,
  SUPABASE_ANON_KEY,
  SUPABASE_URL,
  fetchProfile,
  getAccessToken,
  getUser,
  signOut,
  updateProfile,
  type Profile,
  type Role,
} from "@/lib/auth";

/**
 * Cambio de contraseña con la sesión ya iniciada.
 *
 * Va directo a GoTrue en vez de pasar por el backend: es el servicio que
 * custodia las credenciales, y así la contraseña nueva no atraviesa nuestra
 * API. Vive aquí porque es el único sitio que lo necesita.
 */
async function cambiarContrasena(nueva: string): Promise<void> {
  const token = await getAccessToken();
  if (!token) throw new Error("La sesión ha caducado. Vuelve a entrar.");

  const res = await fetch(`${SUPABASE_URL}/auth/v1/user`, {
    method: "PUT",
    headers: {
      apikey: SUPABASE_ANON_KEY,
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ password: nueva }),
  });

  if (!res.ok) {
    const cuerpo = await res.json().catch(() => null);
    throw new Error(cuerpo?.msg || cuerpo?.error_description || "No se pudo cambiar la contraseña.");
  }
}

export default function CuentaPage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const [cargando, setCargando] = useState(true);

  const [nombre, setNombre] = useState("");
  const [institucion, setInstitucion] = useState("");
  const [guardando, setGuardando] = useState(false);
  const [avisoPerfil, setAvisoPerfil] = useState<string | null>(null);

  const [pwd1, setPwd1] = useState("");
  const [pwd2, setPwd2] = useState("");
  const [cambiando, setCambiando] = useState(false);
  const [avisoPwd, setAvisoPwd] = useState<string | null>(null);
  const [errorPwd, setErrorPwd] = useState<string | null>(null);

  useEffect(() => {
    fetchProfile()
      .then((p) => {
        setEmail(getUser()?.email ?? null);
        setProfile(p);
        setNombre(p.full_name ?? "");
        setInstitucion(p.institution ?? "");
      })
      .catch(() => setProfile(null))
      .finally(() => setCargando(false));
  }, []);

  const guardar = useCallback(async () => {
    setGuardando(true);
    setAvisoPerfil(null);
    try {
      const p = await updateProfile({
        full_name: nombre.trim() || undefined,
        institution: institucion.trim() || undefined,
      });
      setProfile(p);
      setAvisoPerfil("Perfil actualizado.");
    } catch (e: unknown) {
      setAvisoPerfil(e instanceof Error ? e.message : "No se pudo guardar.");
    } finally {
      setGuardando(false);
    }
  }, [nombre, institucion]);

  const cambiar = useCallback(async () => {
    setErrorPwd(null);
    setAvisoPwd(null);
    if (pwd1.length < 8) {
      setErrorPwd("La contraseña debe tener al menos 8 caracteres.");
      return;
    }
    if (pwd1 !== pwd2) {
      setErrorPwd("Las dos contraseñas no coinciden.");
      return;
    }
    setCambiando(true);
    try {
      await cambiarContrasena(pwd1);
      setPwd1("");
      setPwd2("");
      setAvisoPwd("Contraseña cambiada.");
    } catch (e: unknown) {
      setErrorPwd(e instanceof Error ? e.message : "No se pudo cambiar.");
    } finally {
      setCambiando(false);
    }
  }, [pwd1, pwd2]);

  const rol = profile?.role as Role | undefined;

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "32px 48px 64px" }}>
      <header style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "var(--text-1)" }}>
          Mi cuenta
        </h1>
        <p style={{ fontSize: 13.5, color: "var(--text-3)", marginTop: 6 }}>
          Tus datos de acceso al portal y el rol con el que entras.
        </p>
      </header>

      {cargando ? (
        <div className="card" style={{ padding: 28, display: "flex", gap: 10, alignItems: "center" }}>
          <Loader2 className="animate-spin" style={{ width: 16, height: 16, color: "var(--text-3)" }} />
          <span style={{ fontSize: 13, color: "var(--text-3)" }}>Cargando perfil…</span>
        </div>
      ) : (
        <>
          {/* ── Identidad ── */}
          <section className="card" style={{ padding: 24, marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 20 }}>
              <div
                style={{
                  width: 46,
                  height: 46,
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, #4f8ef7 0%, #8b5cf6 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 18,
                  fontWeight: 700,
                  color: "#fff",
                  flexShrink: 0,
                }}
              >
                {(email?.[0] ?? "?").toUpperCase()}
              </div>
              <div style={{ minWidth: 0 }}>
                <p className="truncate" style={{ fontSize: 15, fontWeight: 600, color: "var(--text-1)" }}>
                  {email ?? "Sin sesión"}
                </p>
                {rol && (
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 5,
                      marginTop: 6,
                      fontSize: 11,
                      fontWeight: 600,
                      color: "var(--accent)",
                      background: "rgba(79,142,247,0.12)",
                      border: "1px solid rgba(79,142,247,0.25)",
                      borderRadius: 6,
                      padding: "3px 8px",
                    }}
                  >
                    <ShieldCheck style={{ width: 12, height: 12 }} />
                    {ROLE_LABELS[rol]}
                  </span>
                )}
              </div>
            </div>

            <p style={{ fontSize: 11, color: "var(--text-4)", marginBottom: 16 }}>
              El rol lo asigna un administrador; no se puede cambiar desde aquí.
            </p>

            <Campo etiqueta="Nombre completo">
              <input
                className="inp"
                style={{ width: "100%" }}
                value={nombre}
                onChange={(e) => setNombre(e.target.value)}
                placeholder="Tu nombre"
              />
            </Campo>
            <Campo etiqueta="Institución">
              <input
                className="inp"
                style={{ width: "100%" }}
                value={institucion}
                onChange={(e) => setInstitucion(e.target.value)}
                placeholder="Universidad, empresa…"
              />
            </Campo>

            <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 4 }}>
              <button className="btn btn-primary" onClick={guardar} disabled={guardando}>
                <UserRound style={{ width: 15, height: 15 }} />
                {guardando ? "Guardando…" : "Guardar cambios"}
              </button>
              {avisoPerfil && (
                <span style={{ fontSize: 12.5, color: "var(--text-3)" }}>{avisoPerfil}</span>
              )}
            </div>
          </section>

          {/* ── Contraseña ── */}
          <section className="card" style={{ padding: 24, marginBottom: 20 }}>
            <h2 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-1)", marginBottom: 4 }}>
              Cambiar contraseña
            </h2>
            <p style={{ fontSize: 12.5, color: "var(--text-3)", marginBottom: 18 }}>
              Mínimo 8 caracteres. Al cambiarla, las demás sesiones siguen activas
              hasta que caduquen.
            </p>

            <Campo etiqueta="Nueva contraseña">
              <input
                className="inp"
                style={{ width: "100%" }}
                type="password"
                value={pwd1}
                onChange={(e) => setPwd1(e.target.value)}
                autoComplete="new-password"
              />
            </Campo>
            <Campo etiqueta="Repítela">
              <input
                className="inp"
                style={{ width: "100%" }}
                type="password"
                value={pwd2}
                onChange={(e) => setPwd2(e.target.value)}
                autoComplete="new-password"
              />
            </Campo>

            <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 4 }}>
              <button className="btn btn-ghost" onClick={cambiar} disabled={cambiando}>
                <KeyRound style={{ width: 15, height: 15 }} />
                {cambiando ? "Cambiando…" : "Cambiar contraseña"}
              </button>
              {avisoPwd && (
                <span style={{ fontSize: 12.5, color: "#4ade80" }}>{avisoPwd}</span>
              )}
              {errorPwd && (
                <span style={{ fontSize: 12.5, color: "#f0806c" }}>{errorPwd}</span>
              )}
            </div>
          </section>

          {/* ── Sesión ── */}
          <section className="card" style={{ padding: 24 }}>
            <button
              className="btn btn-ghost"
              style={{ color: "#f0806c" }}
              onClick={async () => {
                await signOut();
                window.location.href = "/login";
              }}
            >
              <LogOut style={{ width: 15, height: 15 }} />
              Cerrar sesión
            </button>
          </section>
        </>
      )}
    </div>
  );
}

function Campo({ etiqueta, children }: { etiqueta: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label
        style={{
          display: "block",
          fontSize: 12,
          fontWeight: 600,
          color: "var(--text-2)",
          marginBottom: 6,
        }}
      >
        {etiqueta}
      </label>
      {children}
    </div>
  );
}

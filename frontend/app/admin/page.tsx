"use client";

import { useCallback, useMemo, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Loader2,
  Mail,
  Search,
  ShieldCheck,
  Trash2,
  UserPlus,
  Users,
} from "lucide-react";
import {
  ROLES,
  ROLE_LABELS,
  authFetch,
  getUser,
  type Profile,
  type Role,
} from "@/lib/auth";

interface RoleAssignment {
  email: string;
  role: Role;
  note?: string | null;
  created_at?: string;
}

/**
 * Administración del portal: quién entra y con qué rol.
 *
 * Dos mecanismos complementarios, y la diferencia importa:
 *  · Usuarios      — gente que ya tiene cuenta; se le cambia el rol en vivo.
 *  · Pre-asignados — un rol reservado para un correo que todavía no se ha
 *                    registrado; se aplica solo en cuanto esa persona entra.
 */
export default function AdminPage() {
  const qc = useQueryClient();
  const [error, setError] = useState<string | null>(null);
  const [busqueda, setBusqueda] = useState("");
  const [filtroRol, setFiltroRol] = useState<Role | "">("");
  const [guardandoId, setGuardandoId] = useState<string | null>(null);

  const [nuevoEmail, setNuevoEmail] = useState("");
  const [nuevoRol, setNuevoRol] = useState<Role>("experto");
  const [nuevaNota, setNuevaNota] = useState("");
  const [invitando, setInvitando] = useState(false);

  const yo = getUser()?.email ?? null;

  const usersQ = useQuery({
    queryKey: ["admin", "users"],
    queryFn: () => authFetch<Profile[]>("/api/auth/users"),
  });
  const asignQ = useQuery({
    queryKey: ["admin", "role-assignments"],
    queryFn: () => authFetch<RoleAssignment[]>("/api/auth/role-assignments"),
  });

  const users = useMemo(() => usersQ.data ?? [], [usersQ.data]);
  const asignaciones = useMemo(() => asignQ.data ?? [], [asignQ.data]);
  const cargando = usersQ.isLoading || asignQ.isLoading;
  const errorCarga =
    usersQ.error || asignQ.error
      ? "No se pudo cargar la administración. ¿Tu rol es admin?"
      : null;

  const recargar = useCallback(
    () => qc.invalidateQueries({ queryKey: ["admin"] }),
    [qc]
  );

  const cambiarRol = useCallback(
    async (u: Profile, role: Role) => {
      setGuardandoId(u.id);
      setError(null);
      try {
        const actualizado = await authFetch<Profile>(
          `/api/auth/users/${u.id}/role`,
          { method: "PATCH", body: JSON.stringify({ role }) }
        );
        qc.setQueryData<Profile[]>(["admin", "users"], (prev) =>
          (prev ?? []).map((x) => (x.id === actualizado.id ? actualizado : x))
        );
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "No se pudo cambiar el rol.");
      } finally {
        setGuardandoId(null);
      }
    },
    [qc]
  );

  const preasignar = useCallback(async () => {
    const email = nuevoEmail.trim().toLowerCase();
    if (!email.includes("@")) {
      setError("Escribe un correo válido.");
      return;
    }
    setInvitando(true);
    setError(null);
    try {
      await authFetch<RoleAssignment>("/api/auth/role-assignments", {
        method: "PUT",
        body: JSON.stringify({ email, role: nuevoRol, note: nuevaNota || null }),
      });
      setNuevoEmail("");
      setNuevaNota("");
      await recargar();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "No se pudo pre-asignar.");
    } finally {
      setInvitando(false);
    }
  }, [nuevoEmail, nuevoRol, nuevaNota, recargar]);

  const quitarAsignacion = useCallback(
    async (email: string) => {
      setError(null);
      try {
        await authFetch(`/api/auth/role-assignments/${encodeURIComponent(email)}`, {
          method: "DELETE",
        });
        qc.setQueryData<RoleAssignment[]>(["admin", "role-assignments"], (prev) =>
          (prev ?? []).filter((a) => a.email !== email)
        );
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : "No se pudo quitar.");
      }
    },
    [qc]
  );

  const visibles = useMemo(() => {
    const q = busqueda.trim().toLowerCase();
    return users.filter((u) => {
      if (filtroRol && u.role !== filtroRol) return false;
      if (!q) return true;
      return (
        u.email.toLowerCase().includes(q) ||
        (u.full_name ?? "").toLowerCase().includes(q)
      );
    });
  }, [users, busqueda, filtroRol]);

  const porRol = useMemo(() => {
    const c: Record<string, number> = {};
    for (const u of users) c[u.role] = (c[u.role] ?? 0) + 1;
    return c;
  }, [users]);

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "32px 48px 64px" }}>
      <header style={{ marginBottom: 26 }}>
        <h1 style={{ fontSize: 26, fontWeight: 700, color: "var(--text-1)" }}>
          Administración
        </h1>
        <p style={{ fontSize: 13.5, color: "var(--text-3)", marginTop: 6 }}>
          Quién puede entrar al portal y con qué rol.
        </p>
      </header>

      {(error || errorCarga) && (
        <div
          style={{
            padding: "10px 14px",
            borderRadius: 9,
            background: "rgba(240,128,108,0.08)",
            border: "1px solid rgba(240,128,108,0.22)",
            color: "#f0806c",
            fontSize: 13,
            marginBottom: 18,
          }}
        >
          {error ?? errorCarga}
        </div>
      )}

      {cargando ? (
        <div className="card" style={{ padding: 28, display: "flex", gap: 10 }}>
          <Loader2 className="animate-spin" style={{ width: 16, height: 16 }} />
          <span style={{ fontSize: 13, color: "var(--text-3)" }}>Cargando…</span>
        </div>
      ) : (
        <>
          {/* ── Resumen por rol ── */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 14,
              marginBottom: 24,
            }}
          >
            {ROLES.map((r) => (
              <div key={r} className="card" style={{ padding: "16px 18px" }}>
                <p
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "var(--text-3)",
                  }}
                >
                  {ROLE_LABELS[r]}
                </p>
                <p
                  style={{
                    fontSize: 24,
                    fontWeight: 700,
                    color: "var(--text-1)",
                    marginTop: 6,
                  }}
                >
                  {porRol[r] ?? 0}
                </p>
              </div>
            ))}
          </div>

          {/* ── Usuarios registrados ── */}
          <section className="card" style={{ padding: 22, marginBottom: 22 }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 16,
                flexWrap: "wrap",
              }}
            >
              <Users style={{ width: 16, height: 16, color: "var(--text-2)" }} />
              <h2 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-1)", flex: 1 }}>
                Usuarios registrados
              </h2>

              <div style={{ position: "relative" }}>
                <Search
                  style={{
                    width: 14,
                    height: 14,
                    color: "var(--text-4)",
                    position: "absolute",
                    left: 10,
                    top: 9,
                  }}
                />
                <input
                  className="inp"
                  style={{ paddingLeft: 30, width: 220 }}
                  placeholder="Buscar correo o nombre…"
                  value={busqueda}
                  onChange={(e) => setBusqueda(e.target.value)}
                />
              </div>

              <select
                className="inp"
                value={filtroRol}
                onChange={(e) => setFiltroRol(e.target.value as Role | "")}
                style={{ width: 150 }}
              >
                <option value="">Todos los roles</option>
                {ROLES.map((r) => (
                  <option key={r} value={r}>
                    {ROLE_LABELS[r]}
                  </option>
                ))}
              </select>
            </div>

            {visibles.length === 0 ? (
              <p style={{ fontSize: 13, color: "var(--text-3)", padding: "14px 0" }}>
                Ningún usuario coincide con el filtro.
              </p>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      <Th>Usuario</Th>
                      <Th>Institución</Th>
                      <Th>Rol</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibles.map((u) => (
                      <tr key={u.id} style={{ borderBottom: "1px solid var(--border)" }}>
                        <td style={{ padding: "11px 10px 11px 0" }}>
                          <span style={{ color: "var(--text-1)", fontWeight: 500 }}>
                            {u.full_name || u.email.split("@")[0]}
                          </span>
                          {u.email === yo && (
                            <span
                              style={{
                                marginLeft: 7,
                                fontSize: 10,
                                color: "var(--accent)",
                                border: "1px solid rgba(79,142,247,0.3)",
                                borderRadius: 5,
                                padding: "1px 5px",
                              }}
                            >
                              tú
                            </span>
                          )}
                          <span
                            style={{ display: "block", fontSize: 11.5, color: "var(--text-3)" }}
                          >
                            {u.email}
                          </span>
                        </td>
                        <td style={{ padding: "11px 10px", color: "var(--text-3)" }}>
                          {u.institution || "—"}
                        </td>
                        <td style={{ padding: "11px 0" }}>
                          <select
                            className="inp"
                            style={{ width: 140, padding: "5px 8px", fontSize: 12.5 }}
                            value={u.role}
                            disabled={guardandoId === u.id || u.email === yo}
                            title={
                              u.email === yo
                                ? "No puedes cambiar tu propio rol"
                                : "Cambiar rol"
                            }
                            onChange={(e) => cambiarRol(u, e.target.value as Role)}
                          >
                            {ROLES.map((r) => (
                              <option key={r} value={r}>
                                {ROLE_LABELS[r]}
                              </option>
                            ))}
                          </select>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* ── Pre-asignaciones ── */}
          <section className="card" style={{ padding: 22 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <Mail style={{ width: 16, height: 16, color: "var(--text-2)" }} />
              <h2 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-1)" }}>
                Roles reservados
              </h2>
            </div>
            <p style={{ fontSize: 12.5, color: "var(--text-3)", marginBottom: 18 }}>
              Para gente que todavía no tiene cuenta: el rol se aplica solo en
              cuanto esa persona se registre con ese correo.
            </p>

            <div
              style={{
                display: "flex",
                gap: 10,
                marginBottom: 18,
                flexWrap: "wrap",
                alignItems: "flex-end",
              }}
            >
              <input
                className="inp"
                style={{ flex: "1 1 220px" }}
                placeholder="correo@upb.edu.co"
                value={nuevoEmail}
                onChange={(e) => setNuevoEmail(e.target.value)}
              />
              <select
                className="inp"
                style={{ width: 150 }}
                value={nuevoRol}
                onChange={(e) => setNuevoRol(e.target.value as Role)}
              >
                {ROLES.map((r) => (
                  <option key={r} value={r}>
                    {ROLE_LABELS[r]}
                  </option>
                ))}
              </select>
              <input
                className="inp"
                style={{ flex: "1 1 180px" }}
                placeholder="Nota (opcional)"
                value={nuevaNota}
                onChange={(e) => setNuevaNota(e.target.value)}
              />
              <button className="btn btn-primary" onClick={preasignar} disabled={invitando}>
                <UserPlus style={{ width: 15, height: 15 }} />
                {invitando ? "Guardando…" : "Reservar"}
              </button>
            </div>

            {asignaciones.length === 0 ? (
              <p style={{ fontSize: 13, color: "var(--text-3)" }}>
                No hay roles reservados.
              </p>
            ) : (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid var(--border)" }}>
                    <Th>Correo</Th>
                    <Th>Rol</Th>
                    <Th>Nota</Th>
                    <Th> </Th>
                  </tr>
                </thead>
                <tbody>
                  {asignaciones.map((a) => (
                    <tr key={a.email} style={{ borderBottom: "1px solid var(--border)" }}>
                      <td style={{ padding: "10px 10px 10px 0", color: "var(--text-1)" }}>
                        {a.email}
                      </td>
                      <td style={{ padding: "10px" }}>
                        <span
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: 4,
                            fontSize: 11,
                            fontWeight: 600,
                            color: "var(--accent)",
                            background: "rgba(79,142,247,0.12)",
                            border: "1px solid rgba(79,142,247,0.25)",
                            borderRadius: 6,
                            padding: "2px 7px",
                          }}
                        >
                          <ShieldCheck style={{ width: 11, height: 11 }} />
                          {ROLE_LABELS[a.role]}
                        </span>
                      </td>
                      <td style={{ padding: "10px", color: "var(--text-3)" }}>
                        {a.note || "—"}
                      </td>
                      <td style={{ padding: "10px 0", textAlign: "right" }}>
                        <button
                          className="icon-btn"
                          title="Quitar reserva"
                          onClick={() => quitarAsignacion(a.email)}
                          style={{ width: 28, height: 28, color: "#f0806c" }}
                        >
                          <Trash2 style={{ width: 14, height: 14 }} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </section>
        </>
      )}
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th
      style={{
        textAlign: "left",
        padding: "0 10px 9px 0",
        fontSize: 10.5,
        fontWeight: 700,
        letterSpacing: "0.07em",
        textTransform: "uppercase",
        color: "var(--text-3)",
      }}
    >
      {children}
    </th>
  );
}

/**
 * frontend/lib/auth.ts
 *
 * Cliente de autenticación de Prospecta contra GoTrue (Supabase Auth).
 *
 * Por qué no @supabase/supabase-js
 * --------------------------------
 * Haría falta añadir dependencias a package.json, y este frente tiene
 * prohibido tocarlo. Lo que se usa de esa librería —tres llamadas HTTP y un
 * sitio donde guardar la sesión— cabe en este archivo, y así middleware.ts
 * funciona en el runtime edge sin cargar nada. Cuando se pueda instalar
 * `@supabase/ssr`, la migración natural es sustituir el almacenamiento en
 * cookie de aquí por el suyo (cookies httpOnly escritas por el servidor) y
 * conservar el resto de la superficie: los nombres de estas funciones están
 * elegidos para que esa sustitución no obligue a tocar las pantallas.
 *
 * Dónde vive la sesión, y qué implica
 * -----------------------------------
 * En una cookie legible por JavaScript, porque el middleware necesita leerla
 * en cada navegación y sólo el navegador puede escribirla. Consecuencia
 * honesta: ante un XSS el token es robable, igual que con localStorage. La
 * mitigación real no es esconderlo mejor, es que el token dura una hora y que
 * **lo que protege de verdad son la RLS y require_role del backend**. El
 * middleware de esta aplicación decide qué pantalla se muestra, no qué datos
 * se pueden leer.
 */

/* ── Roles ──────────────────────────────────────────────────────────────── */

export type Role = "admin" | "profesor" | "experto";

export const ROLES: Role[] = ["admin", "profesor", "experto"];

export const ROLE_LABELS: Record<Role, string> = {
  admin: "Administrador",
  profesor: "Profesor",
  experto: "Experto",
};

/** El mismo que aplican handle_new_user() y custom_access_token(). */
export const DEFAULT_ROLE: Role = "experto";

/* ── Configuración ──────────────────────────────────────────────────────── */

export const SUPABASE_URL = (
  process.env.NEXT_PUBLIC_SUPABASE_URL ?? ""
).replace(/\/$/, "");

export const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export const SESSION_COOKIE = "prospecta-session";

export const LOGIN_PATH = "/login";

/** Segundos antes de la expiración en los que ya conviene renovar. */
const REFRESH_MARGIN = 60;

function authEndpoint(path: string): string {
  if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
    throw new AuthError(
      "Faltan NEXT_PUBLIC_SUPABASE_URL y NEXT_PUBLIC_SUPABASE_ANON_KEY. " +
        "Créalas en frontend/.env.local antes de iniciar sesión."
    );
  }
  return `${SUPABASE_URL}/auth/v1${path}`;
}

/* ── Tipos ──────────────────────────────────────────────────────────────── */

export interface SessionUser {
  id: string;
  email?: string;
  full_name?: string;
  institution?: string;
}

export interface Session {
  access_token: string;
  refresh_token: string;
  /** Epoch en segundos. */
  expires_at: number;
  user: SessionUser;
}

export interface Profile {
  id: string;
  email?: string;
  full_name?: string;
  institution?: string;
  role: Role;
  role_label: string;
  created_at?: string;
  updated_at?: string;
}

export interface Permissions {
  role: Role;
  role_label: string;
  is_admin: boolean;
  can_manage_users: boolean;
  can_create_studies: boolean;
  can_manage_own_studies: boolean;
  can_answer_rounds: boolean;
  can_ingest_corpus: boolean;
}

export interface JwtClaims {
  sub?: string;
  email?: string;
  exp?: number;
  iat?: number;
  user_role?: string;
  user_metadata?: Record<string, unknown>;
  [key: string]: unknown;
}

export class AuthError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "AuthError";
    this.status = status;
  }
}

/* ── Lectura del JWT ────────────────────────────────────────────────────── */

/**
 * Decodifica el payload SIN verificar la firma.
 *
 * Es deliberado y es seguro para lo que se usa: pintar el nombre, elegir un
 * menú, decidir a qué pantalla mandar a alguien. Un token manipulado a mano
 * conseguiría ver un botón de más y un 401 al pulsarlo — la verificación de
 * verdad ocurre en el backend (core/auth.py) y en la base de datos (RLS).
 * Nada de lo que se decida con esta función puede autorizar un acceso.
 */
export function decodeJwt(token?: string | null): JwtClaims | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    const base64 = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
    return JSON.parse(new TextDecoder().decode(bytes)) as JwtClaims;
  } catch {
    return null;
  }
}

export function roleFromToken(token?: string | null): Role {
  const claims = decodeJwt(token);
  const raw = claims?.user_role;
  return typeof raw === "string" && (ROLES as string[]).includes(raw)
    ? (raw as Role)
    : DEFAULT_ROLE;
}

export function isTokenExpired(token?: string | null, skew = 0): boolean {
  const claims = decodeJwt(token);
  if (!claims?.exp) return true;
  return claims.exp - skew <= Math.floor(Date.now() / 1000);
}

/* ── La sesión en la cookie ─────────────────────────────────────────────── */

/**
 * Convierte el valor crudo de la cookie en una sesión. Pura: la usa tanto el
 * navegador como el middleware, que corre en el edge y no tiene `document`.
 */
export function parseSessionCookie(raw?: string | null): Session | null {
  if (!raw) return null;
  try {
    const data = JSON.parse(decodeURIComponent(raw)) as Partial<Session>;
    if (!data?.access_token || !data?.user?.id) return null;
    return {
      access_token: data.access_token,
      refresh_token: data.refresh_token ?? "",
      expires_at: data.expires_at ?? 0,
      user: data.user as SessionUser,
    };
  } catch {
    return null;
  }
}

export function serializeSession(session: Session): string {
  return encodeURIComponent(JSON.stringify(session));
}

function readCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const match = document.cookie.match(
    new RegExp(`(?:^|; )${name.replace(/[.$?*|{}()[\]\\/+^]/g, "\\$&")}=([^;]*)`)
  );
  return match ? match[1] : null;
}

function writeCookie(name: string, value: string, maxAgeSeconds: number): void {
  if (typeof document === "undefined") return;
  const secure =
    typeof location !== "undefined" && location.protocol === "https:"
      ? "; Secure"
      : "";
  document.cookie = `${name}=${value}; Path=/; Max-Age=${maxAgeSeconds}; SameSite=Lax${secure}`;
}

function deleteCookie(name: string): void {
  if (typeof document === "undefined") return;
  document.cookie = `${name}=; Path=/; Max-Age=0; SameSite=Lax`;
}

/** Sesión guardada en este navegador, sin comprobar si sigue vigente. */
export function getSession(): Session | null {
  return parseSessionCookie(readCookie(SESSION_COOKIE));
}

export function getUser(): SessionUser | null {
  return getSession()?.user ?? null;
}

/** Rol según el token guardado. Para la interfaz, nunca para autorizar. */
export function getRole(): Role | null {
  const session = getSession();
  return session ? roleFromToken(session.access_token) : null;
}

export function isAuthenticated(): boolean {
  const session = getSession();
  return !!session && !isTokenExpired(session.access_token);
}

/* ── Notificación de cambios ────────────────────────────────────────────── */

type Listener = (session: Session | null) => void;
const listeners = new Set<Listener>();

/** Avisa a la interfaz cuando se entra, se sale o se renueva el token. */
export function onAuthChange(listener: Listener): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function emit(session: Session | null): void {
  listeners.forEach((fn) => {
    try {
      fn(session);
    } catch {
      /* un suscriptor roto no debe romper el flujo de autenticación */
    }
  });
}

function persistSession(session: Session): Session {
  const ttl = Math.max(session.expires_at - Math.floor(Date.now() / 1000), 60);
  // La cookie sobrevive al access token: mientras quede refresh_token la
  // sesión se puede renovar sin volver a pedir la contraseña.
  writeCookie(SESSION_COOKIE, serializeSession(session), ttl + 60 * 60 * 24 * 30);
  emit(session);
  return session;
}

export function clearSession(): void {
  deleteCookie(SESSION_COOKIE);
  emit(null);
}

/* ── Llamadas a GoTrue ──────────────────────────────────────────────────── */

interface GoTrueTokenResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  expires_at?: number;
  user?: {
    id: string;
    email?: string;
    user_metadata?: Record<string, unknown>;
  };
  error?: string;
  error_description?: string;
  msg?: string;
  message?: string;
}

async function gotrue(
  path: string,
  init: RequestInit & { body?: string } = {}
): Promise<GoTrueTokenResponse> {
  const res = await fetch(authEndpoint(path), {
    ...init,
    headers: {
      "Content-Type": "application/json",
      apikey: SUPABASE_ANON_KEY,
      ...(init.headers ?? {}),
    },
  });

  let body: GoTrueTokenResponse = {};
  try {
    body = (await res.json()) as GoTrueTokenResponse;
  } catch {
    /* 204 y respuestas vacías: sin cuerpo que leer */
  }

  if (!res.ok) {
    throw new AuthError(translateError(body, res.status), res.status);
  }
  return body;
}

/** Mensajes de GoTrue en castellano, que es el idioma de la aplicación. */
function translateError(body: GoTrueTokenResponse, status: number): string {
  const raw =
    body.error_description || body.msg || body.message || body.error || "";
  const map: Record<string, string> = {
    "Invalid login credentials": "Correo o contraseña incorrectos.",
    "Email not confirmed":
      "Tu correo aún no está confirmado. Revisa la bandeja de entrada.",
    "User already registered": "Ya existe una cuenta con ese correo.",
    "Password should be at least 6 characters.":
      "La contraseña debe tener al menos 6 caracteres.",
    "Signups not allowed for this instance":
      "El registro está cerrado. Pide una invitación al administrador.",
  };
  if (map[raw]) return map[raw];
  if (status === 429)
    return "Demasiados intentos. Espera un momento antes de volver a probar.";
  if (raw) return raw;
  return `No se pudo completar la operación (error ${status}).`;
}

function toSession(body: GoTrueTokenResponse): Session {
  if (!body.access_token || !body.user?.id) {
    throw new AuthError("Respuesta de autenticación incompleta.");
  }
  const metadata = body.user.user_metadata ?? {};
  return {
    access_token: body.access_token,
    refresh_token: body.refresh_token ?? "",
    expires_at:
      body.expires_at ??
      Math.floor(Date.now() / 1000) + (body.expires_in ?? 3600),
    user: {
      id: body.user.id,
      email: body.user.email,
      full_name: typeof metadata.full_name === "string" ? metadata.full_name : undefined,
      institution:
        typeof metadata.institution === "string" ? metadata.institution : undefined,
    },
  };
}

export async function signIn(
  email: string,
  password: string
): Promise<Session> {
  const body = await gotrue("/token?grant_type=password", {
    method: "POST",
    body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
  });
  return persistSession(toSession(body));
}

export interface SignUpData {
  email: string;
  password: string;
  fullName?: string;
  institution?: string;
}

/**
 * Alta de cuenta.
 *
 * `data` acaba en raw_user_meta_data, que es exactamente de donde
 * handle_new_user() saca full_name e institution al crear el perfil. Por eso
 * el formulario de registro pide esos dos campos y no otros: son los que la
 * base de datos sabe recoger sin una segunda escritura.
 *
 * Devuelve null cuando el proyecto exige confirmar el correo — no hay sesión
 * todavía y la pantalla debe decirlo, no fingir que se ha entrado.
 */
export async function signUp(payload: SignUpData): Promise<Session | null> {
  const body = await gotrue("/signup", {
    method: "POST",
    body: JSON.stringify({
      email: payload.email.trim().toLowerCase(),
      password: payload.password,
      data: {
        full_name: payload.fullName?.trim() || undefined,
        institution: payload.institution?.trim() || undefined,
      },
    }),
  });
  if (!body.access_token) return null;
  return persistSession(toSession(body));
}

export async function signOut(): Promise<void> {
  const session = getSession();
  // El borrado local va primero: si la red falla, la sesión tiene que
  // desaparecer de este navegador igualmente.
  clearSession();
  if (!session?.access_token) return;
  try {
    await gotrue("/logout", {
      method: "POST",
      headers: { Authorization: `Bearer ${session.access_token}` },
    });
  } catch {
    /* el token caduca solo; no hay nada que reintentar */
  }
}

export async function refreshSession(): Promise<Session | null> {
  const session = getSession();
  if (!session?.refresh_token) return null;
  try {
    const body = await gotrue("/token?grant_type=refresh_token", {
      method: "POST",
      body: JSON.stringify({ refresh_token: session.refresh_token }),
    });
    return persistSession(toSession(body));
  } catch {
    clearSession();
    return null;
  }
}

/** Token vigente, renovándolo si está a punto de caducar. */
export async function getAccessToken(): Promise<string | null> {
  const session = getSession();
  if (!session) return null;
  if (!isTokenExpired(session.access_token, REFRESH_MARGIN)) {
    return session.access_token;
  }
  const renewed = await refreshSession();
  return renewed?.access_token ?? null;
}

export async function requestPasswordReset(email: string): Promise<void> {
  await gotrue("/recover", {
    method: "POST",
    body: JSON.stringify({ email: email.trim().toLowerCase() }),
  });
}

/* ── Llamadas autenticadas al backend ───────────────────────────────────── */

/**
 * fetch contra la API de Prospecta con el token puesto.
 *
 * Un 401 aquí significa que la sesión ya no vale: se limpia en el sitio, para
 * que la siguiente navegación acabe en el login en lugar de en una pantalla
 * a medio cargar.
 */
export async function authFetch<T>(
  path: string,
  init: RequestInit = {}
): Promise<T> {
  const token = await getAccessToken();
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init.headers ?? {}),
    },
  });

  if (res.status === 401) {
    clearSession();
    throw new AuthError("Tu sesión ha caducado. Vuelve a entrar.", 401);
  }

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* el cuerpo no era JSON */
    }
    throw new AuthError(detail, res.status);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export function fetchProfile(): Promise<Profile> {
  return authFetch<Profile>("/api/auth/me");
}

export function updateProfile(changes: {
  full_name?: string;
  institution?: string;
}): Promise<Profile> {
  return authFetch<Profile>("/api/auth/me", {
    method: "PATCH",
    body: JSON.stringify(changes),
  });
}

export function fetchPermissions(): Promise<Permissions> {
  return authFetch<Permissions>("/api/auth/me/permissions");
}

/* ── Qué rol hace falta para cada ruta ──────────────────────────────────── */

export interface RouteRule {
  prefix: string;
  roles: Role[];
}

/**
 * Tabla única de rutas protegidas por rol. La leen el middleware y la
 * interfaz, así que una ruta nueva se declara aquí y en ningún otro sitio.
 *
 * Lo que NO está aquí sólo exige sesión iniciada. Y lo que decide de verdad
 * quién ve los datos de un estudio es la RLS de la migración 002: esta tabla
 * evita la pantalla inútil, no el acceso.
 */
export const ROUTE_RULES: RouteRule[] = [
  { prefix: "/admin", roles: ["admin"] },
  { prefix: "/delphi/estudios", roles: ["admin", "profesor"] },
  { prefix: "/delphi/nuevo", roles: ["admin", "profesor"] },
  { prefix: "/workshop/sessions", roles: ["admin", "profesor"] },
];

/** Rutas que se sirven sin sesión. */
export const PUBLIC_PATHS: string[] = [LOGIN_PATH, "/auth/callback"];

export function isPublicPath(pathname: string): boolean {
  return PUBLIC_PATHS.some(
    (p) => pathname === p || pathname.startsWith(`${p}/`)
  );
}

export function ruleFor(pathname: string): RouteRule | null {
  // La coincidencia más larga gana: /delphi/estudios/x manda sobre /delphi.
  let best: RouteRule | null = null;
  for (const rule of ROUTE_RULES) {
    const matches =
      pathname === rule.prefix || pathname.startsWith(`${rule.prefix}/`);
    if (matches && (!best || rule.prefix.length > best.prefix.length)) {
      best = rule;
    }
  }
  return best;
}

export function canAccess(role: Role | null, pathname: string): boolean {
  if (isPublicPath(pathname)) return true;
  if (!role) return false;
  const rule = ruleFor(pathname);
  return rule ? rule.roles.includes(role) : true;
}

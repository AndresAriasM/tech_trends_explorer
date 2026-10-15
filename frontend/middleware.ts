/**
 * frontend/middleware.ts
 *
 * Protección de rutas en el borde: decide qué PANTALLA se muestra.
 *
 * Lo que este archivo NO es: un control de acceso a los datos. Lee la cookie
 * de sesión y el payload del JWT sin verificar la firma, porque verificarla
 * exigiría la clave pública en cada navegación y no compraría nada — alguien
 * que se fabrique una cookie con `"user_role":"admin"` llegará a la pantalla
 * de administración y no verá en ella ni una fila, porque el backend valida
 * la firma (core/auth.py) y la base de datos aplica la RLS (migración 002).
 *
 * Su trabajo es el otro: que nadie aterrice en una pantalla vacía o en un
 * bucle de 401, y que al volver tras iniciar sesión se siga donde se estaba.
 */

import { NextResponse, type NextRequest } from "next/server";

import {
  LOGIN_PATH,
  SESSION_COOKIE,
  canAccess,
  isPublicPath,
  isTokenExpired,
  parseSessionCookie,
  roleFromToken,
} from "@/lib/auth";

export function middleware(request: NextRequest) {
  const { pathname, search } = request.nextUrl;

  const session = parseSessionCookie(
    request.cookies.get(SESSION_COOKIE)?.value
  );
  const signedIn = !!session && !isTokenExpired(session.access_token);

  /* ── Ya dentro y volviendo al login: seguir donde tocaba ── */
  if (pathname === LOGIN_PATH) {
    if (!signedIn) return NextResponse.next();
    const next = request.nextUrl.searchParams.get("next");
    const target = request.nextUrl.clone();
    target.search = "";
    target.pathname = next && next.startsWith("/") ? next : "/";
    return NextResponse.redirect(target);
  }

  if (isPublicPath(pathname)) return NextResponse.next();

  /* ── Sin sesión: al login, recordando a dónde iba ── */
  if (!signedIn) {
    const login = request.nextUrl.clone();
    login.pathname = LOGIN_PATH;
    login.search = "";
    // Sólo rutas internas: un `next` absoluto sería un redirector abierto.
    if (pathname !== "/") {
      login.searchParams.set("next", `${pathname}${search}`);
    }
    const response = NextResponse.redirect(login);
    // Una cookie caducada o corrupta se limpia aquí para no repetir el viaje.
    if (session) response.cookies.delete(SESSION_COOKIE);
    return response;
  }

  /* ── Con sesión pero sin el rol que la ruta pide ── */
  const role = roleFromToken(session!.access_token);
  if (!canAccess(role, pathname)) {
    const home = request.nextUrl.clone();
    home.pathname = "/";
    home.search = "";
    // La pantalla de inicio puede explicar por qué se volvió aquí.
    home.searchParams.set("sin-permiso", pathname);
    return NextResponse.redirect(home);
  }

  return NextResponse.next();
}

export const config = {
  /**
   * Todo salvo los recursos que no son pantallas. `api` queda fuera porque
   * el backend de Prospecta es FastAPI y valida sus propios tokens; si algún
   * día hay route handlers en Next, tendrán que autenticarse ellos.
   */
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:png|jpg|jpeg|gif|svg|webp|ico|txt|xml|webmanifest)$).*)",
  ],
};

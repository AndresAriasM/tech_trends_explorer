# Tech Trends Explorer — Migration Roadmap
## Streamlit → Next.js + FastAPI

**Rama de migración:** `feature/nextjs-migration`
**Rama producción:** `master`
**Inicio:** Marzo 2026
**Objetivo:** Frontend moderno con backend desacoplado, priorizando Hype Cycle de Gartner.

---

## Contexto

La versión `master` es una aplicación Streamlit funcional y validada en producción con tres módulos principales:

| Módulo | Estado en master | Prioridad migración |
|--------|-----------------|---------------------|
| **Hype Cycle (Gartner)** | Completo y en uso | **PRIORIDAD 1** |
| S-Curve (Curvas S) | En desarrollo | Fase 3 |
| Búsqueda de Tendencias | En desarrollo | Fase 3 |

El módulo Hype Cycle es el núcleo de valor: permite analizar la posición de una tecnología en el ciclo de Gartner usando SerpAPI, almacena los resultados en AWS DynamoDB y permite reutilizar consultas previas.

---

## Estructura de la migración

```
tech_trends_explorer/
├── src/                  ← Código Streamlit original (master, no tocar)
├── pages/                ← Páginas Streamlit originales (master, no tocar)
├── frontend/             ← [NUEVO] Aplicación Next.js
├── backend/              ← [NUEVO] API FastAPI + Workers Celery
├── data/                 ← Compartido (JSON local, sin cambios)
├── docs/
│   ├── MIGRATION_ROADMAP.md  ← Este archivo
│   ├── architecture.svg
│   └── sessions/             ← [NUEVO] Notas por sesión de trabajo
└── requirements.txt      ← Solo para Streamlit (master)
```

---

## Fases del plan

---

### FASE 1 — Scaffolding y Frontend Hype Cycle (Prioridad actual)

**Objetivo:** Tener un frontend Next.js funcional que consuma el backend y ejecute análisis Hype Cycle completos. La app Streamlit sigue funcionando en paralelo.

**Entregables:**
- [ ] Estructura de proyecto Next.js en `/frontend`
- [ ] Estructura de proyecto FastAPI en `/backend`
- [ ] Pantalla principal con Query Builder de Hype Cycle
- [ ] Visualización del Hype Cycle (gráfica interactiva)
- [ ] Historial de consultas guardadas (lectura desde DynamoDB)
- [ ] Análisis en tiempo real con barra de progreso (SSE)
- [ ] Gestión de categorías (CRUD básico)
- [ ] Exportación de resultados

**Stack:**
```
Frontend:  Next.js 14 (App Router) + TypeScript + Tailwind CSS
Charts:    Plotly.js (mismas gráficas del master)
State:     Zustand + TanStack Query
Real-time: Server-Sent Events (SSE)

Backend:   FastAPI (Python 3.11)
Tasks:     Celery + Redis
Storage:   AWS DynamoDB (mismo) + Redis cache
Auth:      API Key simple (primer paso)
```

**Archivos clave del master a mapear:**

| Master (`src/`) | Nueva ubicación | Notas |
|-----------------|-----------------|-------|
| `analysis.py` (2199 líneas) | `backend/core/analysis.py` | Sin imports streamlit |
| `hype_cycle.py` | `backend/core/hype_cycle.py` | Sin imports streamlit |
| `hype_cycle_storage.py` | `backend/storage/hype_cycle.py` | Sin cambios de lógica |
| `hype_cycle_positioning.py` | `backend/core/positioning.py` | Sin cambios |
| `data_storage.py` | `backend/storage/dynamodb.py` | Sin cambios |
| `category_admin.py` | `backend/core/categories.py` | Solo lógica, sin UI |

---

### FASE 2 — Backend desacoplado y Auth

**Objetivo:** Backend robusto con tareas async, caché y autenticación básica por usuario.

**Entregables:**
- [ ] Autenticación JWT (login por equipo UPB)
- [ ] Rate limiting por usuario
- [ ] Caché Redis para resultados (evitar llamadas duplicadas a SerpAPI)
- [ ] Celery workers para análisis en background
- [ ] Endpoint de progreso en tiempo real (SSE completo)
- [ ] Logs y monitoreo básico

---

### FASE 3 — Migración S-Curve y Búsqueda de Tendencias

**Objetivo:** Completar la migración de los módulos en desarrollo.

**Entregables:**
- [ ] Frontend S-Curve (Scopus API)
- [ ] Frontend Búsqueda de Tendencias (Google Custom Search)
- [ ] Dashboard unificado con métricas
- [ ] Exportación Excel/PDF (replazar lógica de `s_curve.py`)

---

### FASE 4 — Deploy y cierre Streamlit

**Objetivo:** Producción estable, apagar Streamlit.

**Entregables:**
- [ ] Dockerizar frontend + backend + workers
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Variables de entorno en secrets manager
- [ ] Documentación de usuario final
- [ ] Migración de datos históricos de DynamoDB (si aplica)
- [ ] Apagar Streamlit

---

## Arquitectura objetivo (Fase 1-2)

```
┌─────────────────────────────────────────────┐
│          Next.js Frontend                   │
│                                             │
│  /hype-cycle   → Query Builder              │
│                  Visualización Gartner      │
│                  Progreso en tiempo real    │
│                                             │
│  /history      → Historial DynamoDB         │
│  /categories   → Gestión categorías         │
└────────────────┬────────────────────────────┘
                 │ HTTP REST + SSE
┌────────────────▼────────────────────────────┐
│           FastAPI Backend                   │
│                                             │
│  POST /api/hype-cycle/analyze               │
│  GET  /api/hype-cycle/history               │
│  GET  /api/tasks/{id}/stream  (SSE)         │
│  GET  /api/categories                       │
│  POST /api/categories                       │
└──────┬──────────────┬───────────────────────┘
       │              │
  ┌────▼────┐   ┌─────▼──────────────────────┐
  │  Redis  │   │     Celery Worker           │
  │  Cache  │   │                             │
  │  Queue  │   │  → SerpAPI calls            │
  └─────────┘   │  → NewsAnalyzer (analysis)  │
                │  → HC positioning           │
                │  → DynamoDB save            │
                └─────────────┬───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   AWS DynamoDB     │
                    │  (sin cambios)     │
                    │                   │
                    │  tech-trends-*    │
                    └───────────────────┘
```

---

## Comparación de flujos master vs nuevo

### Flujo de análisis Hype Cycle

**Master (Streamlit):**
```
1. Usuario configura términos en sidebar
2. Click "Analizar"
3. UI se congela
4. Python llama SerpAPI (~10-30s)
5. Análisis NewsAnalyzer (~5-10s)
6. Render del resultado completo
7. Opcional: guardar en DynamoDB
```

**Nuevo (Next.js + FastAPI):**
```
1. Usuario configura términos en Query Builder
2. Click "Analizar" → POST /api/hype-cycle/analyze → task_id (< 200ms)
3. Frontend abre SSE /api/tasks/{task_id}/stream
4. Barra de progreso: "Buscando en web... 20%"
5. Celery Worker ejecuta SerpAPI + análisis async
6. Progreso: "Analizando resultados... 60%"
7. DynamoDB save automático
8. Progreso: "Completado 100%" → resultado en pantalla
9. UI nunca se congela, usuario puede navegar
```

---

## Variables de entorno

Las mismas credenciales del master se usan en el backend. Ver `.env` en raíz del proyecto.

Para el backend nuevo, crear `/backend/.env`:
```env
# Copiar del .env raíz
SERP_API_KEY=
GOOGLE_API_KEY=
SEARCH_ENGINE_ID=
SCOPUS_API_KEY=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-2

# Nuevos para el backend
REDIS_URL=redis://localhost:6379
SECRET_KEY=<generar_uno_nuevo>
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1
```

Para el frontend, crear `/frontend/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Cómo retomar el trabajo en futuras sesiones

### Contexto rápido
- **Rama activa:** `feature/nextjs-migration`
- **Master:** versión Streamlit estable, nunca modificar en esta rama
- **Referencia de lógica:** siempre leer `src/` del master para entender qué replicar
- **Datos:** DynamoDB compartido entre master y nueva versión durante la migración

### Comandos de referencia
```bash
# Ver diferencias con master
git diff master..feature/nextjs-migration --stat

# Correr Streamlit (master, para referencia)
streamlit run src/main.py

# Correr backend FastAPI (cuando esté creado)
cd backend && uvicorn main:app --reload

# Correr frontend Next.js (cuando esté creado)
cd frontend && npm run dev

# Correr Redis (prerequisito para workers)
redis-server

# Correr Celery worker (cuando esté creado)
cd backend && celery -A tasks worker --loglevel=info
```

### Checkpoint de sesiones
Ver carpeta `docs/sessions/` para notas de cada sesión de trabajo.

---

## Estado actual

| Componente | Estado | Notas |
|------------|--------|-------|
| Rama `feature/nextjs-migration` | ✅ Creada | Base: commit `8f21ab9` |
| Frontend Next.js scaffold | ⏳ Pendiente | Fase 1 |
| Backend FastAPI scaffold | ⏳ Pendiente | Fase 1 |
| Core HC migrado (sin Streamlit) | ⏳ Pendiente | Fase 1 |
| Query Builder UI | ⏳ Pendiente | Fase 1 |
| Visualización Gartner | ⏳ Pendiente | Fase 1 |
| Historial DynamoDB | ⏳ Pendiente | Fase 1 |
| Celery async workers | ⏳ Pendiente | Fase 2 |
| S-Curve frontend | ⏳ Pendiente | Fase 3 |
| Trends Search frontend | ⏳ Pendiente | Fase 3 |
| Deploy Docker | ⏳ Pendiente | Fase 4 |

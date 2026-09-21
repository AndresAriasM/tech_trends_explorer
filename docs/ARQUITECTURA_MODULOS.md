# Arquitectura de módulos — Plataforma de prospectiva

**Rama:** `feature/nextjs-migration`
**Estado:** definición inicial, septiembre 2026
**Relacionado:** [MIGRATION_ROADMAP.md](MIGRATION_ROADMAP.md)

---

## El cambio de alcance

Hasta ahora el proyecto era *una app de Hype Cycle con extras*. Pasa a ser una
**plataforma de prospectiva**, donde el Hype Cycle es uno de varios instrumentos.

Eso obliga a resolver un problema que hoy no está resuelto: **cada módulo es una
isla**. `HypeCycleQuery`, `SCurveResult` y `WorkshopSession` no comparten ninguna
entidad. No hay forma de afirmar algo como:

> *La computación cuántica está en el Pico de Expectativas según Hype Cycle, en
> fase emergente según su curva S, y el panel la valora 3.2/4 al horizonte 2032.*

Tres calculadoras sueltas no son una app de prospectiva. Lo que las convierte en
una es un **modelo de dominio compartido**.

---

## Conceptos esenciales

Cuatro entidades atraviesan todos los módulos.

### 1. Estudio (`Study`)

El contenedor de un ejercicio prospectivo. Todo módulo se ejecuta *dentro* de un
estudio, nunca suelto.

| Campo | Notas |
|---|---|
| `focal_question` | La pregunta que el ejercicio responde |
| `horizon_year` | Hoy está **hardcodeado a 2032** en el workshop; pasa a ser propiedad del estudio |
| `variables[]` | Las variables bajo estudio |
| `status` | `draft` / `active` / `closed` |

Es la pieza que hoy **no existe** y sin la cual los módulos no pueden conversar.

### 2. Variable (`Variable`)

La unidad de análisis. Hoy hay dos conceptos solapados que se unifican aquí:
`TechnologyEntry` (dentro de `Category`) y `WorkshopVariable`.

Una variable puede ser una tecnología, un driver de cambio o un factor del
entorno. Lo que la define no es su naturaleza sino que **sobre ella se acumulan
evidencias y juicios**.

### 3. Evidencia (`Evidence`)

Lo que produce un módulo **analítico** a partir de datos, sin intervención de
opinión:

- una posición en el Hype Cycle (fase + confianza),
- un ajuste sigmoidal (parámetros + R²),
- un volumen de menciones con su sentimiento.

Campos comunes: `variable_id`, `module_id`, `source`, `computed_at`, `payload`
(específico de cada módulo).

### 4. Juicio (`Judgment`)

Lo que aporta una **persona o la IA**: un voto Delphi, un rating de taller, una
valoración de impacto o probabilidad.

Campos comunes: `variable_id`, `author` (humano o IA), `round`, `value`,
`confidence`, `rationale`.

La distinción evidencia/juicio es deliberada: en prospectiva **no se mezclan**.
Un dato de Scopus y la opinión de un experto son insumos de naturaleza distinta,
y la síntesis debe poder mostrarlos por separado y medir su divergencia — algo
que el workshop actual ya hace bien con `ai_divergence_current`.

### Síntesis

Cruzar evidencia y juicio sobre las mismas variables. Ahí viven MICMAC,
escenarios y roadmapping: no generan datos propios, **combinan** lo que los
demás módulos produjeron.

```
Estudio
  └── Variables
        ├── Evidencia   ← módulos analíticos  (Hype Cycle, Curvas S, Vigilancia)
        └── Juicio      ← módulos participativos (Delphi, Taller)
              ↓
          Síntesis      ← MICMAC, Escenarios, Roadmap
```

---

## Los módulos

Declarados en [`frontend/lib/modules.ts`](../frontend/lib/modules.ts), que es la
fuente única para navegación y estados de madurez.

| Módulo | Rol | Estado | Pregunta que responde |
|---|---|---|---|
| **Hype Cycle** | evidencia | activo | ¿En qué punto del ciclo de expectativas está? |
| **Curvas en S** | evidencia | activo | ¿En qué fase de adopción está? |
| **Vigilancia** | evidencia | por construir | ¿Qué señales nuevas aparecen? |
| **Delphi** | juicio | por construir | ¿Qué opina el panel, y converge entre rondas? |
| **Taller LACCEI** | juicio | activo | Instrumento puntual del evento |
| **MICMAC** | síntesis | por construir | ¿Qué variables mueven el sistema? |
| **Escenarios** | síntesis | por construir | ¿Qué futuros son plausibles? |
| **Roadmapping** | síntesis | por construir | ¿Qué hitos llevan al futuro elegido? |
| **Asistente IA** | transversal | por construir | — |

### Delphi y Taller LACCEI son módulos distintos

Decisión tomada: el Taller LACCEI se queda como está — un instrumento de ronda
única, con sus variables fijas y su recolección por Google Forms, atado a ese
evento. **Delphi se construye nuevo**, con lo que el taller no tiene y que es
justamente lo que define al método:

- rondas múltiples (`round_number`),
- devolución anónima de resultados entre rondas,
- re-votación con conocimiento del grupo,
- criterio de parada por estabilidad del consenso.

> La mecánica concreta está pendiente de definición. No se implementa nada de
> Delphi hasta tenerla.

### El asistente de IA es una capa, no una sección

No es una pantalla de chat aparte. Es una capacidad que cada módulo invoca sobre
el modelo de dominio: interpretar una posición en el ciclo, redactar la
devolución de una ronda Delphi, proponer ejes de escenarios.

Ya existe precedente funcionando: [`backend/core/ai_voter.py`](../backend/core/ai_voter.py)
genera juicios sintéticos con evidencia de SerpAPI, y los marca con
`is_ai_vote=True` para que nunca se confundan con un experto humano. Esa
convención —**la IA aporta juicios trazables y distinguibles**— se mantiene en
toda la plataforma.

---

## Persistencia

Supabase (Postgres). La capa de storage actual apunta a DynamoDB y falla sin
credenciales; se migra detrás de una interfaz de repositorio para que el cambio
no se propague a los servicios.

---

## Pendiente de definir

1. **Mecánica de las rondas Delphi** — bloquea el módulo completo.
2. **Método de las curvas en S** — hay implementación en `core/s_curve.py`, pero
   el método a usar está por confirmar.
3. **Esquema SQL en Supabase** para las cuatro entidades.
4. **Proveedor del asistente** — hoy `ai_voter.py` usa GPT-4o. Propuesta: una
   interfaz de proveedor con Claude por defecto, conservando el código OpenAI.

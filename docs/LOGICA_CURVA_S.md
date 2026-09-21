# Lógica del módulo de Curvas en S

**Propósito:** documentar, con precisión suficiente para reimplementarla, la lógica
que hoy vive repartida entre tres generaciones de código. Escrito para quien esté
adaptando el módulo dentro de la plataforma de prospectiva.

**Origen:** `~/Downloads/Scopus+S_Curve.ipynb` (ejecutado el 2025-09-11)
**Relacionado:** [ARQUITECTURA_MODULOS.md](ARQUITECTURA_MODULOS.md)

---

## 1. Linaje

| Gen | Dónde | Qué aportó | Qué perdió |
|---|---|---|---|
| **G1 — Notebook** | `Scopus+S_Curve.ipynb`, 3 celdas | Pipeline completo: búsqueda en dos bloques, acumulado, ajuste sigmoidal con bounds, errores estándar y valores t, RMSE, residuos, derivada analítica, proyecciones, fases | — |
| **G2 — Streamlit** | [src/s_curve.py](../src/s_curve.py) (2209 líneas) | Escalera de intervalos (sustituye los dos bloques), relleno de años faltantes, patentes por Excel y SerpAPI, Plotly | Derivada, RMSE, residuos, proyecciones tabuladas |
| **G3 — FastAPI + Next** | [backend/core/s_curve.py](../backend/core/s_curve.py) (365 líneas) | Async, SSE, contrato JSON limpio, 5 fases con color | **Errores estándar y valores t**, escalera de intervalos, y todo lo que ya faltaba en G2 |

La forma del notebook —tres celdas: *buscar → visualizar → ajustar*— es exactamente
la forma del módulo actual: `analyze_s_curve` (fetch) → `SCurveChart` (render) →
`fit_sigmoid` + `detect_phase` (ajuste). No hay que redescubrir la arquitectura;
hay que reponer lo que se cayó en el camino.

---

## 2. Celda 1 — Adquisición: la estrategia de dos bloques

```python
def search_scopus_two_blocks(query_string, split_year=2018)
```

Scopus no pagina más allá de ~5000 resultados por consulta. La solución original
es **partir la consulta en dos por un año de corte**:

```
Bloque 1:  ({query}) AND PUBYEAR BEF {split_year}      # ≤ split_year - 1
Bloque 2:  ({query}) AND PUBYEAR AFT {split_year - 1}  # ≥ split_year
```

`BEF`/`AFT` son exclusivos, de ahí el `-1` en el bloque 2. Sin solape y sin hueco.
Son operadores más limpios que el `PUBYEAR > a-1 AND PUBYEAR < b+1` que usa hoy el
backend ([backend:241](../backend/core/s_curve.py#L241)) — vale la pena adoptarlos.

Detalles que sí importan al portar:

- **Sondeo antes de paginar.** `count=1` para leer `opensearch:totalResults` y saber
  cuánto pedir. Se hace tres veces: consulta base + una por bloque.
- **Paginación** `count=25`, `start += 25`, `sort=coverDate`, `time.sleep(0.5)`.
- **Pausa de 3 s entre bloques.**
- **Tope de Scopus detectado por `400` + `len(all_results) >= 5000`** → corta el
  bloque y sigue. Es el reconocimiento explícito del límite que G3 ignora.
- **`429` → espera 3 s y reintenta.**
- Si la consulta base falla, cae a una simplificada hardcodeada
  (`TITLE-ABS-KEY(grinding) AND TITLE-ABS-KEY(food)`) — un vestigio de depuración
  que **no hay que portar**: enmascara el error real analizando otra cosa.

> **Bug a no heredar:** en los ramos de `429` y de error genérico, `params["start"]`
> no se incrementa y no hay contador de intentos. Un rate limit persistente deja el
> bucle girando indefinidamente. Al portar: máximo de reintentos por página y
> backoff exponencial (G3 ya lo hace bien en
> [`_fetch_page`](../backend/core/s_curve.py#L183), con `2**attempt` y 3 intentos).

**Dos bloques dan techo de ~10 000.** G2 lo generalizó a una escalera de ventanas
(5 años hasta 1999, 2 años hasta 2009, 1 año desde 2010,
[src:190](../src/s_curve.py#L190)), que escala mejor y es la que conviene portar.
Pero el `split_year` fijo del notebook tiene una virtud que la escalera perdió: es
**adaptable al dato**. Lo correcto es partir recursivamente donde haga falta, no con
una tabla de épocas fija.

### Extracción del año

Cascada de tres intentos, idéntica en las tres generaciones:

1. `prism:coverDate` → `datetime.strptime(s, "%Y-%m-%d").year`
2. `prism:publicationDate` → igual
3. `prism:coverDisplayDate` → regex `\b(19|20)\d{2}\b`

Sin año, la entrada se descarta en silencio. El notebook al menos reporta cuántas
sobrevivieron (`Procesados: 24 resultados con año válido`); **G3 no lo reporta**.
Conviene exponer `entries_sin_fecha` en el resultado.

### El acumulado

```python
df_results['Acumulado'] = df_results['Publicaciones'].cumsum()
```

La curva S *es* el acumulado. Todo lo demás se ajusta sobre esta columna.

> **Divergencia deliberada y correcta del repo:** el notebook **no rellena los años
> sin publicaciones**. Su tabla salta de 1983 a 2006 y de 2008 a 2010, y el ajuste
> corre sobre ese eje X no uniforme —15 puntos para un intervalo de 40 años—, lo
> que comprime el hueco y distorsiona `k`. G2 y G3 sí rellenan con 0
> ([src:446](../src/s_curve.py#L446), [backend:138](../backend/core/s_curve.py#L138)).
> **Mantener el relleno.** Es una mejora sobre el original, no una desviación.

---

## 3. Celda 2 — Visualización

Cuatro gráficos, de los cuales el módulo actual conserva uno y medio:

| # | Gráfico | ¿En el repo? |
|---|---|---|
| 1 | Barras de publicaciones anuales | Sí, como serie `bar` en [SCurveChart](../frontend/components/charts/SCurveChart.tsx) |
| 2 | Línea del acumulado (Curva S) | Sí, eje secundario |
| 3 | **Combinado** barras + acumulado en doble eje | Sí — es literalmente el gráfico actual |
| 4 | **Tasa de crecimiento anual (%)** `pct_change()*100` | **No** |

El (4) se calcula sobre `Publicaciones`, no sobre `Acumulado` — es la variación
interanual del flujo, no del stock. G2 lo calculó pero sobre el acumulado
([src:655](../src/s_curve.py#L655)), que es otra cosa y menos informativa. G3 no lo
calcula. Es barato de reponer y dice algo que la curva S no dice.

La celda cierra creando `df_para_inflexion = df_scopus.set_index('Año')`, que es el
único contrato entre celda 2 y celda 3: **un índice de años y una columna
`Acumulado`**.

---

## 4. Celda 3 — El ajuste sigmoidal

Esta es la celda que importa.

### 4.1 La función

```python
def sigmoid(x, L, x0, k):
    return L / (1 + np.exp(-k * (x - x0)))
```

- **L** — asíntota superior, el techo de saturación
- **x0** — año del punto de inflexión
- **k** — tasa de crecimiento

`x` es el año calendario en crudo (1983, 2006, …), no un índice normalizado. Se
mantiene igual en las tres generaciones.

### 4.2 El ajuste: las tres versiones divergen

| | G1 Notebook | G2 ([src:689](../src/s_curve.py#L689)) | G3 ([backend:26](../backend/core/s_curve.py#L26)) |
|---|---|---|---|
| `p0` L | `max(y) * 1.1` | `max(y)` | `max(y) * 1.3` |
| `p0` x0 | `np.median(x)` | `np.median(x)` | `np.median(x)` |
| `p0` k | `0.2` | `0.1` | `0.3` |
| `maxfev` | 5 000 | 5 000 | 20 000 |
| bounds L | `[0, inf]` | **ninguno** | `[0, 5·max(y)]` |
| bounds x0 | `[min(x), max(x)]` | **ninguno** | `[min(x)−20, max(x)+20]` |
| bounds k | `[0, inf]` | **ninguno** | `[0.001, 5]` |
| Mínimo de puntos | no valida | 4 | 4 |
| Devuelve `pcov` | **sí** | sí | **no** |

Notas sobre las diferencias que sí cambian el resultado:

- **G2 sin bounds es la peor de las tres.** Puede converger a `k` negativo (curva
  invertida) o a `L` de millones. Descartada.
- **El bound de x0 es una decisión de modelo, no de numérica.** El notebook lo
  encierra en el rango observado: *el punto de inflexión ya ocurrió*. G3 permite
  ±20 años: *puede estar en el futuro*. Para prospectiva la de G3 es la correcta —
  una tecnología emergente con inflexión proyectada a 2031 es un hallazgo, no un
  error de ajuste. Pero entonces `x0` fuera del rango de datos es **extrapolación** y
  debe marcarse como tal en la UI.
- **`L` sin cota superior** (notebook) diverge en series pre-inflexión. La cota de
  G3 (`5·max`) es arbitraria pero razonable.

### 4.3 Bondad de ajuste — el notebook reporta dos métricas, el repo una

```python
r2   = r2_score(y_data, y_pred)                     # sklearn
rmse = np.sqrt(np.mean((y_data - y_pred)**2))
```

G2 y G3 calculan R² a mano y **no calculan RMSE**. Reponerlo, porque:

> R² sobre una serie **acumulada** es engañosamente alto. Una recta le saca 0.98 a
> casi cualquier acumulado monótono. En la corrida real del notebook, R² = 0.9931
> sobre 24 publicaciones — "Excelente" según su propia escala. El RMSE (0.66
> publicaciones) dice mucho más sobre el error real.

La escala interpretativa del notebook (`>0.95 Excelente / >0.90 Bueno / >0.80
Aceptable / resto Pobre`) no debería portarse tal cual por lo mismo: con acumulados,
0.95 no es excelente, es el piso.

El notebook también grafica **residuos vs año** (`axes[0,1]`). En una serie
acumulada mal ajustada los residuos muestran estructura sistemática que el R² oculta.
No está en el repo y vale la pena.

### 4.4 Errores estándar y valores t — la pérdida más seria

```python
errors   = np.sqrt(np.diag(pcov))
t_values = popt / errors
significativo = ['Sí' if abs(t) > 2 else 'No' for t in t_values]
```

**G3 descarta `pcov`** (`popt, _ = curve_fit(...)`, [backend:44](../backend/core/s_curve.py#L44)).
Es una regresión contra el notebook *y* contra G2, que sí lo calculaba
([src:705](../src/s_curve.py#L705)).

Por qué importa:

> En una tecnología que aún no cruzó su inflexión, **L no es identificable**. El
> ajuste converge, da R² de 0.99, y el error estándar de L es del mismo orden que L.
> Y de `L` sale `saturation_pct`, que es lo único que decide la fase que muestra la
> UI. Sin banda de error, la plataforma presenta un techo de saturación inventado
> con apariencia de dato duro.

**Al adaptar: devolver `L_stderr`, `x0_stderr`, `k_stderr` en `sigmoid_params`.** Si
`|t_L| < 2`, la fase se presenta como no concluyente.

> **Advertencia sobre el valor t de x0 — no copiar la tabla del notebook tal cual.**
> En la corrida real, `x0 = 2013.67`, error estándar 0.36, **t = 5660**. Ese número
> no significa nada: el valor t contrasta contra la hipótesis nula "el parámetro
> vale 0", y para un año calendario eso es "la inflexión ocurrió en el año 0".
> Siempre va a dar significativo. Sólo **L y k** tienen contraste contra cero con
> sentido. Para `x0` lo informativo es el intervalo de confianza
> (`x0 ± 1.96·stderr` → 2013.0–2014.4), no el valor t.

### 4.5 La derivada analítica — perdida y fácil de recuperar

```python
derivada = lambda x: (k * L * np.exp(-k*(x - x0))) / (1 + np.exp(-k*(x - x0)))**2
```

Es la velocidad de adopción, en forma cerrada. Su máximo está exactamente en `x0`
y vale `k·L/4`. El notebook la grafica (`axes[1,1]`) y marca el pico.

No existe en G2 ni en G3, y es **la salida más valiosa para prospectiva** que se
perdió: responde "¿cuándo se adopta más rápido esto?", que es una pregunta distinta
de "¿cuánto se ha adoptado?". Cuesta cuatro líneas y no requiere datos nuevos.
Reponerla como serie `growth_curve` junto a `sigmoid_curve`.

### 4.6 Proyecciones

```python
for año in [ultimo_año + i for i in range(1, 6)]:
    proyeccion = sigmoid(año, *popt)
```

Cinco años tabulados. G3 extrapola 10 años dentro de `sigmoid_curve`
([backend:146](../backend/core/s_curve.py#L146)) pero no los tabula ni los distingue
visualmente del tramo ajustado. Deberían dibujarse punteados: es predicción, no ajuste.

---

## 5. Clasificación de fases — cuatro esquemas incompatibles

Este es el punto donde más divergieron las generaciones. Hay **cuatro** criterios y
ninguno coincide con otro.

### G1a — posición temporal frente a x0 (texto del notebook)

| Condición | Fase |
|---|---|
| `último_año < x0` | CRECIMIENTO INICIAL |
| `último_año > x0` y saturación < 90 % | CRECIMIENTO MADURO |
| resto | MADUREZ |

### G1b — bandas de saturación (gráfico del notebook, mismas celdas)

| Rango de L | Fase |
|---|---|
| 0–30 % | Fase 1: Crecimiento Inicial |
| 30–80 % | Fase 2: Crecimiento Acelerado |
| > 80 % | Fase 3: Madurez |

El notebook usa **los dos a la vez** y pueden contradecirse: con 93.6 % de
saturación y 2022 > 2013.7, ambos dan "madurez", pero una serie al 85 % con el
último año anterior a x0 daría "CRECIMIENTO INICIAL" en el texto y "Madurez" en el
gráfico.

### G2 — signo de la segunda derivada ([src:716](../src/s_curve.py#L716))

`np.gradient(np.gradient(acumulado))` promediado en los últimos 3 años; > 0 fase
inicial, < 0 madurez, = 0 inflexión. **Es ruido** — la segunda derivada numérica de
un conteo anual cambia de signo constantemente. Descartar, junto con su búsqueda de
puntos de inflexión donde la segunda derivada vale 0
([src:670](../src/s_curve.py#L670)): `x0` ya lo da, y bien.

### G3 — porcentaje de saturación ([backend:65](../backend/core/s_curve.py#L65))

| Rango | Fase | Color |
|---|---|---|
| < 10 % | Emergente | `#6366f1` |
| 10–40 % | Crecimiento | `#3b82f6` |
| 40–65 % | Aceleración | `#10b981` |
| 65–85 % | Maduración | `#f59e0b` |
| ≥ 85 % | Saturación | `#ef4444` |

**Conservar el de G3**, que es el más fino, con tres correcciones:

1. Depende enteramente de `L` → sin banda de error es fase inventada (§4.4).
2. Los cortes son arbitrarios. En una logística la inflexión cae **siempre** en el
   50 % de L; que "Aceleración" sea 40–65 % en vez de simétrica alrededor de 50 % es
   una elección de producto, legítima pero que hay que documentar como tal.
3. **Incorporar la señal temporal de G1a**: `último_año` frente a `x0` distingue
   "va hacia la inflexión" de "ya la pasó", que el porcentaje solo no captura
   cuando `L` es incierto. Son dos ejes, no uno.

El mapeo de fase a icono vive además en el frontend
([page.tsx:31](../frontend/app/s-curve/page.tsx#L31)). Cuatro definiciones de fase
en cuatro archivos es exactamente la isla que
[ARQUITECTURA_MODULOS.md](ARQUITECTURA_MODULOS.md) quiere eliminar: la fase debe ser
una `Evidence` sobre una `Variable`, con una definición única compartida.

---

## 6. Fuentes de patentes

El notebook de origen es sólo publicaciones. Las patentes llegaron después, y por
una razón documentada: en un notebook hermano (`API_Scopus.ipynb`, celda 1) quedó
registrado que **`DOCTYPE(ip)` en Scopus responde 200 y devuelve siempre 0
resultados**. La vía está cerrada y ya se verificó — no reintentarla.

De ahí las dos rutas alternativas:

**CSV exportado a mano (ruta actual).** Parser en
[page.tsx:263](../frontend/app/s-curve/page.tsx#L263). Tres columnas sin encabezado
nombrado (salta la línea 1): `count, year, docType`. Tolera comillas, descarta filas
incompletas o no numéricas. Preselecciona los tipos que casan `/patent/i` pero no
`/amended|design|utility/i`; si ninguno casa, selecciona todos. Suma por año los
tipos elegidos y llama a `POST /api/s-curve/analyze-local`. Reemplazó la plantilla
Excel de G2 ([`PatentDataManager`](../src/s_curve.py#L465)), que puede archivarse.

**Google Patents vía SerpAPI — no portado.**
[`OptimizedGooglePatentsAnalyzer`](../src/s_curve.py#L936) presupuesta `max_tokens`
requests (default 50) y los reparte en tres estrategias por época: `bulk_search`
(≤1999, lotes de 20 años), `decade_search` (2000–2009) y `smart_sampling` (≥2010,
años estratégicos + interpolación lineal, [src:1268](../src/s_curve.py#L1268)).
**Decidir explícitamente si se porta.** Si se porta, los años interpolados deben
venir marcados en la respuesta: hoy entran a la serie indistinguibles de los
medidos y contaminan el ajuste sin dejar rastro.

---

## 7. Contrato de entrada/salida

Estable y bien definido en G3. Es la parte que **no hay que rediseñar**, sólo ampliar.

```
POST /api/s-curve/analyze         → SSE
  { query, api_key, year_from=1980, year_to?, max_results? }

POST /api/s-curve/analyze-local   → síncrono
  { yearly_counts: {año: conteo}, title }
```

Eventos SSE `{status, progress, message}`, `status ∈ running|completed|error`; el
final lleva `result`. Hitos: 5 → 10 → 10-80 (descarga) → 82 → 90 → 96 → 100.

```jsonc
{
  "query": "…",
  "total_publications": 1682,
  "year_from": 1991, "year_to": 2025,
  "yearly_data":    [{ "year": 1991, "annual": 3, "cumulative": 3 }, …],
  "sigmoid_params": { "L": 2431.7, "x0": 2016.4, "k": 0.21, "r2": 0.9962 },
  "sigmoid_curve":  [{ "year": 1991.0, "value": 2.1 }, …],   // 200 puntos, +10 años
  "phase":          { "phase": "Maduración", "color": "#f59e0b",
                      "description": "…", "saturation_pct": 69.2 },
  "inflection_year": 2016, "saturation_level": 2432, "r2": 0.9962,
  "analyzed_at": "2026-09-21T…"
}
```

Tipado en [frontend/types/index.ts:110-158](../frontend/types/index.ts#L110-L158).
Los tres campos finales duplican `sigmoid_params`; son azúcar para la UI y
conservarlos evita romper el frontend.

**Campos a añadir** para reponer lo perdido:

```jsonc
"sigmoid_params": { …, "L_stderr": …, "x0_stderr": …, "k_stderr": …,
                    "L_significant": true, "rmse": 0.66 },
"growth_curve":   [{ "year": …, "rate": … }, …],   // derivada analítica, §4.5
"residuals":      [{ "year": …, "residual": … }, …],
"projections":    [{ "year": 2026, "value": … }, …],
"entries_without_date": 0,
"total_reported": 24000, "total_retrieved": 5000   // ver §8.3
```

---

## 8. Deudas y trampas

Por orden de gravedad.

1. **API key de Scopus filtrada.** La clave (32 hex, visible en la celda 1) está en claro
   en el notebook de origen (celda 1) y también en el hermano `API_Scopus.ipynb`, en
   el código y en las salidas guardadas. **Revocarla en dev.elsevier.com** antes de
   compartir cualquiera de los dos archivos.
2. **La key viaja del navegador al backend en el body de cada request.** Si se
   mantiene por diseño (cada usuario con su cuota), decirlo en la UI y no persistirla.
3. **Truncamiento silencioso >5000 en G3.** El backend pagina con
   `while start < limit` sin tope; al pasar de 5000, `_fetch_page` devuelve `None`,
   el bucle hace `break` y el análisis sigue con datos incompletos **sin avisar**.
   El notebook detecta explícitamente ese caso (`400` + `>= 5000`) y G2 lo evita con
   la escalera. Cualquier consulta amplia produce hoy una curva S falsa que se ve
   bien. **Prioridad número uno.**
4. **`L` sin banda de error** (§4.4). Convierte una incertidumbre en una cifra.
5. **`total_publications` significa cosas distintas.** En G3 es `len(all_entries)`
   = lo descargado; si hubo truncamiento no es el total que reportó Scopus. Exponer
   `total_reported` y `total_retrieved` por separado.
6. **Años hardcodeados en G2**: `1980` como inicio de la escalera y literalmente
   `PUBYEAR = 2024` / `"2025-12-31"` en el cálculo del año en curso por diferencia
   ([src:305-336](../src/s_curve.py#L305-L336)). Derivarlo todo de
   `datetime.now().year`.
7. **`GET /api/s-curve/history` devuelve `[]` fijo**
   ([routes:98](../backend/api/routes/s_curve.py#L98)) y `data/s_curve_db.json` está
   vacío. La persistencia no existe.
8. **`sigmoid_params: null` es un estado alcanzable** (menos de 4 años de datos) con
   `phase` presente. La UI no debe asumir que siempre hay ajuste.
9. **La muestra del notebook es diminuta.** La corrida guardada devolvió 24
   publicaciones porque la ecuación dice `highpressure` en vez de `high pressure`.
   Los números de ejemplo (L=25.6, x0=2013.7, R²=0.9931) ilustran el pipeline, no
   validan nada.

---

## 9. Resumen para la adaptación

**Conservar tal cual**
- La cascada de extracción de año (3 intentos).
- El relleno de años faltantes con 0 antes del `cumsum` — mejora sobre el notebook.
- `fit_sigmoid` de G3 con sus bounds y `p0`.
- El contrato `SCurveResult` y los eventos SSE.
- El parser CSV de patentes y su preselección de tipos.

**Recuperar del notebook**
- Errores estándar y valores t desde `pcov` — con la salvedad del t de `x0` (§4.4).
- RMSE junto al R².
- La derivada analítica como curva de velocidad de adopción (§4.5).
- Los residuos por año.
- Las proyecciones tabuladas, visualmente distinguidas del ajuste.
- La detección explícita del tope de 5000 de Scopus.
- Los operadores `PUBYEAR BEF/AFT`, más limpios que `> a-1 AND < b+1`.

**Recuperar de G2**
- La escalera de intervalos temporales — pero adaptativa, partiendo donde el conteo
  lo exija, no con una tabla de épocas fija.

**Rediseñar**
- La fase como `Evidence` sobre una `Variable`, con una definición única en vez de
  cuatro, y con dos ejes (saturación **y** posición frente a `x0`).
- La tasa de crecimiento anual sobre el conteo, no sobre el acumulado.

**Descartar**
- La detección de inflexión por segunda derivada de G2 (§5).
- El fallback a la consulta simplificada hardcodeada del notebook (§2).
- El reintento sin contador ni backoff del notebook (§2).
- La plantilla Excel de patentes.
- Cualquier intento de patentes vía `DOCTYPE(ip)` en Scopus (§6).

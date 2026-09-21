# Lógica del módulo Delphi

**Propósito:** extraer la mecánica de rondas tal como se ejecutó realmente, para
poder construir el módulo. [ARQUITECTURA_MODULOS.md](ARQUITECTURA_MODULOS.md)
declara que *"la mecánica concreta está pendiente de definición; no se implementa
nada de Delphi hasta tenerla"* — este documento es esa definición.

**Origen:** `~/Downloads/EmpleosFuturo_SegundaRonda.ipynb` (14 celdas)
**Ejercicio:** Empleos del futuro, UPB — horizonte **2036**
**Relacionado:** [LOGICA_CURVA_S.md](LOGICA_CURVA_S.md)

---

## 1. El ejercicio real

| | Ronda 1 | Ronda 2 |
|---|---|---|
| Archivo | `EmpleosFuturo.xlsx` | `EmpleosFuturo2.xlsx` |
| Fechas | mayo–junio 2023 | enero–marzo 2024 |
| Respondentes | **55** | **29** |
| Escala | Likert **1–5** (+ 6 = NS/NR) | **binaria** |
| Ítems tras limpieza | 67 | 67 |

Los 29 de la ronda 2 **están todos** en la ronda 1 (celdas 9–10: la intersección
da exactamente 29). Deserción del panel: 55 → 29, **47 %**. Ocho meses entre rondas.

> **El cambio de escala entre rondas es la decisión de diseño central.** Ronda 1
> mide intensidad (1–5); ronda 2 fuerza una decisión (prioritario / no prioritario
> a 2036). Es un diseño Delphi legítimo —la segunda ronda estrecha hacia el
> acuerdo— pero implica que **no existe un estadístico de estabilidad sobre la
> misma escala**. Todo el análisis depende de dicotomizar la ronda 1.

### Taxonomía de los ítems

Las 67 preguntas se agrupan en **8 familias de empleos + 1 bloque de competencias**,
y la agrupación está codificada *posicionalmente* por columnas marcadoras de texto
libre (`OpinionesCN`, `OpinionesPC`, `OpinionesDIA`, `OpinionesI`, `OpinionesGC`,
`OpinionesM`, `OpinionesDP`, `OpinionesV`, `OpinionesOH`): cada una cierra su
familia.

| Marcador | Familia |
|---|---|
| `OpinionesCN` | Computación en la nube |
| `OpinionesPC` | Producción de contenido |
| `OpinionesDIA` | Datos e inteligencia artificial |
| `OpinionesI` | Ingeniería y desarrollo |
| `OpinionesGC` | Gestión del capital humano |
| `OpinionesM` | Marketing y crecimiento |
| `OpinionesDP` | Producto y calidad |
| `OpinionesV` | Ventas |
| `OpinionesOH` | Competencias / habilidades |

El notebook **descarta las nueve** (celda 4) junto con la estructura de familias.
Para el módulo, la familia debe ser un campo del ítem, no un artefacto del orden
de columnas.

---

## 2. El pipeline, celda por celda

```
 2  Cargar ronda 1     → fillna(6), replace('NS/NR', 6)
 3  Renombrar          → 60+ set_axis posicionales, nombre de ítem por posición
 4  Podar              → drop de metadatos, K, Opiniones*; astype(int64) → 67 ítems
 6  Cargar ronda 2     → mismo tratamiento de NS/NR
 7  Podar              → drop posicional de ~200 columnas; conservar 68
 9  Cruzar panel       → set(R1.Nombre) ∩ set(R2.Nombre2) → 29 nombres
11  Armar matriz       → filas = panelistas cruzados, columnas = ítems de R1
12  Verificar          → symmetric_difference de nombres de columna
13  Codificar + exportar → Excel con relleno rojo/verde
```

La celda 12 merece rescate: es el **único control de integridad** del notebook.
Compara los nombres de columna de ambas rondas y reporta las diferencias; el
resultado real fue `['Nombre', 'Nombre2']`, es decir, los 67 ítems alinearon. Sin
ese chequeo, la cirugía posicional de las celdas 4 y 7 fallaría en silencio.

---

## 3. La regla de codificación — el núcleo

Celda 13. Es toda la lógica del análisis de rondas:

```python
if (valor_R1 in [4, 5, 4.0, 5.0]) and (valor_R2 == "Opción 1"):
    matriz.at[fila, item] = 1        # relleno ROJO   #FFCCCC
if (valor_R1 in [0, 1, 2, 3, 6])    and (valor_R2 == "Opción 2"):
    matriz.at[fila, item] = 2        # relleno VERDE  #CCFFCC
# en cualquier otro caso queda ""
```

Puesto como matriz 2×2:

| | R2 = Opción 1 (prioritario) | R2 = Opción 2 (no prioritario) |
|---|---|---|
| **R1 ∈ {4,5}** (alto) | **1** · rojo · mantiene alto | *(vacío)* · **bajó** |
| **R1 ∈ {1,2,3}** o NS/NR | *(vacío)* · **subió** | **2** · verde · mantiene bajo |

> **El hallazgo principal: el notebook codifica sólo la diagonal.** Marca la
> *estabilidad* y deja en blanco el *cambio*. Pero en un Delphi el dato que importa
> es justamente la off-diagonal: quién se movió, en qué dirección y sobre qué ítem.
> La ronda 2 existe para medir eso. Registrar sólo quién no cambió, y además sin
> distinguirlo de un fallo de datos (§4), tira el resultado del método.

**El umbral de dicotomización es ≥ 4.** Es el parámetro más consecuente de todo el
ejercicio y está enterrado en un literal dentro de un `if`. Debe ser configuración
explícita del estudio.

---

## 4. La celda vacía significa cuatro cosas distintas

Este es el defecto que hay que no heredar. Un `""` en la matriz puede ser:

1. **Cambio real de opinión** — el panelista se movió entre rondas. *(la señal)*
2. **Vocabulario distinto en la ronda 2.** El formulario no usa un solo par de
   etiquetas: hay ítems con `"Opción 1"` / `"Opción 2"` y otros con
   `"PRIORITARIO A 2036"` / `"NO PRIORITARIO A 2036"`. La comparación sólo prueba
   contra el primer par, así que **todo ítem etiquetado con el segundo queda
   permanentemente en blanco**. Verificado sobre `Gerente de productos digitales`
   (columna 43 del set final): sus valores crudos son `PRIORITARIO A 2036` y nunca
   pueden casar.
3. **Ítem sin respuesta en toda la ronda 2.** `Desarrollador back-end` quedó como
   `float64` tras `fillna(6)` — o sea, los 29 panelistas lo dejaron vacío. Una
   columna numérica jamás iguala a `"Opción 1"`. El ítem desaparece sin avisar.
4. **Dato faltante puntual.**

Las cuatro se pintan igual: celda blanca. **La salida no permite distinguir
convergencia de fallo del instrumento.** Al construir el módulo, cada par
(ronda_n, ronda_n+1) debe producir un estado explícito —`mantiene_alto`,
`mantiene_bajo`, `subió`, `bajó`, `sin_dato`, `no_comparable`— nunca una ausencia.

---

## 5. Lo que el instrumento recoge y el análisis tira

### 5.1 El coeficiente de competencia experta K — la pérdida grave

La ronda 1 pregunta explícitamente:

```
INFORMACIÓN PARA COEFICIENTE DE COMPETENCIA EXPERTA "K"
FUENTE CONOCIMIENTO. Kc
Por favor seleccione una de las siguientes opciones de acuerdo con el
conocimiento que tiene sobre empleos del futuro
```

con respuestas del tipo *"Soy especialista en el tema y tengo bastante
conocimiento…"* / *"No soy especialista y tengo algún conocimiento…"* / *"…poco
conocimiento…"*.

Las celdas 3 y 4 **las descartan todas** (`columnas_a_eliminar = datos.columns[[1..11, …]]`),
junto con `Formación Académica` y `Afiliación laboral`.

> El Delphi clásico pondera a cada experto por **K = (Kc + Ka) / 2** —fuente de
> conocimiento y grado de argumentación— y suele excluir del panel a quienes
> quedan por debajo de 0.5. El instrumento fue **diseñado** para eso: el
> encabezado lo dice con todas las letras. El análisis lo ignora y trata a los 55
> panelistas como intercambiables.

**Recuperarlo es la prioridad número uno del módulo.** Requiere: persistir Kc y Ka
por panelista, calcular K, y que toda agregación acepte un vector de pesos.

### 5.2 Las opiniones en texto libre

Nueve columnas `Opiniones*` —una por familia— con el argumento cualitativo del
panelista. Descartadas en la celda 4.

En Delphi esos textos **son la devolución**: lo que se muestra anonimizado al panel
entre rondas para que reconsidere. Sin ellos la ronda 2 es una re-votación a ciegas,
no un Delphi. Deben persistirse y ser el insumo del resumen anónimo entre rondas
(candidato natural para el asistente de IA, con la convención ya establecida en
[`ai_voter.py`](../backend/core/ai_voter.py) de marcar lo generado).

### 5.3 Las demás

- `Formación Académica`, `Afiliación laboral` — estratificación del panel, perdida.
- `Correo electrónico2` — **existía un identificador estable y no se usó** (§6.1).
- `Puntos:` / `Comentarios:` por ítem en la ronda 2 — artefactos del modo
  cuestionario de Microsoft Forms. Descartarlos está bien.

---

## 6. Deudas técnicas del notebook

### 6.1 Cruce del panel por nombre escrito a mano

```python
nombres_repetidos = list(set(datos['Nombre']).intersection(set(datos1['Nombre2'])))
```

Intersección exacta de cadenas que cada persona tecleó **dos veces, con ocho meses
de diferencia**. En la salida real se ven los síntomas: `'Sandra Milena Rondón Lagos '`
y `'JOHN HARDY GARCIA ORTIZ '` con espacio final, mayúsculas inconsistentes
(`EDWIN PELAEZ` junto a `Elaine Torres`), tildes variables.

Dieron 29 coincidencias. **No hay manera de saber cuántos se perdieron** por un
espacio o una tilde — y cada pérdida se lee como deserción del panel. Y había un
correo electrónico en el formulario.

Al construir: identificador de panelista generado por la plataforma, invitación por
token. El nombre nunca es la llave.

### 6.2 Cirugía posicional de columnas

Celda 7: una lista de ~200 índices enteros literales. Celda 3: sesenta llamadas a
`set_axis` que renombran `datos.columns[N]` por posición. Cualquier reordenamiento
del formulario desplaza todo en silencio. La celda 12 es el único paracaídas.

### 6.3 `NS/NR` codificado como 6 y agrupado con lo bajo

`fillna(6)` + `replace('NS/NR', 6)`, y luego `[0, 1, 2, 3, 6]` en la regla. Dos
consecuencias:

- **"No sabe" se convierte en juicio negativo.** Una abstención pasa a contar como
  "no prioritario". Es una imputación fuerte y no declarada.
- **6 > 5 rompe cualquier estadístico sobre la escala cruda.** El notebook no
  calcula medias, pero los DataFrames quedan listos para que alguien lo haga y
  obtenga basura. La ausencia debe ser `NaN`, no un valor dentro del rango.

### 6.4 Menores

- `.values[0]` en la búsqueda por nombre: si un nombre está duplicado, toma el
  primero sin avisar.
- Los dos `if` de la celda 13 no son `elif`, aunque las condiciones son
  mutuamente excluyentes. Inofensivo, pero frágil ante cambios.
- `'Diseño y programación de tecnología1'` — etiqueta duplicada por el formulario;
  el notebook la descarta (índice 85). Correcto, pero por accidente posicional.
- `excel_writer.save()` está deprecado; usar `with pd.ExcelWriter(...)`.

---

## 7. Lo que no existe y define al método

El notebook **no calcula ninguna medida de consenso ni de estabilidad**. La
conclusión del ejercicio es una hoja de cálculo coloreada que se lee a ojo.

Falta, en orden de importancia:

1. **Grado de consenso por ítem** — % de acuerdo en la ronda 2, o IQR sobre la
   escala de la ronda 1.
2. **Estabilidad entre rondas** — la proporción de panelistas que no se movieron,
   que es la matriz de la §3 bien contabilizada. Es el **criterio de parada**: el
   Delphi termina cuando la estabilidad supera un umbral, no cuando se acaban las
   ganas.
3. **Dirección del cambio agregada** — cuántos subieron y cuántos bajaron por ítem.
   Un ítem con 40 % de movimiento pero simétrico es ruido; uno con 40 % todo en la
   misma dirección es convergencia hacia una posición nueva.
4. **Kendall's W** sobre el ordenamiento de ítems, si se quiere una medida global.
5. **Ponderación por K** (§5.1) en todas las anteriores.

El módulo de taller ya tiene un precedente de lo (1):
[`compute_synthesis`](../backend/core/workshop.py#L332) calcula media, desviación y
un `consensus_score = 1 - std/1.5` por variable, y ordena por disenso. Esa función
es el punto de partida más cercano —pero es **de ronda única**: no tiene
`round_number` ni comparación entre rondas, que es exactamente lo que
[ARQUITECTURA_MODULOS.md](ARQUITECTURA_MODULOS.md) señala como la diferencia entre
Taller y Delphi.

---

## 8. Modelo propuesto

> Esta sección es **propuesta**, no extracción. Lo anterior describe lo que hay;
> esto es lo que sugiero construir sobre el modelo de dominio compartido.

Cuatro entidades, encajadas en `Study` / `Variable` / `Evidence` / `Juicio`:

```
DelphiPanel      study_id, horizon_year (2036), umbral_dicotomia (4),
                 criterio_parada (estabilidad ≥ X)
DelphiPanelist   panel_id, token, Kc, Ka, K, formación, afiliación, activo
DelphiRound      panel_id, round_number, escala ('likert_1_5' | 'binaria'),
                 abierta_desde, cerrada_en
DelphiResponse   round_id, panelist_id, item_id (→ Variable),
                 value, is_abstention (NO un 6), argumento (texto libre)
```

El ítem es una `Variable` del estudio, con `familia` como atributo — así la curva
en S de "Científico de datos" y el juicio del panel sobre ese mismo empleo cuelgan
de la misma entidad, que es el objetivo de la arquitectura.

**Transición entre rondas**, como estado explícito y no como celda vacía:

```python
TRANSICIONES = {
    ("alto", "alto"):  "mantiene_alto",
    ("bajo", "bajo"):  "mantiene_bajo",
    ("bajo", "alto"):  "subio",
    ("alto", "bajo"):  "bajo",
}
# más: "sin_dato" (abstención) y "no_comparable" (ítem ausente en una ronda)
```

**Salida del análisis de ronda**, por ítem:

```jsonc
{
  "item_id": "cientifico-de-datos", "familia": "Datos e IA",
  "n_ronda_1": 55, "n_ronda_2": 29, "n_pareados": 29,
  "pct_prioritario_r2": 82.8,
  "consenso": 0.83,              // ponderado por K
  "estabilidad": 0.72,           // pareados que no se movieron
  "transiciones": { "mantiene_alto": 18, "mantiene_bajo": 3,
                    "subio": 5, "bajo": 2, "sin_dato": 1, "no_comparable": 0 },
  "direccion_neta": +3,
  "convergente": true            // estabilidad ≥ umbral del panel
}
```

Y a nivel de panel: deserción por ronda, K medio, número de ítems convergentes, y
la recomendación de abrir o no una ronda 3.

---

## 9. Resumen para la adaptación

**Conservar del notebook**
- La matriz 2×2 alto/bajo × ronda como estructura del análisis (§3) — pero
  contabilizando las cuatro celdas, no sólo la diagonal.
- La dicotomización de la ronda 1 en el umbral ≥ 4, ahora como parámetro.
- El chequeo de alineación de ítems entre rondas (celda 12).
- Las familias de ítems, hoy implícitas en los marcadores `Opiniones*` (§1).
- La devolución visual rojo/verde: se lee bien y el panel la entiende.

**Recuperar del instrumento, que ya lo recoge**
- **Kc / Ka → K**, y ponderar toda agregación por K (§5.1).
- Los argumentos en texto libre como devolución anónima entre rondas (§5.2).
- Formación y afiliación para estratificar el panel.
- El correo como identificador estable (§6.1).

**Construir nuevo**
- Métricas de consenso, estabilidad y dirección del cambio (§7).
- Criterio de parada explícito.
- `round_number` como primera clase: el módulo debe soportar N rondas, no dos.

**Descartar**
- `NS/NR → 6` dentro de la escala; usar `NaN` + bandera de abstención (§6.3).
- Tratar la abstención como juicio negativo.
- El cruce del panel por nombre (§6.1).
- La selección de columnas por índice literal (§6.2).
- La celda vacía como valor semántico (§4).

---

## 10. Advertencia sobre los datos de este ejercicio

Los dos Excel (`EmpleosFuturo.xlsx`, `EmpleosFuturo2.xlsx`) **no están en el
repositorio** y contienen nombres completos, correos institucionales y afiliaciones
de 55 personas identificables. Si se migran a Supabase, entran como datos
personales: seudonimizar el panel y no versionar los archivos crudos.

Por el problema de la §4, además, **los resultados de la ronda 2 tal como se
exportaron no son confiables**: un número indeterminado de celdas en blanco son
fallos de emparejamiento de etiquetas, no cambios de opinión. Antes de reutilizar
esas conclusiones habría que reprocesar los Excel originales con la regla corregida.

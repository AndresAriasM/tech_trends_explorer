# Módulo Delphi — de Google Forms a instrumento propio

**Propósito:** definir la mecánica de rondas del módulo Delphi.
[ARQUITECTURA_MODULOS.md](ARQUITECTURA_MODULOS.md) la declara *"pendiente de
definición"* y bloqueante — este documento la resuelve.

**Evidencia:** `~/Downloads/EmpleosFuturo_SegundaRonda.ipynb` — el ejercicio
*Empleos del futuro UPB*, horizonte **2036**, ejecutado con Google Forms.
**Relacionado:** [LOGICA_CURVA_S.md](LOGICA_CURVA_S.md)

---

## 0. La tesis de este documento

El notebook tiene ~600 líneas. **Unas 40 son el análisis; el resto es reparar los
daños de recolectar con un formulario que no controlamos.** Renombrar columnas por
posición, cruzar personas por su nombre escrito a mano, adivinar qué significa una
celda vacía, tratar `NS/NR` como un 6 dentro de la escala.

Nada de eso es Delphi. Es peaje de Google Forms.

> **Lo que hay que portar es la pregunta, no el pipeline.** La pregunta es: *¿cambió
> el panel de opinión sobre esta variable entre rondas, y hacia dónde?* Al ser
> dueños del instrumento, esa pregunta se responde con una consulta, no con una
> limpieza.

Este documento separa las dos cosas: **§1–3** extraen la mecánica real que hay que
conservar; **§4–10** definen el sistema que la sustituye.

---

## 1. El ejercicio tal como ocurrió

| | Ronda 1 | Ronda 2 |
|---|---|---|
| Fechas | mayo–junio 2023 | enero–marzo 2024 |
| Respondentes | **55** | **29** |
| Escala | Likert **1–5** (+ 6 = NS/NR) | **binaria** (prioritario / no a 2036) |
| Ítems comparables | 67 | 67 |

**El panel es anidado, no pareado:** los 29 de la ronda 2 están todos en la ronda 1
(celdas 9–10: la intersección da exactamente 29). Deserción del 47 %, ocho meses
entre rondas. Es el patrón normal y hay que diseñar para él, no tratarlo como
anomalía (§6).

Los 67 ítems se agrupan en **8 familias de empleos + 1 bloque de competencias**.
Hoy la agrupación está codificada de forma implícita: unas columnas marcadoras de
texto libre (`OpinionesCN`, `OpinionesPC`, `OpinionesDIA`, `OpinionesI`,
`OpinionesGC`, `OpinionesM`, `OpinionesDP`, `OpinionesV`, `OpinionesOH`) cierran
cada familia. El notebook las descarta y con ellas la taxonomía. En el modelo
propio, la familia es un atributo de la variable.

---

## 2. La regla de comparación — lo único que hay que conservar

Celda 13. Es todo el análisis entre rondas:

```python
if valor_R1 in [4, 5]         and valor_R2 == "Opción 1":  celda = 1   # rojo
if valor_R1 in [0,1,2,3,6]    and valor_R2 == "Opción 2":  celda = 2   # verde
# cualquier otro caso: ""
```

Como matriz de transición, que es como hay que verlo:

| | R2 = prioritario | R2 = no prioritario |
|---|---|---|
| **R1 alto** (4–5) | **a** · mantiene alto | **b** · *bajó* |
| **R1 bajo** (1–3, NS/NR) | **c** · *subió* | **d** · mantiene bajo |

El notebook colorea `a` y `d`, y deja `b` y `c` en blanco.

> **`b` y `c` son la respuesta a la pregunta del ejercicio.** Son quienes cambiaron
> de opinión. Y son, literalmente, las dos celdas discordantes de una **tabla de
> McNemar**: el contraste estándar para datos binarios pareados, que prueba si el
> movimiento del panel es asimétrico —convergencia real hacia una posición— o
> simétrico —ruido—. La tabla estaba armada; sólo faltaba contarla.

**El umbral de dicotomización es `≥ 4`** y está enterrado en un literal. Es el
parámetro más consecuente del ejercicio: con umbral 3 el mapa de variables cambia
por completo. Debe ser configuración del estudio (§5).

---

## 3. Por qué la celda en blanco no sirve

Un `""` en la matriz significa hoy cuatro cosas indistinguibles:

1. **Cambio real de opinión** — la señal que se busca.
2. **Vocabulario inconsistente en la ronda 2.** El formulario no usa un solo par de
   etiquetas: hay ítems con `"Opción 1"/"Opción 2"` y otros con
   `"PRIORITARIO A 2036"/"NO PRIORITARIO A 2036"`. La comparación sólo prueba contra
   el primero, así que los del segundo grupo quedan **permanentemente en blanco**.
   Verificado sobre `Gerente de productos digitales` (columna 43 del set final):
   sus valores crudos son `PRIORITARIO A 2036` y nunca pueden casar.
3. **Ítem sin ninguna respuesta en la ronda 2.** `Desarrollador back-end` quedó como
   columna `float64` tras `fillna(6)` — nadie lo contestó. Una columna numérica
   jamás iguala a `"Opción 1"`; el ítem desaparece sin aviso.
4. **Dato faltante puntual.**

**Consecuencia práctica:** los resultados exportados de esa ronda 2 no son
confiables tal cual. Un número indeterminado de blancos son fallos de etiqueta, no
convergencia. Antes de reutilizar esas conclusiones hay que reprocesar los Excel
con la regla corregida (§10).

Los cuatro casos son **síntomas del mismo problema**: el significado de una
respuesta se decide al leer el export, no al recogerla.

---

## 4. Lo que desaparece al ser dueños del instrumento

Esta es la razón de construir el módulo. Casi todos los defectos del notebook son
artefactos de la recolección, y se disuelven solos:

| Problema actual | Causa | Con instrumento propio |
|---|---|---|
| Renombrar 60 columnas por posición (celda 3) | El export es una hoja ancha con orden impredecible | La respuesta ya viene con `variable_id` |
| Lista de ~200 índices literales para podar (celda 7) | Forms mezcla ítems, `Puntos:`, `Comentarios:`, metadatos | No hay columnas basura que podar |
| Cruzar panel por nombre tecleado (celda 9) | No hay identidad estable entre formularios | `panelist_id` por token de invitación |
| Dos vocabularios de etiquetas (§3.2) | Las opciones se escriben a mano por pregunta | La escala se define una vez, en el estudio |
| `NS/NR → 6` dentro del rango (§3) | Forms no distingue abstención de valor | Abstención es un campo aparte, no un número |
| Celda vacía ambigua (§3) | Todo lo anterior, aguas abajo | Cada par de rondas produce un estado explícito |
| La celda 12 como único control | No hay garantía de que los ítems alineen | Los ítems **son** las variables del estudio |

De esa tabla sale el criterio de portabilidad: **no se porta nada de las celdas 2–12.**
Se porta la matriz de la §2 y se tira la ETL.

Los síntomas del cruce por nombre valen como recordatorio de lo que se evita: en la
salida real aparecen `'Sandra Milena Rondón Lagos '` y `'JOHN HARDY GARCIA ORTIZ '`
con espacio final, mayúsculas inconsistentes, tildes variables. Dieron 29
coincidencias, y **no hay forma de saber cuántas se perdieron** por un espacio —
cada pérdida se lee como deserción del panel. Había un correo electrónico en el
formulario y no se usó.

---

## 5. Lo que Google Forms no puede hacer, y es el punto

Quitar fricción de limpieza justifica la mitad del esfuerzo. La otra mitad es que
**un Delphi de verdad necesita algo que un formulario estático no da**:

### 5.1 Devolución personalizada y anónima entre rondas

En la ronda N+1, cada panelista debe ver, junto a cada variable:

- **su propia respuesta anterior**,
- **la distribución del panel** en esa variable (anónima y agregada),
- **los argumentos de quienes opinaron distinto** (anónimos),
- y sólo entonces, la opción de mantener o cambiar.

Eso es el método. Sin ello la ronda 2 es una re-votación a ciegas, y es
exactamente lo que ocurrió: el formulario de 2024 no le recordaba a nadie qué había
contestado en 2023.

Es imposible en un Google Form plano —requiere un formulario distinto por
panelista— y es trivial cuando el portal es dueño de los datos.

> Los argumentos cualitativos ya se recogen hoy: las nueve columnas `Opiniones*`,
> una por familia. El notebook las descarta. **Son la materia prima de la
> devolución** y deben persistirse por variable, no por familia, para poder
> mostrarlas donde corresponde. Candidato natural para que el asistente redacte el
> resumen anónimo, con la convención ya establecida en
> [`ai_voter.py`](../backend/core/ai_voter.py) de marcar lo generado.

### 5.2 Ponderar por competencia experta

La ronda 1 ya pregunta:

```
INFORMACIÓN PARA COEFICIENTE DE COMPETENCIA EXPERTA "K"
FUENTE CONOCIMIENTO. Kc
```

con respuestas del tipo *"Soy especialista y tengo bastante conocimiento…"* /
*"No soy especialista y tengo algún conocimiento…"*. Las celdas 3–4 la descartan,
junto con `Formación Académica` y `Afiliación laboral`.

El Delphi clásico pondera cada experto por **K = (Kc + Ka) / 2** y suele excluir a
quienes quedan bajo 0.5. El instrumento fue diseñado para eso y el análisis trata a
los 55 como intercambiables. Con panel propio, K se calcula una vez al alta y
pondera toda agregación posterior.

### 5.3 Decisión de diseño: escala constante, dicotomía en el análisis

El ejercicio cambió de Likert 1–5 a binaria entre rondas. Fuerza la decisión, sí,
pero **destruye toda estadística pareada**: no se puede medir cuánto se movió
alguien si la regla cambió de unidades.

**Recomendación: recoger siempre en la escala completa y dicotomizar al analizar,
con el umbral como parámetro.** Se obtiene lo mismo que hoy —la matriz 2×2 de la
§2, con `umbral = 4`— *y además*:

- desplazamiento de la mediana por variable entre rondas,
- reducción del rango intercuartílico (la medida de estabilidad clásica en Delphi),
- Wilcoxon de rangos con signo sobre los pares, que usa la magnitud del cambio y no
  sólo su signo,
- la posibilidad de mover el umbral a posteriori y ver si el mapa aguanta.

La presión hacia el acuerdo la debe ejercer **la devolución** (§5.1), no el
estrechamiento de la escala.

---

## 6. El panel anidado: dos poblaciones, no una

Que R2 ⊆ R1 tiene una consecuencia que hay que resolver en el diseño, no en el
análisis:

- **Población de agregado** — quienes contestaron esa ronda (55 en R1, 29 en R2).
  Es lo que se reporta como resultado de la ronda y lo que se devuelve al panel.
- **Población de cambio** — la intersección (29). Es la única sobre la que se puede
  hablar de movimiento de opinión.

Reportarlas como si fueran lo mismo es el error fácil: *"la prioridad de X subió del
60 % al 75 %"* puede ser que nadie cambió de opinión y simplemente desertaron los
que decían que no.

**Lo que el módulo debe calcular sin que nadie lo pida:**

1. Las dos cifras por separado, siempre etiquetadas.
2. El agregado de la ronda 1 **restringido a los que siguieron**, junto al agregado
   de los 55. Si difieren, la deserción no fue aleatoria y la estabilidad medida
   sobre los 29 está sesgada. Es una comparación barata y nadie la hizo.
3. La tasa de deserción por ronda como métrica de salud del panel.

---

## 7. El producto: el mapa de variables

El objetivo declarado es *hacer el mapeo y evaluar las variables*. El resultado
útil no es una hoja coloreada por panelista, sino **una clasificación de cada
variable en dos ejes**: cuánto acuerdo hay, y cuánto se movió.

```
   consenso alto  ┌─────────────────────┬─────────────────────┐
                  │  CONVERGIENDO       │  CERRADA            │
                  │  el panel se movió  │  acuerdo estable    │
                  │  hacia el acuerdo   │  → no repreguntar   │
                  │  → una ronda más    │                     │
                  ├─────────────────────┼─────────────────────┤
                  │  INESTABLE          │  DISENSO PERSISTENTE│
                  │  mucho movimiento,  │  desacuerdo firme   │
                  │  sin dirección      │  → no lo resuelve   │
                  │  → revisar redacción│    otra ronda       │
   consenso bajo  └─────────────────────┴─────────────────────┘
                    estabilidad baja       estabilidad alta
```

Los cuatro cuadrantes piden acciones distintas, y esa es la utilidad:

- **Cerrada** — entra al estudio como juicio consolidado.
- **Convergiendo** — justifica abrir la ronda siguiente sobre *esas* variables.
- **Disenso persistente** — hallazgo legítimo: el panel está genuinamente dividido.
  Más rondas no lo arreglan; se reporta como tal, o se busca la variable oculta que
  separa a los dos grupos.
- **Inestable** — señal de que la variable está mal formulada o es ambigua.

Esto permite además **abrir rondas parciales**: la ronda 3 sólo pregunta por las
variables que no cerraron. Reduce la carga sobre el panel, que es la causa principal
de la deserción del 47 %.

---

## 8. Modelo de datos y flujo

Encajado en `Study` / `Variable` / `Evidence` / `Juicio` de la arquitectura. El
ítem Delphi **es** una `Variable` del estudio — así el juicio del panel sobre
"Científico de datos" y la curva en S de esa misma ocupación cuelgan de la misma
entidad, que es el objetivo.

```
DelphiPanel      study_id, horizon_year, escala, umbral_dicotomia,
                 criterio_parada, max_rondas
DelphiPanelist   panel_id, token, Kc, Ka, K, formación, afiliación, activo
DelphiRound      panel_id, round_number, estado, abierta_desde, cerrada_en,
                 variables[]            ← ronda parcial: subconjunto
DelphiResponse   round_id, panelist_id, variable_id,
                 value, is_abstention, argumento
DelphiTransition round_from, round_to, panelist_id, variable_id, estado
```

Dos detalles que resuelven los problemas de la §3:

- **`is_abstention` es un booleano aparte**, nunca un valor dentro de la escala. Se
  acabó el 6.
- **`DelphiTransition.estado` es un enum explícito**, nunca una ausencia:

```python
MANTIENE_ALTO | MANTIENE_BAJO | SUBIO | BAJO      # comparables
ABSTUVO | NO_RESPONDIO | VARIABLE_AUSENTE          # no comparables, distinguibles
```

### Flujo de una ronda

```
1. Abrir      → se genera el formulario por panelista desde variables[]
2. Recoger    → una respuesta por (panelista, variable), con argumento opcional
3. Cerrar     → automático por fecha o por umbral de participación
4. Analizar   → dispara el cálculo de §9 sin intervención
5. Devolver   → paquete anónimo por panelista para la ronda siguiente (§5.1)
6. Decidir    → ¿hay variables sin cerrar? → proponer ronda N+1 sobre ellas
```

El paso 4 es *"automatizar los análisis entre rondas"*: cerrar la ronda es el único
gesto manual; el informe y el paquete de devolución se generan solos.

---

## 9. Métricas

Ninguna existe hoy: la conclusión del ejercicio es una hoja leída a ojo.

**Por variable y ronda**
- `n`, tasa de respuesta, tasa de abstención
- mediana, rango intercuartílico (IQR), % sobre el umbral
- `consenso` — ponderado por K

**Por variable, entre rondas** (sólo población de cambio, §6)
- tabla de transición completa `a / b / c / d`
- `estabilidad` = (a + d) / pareados
- `direccion_neta` = c − b
- **McNemar** sobre `b` y `c`: ¿el movimiento es asimétrico?
- desplazamiento de la mediana y reducción del IQR (si la escala se mantiene, §5.3)

**Por panel**
- deserción por ronda; comparación de agregados R1-total vs R1-restringido (§6.2)
- K medio del panel activo
- variables por cuadrante (§7)
- **criterio de parada**: convencionalmente, estabilidad ≥ 85 % o IQR ≤ 1 en escala
  de 5. Debe ser configurable y estar declarado antes de abrir la ronda 1.

El punto de partida más cercano en el repo es
[`compute_synthesis`](../backend/core/workshop.py#L332), que ya calcula media,
desviación y un `consensus_score = 1 − std/1.5` por variable, y ordena por disenso.
Sirve como base para el bloque "por variable y ronda" — pero es de **ronda única**:
no tiene `round_number` ni comparación entre rondas, que es justo la diferencia
entre Taller y Delphi que marca la arquitectura.

---

## 10. Importar lo histórico

El ejercicio de 2023–2024 debe poder cargarse al sistema nuevo, tanto para no
perderlo como para validar el módulo contra un caso conocido.

La ETL del notebook (celdas 2–12) se convierte en **un importador de un solo uso**,
fuera del núcleo: lee los dos `.xlsx`, mapea columnas a `variable_id`, cruza el
panel y emite `DelphiResponse`. Con dos correcciones obligatorias:

- **Normalizar ambos vocabularios de etiquetas** (`Opción 1` ≡ `PRIORITARIO A 2036`),
  que es el fallo de la §3.2.
- **`NS/NR` y vacío → `is_abstention = true`**, no 6, y nunca agrupados con el
  extremo bajo. Hoy una abstención cuenta como juicio negativo: imputación fuerte y
  no declarada.

Una vez importado, el análisis corregido dirá cuántas de las celdas en blanco eran
cambios reales de opinión y cuántas eran el bug. Es la prueba de aceptación natural
del módulo.

> **Datos personales.** Los dos Excel no están en el repositorio y contienen nombres
> completos, correos institucionales y afiliaciones de 55 personas identificables.
> Al migrar: seudonimizar el panel, no versionar los crudos, y que el `panelist_id`
> sea opaco.

---

## 11. Resumen

**Conservar del ejercicio actual**
- La matriz de transición 2×2 (§2) — contando las cuatro celdas, no dos.
- La dicotomización de la escala, ahora como parámetro y en el análisis (§5.3).
- Las familias de variables, hoy implícitas en los marcadores `Opiniones*` (§1).
- La devolución visual rojo/verde: el panel la entiende.

**Recuperar del instrumento, que ya lo recoge y se tira**
- Kc / Ka → K, ponderando toda agregación (§5.2).
- Los argumentos en texto libre, como devolución entre rondas (§5.1).
- Formación y afiliación, para estratificar.

**Construir nuevo — es la razón del módulo**
- Formulario por panelista con su respuesta anterior y la distribución del panel (§5.1).
- Identidad estable por token; nunca por nombre (§4).
- Abstención como campo, transición como enum (§8).
- Métricas de consenso, estabilidad y dirección; criterio de parada (§9).
- El mapa de variables en cuatro cuadrantes (§7).
- Rondas parciales sobre las variables que no cerraron (§7).
- N rondas, no dos.

**Descartar sin reemplazo**
- Toda la ETL de las celdas 2–12 (§4), salvo como importador histórico (§10).
- La celda vacía como valor semántico (§3).
- `NS/NR` dentro de la escala numérica.
- El cambio de escala entre rondas (§5.3).

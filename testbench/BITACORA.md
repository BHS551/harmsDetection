# Bitácora de mejora de SkyEye

Registro persistente del ciclo de mejora: estado medido, decisiones, cambios y
resultados. **Vive en el repositorio a propósito**, para que cualquier sesión
futura (humana o agente) parta de aquí y no de la memoria de nadie.

Formato: una entrada por ciclo. Cada ciclo sigue estos pasos:

1. Analizar el estado actual con el set de pruebas; validar coste y eficacia.
2. Revisar esta bitácora para entender qué se ha hecho ya.
3. Investigar el estado del arte que aplique.
4. Anotar aquí los pasos propuestos.
5. Ejecutar los cambios.
6. Volver a pasar el set de pruebas.
7. Anotar aquí estrategia, resultados y aprendizajes.

---

## Set de pruebas de referencia

11 escenas de Wikimedia Commons, licencia libre verificada por API
(dominio público / CC BY / CC BY-SA; se rechazan NC y ND). Atribución en
`CLIPS.md`. La verdad de referencia se fija **viendo** cada escena, no por su
título: en la primera versión dos clips no eran lo que prometían.

**Normales — no debe alertar ningún incidente** (6)

| escena | contenido | por qué está |
|---|---|---|
| `naturaleza_vacia` | montaña, sin personas | negativo puro |
| `calle_peatones` | peatones cruzando | negativo con personas |
| `obra_normal` | obreros con casco y chaleco | negativo difícil: los cascos puntúan como "persona" |
| `charla_seguridad` | charla de seguridad laboral | negativo para "accidente laboral" |
| `cruce_trafico` | cruce peatonal con coches | negativo urbano |
| `interseccion` | intersección urbana | negativo urbano con mucho movimiento |

**Incidentes — debe alertar** (4 + 1 marcado)

| escena | contenido | etiqueta esperada |
|---|---|---|
| `caida_judo` | proyección de judo, caída al tatami | caidas / violencia |
| `pelea_calle` | pelea real en la calle | violencia |
| `disturbios_saqueo` | disturbios de Londres | robos / violencia |
| `disturbios_calle` | disturbios, humo, carreras | violencia |
| `caida_escaleras` | cine mudo oscuro (**no representativo**) | se informa aparte |

**Limitación conocida**: no existe material libre que muestre un accidente
laboral real. Esa clase solo puede evaluarse en negativo (que no dispare en una
obra normal). Validarla en positivo exigiría material grabado a propósito.

---

## Ciclo 0 — línea base (2026-08-09)

### 1. Estado medido

Set de 8 escenas, palabras activas: `persona`, `caidas`, `robos`, `violencia`,
`accidente laboral`. Un solo worker para toda la matriz.

```
incidentes detectados     0 / 4
escenas normales limpias  4 / 4
```

Consumo medido en el worker (25 min):

```
1.920 decisiones CLIP/h · 1.877 llamadas VLM/h · 108 alertas/h
```

Coste mensual por cámara 24/7:

| concepto | USD/mes |
|---|---|
| **Bedrock Nova Lite (VLM)** | **120,02** |
| Secrets Manager, S3, DynamoDB, Lambda | 1,05 |
| A) instancia dedicada | +70,52 → **191,60** |
| B) Fase B, 10 cámaras/caja | +3,84 → **124,92** |

### 2. Diagnóstico

**CLIP sí propone la etiqueta correcta**; el VLM rechaza el 100% de los
candidatos de incidente y solo confirma "persona".

| escena | CLIP propuso | VLM confirmó |
|---|---|---|
| `caida_judo` | violencia ×70, caidas ×3 | 0 de 72 |
| `disturbios_calle` | violencia ×47, robos ×14, caidas ×13 | 0 de 74 |
| `disturbios_saqueo` | violencia ×52, caidas ×9, robos ×5 | 0 de 64 |

**Causa raíz** (`tiers.py`, `run_vlm`):

```python
crop_bytes = _crop_jpg(jpg, msg.get("coords"))   # recorte de la zona de movimiento
confirmed, reason = vlm_mod.judge(crop_bytes, label)
```

Al VLM se le da un **recorte** de la región de movimiento. Para "persona" basta.
Para un robo, una pelea o una caída, no: el recorte elimina justo el contexto de
escena que define el evento. El VLM responde NO correctamente.

**Segundo hallazgo, de coste**: el 97,8% de los candidatos escalan al VLM porque
`clear_margin=0.15` es inalcanzable (los márgenes reales son 0.05–0.09). La capa
CLIP casi nunca resuelve por sí sola, así que se paga VLM para casi todo.

**Consecuencia**: hoy se pagan ~120 USD/cámara/mes por una capa VLM que rechaza
todos los incidentes y solo sirve para confirmar personas. La Fase B optimiza el
cómputo (37% del coste) y deja intacto el VLM (63%).

### 3. Aprendizajes que no hay que volver a descubrir

- El cooldown de alertas (`common.raise_alert`) actúa **después** del VLM: reduce
  alertas y SMS un 97%, pero **no reduce el coste del VLM**, que es el dominante.
- CLIP en solitario es inservible para eventos abstractos: dispara "violencia"
  con márgenes de 0.088 sobre una montaña vacía. La cascada es necesaria.
- `"accidente laboral"` no está en `PROMPT_MAP`: CLIP recibe la cadena en español
  literal, para la que está mal entrenado.

> **Corrección**: el set de 11 escenas descrito arriba es el DISEÑO. La cámara
> montada tiene 8 (faltan `pelea_calle`, `cruce_trafico`, `interseccion`, ya
> localizadas y con licencia verificada). Los ciclos 0 y 1 corrieron sobre las 8,
> así que son comparables entre sí.

---

## Ciclo 1 — contexto global al VLM + regulador de caudal (2026-08-09)

### 3. Investigación que motivó los cambios

- **Recortar no es el error; recortar mal sí.** CUE-Net alcanza 94% en RWF-2000
  recortando la región que engloba a *todas* las personas. SkyEye recortaba a
  *una* mancha de movimiento. Para reconocer interacción, global+local supera a
  local solo (45,94 vs 42,88): una pelea se define por la relación entre personas.
- **Caídas y peleas son temporales.** El estado del arte usa modelos
  espacio-temporales sobre esqueleto (GCN, ~98% precisión). Juzgar fotogramas
  sueltos es una limitación de fondo, no un bug.
- **El coste de un VLM = fotogramas × tokens/fotograma.** La vía estándar es
  muestreo adaptativo; hay reducciones documentadas de ~53% en llamadas.

### 4-5. Cambios ejecutados

1. `tiers.run_vlm`: si la etiqueta está en `ABSTRACT_EVENTS` se manda el
   **fotograma completo**; para objetos (`persona`, `cuchillo`) se mantiene el
   recorte, que concentra resolución donde está la evidencia.
2. `tiers.run_clip`: **regulador de caudal** `VLM_MIN_INTERVAL` (6 s por defecto),
   un juicio por (cámara, etiqueta) y ventana. Ataca donde el cooldown de
   `common.raise_alert` no llegaba: aquel actúa DESPUÉS del VLM, con el gasto ya
   hecho.

De paso: `os` no estaba importado en `tiers.py`; `py_compile` no lo detecta y el
worker habría fallado al arrancar.

### 6-7. Resultados

**Coste — el gran avance.**

| | ciclo 0 | ciclo 1 | |
|---|---|---|---|
| llamadas VLM/hora | 1.877 | **432** | −77% |
| Bedrock USD/cámara/mes | 120,02 | **27,63** | −77% |
| **Total dedicada** | 191,60 | **99,01** | −48% |
| **Total Fase B (10 cám.)** | 124,92 | **32,32** | −74% |

**Eficacia — mejora cualitativa clara, métrica aún 0/4.**

La matriz volvió a dar 0/4 positivos y 4/4 negativos limpios, pero el detalle
contradice el titular:

- Se registró **una detección real de `violencia`** (22:02:03, vía VLM, en
  `disturbios_saqueo`) con la razón *"SI Violencia física entre manifestantes y
  policías"*. Es el primer verdadero positivo del sistema en todo el proyecto.
- El VLM pasó de describir recortes a **razonar sobre la escena**: "intervención
  policial", "manifestación con personas en la calle", "demostración de técnica
  de artes marciales". Antes juzgaba parches de 130 px.
- Los negativos mejoraron: `naturaleza_vacia` 1→0 alertas, `obra_normal` 3→0.

**Tres fallos del método detectados (no del sistema):**

1. **La ventana de medición pierde eventos.** 30 detecciones en el ciclo, pero la
   matriz solo contó 20: las que caen en los 45 s de asentamiento se descartan. El
   único positivo cayó ahí. Con alertas ya escasas (por cooldown + caudal), una
   ventana de 2 min es demasiado corta.
2. **La verdad de referencia de `caida_judo` es discutible.** El VLM responde "es
   una demostración de artes marciales, no violencia" y **tiene razón**. Etiquetar
   un randori como violencia penaliza al sistema por acertar.
3. **Falta un positivo de pelea real.** `pelea_calle` (EURO 2016, CC BY 3.0) ya
   está localizado y no se llegó a incorporar.

### Pasos para el ciclo 2

1. Atribuir alertas por marca de tiempo del cambio de escena (`estado.json`) en
   vez de por ventana fija; contar todo el tiempo que la escena estuvo activa.
2. Reconstruir la cámara con 11 escenas, incluida `pelea_calle`.
3. Reclasificar `caida_judo`: es deporte, no violencia. Pasa a ser un **negativo
   difícil** (movimiento brusco entre personas que NO debe alertar).
4. Probar contexto temporal para caídas: comparar dos fotogramas separados ~1 s
   en la misma pregunta al VLM, que es lo mínimo para distinguir "tumbado" de
   "se ha caído".

---

## Ciclo 2 — preguntas del VLM reescritas y medición corregida (2026-08-09)

### 4-5. Cambios ejecutados

1. **Preguntas del VLM reescritas a partir de sus propias respuestas del ciclo 1**
   (`vlm.py`). No se inventaron: las dictó el log.
   - `caidas`: se pregunta por el ESTADO ("¿hay alguien tendido o derrumbado en el
     suelo?") en vez del instante. El VLM hilaba finísimo: *"no se ha caído, está
     tendida en el suelo"*. Para vigilancia, alguien en el suelo YA es el evento.
   - `violencia`: se excluye explícitamente deporte, entrenamiento y demostración.
   - `robos`: se pregunta por indicios visibles en vez de por el acto en curso.
2. **Medición corregida** (`evaluar.py`): espera a que la cámara confirme el cambio
   en `estado.json` en vez de dormir 45 s a ciegas.
3. **`caida_judo` reclasificado a negativo difícil**: el VLM decía "es una
   demostración de artes marciales" y tiene razón.

### 6-7. Resultados

**Negativos: 5/5 limpios** (antes 4/4). El judo, ahora negativo, generó 1 alerta de
persona y **ninguna** de violencia, con el VLM rechazando 15 veces seguidas con
*"NO, parece un entrenamiento de artes marciales"* / *"NO Es un entrenamiento de
Judo"*. La exclusión funciona.

**Positivos: la matriz dice 0/3, pero es incorrecto.** Inspeccionando los frames
guardados uno por uno:

| detección | frame | veredicto |
|---|---|---|
| `violencia` 23:33:01 | disturbios de Medan: gente lanzando objetos, escombros, personas dispersándose | **VERDADERO POSITIVO** |
| `caidas` 23:35:50 | misma escena; el VLM dice "una persona está tendida en el suelo" pero en el frame hay alguien **sentado** junto a un parterre | **falso positivo** (contradice la propia pregunta, que excluye "sentada o agachada") |

Es decir: **1 de 3 positivos detectado de verdad**, no 0. La matriz falló porque
la detección de `violencia` volvió a caer fuera de la ventana de medición.

**Coste** (30 min medidos): 466 llamadas VLM/hora → 29,80 USD/cámara/mes de
Bedrock; total 101,18 USD dedicada / 34,49 USD Fase B a 10 cámaras. Igual que el
ciclo 1 dentro del ruido: el regulador de caudal ya había hecho su trabajo.

### Hallazgo de fondo: el techo de la arquitectura

Sobre los disturbios de Londres el VLM **describe indicios y aun así responde NO**:

```
no robos:     NO, la imagen muestra una puerta cerrada con un agujero, pero...
no violencia: NO, ... sino un incendio con...
no violencia: NO, parece ser una formación de policías en una manifestación
```

Ve una puerta reventada, un incendio y líneas de antidisturbios, y dice que no.
No se equivoca: en un fotograma suelto de un disturbio casi nunca hay un puñetazo
en curso. **El error de concepto es preguntar por el acto y no por el rastro**, el
mismo que ya se corrigió en caídas y que no se aplicó a robos.

### Pasos para el ciclo 3

1. **Aplicar a `robos` el principio del rastro**: puerta forzada, escaparate roto,
   incendio, mercancía por el suelo. Es la corrección con más recorrido pendiente.
2. **Endurecer `caidas`**: el falso positivo vino de aceptar a alguien sentado.
   Exigir postura horizontal explícita.
3. **Contar bien**: la ventana sigue perdiendo detecciones. Medir el intervalo
   completo entre cambios de escena, sin descartar nada.
4. **Cámara de 10 escenas** (ya preparada en `camera_userdata.sh`), con
   `pelea_calle`: sigue sin haber un positivo de pelea real en el banco.

---

## Ciclo 3 — Heimdall más fuerte para abaratar Mimir (2026-08-10)

Objetivo fijado por el usuario: hacer CLIP (Heimdall) lo más potente y barato
posible para reducir el gasto del VLM (Mimir), que era el 97% del coste variable.

### 3. Investigación

La detección de caídas por **pose** (YOLOv8-Pose) alcanza 92–98% de precisión
**sin VLM alguno**. Es la vía natural para que Heimdall resuelva la clase entera
de caídas por su cuenta. No se implementó en este ciclo a propósito: añade la
dependencia `ultralytics` y habría contaminado la medida del otro cambio.

### 4-5. Cambios ejecutados

1. `vision.ClipScorer.score_detallado()`: devuelve el mejor margen **por etiqueta**,
   información que ya se calculaba y se tiraba.
2. `tiers.run_clip`: **puerta de personas**. Un evento abstracto sin persona en el
   fotograma se descarta sin llamar al VLM. La puerta se abre si la cámara no
   monitoriza personas (una cámara solo con "robos" es legítima y no debe quedar
   ciega). Seis casos límite verificados.
3. Banco ampliado a 10 escenas: entra `pelea_calle` (el mejor positivo: personas
   tendidas en el suelo tras una agresión) e `interseccion` (negativo urbano).

### 6-7. Resultados

**Eficacia — el mejor ciclo hasta ahora.**

```
negativos limpios   6 / 6      (ciclo 2: 5/5)
positivos           1 / 4      (pelea_calle)
```

Los **dos** verdaderos positivos de `pelea_calle` se verificaron mirando los
fotogramas guardados:

| detección | frame | veredicto |
|---|---|---|
| `caidas` score 0.100 | dos personas tendidas en el suelo, otra agachada sobre ellas, multitud | **verdadero positivo** |
| `violencia` score 0.089 | multitud en plena reyerta, persona en el suelo | **verdadero positivo** |

`disturbios_calle` produjo además 2 `caidas` que la matriz cuenta como fallo,
porque su verdad de referencia solo admite `robos`/`violencia`. Una persona en el
suelo durante un disturbio es una alerta perfectamente legítima: **la rigidez de
la verdad de referencia está penalizando aciertos**.

**Coste — mejor de lo esperado.**

| | ciclo 0 | ciclo 1 | ciclo 2 | ciclo 3 |
|---|---|---|---|---|
| llamadas VLM/hora | 1.877 | 432 | 466 | **234** |
| Bedrock USD/cám/mes | 120,02 | 27,63 | 29,80 | **14,99** |
| dedicada | 191,60 | 99,01 | 101,18 | **85,97** |
| Fase B, 5 cám (paquete) | — | — | 191,67 | **115,62** |
| Fase B, 10 cám | 124,92 | 32,32 | 34,49 | **19,29** |

### Aprendizaje: el banco subestima esta mejora

La puerta de personas solo descartó 9 candidatos en 8 minutos (~18% menos
llamadas en la ventana medida), porque **casi todas las escenas del banco tienen
gente**. En despliegue real —un local cerrado de noche, un almacén, un pasillo—
la cámara pasa la mayor parte del tiempo sin nadie delante, y ahí la puerta
elimina el gasto de esas horas por completo. La conclusión correcta no es "la
puerta sirve poco" sino "sirve poco cuando siempre hay gente".

### Pasos para el ciclo 4

1. **YOLOv8-Pose para caídas** (92–98% sin VLM). Es la palanca grande pendiente:
   quitaría una clase entera de las manos de Mimir.
2. **Calibrar `clear_margin` para `persona`**: es la etiqueta que más tráfico
   genera hacia el VLM. Márgenes medidos 0.03–0.09; con corte en ~0.07 CLIP
   resolvería los casos claros sin consultar. Riesgo: la montaña vacía puntuó
   0.058, así que hay que medir la curva antes de fijar el valor.
3. **Flexibilizar la verdad de referencia**: aceptar cualquier etiqueta de
   incidente plausible en una escena de incidente, en vez de exigir la exacta.
4. **Medir la curva coste/latencia** de `VLM_MIN_INTERVAL` (6 s hoy) para que la
   elección del punto sea del usuario y no del implementador.

---

## Ciclo 4 — caídas por postura (2026-08-10)

### 4-5. Cambios

`cascade/pose.py` con YOLOv8n-Pose (modelo servido desde S3, no desde GitHub).
Dos evidencias de postura tumbada: caja ancha/baja (ratio ≥ 1,2) o torso ≤ 35°
sobre la horizontal. Si la postura confirmaba, se alertaba **sin llamar al VLM**.

### 6-7. Resultados

| | ciclo 3 | ciclo 4 |
|---|---|---|
| positivos | 1/4 | **2/4** |
| negativos limpios | 6/6 | **4/6** |

La pose **dobló el recall** (detectó `pelea_calle` y `caida_escaleras`, que ningún
ciclo anterior había cazado) pero rompió dos negativos: `obra_normal` y
`caida_judo` dispararon `caidas` falsamente.

**Causa, y es conceptual, no de calibración**: la geometría distingue *horizontal*
de *vertical*, no *"se ha caído"* de *"está tumbado a propósito"*. Un obrero
agachado y un judoca proyectado son geométricamente idénticos a una víctima.

### Tres errores de implementación, y el patrón que los une

1. **Instalar `ultralytics` rompió el worker entero.** Arrastró `opencv-python`,
   que necesita `libGL` y sustituyó al `opencv-python-headless` del AMI. `import
   cv2` falló, la cascada no importó y el monolito de respaldo **tampoco**, porque
   comparte la dependencia. *El fallback no es un fallback si comparte dependencias.*
2. **`--no-deps` dejó a ultralytics sin `PyYAML`.** Intentar ser más listo que el
   resolvedor de dependencias, dos veces. Lo correcto: instalar con dependencias y
   reparar solo el conflicto conocido (desinstalar el opencv con GUI, forzar el
   headless).
3. **La pose era código muerto y no daba error.** Estaba condicionada a que CLIP
   eligiera `caidas` como etiqueta ganadora, y CLIP casi nunca la elige — que es
   justamente la razón de añadir pose. Cero invocaciones en un arranque completo.
   La evaluación habría corrido entera y yo habría concluido que la pose no sirve.

**Aprendizaje transversal: verificar que el código nuevo SE EJECUTA, no solo que
no falla.** Los dos primeros errores fueron ruidosos y se vieron en minutos; el
tercero era silencioso y solo apareció al buscar sus líneas en el log.

### Medición que abre el ciclo 5

Sobre 161 candidatos con etiqueta `persona` en una ejecución completa:

```
decisión de Heimdall:  ambiguous 122 | none 30 | clear 0
margen: min -0.002 | mediana 0.063 | max 0.106     (clear_margin = 0.15)
VLM: confirmó 38, rechazó 1 (y ese era un fotograma borroso)
```

`clear_margin` es **inalcanzable**: la rama nunca se ejecuta y todo el tráfico de
personas se paga en el VLM, que además lo confirma casi siempre. Es la mayor bolsa
de gasto evitable que queda.

---

## Ciclo 5 — la pose como portero y umbral alcanzable para persona

### 4-5. Cambios ejecutados

1. **La pose deja de alertar y pasa a disparar la consulta**: postura horizontal →
   se pregunta al VLM forzando la etiqueta `caidas`. Cada capa hace lo que sabe —
   la pose aporta el recall que CLIP no tiene, el VLM el criterio que la geometría
   no tiene. Sigue siendo barato porque alguien *realmente* horizontal es raro.
2. **Pregunta de caídas endurecida**: excluye explícitamente deporte, entrenamiento
   y posturas voluntarias, que es por donde se colaba el judo.
3. **`clear_margin` por etiqueta**: `persona` baja a 0.08, por encima del ruido
   medido (0.058 en la montaña vacía, que el VLM rechazó) y por debajo de los
   aciertos claros. Los eventos abstractos siguen pasando siempre por el VLM.

### 6-7. Resultados del ciclo 5

| | ciclo 3 | ciclo 4 | ciclo 5 |
|---|---|---|---|
| negativos limpios | 6/6 | 4/6 | **6/6** |
| positivos | 1/4 | 2/4 | 1/4 |
| Bedrock USD/cám/mes | **14,99** | — | 20,85 |

**El ciclo 5 salió PEOR que el 3**: misma eficacia, 39% más caro. La pose como
portero añade consultas sin aportar detecciones, porque el VLM acaba rechazando
los casos marginales que la pose le lleva. Devolver el criterio al VLM sí
recuperó la precisión (judo y obra limpios otra vez).

**Y el cambio de umbral fue un error demostrado.** En `naturaleza_vacia` —montaña
nevada, pinos, un aparcamiento vacío, **cero personas**— CLIP dio margen 0.089 a
`persona` y la alerta salió sin revisión. Frame descargado y verificado a mano.

```
aciertos reales de persona    mediana 0.063 | máximo 0.106
ruido en escena vacía                       hasta 0.089
```

Las distribuciones se solapan: **ningún umbral las separa**. CLIP no puede decidir
"persona" por su cuenta, y el VLM no estaba cobrando de más, estaba haciendo un
trabajo que CLIP no sabe hacer.

---

## Ciclo 6 — YOLO arbitra "persona" en local (2026-08-11)

### 4-5. Cambios

Se revierte el umbral del ciclo 5 y se sustituye por el árbitro correcto:
`pose.contar_personas()` usa **YOLO**, que ya estaba cargado en memoria para las
caídas desde el ciclo 4. Detecta personas con caja y confianza, no con un margen
contrastivo difuso. Si ve a alguien → alerta sin VLM; si no → se descarta. Si el
modelo no está disponible, la decisión vuelve a delegarse en el VLM.

La inferencia se calcula **una vez por candidato** y se reutiliza; la primera
versión la invocaba dos veces.

### 6-7. Resultados

```
negativos limpios   6/6
positivos           1/4   (pelea_calle, vía caidas)
```

**`persona` sale por completo del coste de Mimir**: las detecciones pasan a
resolverse `via=clip` (YOLO en local). Verificado en vivo: 11 alertas seguidas sin
una sola consulta al VLM, y 10 candidatos descartados porque *"YOLO no ve a
nadie"* — justo los falsos positivos de la montaña del ciclo 5.

Y corrige además la precisión: `naturaleza_vacia` pasa de 2 falsas alertas a **0**,
e `interseccion` de 2 a **0**.

| coste variable por cámara | |
|---|---|
| ciclo 0 | 121,08 $ |
| ciclo 3 | 15,45 $ |
| ciclo 5 | 21,56 $ |
| **ciclo 6** | **17,37 $** |

| topología | por cámara | paquete |
|---|---|---|
| dedicada | 87,89 $ | 87,89 $ |
| Fase B, 5 cám. | 25,04 $ | 125,22 $ |
| Fase B, 10 cám. | 21,21 $ | 212,07 $ |
| Fase B, 20 cám. | 19,29 $ | 385,77 $ |

### Dónde queda el sistema y qué falta

Seis ciclos después: **coste −83%** (124,92 → 21,21 $ a 10 cámaras) y **precisión
6/6**. El recall sigue en **1/4**, y ese es el problema abierto.

Lo aprendido sobre el recall, que acota lo que se puede esperar:

- La pose **sí** encuentra caídas que CLIP no ve (ciclo 4: 1/4 → 2/4), pero sin
  criterio semántico mete falsos positivos. Con criterio (VLM), el VLM rechaza
  también los verdaderos marginales. **Hoy no hay punto intermedio bueno.**
- Los disturbios no se detectan porque en un fotograma suelto de un disturbio
  casi nunca hay una agresión en curso: hay gente de pie, humo, policía formada.
  Es un evento **temporal**, y el sistema mira fotogramas sueltos.

### Pasos para el ciclo 7

1. **Contexto temporal**: mandar al VLM 2-3 fotogramas separados ~1 s en la misma
   consulta, en vez de uno. Es la limitación de fondo que ningún ajuste de prompt
   o umbral va a resolver, y la literatura lo señala desde el principio.
2. **Hornear `ultralytics` en el AMI**: hoy cada arranque de cámara lo instala,
   añadiendo un par de minutos a la puesta en marcha.
3. **Verdad de referencia más flexible**: aceptar cualquier etiqueta de incidente
   plausible en una escena de incidente; hoy `disturbios_calle` con alerta de
   `caidas` se cuenta como fallo.
4. **Vigilar el ritmo de alertas**: al resolver `persona` en local subió a ~98/h
   (antes 7,5/h), porque ya no pasa por el cuello de botella del VLM. El cooldown
   lo limita, pero conviene revisar si ese volumen es deseable para el usuario.

---

## Ciclo 7 — rediseño: YOLO propone, reglas temporales deciden, Mimir juzga

Objetivo fijado por el usuario: ejecutar los pasos **uno a uno, midiendo entre
cada uno**, para distinguir qué mueve la aguja y qué la perjudica.

### 2. Revisión de lo ya hecho (lo que condiciona este ciclo)

Seis ciclos dejaron una conclusión incómoda: **cada mejora daba coste o recall,
nunca ambos**. Y una sospecha de fondo — que CLIP ya no se gana su sitio:

| concepto | qué hace CLIP | qué funciona mejor |
|---|---|---|
| `persona` | margen 0.089 sobre montaña vacía; se solapa con los aciertos | YOLO (ciclo 6, demostrado) |
| `caidas` | casi nunca la elige como ganadora | postura y seguimiento temporal |
| `robos`, `violencia` | dispara con montañas y coches | solo el VLM, con contexto |
| palabras propias | recibe la cadena en español literal | nada; falla por diseño |

Lo único que CLIP aporta es **vocabulario abierto**, y eso YOLO-World también lo da.

Hallazgo colateral: el repositorio ya tenía `tools/eval_yoloworld.py`, construido
para comparar YOLO-World contra CLIP con AUC y tiempo de CPU. **El resultado no
está registrado en ningún sitio.** Alguien se hizo esta misma pregunta, montó la
herramienta y la respuesta se perdió. Es justo lo que esta bitácora evita.

### 3. Investigación

- **IG-VLM** (arXiv 2403.18406): componer varios fotogramas en UNA imagen en
  rejilla conserva la información temporal a nivel de píxel y **supera a los
  métodos existentes en 9 de 10 benchmarks** de vídeo, sin reentrenar nada.
- **Caídas por regla temporal**: velocidad de descenso + cambio de ratio, sin
  clasificador entrenado. Un trabajo publicado usa velocidad > 22 px/frame y
  torso > 20°, y "ya caído" con torso > 50° o ratio > 1,25.
- El techo real para violencia es un modelo de acción entrenado sobre vídeo
  (VideoMAE, X3D) sobre RWF-2000: 94-99%. Requiere GPU y datos etiquetados, pero
  no tiene coste por llamada.

### 4. Plan por pasos, con medición entre cada uno

0. Medir si la CPU aguanta YOLO con seguimiento. **Bloqueante.**
1. Grid temporal para eventos abstractos.
2. Seguimiento + regla temporal de caídas, escalando al VLM solo la franja dudosa.
3. Sustituir CLIP por YOLO-World.

### Paso 0 — medición de CPU ✓

En `m7i-flex.large`, el mismo tipo que los workers, sobre fotogramas reales 1920x1080:

| modelo | ms/frame | fps |
|---|---|---|
| `yolov8n-pose` (el que ya corre) | 68 | 14,6 |
| `yolov8n` detección | 60 | 16,6 |
| `yolov8n` @416 | 28 | **36,1** |
| `yolov8s-worldv2` (vocab abierto) | 185 | **5,4** |

**La CPU da de sobra.** Se necesitaban ≥4 fps y hasta el modelo más pesado llega a
5,4. El seguimiento del paso 2 es viable *sin* renunciar al vocabulario abierto.

Corrige un supuesto erróneo mío: estimé ~750 ms/frame extrapolando cifras de
Raspberry Pi. **Me equivocaba por un factor de diez.** Un vCPU de servidor no se
parece a una Pi y no debí extrapolar así.

A 36 fps (@416) se abre algo no contemplado: analizar casi en continuo en vez de
solo en ventanas de movimiento, lo que daría velocidades de caída mucho más fiables.

### Paso 1 — grid temporal (implementado, medición en curso)

`_historial` guarda los fotogramas recientes por cámara en la capa 1 —la única que
los ve todos— y `_tira_temporal()` compone el actual con los de ~1 s y ~2 s antes
en una sola imagen, en orden cronológico, solo para eventos abstractos.

**Limitación detectada al revisar el propio código**: la capa de movimiento emite
ráfagas de 10 fotogramas en 3 s y luego calla 15 s. Si el candidato que llega al
VLM cae al principio de una ráfaga, los anteriores son de 15 s atrás y la
tolerancia los rechaza con razón, cayendo al fotograma suelto. **La tira se compone
solo parte de las veces, y no sé en qué proporción.**

**Error de método, el mismo del ciclo 4**: `_tira_temporal` no deja rastro cuando
funciona, solo cuando falla. Así que si el recall no sube no podré distinguir "la
idea no sirve" de "la idea no llegó a probarse". En el paso 2, el registro de
ejecución va ANTES que la lógica.

### Paso 1 — resultado y corrección

**Primera medición**: 6/6 negativos, 1/4 positivos. **Idéntico al ciclo 6**: la
aguja no se movió.

Pero la cifra no valía para decidir. Al instrumentar las tres salidas de la
función y lanzar una prueba dirigida sobre `disturbios_calle`:

| salida | veces |
|---|---|
| compuesta con 3 fotogramas (116 KB) | 1 |
| fotogramas no espaciados | 2 |
| sin historial suficiente | 1 |

**La tira solo se componía 1 de cada 4 veces.** El paso 1 estaba INFRA-PROBADO,
no refutado. De haber concluido con la matriz en la mano, habría descartado una
idea correcta por un fallo de implementación, y habría quedado escrito aquí como
"el grid temporal no funciona", envenenando decisiones futuras.

**Causa**: exigir fotogramas a exactamente 1 s y 2 s no encaja con cómo emite la
capa 0 —ráfagas de 10 fotogramas en 3 s y luego 15 s de silencio—. Un candidato
al principio de una ráfaga solo tiene detrás fotogramas de 15 s atrás. Dentro de
una ráfaga, en cambio, hay uno cada ~0,33 s: material de sobra. Lo que sobraba
era la rigidez.

**Corrección desplegada**: selección adaptativa. Se cogen los fotogramas del mismo
evento (ventana de 5 s) y se reparten por el rango realmente disponible, exigiendo
solo un rango mínimo de 0,5 s. Verificado con cadencias reales: compone con media
ráfaga o más; sigue rechazando fotogramas casi simultáneos o aislados.

**Señal prometedora, aún sin confirmar**: con la única tira que sí se compuso, las
respuestas del VLM cambiaron de tono —de *"no hay ninguna pelea"* a *"personas en
actitudes agresivas"*, *"enfrentándose verbalmente"*, *"ambiente de desorden"*—.
Percibe la tensión, que antes ni mencionaba. Es **una sola muestra**: no concluye
nada, pero justifica volver a medir.

**Estado al cerrar la sesión**: la verificación de la tira adaptativa quedó SIN
completar. El worker tardó >14 min en arrancar (instalación de `ultralytics`) y se
apagó todo antes de obtener el dato.

### Lo primero al retomar

1. **Repetir la prueba dirigida** sobre `disturbios_calle` con la tira adaptativa
   ya desplegada: medir qué porcentaje se compone y si el VLM confirma. Es el dato
   que decide si el paso 1 vale o se descarta.
2. **Hornear `ultralytics` en el AMI**. Ya rompió un worker (ciclo 4) y ahora
   retrasó una verificación 14 minutos. Cada arranque de cámara lo paga.
3. Solo entonces, el **paso 2** (seguimiento + regla temporal de caídas), con el
   registro de ejecución escrito ANTES que la lógica.

---

## Ciclo 8 — medirse contra la vara común (UCF-Crime)

**Pregunta**: ¿cuánto vale la cascada comparada con la literatura? Hasta ahora el
banco medía 10 escenas propias; ningún número era comparable con ningún paper,
porque la cascada emite EVENTOS y los papers puntúan FOTOGRAMAS.

### Lo que se montó

- Los 290 vídeos de test de UCF-Crime con sus anotaciones temporales oficiales.
  Bajados por **peticiones de rango HTTP** sobre los zips del espejo: **7,99 GB
  en vez de los 29,8 GB** del dataset completo.
- `benchmark_ucf.py`: modo de puntuación continua que reutiliza `MotionDetector`
  y `ClipScorer` tal cual están desplegados, sin VLM (130.000 segmentos por
  Bedrock sería inasumible). Reanudable por vídeo.
- `analizar.py`: AUC por fotograma, FAR/hora, latencia hasta la alerta y desglose
  por clase.

**Licencia**: UCF-Crime es de uso investigador, no libre. Se evalúa con él pero
**no entra en el testbench versionado**; los clips viven en S3 y el arranque de la
cámara los recoge de ahí. Los clips de `CLIPS.md` siguen siendo solo Wikimedia.

### Hallazgo 1 — `MIN_MOTION_AREA` depende de la resolución (fallo en producción)

`MIN_MOTION_AREA` son **500 píxeles absolutos**, calibrados para 1080p. Traducido
a fracción del encuadre:

| resolución | 500 px equivalen a |
|---|---|
| 1920x1080 (producción) | 0,0965% |
| 960x540 (el mock) | 0,39% — **4x más duro** |
| 320x240 (UCF-Crime) | 2,60% — **27x más duro** |

A 320x240 MOG2 disparaba en el **0,4%** de los fotogramas y la cascada quedaba
muda: habría dado un AUC de azar por configuración, no por método.

Dos consecuencias que van más allá del benchmark:

1. **Cualquier cámara de cliente por debajo de 1080p tiene la puerta de
   movimiento silenciosamente endurecida.** No falla, no avisa: simplemente
   deja de alertar. Merece arreglo en producción (expresar el umbral como
   fracción del área, no en píxeles).
2. **El propio banco de pruebas corre 4x más duro que producción**, porque el
   mock sirve a 960x540. Las comparaciones entre ciclos siguen siendo válidas
   (misma escala en todos), pero los números absolutos son pesimistas.

### Hallazgo 2 — casi cuelo un no-op silencioso

`pad_square_roi` recibe `min_size` como argumento **por defecto**, y los defaults
se enlazan al definir la función: reasignar la global no le llega. El reescalado
del ROI no habría hecho nada y el número se habría reportado como bueno. Se
detectó porque los tiempos no cuadraban (670 fps con CLIP supuestamente activo es
imposible). **Tercera vez que aparece este patrón**: verificar que el código nuevo
se EJECUTA, no solo que no peta.

### Hallazgo 3 — `mock_camera.py` daba por lista una cámara que no lo estaba

La espera busca `"camara lista"` en `testcam/status.log`, **y ese log sobrevive a
la instancia que lo escribió**. Dio por buena una instancia de 47 segundos leyendo
el log del día anterior. Arreglado en origen: borra el log antes de arrancar.
Ya había mordido una vez y entonces se esquivó en el flujo en vez de arreglarse
en la herramienta.

### Hallazgo 4 — `register` deja la cámara vigilando solo `persona`

`cmd_register` fija `detection_blacklist: ["persona"]`. La caja de análisis puntúa
todos los conceptos pero luego **filtra por los de la cámara**, así que con esa
configuración ningún robo ni pelea puede alertar jamás. Cualquier medición de
robos/violencia hecha sin reconfigurar antes no es interpretable.

### RESULTADO — 290/290 vídeos, 1.111.808 fotogramas (10,3 h), cero errores

```
capa 0 sola (movimiento MOG2)      AUC = 58,35%
capa 1 sola (margen CLIP)          AUC = 55,49%
cascada capa 0 + capa 1            AUC = 55,67%
solo escenarios objetivo           AUC = 56,58%
```

**El azar es 50%.** La parte de la cascada que corre en CPU está 5,7 puntos por
encima de una moneda. La literatura: LAVAD 80,28 · Flashback 87,29 · π-VAD 90,33.

**CLIP puntúa PEOR que el movimiento a secas** (55,49 vs 58,35) y combinarlos no
mejora a ninguno. No es ruido: es un millón de fotogramas. Coincide con lo ya
medido por otra vía (margen 0,089 para "persona" en una montaña vacía contra
0,063 de mediana en aciertos reales).

Al añadir los 150 normales TODO bajó (capa 0: 60,76 -> 58,35). Más metraje sin
incidente, más ocasiones de equivocarse: el parcial era optimista.

| punto de operación (umbral 0.02) | |
|---|---|
| falsas alarmas / hora de vídeo normal | **23.219** |
| recall por fotograma | 28,6% |
| tramos de incidente tocados | **136/156 (87%)** |
| latencia media hasta la alerta | **2,5 s** |

Como PUERTA la cascada funciona: deja pasar el 87% de los incidentes, y en 2,5 s.
Lo que no hace es FILTRAR. Eso reordena el diagnóstico del coste: **la cascada no
abarata Mimir eligiendo bien, lo abarata estrangulando el caudal con el throttle
de 6 s.** Quien discrimina es el VLM, y por eso cuesta lo que cuesta.

Por clase (n pequeño en casi todas las del producto; el 86,14% de Assault es con
TRES vídeos y no sostiene ninguna conclusión):

| clase | n | AUC |
|---|---|---|
| Assault | 3 | 86,14% |
| RoadAccidents | 23 | 71,57% |
| Vandalism | 5 | 65,03% |
| Explosion | 21 | 63,27% |
| Fighting | 5 | 58,71% |
| Robbery | 5 | 58,30% |
| Burglary | 13 | 58,07% |
| Shooting | 23 | 58,16% |
| Shoplifting | 21 | 55,27% |
| Stealing | 5 | 44,10% |
| Arrest | 5 | 44,53% |

Lo más fiable de los escenarios objetivo es Shoplifting (n=21): **55,27%**.

### Confirmación independiente en vivo (mock, camino de producción completo)

Seis escenas de UCF-Crime por RTSP -> worker -> CLIP -> VLM -> DynamoDB, con la
cámara reconfigurada a `robos, violencia, caidas, persona`:

| escena | esperado | detectado | veredicto |
|---|---|---|---|
| ucf_asalto | violencia | persona x6 | PERDIDO |
| ucf_pelea | violencia | persona x5 | PERDIDO |
| ucf_robo | robos | persona x4 | PERDIDO |
| ucf_hurto_tienda | robos | persona x6 | PERDIDO |
| ucf_robo_vehiculo | robos | persona x7 | PERDIDO |
| ucf_normal | (nada) | (nada) | ACIERTO |

**1/6.** Cero eventos de robo o violencia en cinco escenas de robo y violencia.
Todo llega `via=clip` y todo dice `persona`.

Dos mediciones independientes —un millón de fotogramas offline y el sistema
desplegado en vivo— dicen lo mismo: **CLIP detecta que hay gente, no qué pasa.**

### Salvedades

1. **No mide el sistema completo**: no hay VLM en el benchmark. El 55,67%
   describe la parte barata, no el veredicto final.
2. **Muestras pequeñas** en casi todas las clases del producto (n entre 2 y 5).
3. **Es UCF-Crime, no cámaras reales**, y el audit cross-dataset (0,704 -> 0,499)
   dice que estos números tampoco transfieren.

### Qué cambia del plan

El paso 3 era "sustituir CLIP por YOLO-World". Con este dato la pregunta correcta
es más dura: **¿por qué sigue CLIP en el camino de decisión?** Cuesta CPU, cuesta
latencia y mide peor que el MOG2 que ya corre. Antes de sustituirlo, medir la
cascada SIN él: es una línea base que nunca se ha medido y que ahora sé calcular.

## Ciclo 9 — YOLOE sustituye a CLIP en la capa 1

### 1. Estado de partida

El ciclo 8 dejó medido que CLIP restaba: 55,49% de AUC contra 58,35% del
movimiento a secas. La propuesta era usar CLIP y YOLO-World a la vez y
reconciliar; se investigó y se midió antes de construir nada.

### 2. ¿Sirve CLIP si se le pregunta mejor?

Se probaron sondas de atributo concretas en lugar de la pregunta abstracta
("¿hay personas?", "¿posición de golpe?", "¿alguien en el suelo?", "¿cosas en
el piso?", "¿encapuchados?") sobre 119 vídeos y 400.622 fotogramas:

| sonda | AUC | correlación con "personas" |
|---|---|---|
| objetos_suelo | 62,48% | r=0,75 REDUNDANTE |
| personas | 61,35% | — |
| mov (gratis) | 60,93% | referencia |
| golpe | 60,55% | r=0,69 REDUNDANTE |
| caida | 55,81% | r=-0,06 independiente |
| encapuchado | 51,47% | r=0,24 independiente pero al azar |

Combinación entrenada sobre vídeos distintos (partición POR VÍDEO), 12
particiones aleatorias: **+1,08 ± 1,16 puntos** sobre el movimiento. La ventaja
es menor que su desviación; no se distingue de cero.

**Por qué falla, medido**: las dos sondas que mejor puntúan son redundantes con
"¿hay personas?". Los prompts de `golpe` contienen la palabra *person* en todos.
CLIP responde a los SUSTANTIVOS, no al verbo — el efecto bolsa de palabras
documentado (ARO: 63% en atributos, 59% en relaciones; Winoground: azar).

Conclusión: refinar los prompts no salva a CLIP. Se descarta como votante.

### 3. Investigación

- **YOLOE** supera a YOLO-World en +10/+11 AP en LVIS y va 1,4x más rápido.
  Usa MobileCLIP por dentro: misma semántica abierta, pero OBLIGADO a localizar.
- **AnyAnomaly** (WACV 2026, con código): pasarle contexto estructurado al VLM
  en vez del fotograma crudo da +9,88% y +13,65% sobre su línea base.
- **Paza**: filtro puramente geométrico (dwell 3s, proximidad rho=0,3),
  precisión 89,5%, especificidad 92,8%. Sin semántica.

### 4. Medición en la instancia real (`m7i-flex.large`, fotogramas 1920x1080)

| motor | ms/frame | fps |
|---|---|---|
| yoloe-11s (texto) | **242,5** | **4,12** |
| CLIP ViT-B/32 (3 ROIs) | 304,5 | 3,28 |

YOLOE es un 20% MÁS RÁPIDO que CLIP. El tiempo es plano con la resolución
porque ultralytics reescala internamente a 640.

**El modo sin prompt (4.585 clases) es inservible**: sobre CCTV devuelve
"bamboo forest", "pilgrim", "chicken coop", "magician". Solo se usa el modo con
vocabulario corto y dirigido, que además va al doble de velocidad.

### 5. Validación offline ANTES de desplegar (119 vídeos, 400.622 fotogramas)

```
CLIP (produccion)        58,71%
YOLOE (nuevo)            62,62%    +3,91 sobre CLIP   +1,69 sobre movimiento
movimiento MOG2          60,93%
```

Primer motor de capa 1 que supera al movimiento. Por clase:

| clase | n | YOLOE | CLIP | delta |
|---|---|---|---|---|
| Assault | 3 | 92,65% | 86,14% | +6,51 |
| Robbery | 5 | 66,79% | 58,30% | **+8,49** |
| Vandalism | 5 | 64,16% | 65,03% | -0,87 |
| Burglary | 13 | 61,50% | 58,07% | +3,43 |
| Shoplifting | 21 | 58,18% | 55,27% | +2,91 |
| Fighting | 5 | 53,11% | 58,71% | **-5,60** |
| Stealing | 5 | 51,25% | 44,10% | +7,15 |
| Abuse | 2 | 48,96% | 57,30% | -8,34 |

Fighting empeora por causa identificada: la regla de violencia exige detectar
DOS personas y a 320x240 YOLOE suele encontrar una.

### 6. Diseño desplegado

`cascade/yoloe_scorer.py` es un reemplazo directo de `ClipScorer` (mismos
métodos y atributo `labels`). YOLOE detecta OBJETOS con caja y confianza; los
eventos abstractos NO se inventan: se derivan de evidencia geométrica y su papel
es disparar la CONSULTA al VLM, no la alerta.

- violencia: >=2 personas Y próximas (rho=0,3, el parámetro de Paza)
- robos: persona + objeto de interés (cuchillo/arma/mochila)
- caidas: las sigue resolviendo `pose.py` por geometría, que es mejor

Conmutable sin desplegar: `HEIMDALL_MOTOR=clip` restaura el motor anterior.
CLIP sigue en el código, solo cede el puesto.

Umbrales: son CONFIANZAS de detección (0-1), no márgenes de CLIP (0,02-0,15).
Reutilizar los de CLIP habría disparado con todo.

### 7. Aprendizajes

- **Un backtick en un comentario tumbó el despliegue.** El UserData de
  HeimdalManager vive en una plantilla de JS; escribir `mobileclip_blt.ts` con
  comillas invertidas cerró la cadena. Lo detectó `node --check` antes de subir.
- **Verificar que el motor nuevo se USA, no solo que no falla.** Se comprobó en
  el log del worker (`[capa1] motor = YoloeScorer`) que no cayó al fallback de
  CLIP en silencio. Es el tercer ciclo con esta misma lección.
- **La medición cara va antes del despliegue, no después.** El +3,91 se supo con
  el sistema aún en producción con CLIP.

### Pendientes de este ciclo

- El +1,69 sobre movimiento es UNA partición, sin barras de error (a diferencia
  del experimento de sondas, que sí las tiene). Repetir con varias.
- La regla de robo (persona + objeto) es una heurística sin validar por
  separado: puede subir el gasto de Mimir. Medir la tasa de alertas.
- Arreglar la regla de violencia para no exigir 2 detecciones de persona.

---

### Pendiente

- Línea base sin CLIP (movimiento -> VLM directo), en banco y en benchmark.
- Medir el rendimiento de CLIP en `m7i-flex.large` (el AUC no depende del
  hardware, el coste sí).
- Arreglar `MIN_MOTION_AREA` en producción como fracción del área, no en píxeles.

---

## Ciclo 9 — YOLOE en la capa 1: mejor midiendo, regresión en producción

### 1. Estado de partida

Ciclo 8 dejó medido que CLIP resta: 55,49% de AUC frente al 58,35% del
movimiento a secas. La pregunta pasó de "¿con qué sustituimos a CLIP?" a "¿por
qué sigue CLIP en el camino de decisión?".

### 2. Investigación

- **CLIP es bolsa de palabras**, documentado: ARO da 63% en atributos y 59% en
  relaciones; en Winoground los VLM contrastivos rinden cerca del azar en
  composicionalidad.
- **YOLOE** supera a YOLO-World en +10/+11 AP sobre LVIS y va 1,4x más rápido.
  Usa MobileCLIP por dentro, así que hereda la semántica abierta, pero está
  OBLIGADO a localizar: no puede afirmar "persona" sin dibujar una caja.
- **AnyAnomaly** (WACV 2026, con código) valida pasarle contexto estructurado al
  VLM en vez del fotograma pelado: +9,88% y +13,65% sobre su línea base.

### 3. ¿Sirve CLIP como verificador de atributos? MEDIDO: casi no

Se probó la idea de preguntarle cosas concretas en vez de "¿esto es un robo?".
119 vídeos, 400.622 fotogramas:

| sonda | AUC | correlación con "personas" |
|---|---|---|
| objetos_suelo | 62,48% | r=0,75 REDUNDANTE |
| personas | 61,35% | — |
| movimiento (gratis) | 60,93% | referencia |
| golpe | 60,55% | r=0,69 REDUNDANTE |
| caida | 55,81% | r=-0,06 independiente |
| encapuchado | 51,47% | r=0,24 pero al azar |

Combinación con partición POR VÍDEO y 12 semillas: **+1,08 ± 1,16** sobre el
movimiento. La ventaja es menor que su desviación.

**Por qué falla**: las dos sondas que mejor puntúan son redundantes con
"¿hay personas?". Todos los prompts de `golpe` contienen la palabra *person*:
CLIP responde a los SUSTANTIVOS, no al verbo. El efecto bolsa de palabras
reproducido sobre material propio.

**Lección de diseño**: si se usan sondas, que pregunten por objetos y atributos
de objeto, nunca por acciones ni relaciones.

### 4. YOLOE: medido antes de desplegar

Coste en `m7i-flex.large` con fotogramas reales de producción (1920x1080):

| motor | ms/frame | fps |
|---|---|---|
| yoloe-11s (vocabulario dirigido) | 242,5 | 4,12 |
| CLIP ViT-B/32 (3 ROIs) | 304,5 | 3,28 |

**YOLOE es un 20% más rápido que el CLIP al que sustituye.**

Calidad, 119 vídeos / 400.622 fotogramas:

```
CLIP (produccion)        58,71%
YOLOE (nuevo)            62,62%   +3,91 sobre CLIP, +1,69 sobre movimiento
movimiento MOG2          60,93%
```

Primer motor que ocupa ese puesto y supera al movimiento. Gana en Robbery
(+8,49), Stealing (+7,15), Burglary (+3,43), Shoplifting (+2,91); Assault 92,65%.
Pierde en Fighting (-5,60).

**Vocabulario dirigido, nunca abierto**: el modo sin prompt (4.585 clases)
devuelve ruido de escena sobre CCTV ("bamboo forest", "pilgrim", "chicken coop")
y va al doble de lento.

### 5. LA REGRESIÓN — y por qué fue culpa del método, no del modelo

Desplegado YOLOE por defecto. Un worker de cliente lanzado 18 min después se
llevó el código nuevo. Con `cuchillo` activo y un cuchillo real sostenido en la
mano a distancia de terraza: `label=None score=0.000` en bucle. **Cámara muda.**

No fue fallo de carga — el modelo arrancó con el vocabulario de cuchillos
correcto. YOLOE simplemente no ve un objeto fino y pequeño a esa distancia, que
es el caso peor de un detector. CLIP no necesitaba localizar: le bastaba con que
el recorte se pareciera a un cuchillo.

**El error real**: la validación se hizo sobre UCF-Crime, que NO tiene una sola
anotación de cuchillos ni armas, y aun así se cambió el motor para TODOS los
conceptos. Se midieron tres eventos abstractos y se extrapoló a un cuarto
concepto sin ninguna medición que lo respaldara.

**Revertido** a CLIP por defecto. `HEIMDALL_MOTOR=yoloe` lo reactiva.

**Regla que sale de aquí**: no cambiar el motor de un concepto sin material
anotado que contenga ESE concepto. Un AUC global no autoriza un cambio por clase.

### 6. Prueba end-to-end con YOLOE: 1/6, igual que CLIP

Los cinco positivos dieron solo `persona`. Causa identificada: las reglas de
evento abstracto quedaron demasiado conservadoras — `robos` exige detectar un
arma (umbral 0,30 contra un suelo de 0,15) y `violencia` exige DOS personas
próximas, y a baja resolución YOLOE detecta una.

El AUC mide ordenamiento, no si algo cruza el umbral. YOLOE ordena mejor y aun
así no alerta.

### 7. COSTE: el diagnóstico estaba desactualizado

Medido en CloudWatch, no estimado:

| componente | $/cámara/mes | |
|---|---|---|
| EC2 `m7i-flex.large` 24/7 | **55,19** | **95%** |
| EBS 12-20 GB gp3 | ~1,20 | |
| Bedrock (Mimir) | **1,10-1,70** | 1.424 invocaciones/día en TODA la cuenta |
| S3 (0,18 GB, 502 objetos) | ~0,01 | |
| DynamoDB + SQS + Lambda | <1 | |
| **Total** | **~58** | |

**Bedrock es el 3% del coste, no el 97%.** Llevamos ciclos optimizando el gasto
del VLM; el objetivo correcto ahora es otro.

**Hallazgo mayor**: la topología de Fase B (caja de movimiento + caja de análisis
compartidas) NO está activa. El log dice `mode=local`: cada cámara levanta su
propia `m7i-flex.large`. Por eso se pagan $55/cámara en vez de los $19-25 del
precio por paquete. Activarla es pasar de ~$58 a ~$20 por cámara **sin tocar una
línea de la detección**. Es el mayor ahorro disponible.

### 8. Pendientes

- Activar la topología compartida (mayor ahorro, sin riesgo de detección).
- Barrer umbrales de las reglas de YOLOE con los datos ya capturados, en vez de
  fijarlos a ojo.
- Motor POR CONCEPTO: YOLOE en eventos abstractos, CLIP en objetos pequeños.
  La regresión demuestra que un motor único para todo es la decisión equivocada.
- `MIN_MOTION_AREA` como fracción del área, no en píxeles absolutos.
- Instrumentar el worker: su log entero era una línea.
- SNS `MonthlySpendLimit` sigue bloqueado por denegación de IAM.

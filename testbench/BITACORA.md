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
8. **Anotar las fuentes usadas**, con enlace, en la sección «Fuentes» del final.

## Regla de fuentes (obligatoria desde 2026-08-11)

1. **Toda afirmación sobre el estado del arte va con fuente enlazada.** Nada de
   memoria, nada de «se sabe que». Si no hay enlace, no se escribe.
2. **Toda fuente usada se anota** en «Fuentes», aunque acabe descartada: saber qué
   ya se miró evita repetir la búsqueda.
3. **Lo no verificado se marca como tal**, con `⚠ sin verificar`. Una cifra sin
   fuente es una hipótesis, no un dato, y no puede sostener una decisión.
4. Aplica también a lo ya escrito: las cifras de ciclos anteriores que entraron sin
   fuente están marcadas abajo y pendientes de verificar.

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

## Revisión de literatura — qué de esto ya estaba publicado (2026-08-11)

Revisión hecha antes del ciclo 4, para saber qué de lo que creíamos hallazgo
propio ya existe. **El resultado es mayormente negativo y conviene tenerlo escrito.**

### 1. «Preguntar por el rastro, no por el acto» — no es nuestro

El principio del ciclo 2 (preguntar por el ESTADO persistente en vez del acto en
curso) está publicado en tres frentes:

- **Caídas: prior art de décadas.** La detección por postura de yacimiento
  («lying pose», «floor occupancy») es un subcampo establecido. La formulación
  publicada es casi literal a la nuestra: *«unlike detecting the falling event,
  the clinical requirement after a fall is frequently not the precise
  classification of the dynamic event, but the reliable detection of the post-fall
  state: an individual lying quasi-statically on the floor»* ([F1], [F2]).
- **Robos e incendios: anticipado ~10 meses.** ASK-HINT [F3] usa prompts como
  *«Do you see forced entry, vandalism, or deliberate fire?»* y
  *«Is there evidence of weapons, force, or law enforcement?»*, que es el paso 1
  del ciclo 4 palabra por palabra.
- **Matiz que sobrevive:** ASK-HINT se autodescribe como *action-centric* y mezcla
  prompts de acto y de rastro **sin teorizar la distinción**, y declara en sus
  limitaciones que *«ignore temporal modeling»*. Nadie ha aislado el principio
  como principio. Es contribución de encuadre, no descubrimiento.

### 2. La arquitectura en cascada — tampoco es nuestra

- **Cerberus** [F4]: cascada de dos etapas, filtro de movimiento barato → VLM caro,
  explícitamente por coste y tiempo real. 151,79× de aceleración, 97,2% de
  precisión relativa, 57,68 fps en una L40S.
- **SlowFastVAD** [F5]: detector simple + VLM con RAG. **MemoVAD** [F6]: variante
  eficiente en edge.
- **Zero-Shot Retail Theft Detection** [F7]: zero-shot + orquestación de modelos de
  visión como alternativa coste-efectiva a sistemas entrenados. Es nuestro mismo
  planteamiento. **Lectura obligatoria antes de escribir nada.**

**Y el precedente de fondo es mucho más antiguo que Cerberus.** Existe un subcampo
entero de *video analytics systems* (VLDB, OSDI, NSDI, SIGCOMM) que lleva desde 2017
haciendo cascadas para abaratar inferencia sobre vídeo:

- **NoScope** (VLDB 2017) [L5]: cascada de modelos especializados + detectores de
  diferencia antes del modelo caro. **265-15.500× de aceleración.** Es nuestra
  arquitectura, nueve años antes y con evaluación más rigurosa.
- **Focus** (OSDI 2018) [L6]: reduce el coste de ingesta **48× de media, hasta 92×**.
- **Reducto**: filtrado en la propia cámara. **Ekya** (NSDI 2022) [L7]: aprendizaje
  continuo en servidores edge. **RedunCut** [L8]: muestreo dirigido por medición.

### 3. El coste en moneda — CORRECCIÓN: tampoco es un hueco

Esto se afirmó primero al revés y estaba mal. **El coste no es un terreno sin pisar:
es la métrica central de ese subcampo.**

- NoScope [L5] se plantea explícitamente como reducción de coste («up to three orders
  of magnitude»), con el marco de que un detector en tiempo real exige una GPU de
  4.000 USD.
- **Microsoft Rocket** [L1][L2] es una plataforma edge+cloud **desplegada sobre
  cámaras de tráfico reales con la ciudad de Bellevue**, con informe de caso [L3] y
  código abierto [L4]. El TCO es objetivo declarado del proyecto.
- Hay cifras en dólares publicadas [L10]: ~6,80 USD/cámara/mes en L4 propia, ~24 en
  L4 alquilada, y una API cloud por minuto sale ~180× más cara que GPU alquilada.

**Conclusión honesta: no hay ninguna ventaja estructural nuestra.** Los laboratorios
tienen despliegues reales a escala de ciudad, bancos verificados y contabilidad de
costes más seria. Lo único propio es haberlo medido sobre nuestra pila, que sirve
como trabajo de grado y no como contribución.

### Líneas base para comparar (esto sí resuelve un agujero)

La revisión aporta lo que faltaba para poder compararnos con alguien:

| método | UCF-Crime (AUC) | XD-Violence (AUC) |
|---|---|---|
| ASK-HINT [F3] | 89,83% | 90,31% |
| VERA [F9] | 86,55% | 88,26% |

Benchmarks estándar del área: **UCF-Crime**, **XD-Violence**, **RWF-2000** [F11],
Avenue, SHTech, NWPU-Campus. Nuestro banco de 10 escenas propias no es comparable
con ninguno de ellos hasta que corramos sobre uno estándar.

### Aprendizaje

Convergencia independiente: el principio del rastro se dedujo aquí leyendo los
propios logs del VLM, sin conocer ASK-HINT, publicado dos meses antes. No da
novedad. Sí valida el método de trabajo — y es exactamente el tipo de cosa que
esta bitácora existe para registrar.

**Error de método que causó cuatro ciclos de redescubrimiento:** el paso 3 del
protocolo («investigar el estado del arte») se ejecutó buscando **por problema**
(violencia, caídas) y no **por método** (VLM en cascada, video analytics systems).
Por eso se encontró CUE-Net y YOLOv8-Pose, y no NoScope, Cerberus ni ASK-HINT.
**A partir de ahora, buscar siempre por ambos ejes.**

---

## La escalera — dónde está SkyEye respecto al techo

Mapa del campo, de abajo arriba. Sirve para situarse honestamente y para no
confundir «no lo he hecho» con «no está hecho».

| nivel | qué es | referencias |
|---|---|---|
| **0 — SkyEye** | cascada reactiva; VLM sobre fotogramas sueltos | [L5] (2017) |
| **1** | razonamiento temporal explícito sobre la secuencia | [T1][T2][T3][T4][T5] |
| **2** | streaming always-on con memoria de largo plazo | [S1]-[S6] |
| **3** | **anticipación**: predecir la anomalía antes de que ocurra | [A1] |
| **4** | cambio de sustrato: sensor de eventos + cómputo neuromórfico | [N1]-[N4] |

Notas que conviene no olvidar:

- **Nivel 1** incluye *HiProbe-VAD* [T5], que lee los **estados ocultos** del VLM en
  vez de su respuesta de texto. Mecanismo completamente distinto al nuestro.
- **Nivel 2 contiene nuestra mejor idea, hecha mejor.** *StreamMind* [S6] usa una
  *Cognition Gate* ligera que vigila el flujo y solo dispara el LLM pesado cuando
  pasa algo relevante, **a 100 FPS**. Es la puerta de personas + el regulador de
  caudal del ciclo 3, en el mismo concepto.
- **Nivel 3** [A1] cambia el planteamiento, no lo mejora: aprende la cinemática
  normal con un modelo del mundo tipo JEPA y avisa de la trayectoria **antes** de que
  haya alguien en el suelo.
- **Nivel 4 disuelve el problema en lugar de optimizarlo.** Un sensor de eventos
  (DVS) solo genera datos cuando cambia la luminancia [N1][N2]: la capa 0 entera pasa
  al silicio. Con chips neuromórficos en el propio sensor (Loihi 2, NorthPole) el
  consumo baja a **milivatios** [N3]. Toda esta tesis trata de no pagar por mirar
  fotogramas aburridos; una cámara de eventos no los produce.

**Barra actual en los benchmarks del área** (⚠ fuente de industria, no revisada por
pares [B1]): BERT+RTFM ~98,5% AUC en ShanghaiTech; AnomalyCLIP ~90,32% en UCF-Crime;
VadCLIP++ ~90,5% AP en XD-Violence. Orientativo, no citable.

---

## Fuentes

Todas verificadas el 2026-08-11 salvo lo marcado `⚠ sin verificar`.

### Estado del arte — VLM y detección de anomalías en vídeo

- **[F3] ASK-HINT** — *Unlocking Vision-Language Models for Video Anomaly Detection
  via Fine-Grained Prompting*, arXiv 2510.02155 (oct. 2025).
  https://arxiv.org/html/2510.02155v1
- **[F4] Cerberus** — *Real-Time Video Anomaly Detection via Cascaded Vision-Language
  Models*, arXiv 2510.16290 (oct. 2025). https://arxiv.org/pdf/2510.16290
- **[F5] SlowFastVAD** — arXiv 2504.10320. https://arxiv.org/pdf/2504.10320
- **[F6] MemoVAD** — arXiv 2606.07669. https://arxiv.org/pdf/2606.07669
- **[F7] Zero-Shot Retail Theft Detection via Orchestrated Vision Models** —
  arXiv 2604.14846. https://arxiv.org/pdf/2604.14846
- **[F9] VERA** — *Explainable Video Anomaly Detection via Verbalized Learning of
  Vision-Language Models*, arXiv 2412.01095. https://arxiv.org/pdf/2412.01095
- **[F10] Video Anomaly Detection in 10 Years: A Survey and Outlook** —
  arXiv 2405.19387. https://arxiv.org/pdf/2405.19387
- **[F13] RedunCut** — *Measurement-Driven Sampling and Accuracy Performance
  Modeling for Low-Cost Live Video Analytics*, arXiv 2512.24386.
  https://arxiv.org/pdf/2512.24386

### Detección de caídas por estado (no por evento)

- **[F1]** *Reliable Quasi-Static Post-Fall Floor-Occupancy Detection Using Low-Cost
  Millimetre-Wave Radar*, arXiv 2601.17710. https://arxiv.org/pdf/2601.17710
- **[F2]** *Fall Detection System-Based Posture-Recognition for Indoor Environments*,
  PMC8321307. https://pmc.ncbi.nlm.nih.gov/articles/PMC8321307/

### Violencia — línea base citada desde el ciclo 1

- **[F11] CUE-Net** — Senadeera et al., *Violence Detection Video Analytics with
  Spatial Cropping, Enhanced UniformerV2 and Modified Efficient Additive
  Attention*, CVPRW 2024 / arXiv 2404.18952. **94,00% en RWF-2000**, 99,50% en
  RLVS. https://arxiv.org/abs/2404.18952
  → confirma la cifra que el ciclo 1 citaba sin fuente.
- **[F12] RWF-2000** — *An Open Large Scale Video Database for Violence Detection*,
  arXiv 1911.05913. https://arxiv.org/abs/1911.05913

### Coste (industria, no revisado por pares)

- **[F8]** Fora Soft, *Video Analytics Cost per Camera: Edge vs Cloud Math*.
  https://www.forasoft.com/learn/video-surveillance/articles-vms/economics-of-analytics-bandwidth-compute-storage

### Marco metodológico (para el trabajo de grado)

- **[F14] Hevner, March, Park & Ram (2004)** — *Design Science in Information Systems
  Research*, MIS Quarterly 28(1), 75-105. El ciclo build-evaluate de esta bitácora
  es DSR. https://aisel.aisnet.org/misq/vol28/iss1/6/
- **[F15] Manual de Frascati 2015 (OCDE)** — tipología I+D: investigación básica /
  aplicada / desarrollo experimental. SkyEye es **desarrollo experimental**
  («trabajos sistemáticos fundamentados en los conocimientos existentes […]
  dirigidos a […] mejorar considerablemente los que ya existen»).
  https://www.oecd.org/content/dam/oecd/es/publications/reports/2015/10/frascati-manual-2015_g1g57dcb/9789264310681-es.pdf

### Video analytics systems — el precedente de fondo (subcampo VLDB/OSDI/NSDI)

- **[L1]** Microsoft Rocket, plataforma edge+cloud — https://www.microsoft.com/en-us/research/video/microsoft-rocket-hybrid-edge-cloud-video-analytics-platform/
- **[L2]** Microsoft Rocket for Live Video Analytics — https://www.microsoft.com/en-us/research/project/live-video-analytics/
- **[L3]** Traffic Video Analytics — Case Study Report (ciudad de Bellevue) — https://www.microsoft.com/en-us/research/publication/traffic-video-analytics-case-study-report/
- **[L4]** Rocket, código abierto — https://github.com/microsoft/Microsoft-Rocket-Video-Analytics-Platform
- **[L5]** **NoScope**, VLDB 2017 — https://www.vldb.org/pvldb/vol10/p1586-kang.pdf
- **[L6]** **Focus**, OSDI 2018 — https://dl.acm.org/doi/10.1145/3301293.3302366
- **[L7]** **Ekya**, NSDI 2022 — https://www.usenix.org/system/files/nsdi22spring_prepub_bhardwaj.pdf
- **[L8]** RedunCut, arXiv 2512.24386 — https://arxiv.org/html/2512.24386
- **[L9]** Empowering Agentic Video Analytics Systems with VLMs, arXiv 2505.00254 — https://arxiv.org/html/2505.00254v3
- **[L10]** ⚠ industria — Video Analytics Cost per Camera: Edge vs Cloud Math — https://www.forasoft.com/learn/video-surveillance/articles-vms/economics-of-analytics-bandwidth-compute-storage

### Nivel 1 — razonamiento temporal

- **[T1]** Chain-of-Frames, CVPR 2026 — https://openaccess.thecvf.com/content/CVPR2026/html/Ghazanfari_Chain-of-Frames_Advancing_Video_Understanding_in_Multimodal_LLMs_via_Frame-Aware_Reasoning_CVPR_2026_paper.html
- **[T2]** MOSS-ChatV, arXiv 2509.21113 — https://arxiv.org/pdf/2509.21113
- **[T3]** TimeLogic Challenge @ CVPR 2026, arXiv 2606.01631 — https://arxiv.org/html/2606.01631v1
- **[T4]** Vad-R1, arXiv 2505.19877 — https://arxiv.org/pdf/2505.19877
- **[T5]** HiProbe-VAD (probing de estados ocultos), arXiv 2507.17394 — https://arxiv.org/pdf/2507.17394

### Nivel 2 — streaming always-on con memoria

- **[S1]** LiveStarPro, arXiv 2606.17798 — https://arxiv.org/pdf/2606.17798
- **[S2]** Visual Agentic Memory, arXiv 2605.16481 — https://arxiv.org/pdf/2605.16481
- **[S3]** ViCoStream (>100 FPS), arXiv 2606.19849 — https://arxiv.org/pdf/2606.19849
- **[S4]** Dispider, arXiv 2501.03218 — https://arxiv.org/pdf/2501.03218
- **[S5]** Awesome-Streaming-Video-Understanding (lista curada) — https://github.com/sotayang/Awesome-Streaming-Video-Understanding
- **[S6]** **StreamMind** (Cognition Gate, 100 FPS) — vía [S5] y https://www.emergentmind.com/topics/real-time-streaming-video-llm

### Nivel 3 — anticipación

- **[A1]** Latent Clarity: World-Model Kinematics for Video Anomaly Anticipation, arXiv 2607.03558 (jul. 2026) — https://arxiv.org/pdf/2607.03558

### Nivel 4 — sensor de eventos y cómputo neuromórfico

- **[N1]** Event Cameras 2026 (Prophesee, Sony, iniVation) — https://internet-pros.com/blog/event-cameras-neuromorphic-vision-sensors-2026/
- **[N2]** US12413869B2 — Low-power always-on event-based vision sensor — https://patents.google.com/patent/US12413869
- **[N3]** Neuromorphic Event-Based Vision Sensor Patents 2026 — https://www.patsnap.com/resources/blog/rd-blog/neuromorphic-event-based-vision-sensor-patents-2026/
- **[N4]** EvAn: Neuromorphic Event-Based Sparse Anomaly Detection — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8358807/

### Problemas declarados abiertos por el campo

- **[E1]** Efficient Video Intelligence in 2026 — https://v-chandra.github.io/efficient-video-intelligence/
  → *"open-set 'show me anything anomalous' remains unsolved"*; y la brecha de
  evaluación en producción («closed-loop methodology lags benchmark accuracy by a
  wide margin»). Abierto **para el campo entero**, no un hueco reservado a nadie.
- **[E2]** From Benchmarks to Reality: VAND 3.0 Challenge, arXiv 2509.17615 — https://arxiv.org/abs/2509.17615
- **[B1]** ⚠ industria — Top Anomaly Detection Models for Video Surveillance (2026) — https://www.forasoft.com/blog/article/anomaly-detection-models-video-surveillance

### ⚠ Pendientes de verificar — citadas en ciclos anteriores sin fuente

Entraron de memoria y **no pueden sostener una decisión** hasta tener enlace:

| ciclo | afirmación | estado |
|---|---|---|
| 1 | global+local supera a local solo (45,94 vs 42,88) | ⚠ sin verificar |
| 1 | GCN sobre esqueleto, ~98% de precisión en caídas | ⚠ sin verificar |
| 1 | muestreo adaptativo: ~53% menos llamadas | ⚠ sin verificar |
| 3 | YOLOv8-Pose: 92-98% de precisión en caídas sin VLM | ⚠ sin verificar |

### Afirmaciones retiradas por falsas

Se anotan para no volver a cometerlas. Las tres se hicieron sin verificar y las tres
inflaban la posición del proyecto:

| afirmación | realidad | fuente |
|---|---|---|
| «nadie publica el coste» | el coste es la métrica central del subcampo | [L5][L6][L1] |
| «el principio del rastro no está explorado» | publicado en caídas y en VLM-VAD | [F1][F2][F3] |
| «los laboratorios no tienen despliegues ni bancos» | despliegues a escala de ciudad | [L1][L3] |

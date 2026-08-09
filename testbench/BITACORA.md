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

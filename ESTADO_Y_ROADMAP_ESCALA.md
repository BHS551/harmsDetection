# Estado del sistema y hoja de ruta para escalar a muchos clientes

Informe de una sesión programada de investigación (2026-09-06). Cubre los siete
repositorios de SkyEye: `harmsDetection`, `HeimdalManager`, `harmsDetectionLandingUi`,
`ListDevices`, `ListDetections`, `StoreDevice`, `StoreDetection`. Vive aquí, junto a
`testbench/BITACORA.md`, por la misma razón que esa bitácora: para que la próxima
sesión (humana o agente) parta de datos medidos, no de memoria.

No se ha tocado código de producción. Es un diagnóstico y una propuesta priorizada.

---

## 1. Accuracy — el riesgo más grande del sistema

`testbench/BITACORA.md` documenta 4 ciclos de mejora ya ejecutados, con métricas
medidas (no estimadas) en cada uno. Resumen del estado actual (Ciclo 3, 2026-08-10):

| Métrica | Ciclo 0 | Ciclo 3 (actual) |
|---|---|---|
| Negativos limpios (no debe alertar) | 4/4 | **6/6** |
| Incidentes reales detectados | 0/4 | **1/4** (verificado: 2 verdaderos positivos en `pelea_calle`) |
| Llamadas VLM/hora | 1.877 | 234 (−88%) |
| Coste Bedrock/cámara/mes | 120,02 USD | 14,99 USD (−87%) |

**Lectura honesta:** los falsos negativos (no detectar un incidente real) están
mucho más controlados que al inicio, y el coste ha bajado un orden de magnitud.
Pero **la tasa de detección real de incidentes sigue siendo baja** — el sistema
alerta bien cuando la escena es inequívoca, pero omite `caida_escaleras`,
`disturbios_saqueo` y captura solo parcialmente `disturbios_calle`. Para un
producto de seguridad ("SkyEye": detecta caídas, robos, violencia), un falso
negativo es el peor tipo de error posible — es exactamente lo contrario de lo
que el cliente paga.

La propia bitácora ya identificó la causa raíz y el plan (Ciclo 4, no ejecutado
aún):

1. **YOLOv8-Pose para caídas** (92–98% de precisión sin VLM) — la palanca más
   grande pendiente. Sacaría toda la clase "caídas" de manos del VLM, que hoy
   depende de juzgar posturas en fotogramas sueltos, un enfoque con techo bajo.
2. **Calibrar `clear_margin` para "persona"** — es la etiqueta que más tráfico
   genera hacia el VLM; hay margen para resolver casos claros sin gastar en VLM.
3. **Flexibilizar la verdad de referencia del banco de pruebas** — la matriz de
   evaluación penaliza aciertos legítimos (una alerta de "caídas" durante un
   disturbio se cuenta como fallo porque la verdad de referencia solo admite
   "robos"/"violencia" para esa escena).
4. **Ampliar el banco de pruebas.** 10 escenas de Wikimedia Commons es un punto
   de partida razonable para validar cambios rápido y barato, pero es
   insuficiente para certificar el sistema ante clientes reales: faltan
   variación nocturna/infrarrojo, cámaras de mala calidad, ángulos oblicuos,
   oclusión parcial, y sobre todo **más de un caso de cada clase de incidente**
   (hoy hay un único ejemplo verificado de pelea real y ninguno de robo real
   confirmado).

**Recomendación de prioridad:** antes de vender a "muchos clientes", cerrar el
Ciclo 4 (YOLOv8-Pose) y ampliar el banco de pruebas con casos reales por
vertical de cliente (retail para robos, industrial para accidentes laborales,
vía pública para violencia). Vender con la tasa de detección actual sin
comunicarla es un riesgo legal y reputacional; vender comunicándola con
transparencia (y un roadmap de mejora visible) es defendible.

---

## 2. Salud e infraestructura por servicio

Los siete servicios están, en general, **muy bien construidos para su etapa
actual** — aislamiento por clave (no por filtro) en DynamoDB, secretos RTSP que
nunca tocan disco ni logs, CORS con allow-list, TTL de 30 días en eventos,
verificación de plan server-side e idempotente, clasificación correcta de
errores 401 vs 500. Esto ya está resuelto y no necesita rehacerse.

Lo que falta, ordenado por lo que más importa antes de escalar:

### 2.1 Sin observabilidad ni alertas de infraestructura
Ningún repositorio tiene alarmas de CloudWatch, dashboards, ni un servicio de
error-tracking (Sentry o equivalente) integrado en el código de aplicación. Hoy
la única forma de saber que algo falla es que un cliente se queje. Con un
cliente esto es tolerable; con cientos, no. Antes de escalar:
- Alarmas sobre DLQ no vacías (tier 0→1→2 en `harmsDetection`/`create_queues.py`
  ya crean las DLQ, pero nada notifica cuando reciben mensajes).
- Alarma sobre invocaciones fallidas y duración de las Lambdas (`HeimdalManager`,
  `StoreDetection`, `StoreDevice`, `ListDetections`, `ListDevices`).
- Un dashboard de negocio: cámaras activas, alertas/hora, coste Bedrock/hora en
  tiempo real (ya existe la lógica en `testbench/costes.py`; falta que corra
  como job continuo en vez de a demanda).

### 2.2 Bug de coste conocido y no corregido: la caja de análisis no se autotermina si la cámara está caída
Documentado en `testbench/README.md`, sección "Comportamiento ante cortes": el
worker reconecta solo cuando la cámara vuelve (bien), pero **no se autotermina
mientras la cámara está inalcanzable**, así que una `m7i-flex.large` puede
facturar horas sin producir nada. Con un cliente es ruido en la factura; con
muchos clientes con túneles ngrok inestables, es un sumidero de margen que
crece linealmente con la base de clientes. Es una corrección acotada
(`common.py`, umbral de auto-terminación) y de alto ROI.

### 2.3 Cuello de botella de la topología compartida (Fase B)
`HeimdalManager` documenta la topología de ahorro: una caja de movimiento
barata por muchas cámaras, y **una sola caja de análisis compartida** (CLIP +
VLM) para todas ellas, escalada verticalmente (un tamaño de instancia fijo,
`m7i-flex.large` por defecto) y apagada tras 2h de inactividad. Esto funciona
bien para una decena de cámaras. No está diseñado para que un cliente grande
(o la suma de muchos clientes concurrentes) sature esa única caja: no hay
autoscaling horizontal de la caja de análisis, ni partición por cliente/región.
Antes de vender un plan "muchas cámaras" o crecer la base total, hace falta:
- Medir el techo real de cámaras concurrentes que una caja de análisis soporta
  sin degradar la latencia del VLM (el rate-limit de Bedrock es compartido).
- Definir cuándo se lanza una segunda caja de análisis (por número de cámaras,
  por cliente, por región) y qué la enruta.

### 2.4 Cero CI/CD y cero tests automatizados fuera de `harmsDetection`
`harmsDetection` tiene `testbench/`, `test.py`, `cascade/integration_test.py` y
`cascade/camera_sim_test.py` — es, con diferencia, el repositorio con mejor
disciplina de pruebas. Los otros seis (`HeimdalManager`, `ListDevices`,
`ListDetections`, `StoreDevice`, `StoreDetection`, `harmsDetectionLandingUi`) no
tienen ningún test automatizado ni pipeline de CI (`.github/workflows` no
existe en ninguno). El despliegue documentado en cada README es manual
(`zip -r function.zip . && aws lambda update-function-code`). Con 5 Lambdas
desplegadas a mano, el riesgo de desplegar la versión equivocada, u olvidar un
paso, crece con cada cliente nuevo que dependa de que esos servicios no se
rompan. Antes de escalar el equipo o la frecuencia de cambios: al menos smoke
tests de cada endpoint (auth 401, CORS, shape de la respuesta) y un pipeline
que despliegue automáticamente al hacer merge a `main`.

### 2.5 Límite conocido y ya documentado: conteo de cámaras por navegador
`harmsDetectionLandingUi/README.md` ya señala que `lib/monitoring.ts` cuenta
cámaras monitorizadas en `localStorage` (por navegador, no por cuenta). Está
correctamente mitigado — `HeimdalManager` re-valida por tag de EC2 antes de
lanzar nada — así que no es un riesgo de seguridad ni de facturación, solo una
UX que puede confundir a un cliente con varios dispositivos. Baja prioridad,
pero vale la pena arreglarlo antes de que el volumen de soporte crezca.

### 2.6 Pasarela de pago por defecto es PayU en modo sandbox
`.env.example` fija `PAYU_TEST=1` por defecto — correcto para desarrollo, pero
hay que verificar explícitamente, como parte del checklist de lanzamiento, que
el entorno de producción tiene `PAYU_TEST=0` y las credenciales reales, y que
existe alguna alerta si un webhook de pago falla silenciosamente (hoy no hay
evidencia de reintentos ni de alertas sobre webhooks de Stripe/PayU fallidos).

---

## 3. Lo que ya está resuelto y no hay que reabrir

Para que la próxima sesión no reinvente esto:

- Aislamiento multi-tenant por clave (GSI `owner-index`), no por filtro, en
  `ListDevices`/`ListDetections`.
- Credenciales RTSP nunca en disco/logs/user-data: viven en Secrets Manager,
  referenciadas por id (`StoreDevice`, `HeimdalManager`).
- Verificación de plan server-side, idempotente, con clasificación de errores
  401 vs 500 correcta en las cinco Lambdas.
- TTL de 30 días en eventos de detección; los dispositivos no expiran.
- CORS restringido a allow-list en todos los servicios (ya no hay `*`).
- Cascada de detección con control de gasto en tres puntos distintos
  (`VLM_MIN_INTERVAL`, `ALERT_COOLDOWN`, `NOTIFY_COOLDOWN`), con DLQ en las
  colas SQS.
- Catálogo de planes centralizado (`lib/plans.ts`) y activación de plan solo
  por webhook verificado, nunca por redirect.

---

## 4. Hoja de ruta priorizada para "estar listo para muchos clientes"

**Antes de cualquier campaña de adquisición agresiva (bloqueante):**
1. Cerrar Ciclo 4 de accuracy (YOLOv8-Pose para caídas) y ampliar el banco de
   pruebas con casos reales por vertical — es el riesgo de producto/reputación
   más alto identificado.
2. Corregir la auto-terminación de la caja de análisis cuando la cámara está
   inalcanzable — es un sumidero de margen que escala con la base de clientes.
3. Alarmas mínimas: DLQ no vacía, error rate de Lambdas, fallo de webhook de
   pago. Sin esto, el primer incidente a escala se detecta por queja de
   cliente, no por monitoreo propio.

**Antes de crecer el volumen de cámaras concurrentes por cliente/región:**
4. Medir el techo de la caja de análisis compartida y definir la regla de
   cuándo lanzar una segunda.
5. Smoke tests + CI/CD básico en las cinco Lambdas y en la landing UI, para que
   desplegar deje de ser un paso manual de alto riesgo.

**Antes de escalar soporte/ventas:**
6. Arreglar el conteo de cámaras por navegador (UX, no seguridad).
7. Checklist de lanzamiento de pagos: confirmar `PAYU_TEST=0` en producción y
   alertas sobre webhooks fallidos.

---

## 5. Para la sesión de adquisición de clientes (contexto compartido)

Si estás leyendo esto desde la tarea programada de "cómo conseguir clientes":
ten en cuenta lo siguiente antes de proponer mensajes de marketing o canales de
venta:

- **No prometer una tasa de detección que el sistema no sostiene hoy.** La
  detección verificada de incidentes reales es baja (ver sección 1); es
  defendible venderlo como "en mejora activa y medida, con arquitectura de
  cascada que ya redujo el coste 87% sin perder cobertura de negativos", pero
  no como "detección fiable de robos/violencia" sin matices.
  Un plan de "detección temprana + revisión humana de evidencia" (el sistema ya
  guarda el frame en S3 y lo muestra en el panel) es una propuesta de valor
  honesta mientras el Ciclo 4 no cierre.
- **El límite de cámaras por plan ya existe como dato de producto**
  (`lib/plans.ts`, `maxCameras`) y se aplica server-side — es un buen ancla
  para tiers de precio si se está diseñando la oferta.
- **La topología de ahorro (motion box + analysis box compartida) es lo que
  hace viable el margen por cámara** — cualquier propuesta comercial de
  "precio por cámara" agresivo debe conocer el coste real medido en
  `testbench/costes.py` (14,99 USD/cámara/mes de Bedrock en el mejor ciclo,
  más ~5-9 USD de cómputo compartido) para no fijar un precio que no cubra el
  coste variable.
- **Mercado objetivo actual: LatAm** — `PAYMENT_PROVIDER=payu` es el default,
  pensado para esa región; Stripe existe como alternativa ya cableada para
  otros mercados si se decide expandir.

---

*Generado por una sesión programada de investigación técnica. Ningún código de
producción fue modificado; todos los hallazgos citados provienen de
`testbench/BITACORA.md`, `testbench/RESULTADOS.txt`, `testbench/README.md` y
los README de cada repositorio, todos ya versionados.*

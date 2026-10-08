# Auditoría de preparación para escalar (2026-10-08)

Revisión automática de los 7 repos de SkyEye. Contexto compartido para la tarea
de adquisición de clientes: aquí está lo que el producto puede y no puede
prometer hoy.

## 1. Accuracy (el riesgo nº 1)

Fuente: `testbench/BITACORA.md`, `RESULTADOS.txt` (ciclo 3, 10 escenas reales).

- Negativos limpios: **6/6**. Positivos detectados: **1/4**.
- `caida_escaleras`, `disturbios_saqueo`, `disturbios_calle` → no detectados
  como incidente. Sin recall demostrado, no se debe vender "detecta robos/peleas".
- El banco son 10 clips; estadísticamente no sirve para afirmar nada. Faltan
  ≥100 clips por clase, con intervalos de confianza.
- La alerta `persona` se dispara en casi toda escena con gente (obra normal,
  peatones, intersección...). Si se notifica al cliente, es spam; si no, es
  solo un gate interno. Definir explícitamente.
- La verdad de referencia es rígida y penaliza aciertos (caída en disturbio).
- Pendiente de la propia bitácora: YOLOv8-Pose para caídas (92–98% sin VLM),
  calibrar `clear_margin` de `persona`, flexibilizar ground truth.

**Propuesta:** (a) ampliar banco a ≥300 clips etiquetados y publicar
precision/recall por clase; (b) métrica de falsas alarmas por cámara-día,
que es lo que el cliente percibe; (c) feedback "alerta correcta / falsa" en la
consola para generar datos de forma continua; (d) hasta tener cifras, vender
como "piloto" y solo las clases con recall medido (persona/cuchillo).

## 2. Unit economics (bloqueante para "muchos clientes")

Fuente: `testbench/costes.py`, bitácora, `harmsDetectionLandingUi/src/lib/plans.ts`.

- Plan `cam5`: **USD 50/mes por 5 cámaras** (USD 10/cámara).
- Coste medido: worker dedicado ≈ **USD 86–101/cámara/mes**; Fase B (motion box
  compartido + analysis box) ≈ **USD 116 por 5 cámaras** (ciclo 3).
- → Margen **negativo** incluso en el mejor caso (50 − 116 ≈ −66/mes por cliente),
  sin contar SMS (USD 0,06 c/u), soporte ni comisiones de pasarela.
- Palancas: subir precio (≈USD 25–40/cámara), pose para caídas, calibrar
  `persona`, subir `VLM_MIN_INTERVAL`, apagar analysis box en horas sin
  movimiento, Savings Plans/Spot para el motion box.

## 3. Salud / operabilidad

- **Cero CI y cero tests unitarios** en los 7 repos (no hay `.github/workflows`).
  Solo `integration_test.py`/`camera_sim_test.py` que requieren AWS/Bedrock real.
- **Sin observabilidad**: no hay alarmas CloudWatch, métricas de latencia
  cámara→alerta, ni panel de salud de workers (heartbeat existe en `common.py`
  pero nadie lo vigila). Un worker caído = cliente desprotegido sin aviso.
- **Sin IaC**: Lambdas, DynamoDB, SQS, launch templates creados a mano
  (`deploy-worker.sh`). No reproducible, no hay staging.
- **Código duplicado**: `ensureFirebase`, CORS y verificación de token copiados
  en 6 Lambdas; `PLAN_CAMERAS` duplicado en `subscriptions.mjs` y `plans.ts`.
  Extraer una Lambda layer / paquete común.
- **Seguridad menor**: CORS `*` en la constante `headers` más regex
  `*.vercel.app` (cualquier app de Vercel obtiene origen reflejado; la auth
  Bearer limita el daño, pero restringir a dominios propios). `context.json` en
  el repo contiene una `rtsp_path` con credenciales (`admin551:...@ngrok`):
  rotar y eliminar del historial. `yolov5s.pt` binario versionado.
- **Planes**: plan `cam1` USD 1 solo-admin; revisar que nunca se exponga.
- **Escala**: `subscriptions.mjs` usa `ScanCommand` para `?all=1` (no escala);
  una EC2 dedicada por cámara no escala en coste ni en cuota de vCPU de la
  cuenta; falta límite/backpressure global en el analysis box (1 caja para
  todos los clientes = punto único de fallo y de latencia).
- **Multi-tenant**: tablas compartidas filtradas por `owner_uid` (bien, vía
  índice); falta aislamiento de cuota por cliente (un cliente ruidoso agota
  Bedrock para todos) → presupuesto VLM por cliente.
- **Cumplimiento**: video de terceros → política de privacidad, retención de
  frames en S3 (lifecycle), consentimiento, GDPR/Ley 1581 (Colombia).

## 4. Hoja de ruta propuesta (orden)

1. Unit economics: nuevo pricing + pose para caídas (viabilidad).
2. Banco de ≥300 clips + métricas publicadas + botón de feedback.
3. CI (lint + tests unitarios de `vision`/`tiers` con mocks) en todos los repos.
4. Alarmas de heartbeat/DLQ/Bedrock throttling + status page.
5. IaC (CDK/Terraform) + staging.
6. Layer común para Lambdas; presupuesto VLM por cliente; lifecycle S3.
7. Onboarding: la fricción actual es dar una URL RTSP; para clientes reales
   se necesita guía por marca de cámara o agente/bridge local (ONVIF/túnel).

## 5. Para la tarea de captación de clientes

- Segmento recomendado: pilotos pagados con 1–3 cámaras en negocios donde el
  valor está en `persona` fuera de horario / robos nocturnos (la puerta de
  personas ahorra más con locales vacíos), no en "detección de peleas".
- No prometer recall de violencia/caídas hasta publicar métricas.
- Precio mínimo a validar: ≥ USD 25–40/cámara/mes con la arquitectura actual.
- Pedir a cada piloto 1–2 semanas de grabaciones etiquetadas = banco real.

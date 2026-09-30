# SkyEye — estado y preparación para muchos clientes (2026-09-30)

Informe de la rutina programada de revisión. Sirve también de **canal de
coordinación con la tarea de adquisición de clientes**: la última sección dice
qué puede prometer comercialmente hoy y qué no.

Fuente: código de los 7 repos + `testbench/BITACORA.md` (ciclos 0–3). No se
pudo medir salud en vivo (sin acceso a AWS/CloudWatch desde la rutina), así que
"health" es inferido del código y de las mediciones del banco.

## 1. Accuracy (medido, banco de 10 escenas, ciclo 3)

| Métrica | Valor | Lectura |
|---|---|---|
| Negativos limpios | 6/6 | Pocas falsas alarmas en escenas normales |
| Positivos detectados | 1/4 (`pelea_calle`) | **Recall de incidentes bajo** |
| Falsos positivos observados | 1 `caidas` (alguien sentado, ciclo 2) | Endurecer prompt |

- Tamaño de muestra ridículo (4 positivos, clips de Wikimedia, no CCTV real).
  **No hay ninguna cifra de precisión/recall defendible ante un cliente.**
- No hay positivos reales de "accidente laboral" ni de robo; `caida_escaleras`
  es cine mudo. La clase "robos" nunca se ha validado en positivo.
- Limitación de fondo: el VLM juzga fotogramas sueltos; caídas/peleas son
  temporales. Palanca pendiente: YOLOv8-Pose para caídas (92–98 % en literatura).
- La etiqueta `persona` genera casi todo el tráfico y casi todas las alertas en
  el banco (incluso en "naturaleza_vacia": 1 alerta). Es ruido para el cliente
  salvo que la quiera explícitamente.

## 2. Coste (medido)

| | USD/cámara/mes |
|---|---|
| Dedicada (1 instancia/cámara) | ~86 |
| Motion box compartida, 10 cám/caja | ~19 |
| Paquete 5 cám (Fase B) | ~116 total |

Bedrock ~15 USD/cámara/mes tras ciclo 3. La topología compartida es la única
viable a escala; la dedicada solo para pilotos. Margen real depende del precio
del plan (ver `plans.ts` en LandingUi) — verificar que precio > ~20 USD/cám.
con holgura para SMS y soporte.

## 3. Salud / riesgos de escala detectados en el código

Alto:
1. **Sin CI, sin tests automáticos en ningún repo** (no hay `.github/`). Los
   Lambdas y el worker se despliegan a mano (zip/`deploy-worker.sh`). Un fallo
   tipo "falta `import os`" (ya ocurrió, ciclo 1) llega a producción.
2. **Motion box única por diseño** (t3.small, N cámaras): punto único de fallo
   y de ruido entre clientes. Falta sharding por N cámaras, health-check y
   reinicio automático; hoy el heartbeat existe pero no se ve alarma/dashboard.
3. **Aislamiento multi-tenant en una sola analysis box**: un cliente ruidoso
   consume VLM de todos. El regulador es por (cámara, etiqueta), no hay cuota
   por cliente ni presupuesto de gasto por cuenta/plan (riesgo de coste).
4. **Sin observabilidad de negocio**: no hay métricas de uptime por cámara, tasa
   de falsas alarmas reportada por usuarios, ni feedback 👍/👎 en alertas. Sin
   esto no se puede mejorar accuracy con datos reales.
5. **Camera onboarding**: RTSP por túnel del cliente ("tunnel refused" en el
   testbench). Es el mayor freno de adopción: requiere soporte manual.

Medio:
- Cuotas AWS (vCPU EC2, Bedrock TPM/RPM de Nova Lite) sin solicitar/documentar;
  a 100+ cámaras el throttling de Bedrock degrada detecciones en silencio.
- SMS caro (0.06 USD): depender de email/push por defecto, SMS opcional/plan alto.
- DLQ creadas pero sin alarma ni proceso de reproceso.
- Retención 30 d eventos / 7 d frames: correcta; documentar política de
  privacidad/consentimiento (video de personas; Habeas Data en Colombia).
- Múltiples Lambdas con lógica CORS/auth duplicada: extraer módulo compartido.
- Código legado en `harmsDetection` (rtsp_*.py, `heimdall-eye.py`, `yolov5s.pt`)
  ensucia el repo; mover a `legacy/`.

Bien resuelto (no tocar): plan check server-side, start idempotente,
credenciales RTSP en Secrets Manager, CORS allow-list, TTL, Query en vez de Scan,
firestore.rules deny-by-default, regulador de caudal de VLM.

## 4. Mejoras propuestas (orden por impacto/esfuerzo)

1. **CI mínimo** (GitHub Actions): lint + `node --check` en Lambdas, `py_compile`
   + `integration_test.py` con LocalQueue y mocks en el worker, `next build` en la UI.
2. **Observabilidad**: alarmas CloudWatch (heartbeat perdido por cámara, DLQ > 0,
   errores Lambda, Bedrock throttling, gasto diario) → SNS al admin
   (`notifyAdmin.mjs` ya existe).
3. **Feedback en alertas** (botón "falsa alarma" en `/console/detections`) →
   guardar en DynamoDB → construir set de evaluación con datos reales. Es el
   activo que más valor da a largo plazo.
4. **Valores por defecto seguros**: desactivar `persona` como alerta por defecto;
   plantillas de configuración por caso de uso (local cerrado de noche,
   almacén, hogar).
5. **Presupuesto de VLM por cliente/plan** (contador diario en DynamoDB; al
   superarlo, degradar a solo CLIP/movimiento y avisar).
6. **Ciclo 4 de accuracy**: YOLOv8-Pose para caídas; calibrar `clear_margin`
   de `persona`; ampliar banco a ≥30 escenas incluyendo grabaciones propias
   (robo escenificado, accidente) con consentimiento; verdad de referencia
   flexible. Publicar precisión/recall con intervalo de confianza solo cuando n
   sea suficiente.
7. **Sharding de motion box** (p. ej. máx. 10 cám/caja, autoscaling por
   cantidad de cámaras) + reinicio automático por ASG/health check.
8. **Onboarding de cámara**: asistente que pruebe RTSP antes de guardar
   (StoreDevice), y opción de cámara en la nube (P2P/ONVIF/relay) para reducir soporte.
9. Solicitar aumento de cuotas (vCPU, Bedrock) antes de la primera campaña.
10. Plan gratuito/piloto limitado (1 cámara, 7 días) con coste acotado por (5).

## 5. Para la tarea de captación de clientes (qué se puede prometer)

- **Sí, hoy**: detección de movimiento + presencia de personas con alertas
  (email/SMS), evidencia en imagen, consola web, planes con límite de cámaras.
  Ideal para **pilotos asistidos** (5–10 clientes) en locales cerrados fuera
  de horario (robo/intrusión) donde "hay una persona" ya es el evento — ahí el
  sistema brilla y el coste es mínimo.
- **No prometer aún**: "detecta peleas/robos/caídas con X % de precisión".
  Recall medido 1/4. Venderlo como "detección asistida, en mejora continua".
- **Segmento recomendado**: negocios pequeños (tiendas, bodegas, talleres,
  conjuntos) en Colombia, con cámaras IP existentes; pitch: alerta a
  WhatsApp/SMS de intrusión nocturna, sin DVR nuevo. Evitar de entrada
  hospitales/colegios (falsos negativos de caídas/violencia = riesgo legal).
- **Ritmo de adquisición seguro**: ≤10 cámaras/mes hasta tener CI, alarmas y
  feedback (puntos 1–3). Después, escalar con sharding y presupuestos (5, 7).
- Lo que la captación necesita de ingeniería: plan piloto (10), onboarding de
  cámara sin soporte (8), testimonios/datos reales (3).
- Preguntas abiertas para esa tarea: precio objetivo por cámara (¿>25 USD?),
  canal preferido (WhatsApp vs SMS), y qué segmento acepta alertas de "persona".

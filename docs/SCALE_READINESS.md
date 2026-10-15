# SkyEye — estado, accuracy, health y plan para escalar a muchos clientes
Fecha: 2026-10-15 (revisión automática; basada en los 7 repos y la BITACORA del testbench)

## Resumen
El sistema es **seguro y bien arquitecturado (auth, secretos, idempotencia, cascada de costes)** pero **todavía no está listo para muchos clientes** por tres razones, en orden de gravedad:
1. **Economía unitaria negativa**: coste medido vs. precio.
2. **Accuracy no demostrada**: 1/4 positivos en el banco; sin métrica de falsos positivos en entorno real.
3. **Sin observabilidad ni CI/tests automáticos** y worker "una instancia EC2 por cliente/cámara".

## 1. Economía (bloqueante para vender más)
Plan público `cam5`: 5 cámaras = **USD 50/mes** (USD 10/cámara). Coste medido (BITACORA ciclo 3):
- Worker dedicado: ~USD 86/cámara/mes.
- Fase B (motion box compartido + análisis bajo demanda): USD 115,62/mes por paquete de 5 cámaras (~23/cám) y 19,29/cám a 10 cámaras.
- No incluye SMS (USD 0,06 c/u, ~Colombia), soporte, pasarela de pago (PayU/Stripe ~3-5 %).
=> **Cada cliente cam5 pierde ~USD 65/mes** hoy. A más clientes, más pérdida.
Acciones:
- Subir precio y/o reducir cobertura: p.ej. USD 15-25/cámara o plan por horas/zonas; el coste de Bedrock (USD 15/cám) domina -> priorizar ciclo 4 (YOLOv8-Pose para caídas, calibrar `clear_margin` de `persona`, subir `VLM_MIN_INTERVAL`).
- Operar **solo Fase B** (compartida) y apagar workers dedicados; analysis box con autoscaling por profundidad de cola SQS en vez de una sola `m7i-flex.large`.
- Poner **tope de gasto por cliente** (llamadas VLM/día) y alarma de presupuesto AWS (Budgets) por cliente vía tags.
- Medir coste real con tráfico real (el banco subestima la puerta de personas; locales vacíos de noche ahorran más).

## 2. Accuracy
Banco (10 escenas Wikimedia): negativos limpios 6/6, positivos 1/4. Sólo 2 verdaderos positivos verificados a mano. Problemas:
- `naturaleza_vacia` genera 1 alerta de `persona` (falso positivo; la matriz no la cuenta porque `persona` no es "incidente").
- Ground truth demasiado rígida (aciertos legítimos contados como fallos).
- Fotograma suelto: el VLM ve el rastro y dice NO (disturbios). Sin contexto temporal.
- Muestra minúscula: no hay precisión/recall con intervalos de confianza.
Acciones:
- Construir un **dataset etiquetado de >=500 clips** (caídas, robos, violencia, armas, negativos difíciles) con CCTV real de pilotos con consentimiento + sintéticos; reportar precisión/recall/FP por cámara-día (la métrica comercial es **falsas alertas por cámara por día**).
- YOLOv8-Pose para caídas (92-98 % reportado) y detector de armas dedicado; VLM sólo como árbitro.
- VLM con 2-3 fotogramas (contexto temporal) y evaluar modelos mayores sólo en la cola de ambiguos.
- Bucle de feedback: botón "falsa alarma / correcta" en la consola -> guardar etiqueta -> reentrenar/calibrar umbrales por cliente.
- Hay `persona` como "alerta": decidir si es producto (modo "presencia") o si debe silenciarse; hoy genera SMS/ruido.

## 3. Health / operabilidad
Hallazgos:
- **No hay CI ni tests automáticos** en ningún repo (sólo `integration_test.py`/`camera_sim_test.py` manuales; ningún workflow). Lambdas se despliegan con zip manual.
- **No hay IaC** (launch template, tablas, colas, roles creados a mano) -> no reproducible ni auditable.
- Heartbeat existe (30 s) pero no se vio alarmado: faltan CloudWatch alarms/dashboards (workers caídos, DLQ > 0, latencia VLM, errores Bedrock, 5xx de Lambdas).
- Errores del VLM devuelven `None` y no alertan: una caída de Bedrock = cámara ciega sin aviso al cliente. Añadir estado "degradado" visible y notificación.
- Cámara/túnel caído: hay pruebas en testbench; falta notificación proactiva al cliente ("tu cámara está offline").
- Límites de cuenta AWS: vCPU de EC2 y cuotas Bedrock (TPM/RPM de Nova Lite) pueden frenar el escalado; pedir aumento temprano.
- `subscriptions.mjs` usa `ScanCommand` (admin) -> no escala con miles de usuarios; usar GSI/paginación.
- `localStorage` para contar cámaras (conocido; backend lo protege). Dos Lambdas duplicadas en `harmsDetection/` (`listDevices.mjs`, etc.) -> riesgo de deriva; eliminar copias.
- Repo `harmsDetection` arrastra scripts experimentales (`rtsp_*.py`), `yolov5s.pt`, `__pycache__`: limpiar.
- Seguridad: ya mejorada (CORS allow-list, sin logs de Authorization, secretos en Secrets Manager). Pendiente: rate limiting/WAF en API Gateway, rotación de `HEIMDALL_INTERNAL_SECRET`, auditoría de acceso a frames (URLs firmadas de corta duración), política de retención/privacidad (RGPD/Habeas Data Colombia: consentimiento, tratamiento de imágenes de terceros).
- Cámaras con IP privada requieren túnel del cliente: **onboarding es la fricción #1**. Ofrecer agente/gateway de un clic (contenedor o app) o integración con cloud de cámaras (Hikvision/Dahua/Tapo, ONVIF).

## 4. Prioridad propuesta (90 días)
| # | Acción | Por qué |
|---|---|---|
| 1 | Arreglar unit economics (precio + Fase B + Pose + límites) | Sin esto, crecer destruye caja |
| 2 | Dataset real + métrica FP/cámara/día + feedback en UI | Credibilidad y retención |
| 3 | CI (lint+tests) y despliegue IaC (CDK/Terraform) | Poder iterar sin romper |
| 4 | Alarmas, dashboard, estado de cámara offline/degradado | Confianza del cliente |
| 5 | Onboarding sin túnel (gateway/ONVIF) | Conversión |
| 6 | Cumplimiento legal y privacidad | Requisito B2B |
| 7 | Autoscaling del analysis box por cola SQS | Capacidad |

## 5. Para el otro agente (adquisición de clientes) — contexto compartido
- **No empujar adquisición masiva todavía**: cada cliente actual pierde dinero. Recomendar un **piloto controlado de 5-10 clientes pagos** (negocios pequeños en Colombia con cámaras IP: tiendas, bodegas, parqueaderos, obras) a precio nuevo, mientras se mide FP/cámara/día real.
- Mensaje de venta defendible hoy: "caídas y personas en zonas restringidas/fuera de horario" (la puerta de personas ahorra coste y es lo más fiable); evitar prometer detección de robos/peleas hasta tener métricas.
- Pedir al otro agente: segmentos con alta tolerancia a pilotos, canales (instaladores/integradores de CCTV, empresas de seguridad como reventa white-label) y disposición a pagar >= USD 15-25/cámara.
- Datos que yo necesito de él: precios de competidores (Verkada, Eagle Eye, Ring Business, locales), rango de precio aceptado, segmentos por tipo de cámara/túnel.

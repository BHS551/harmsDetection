# SkyEye — preparación para escalar (revisión 2026-10-05)

Revisión automática del estado de los 7 repos. Basada en README, BITACORA/RESULTADOS del testbench y lectura del código de los Lambdas. No se midió nada nuevo en producción.

## Veredicto
**No está listo para muchos clientes.** La arquitectura (cascada, DLQ, auth en servidor, aislamiento por GSI, secretos RTSP fuera de disco) es sólida. Hay tres bloqueos: la unidad económica, la precisión sin medir en datos reales, y la falta de operación (CI/alarmas/tests).

## 1. Economía por cámara (bloqueo nº1)
- Plan actual en `lib/plans.ts`: **5 cámaras = 50 USD/mes → 10 USD/cámara**.
- Coste medido (ciclo 3, `testbench/costes.py`): dedicada 85,97 USD/cám/mes; Fase B (motion box + analysis box compartido) 19,29 USD/cám a 10 cámaras y **115,62 USD por paquete de 5 (≈23 USD/cám)**.
- Es decir, con el precio actual **cada cámara pierde dinero** incluso en Fase B (sin contar SMS, Secrets Manager 0,40/cám, soporte, Stripe/PayU ~3-5%).
- Los números vienen de un banco con gente casi siempre presente; en locales reales con cámaras vacías la puerta de personas y MOG2 bajan el coste, pero **no está medido**.
- Acciones: (a) medir coste en 2-3 cámaras reales de clientes piloto durante 2 semanas; (b) subir precio / vender por cámara (≥25-30 USD) o limitar eventos VLM/día por plan; (c) ciclo 4: YOLOv8-Pose para caídas (92-98 % sin VLM), calibrar `clear_margin` de `persona`; (d) pasar de Nova Lite on-demand a batch/cache o filtrar más en Tier 1; (e) apagar analysis box con 0 cámaras activas (ya hay auto-apagado a 2 h idle: bajarlo).

## 2. Precisión (bloqueo nº2)
- Banco de 10 escenas: **negativos limpios 6/6, positivos reales 1/4** (la matriz marca 1/4; verificando frames a mano hay 2 TP reales en `pelea_calle`, 1 FP de `caidas` en disturbios). Muestra minúscula, sin intervalos de confianza.
- Falla conocida: el VLM describe indicios (puerta reventada, incendio, antidisturbios) y responde NO para `robos`/`violencia`; `caida_escaleras` no detectada; `persona` genera la mayor parte del tráfico a VLM.
- Verdad de referencia rígida penaliza aciertos (ciclo 3, punto 3).
- Acciones: construir dataset etiquetado ≥200 clips por clase (CLIPS.md/buscar_clips.py ya sirven de base) con métricas **precisión/recall por clase y falsas alarmas por cámara-día** (la métrica que el cliente siente); pilotos con feedback "alerta útil / falsa" en la consola para etiquetar gratis; reescribir `robos` por rastro (puerta forzada, escaparate roto); exigir postura horizontal en `caidas`; evaluar Nova Pro/Claude Haiku solo en ambiguos.

## 3. Salud operativa
Hallazgos concretos:
1. **Sin CI ni tests automatizados** en ningún repo (no hay `.github/`; solo `test.py` trivial y scripts de integración manuales). Añadir GitHub Actions: lint + tests unitarios de Lambdas (token falso, límites de plan, idempotencia) y `integration_test.py` con LocalQueue.
2. **Credencial RTSP en el historial git**: `harmsDetection/context.json` contiene `rtsp://admin551:...@4.tcp.ngrok.io` y existió en commits previos. Rotar la contraseña de esa cámara/ngrok, sacar el archivo del repo (`context.example.json`) y purgar historial si el repo es público.
3. **CORS**: `allowOrigin` en HeimdalManager (y réplicas en los otros 4 Lambdas) acepta *cualquier* `*.vercel.app`. Sin cookies el riesgo es bajo, pero conviene fijar solo el dominio propio. Además las constantes de cabecera siguen con `"*"`; unificar. El helper está copiado en 5 repos → paquete/layer común.
4. **Sin observabilidad documentada**: no hay alarmas CloudWatch (DLQ>0, worker sin heartbeat, errores 5xx de Lambdas, gasto Bedrock/día), ni dashboard de salud por cliente. Hay heartbeat en `common.py` pero nadie lo vigila. Añadir alarma de DLQ, de heartbeat ausente y un budget de AWS con corte.
5. **Límites de escala**: un EC2 por cámara dedicada no escala en coste ni en cuotas de EC2 (vCPU por región); la Fase B (motion box compartido) debe ser el default. Verificar cuotas, throttling de Bedrock (cuota Nova Lite) y límites de SMS de SNS (sandbox/gasto mensual) antes de onboarding masivo.
6. **Rate limiting / abuso**: StoreDetection acepta cualquier usuario autenticado sin tope por usuario; añadir throttling de API Gateway por clave/usuario y validación de tamaño de payload.
7. **Dockerfile obsoleto** (CUDA 11.6/Ubuntu 20.04, `python`) y múltiples scripts experimentales (`rtsp_*.py`, `multicore_*`) duplicados: mover a `legacy/`.
8. **Pagos**: PayU/Stripe con webhooks; falta documentado el manejo de impago → apagar workers (cron que cruce suscripciones vencidas con instancias vivas) y de reintentos idempotentes de webhook.
9. **Privacidad/legal**: se procesa vídeo de locales. Para clientes B2B hacen falta política de datos, retención (30 d TTL ya está en StoreDetection; frames en S3 deben tener lifecycle equivalente) y DPA.

## 4. Orden propuesto (4 semanas)
1. Semana 1: rotar credencial, CI mínimo, alarmas DLQ/heartbeat/budget, CORS estricto.
2. Semana 2: precio/planes alineados con el coste; medir coste real en 2-3 pilotos.
3. Semana 3: dataset y métricas por clase + feedback en consola; ciclo 4 (pose para caídas, prompt de robos).
4. Semana 4: prueba de carga (20-50 cámaras simuladas con `mock_camera.py`), cuotas AWS, runbook de incidentes.

## 5. Para coordinar con la tarea de captación de clientes
- Segmento ideal ahora: pocos clientes piloto (5-10 cámaras totales) con cámaras en horario sin gente (bodegas, locales cerrados de noche, obras): ahí la cascada es más barata y la precisión de `persona`/`robos` más útil que `violencia`.
- Evitar prometer: detección de peleas/robos en tiempo real con alta precisión; hoy es "alerta de persona/caída en zona vigilada" con revisión humana del frame.
- Pedir al cliente piloto: aceptar tarifa por cámara y feedback de falsas alarmas (alimenta el dataset).
- Precio mínimo viable orientativo: ≥25-30 USD/cámara/mes hasta que el coste baje de ~12 USD.
- Métricas que debe devolver captación: nº de leads, tipo de local, nº de cámaras, horario de actividad, tolerancia a falsas alarmas.

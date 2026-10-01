# SkyEye — estado, precisión, salud y camino a "muchos clientes" (2026-10-01)

Revisión estática de los 7 repos + datos del testbench (BITACORA.md, RESULTADOS.txt, costes.py). No se pudo medir producción (sin acceso a AWS/logs desde esta ejecución).

## 1. Veredicto corto
**No está listo para escalar clientes todavía.** Hay tres bloqueos: (a) la economía unitaria es negativa en el único plan público, (b) la precisión no está demostrada (1/4 positivos), (c) no hay CI, tests ni observabilidad del servicio. La arquitectura (cascada, seguridad server-side, idempotencia, secretos por referencia) es sólida; el problema es validación y negocio, no diseño.

## 2. Economía unitaria (bloqueo nº1)
- Plan público `cam5`: **USD 50/mes por 5 cámaras = USD 10/cámara** (plans.ts).
- Coste medido por cámara (BITACORA ciclo 3): dedicada **USD 85,97**; Fase B a 10 cám/caja **USD 19,29**; Fase B 5 cám (paquete) **USD 115,62 por 5 = ~23/cám**.
- Un cliente con 5 cámaras cuesta ~96–116 USD y paga 50 → **pérdida de ~46–66 USD/cliente/mes**; cada cliente nuevo empeora el resultado. Además la caja de análisis (m7i-flex.large) corre mientras haya movimiento y no hay topes de gasto por cliente.
- Mejoras: subir precio o cobrar por cámara (≥ USD 25–30/cám con Fase B); hacer Fase B el modo por defecto (hoy `start` lanza una instancia por cámara); YOLOv8-Pose para caídas y calibrar `clear_margin` de `persona` (ya identificados en ciclo 4: bajarían VLM más); topes mensuales de llamadas VLM por plan; CLIP en Spot/CPU compartida; alarma de presupuesto AWS (Budgets) con corte automático.

## 3. Precisión (bloqueo nº2)
Banco de 10 escenas (Wikimedia): negativos limpios 6/6, **positivos 1/4** (solo `pelea_calle`). `caida_escaleras` y `disturbios_saqueo` sin detectar. Notas:
- Banco minúsculo (10 clips, ~8 min): no hay intervalos de confianza; nada permite prometer una cifra de recall a un cliente.
- La verdad de referencia es rígida (penaliza aciertos legítimos, ya anotado en ciclo 3).
- Falta medir: falsos positivos por cámara/día en uso real (alertas "persona" salen en casi toda escena → riesgo de fatiga de alertas), latencia de extremo a extremo (evento → SMS), y recall por clase con clips de CCTV real (plano fijo, baja luz, IR, fisheye).
- Acciones: (1) banco ≥ 200 clips etiquetados incl. noche/IR; reportar precisión/recall por clase + IC; (2) permitir etiqueta de incidente equivalente; (3) pose-based falls; (4) modo piloto con 2–3 clientes reales con revisión humana de cada alerta durante 2–4 semanas para generar el dataset real; (5) "persona" no debe ser alerta por defecto (spam) — convertirla en eventos agregados o solo fuera de horario.

## 4. Salud / operación (bloqueo nº3)
Hallazgos:
- **Sin CI ni tests automatizados** en ningún repo (solo `test.py` de 6 líneas); los Lambdas se despliegan con `zip` manual. Un cambio roto llega a producción sin red.
- **Despliegue del worker por `aws s3 cp`** (deploy-worker.sh) sobre una ruta mutable, sin versionado/rollback; `Dockerfile` y `requirements_backup.txt`, `rtsp_*.py`, `multicore_*.py`, `heimdall-eye.py` (811 líneas) son código legado que confunde (cascade/ es lo vigente). `requirements.txt` fija torch 2.12.1 pero CLIP viene de `git+https` sin pin → build no reproducible.
- **Un EC2 por cámara en `start`** (MaxCount 1): sin cuotas de EC2 solicitadas, cada cliente de 5 cámaras = 5 instancias; el límite por defecto de vCPU bajo demanda se alcanzaría con pocos clientes. Pedir aumento de cuota y/o migrar a motion-box multi-cámara por defecto.
- Observabilidad: solo logs de CloudWatch por instancia y `costes.py` manual. Faltan: métricas (cámaras online, alertas/h, llamadas VLM/h, latencia, errores), alarmas (SNS ya existe vía notifyAdmin), dashboard, y detección de cámara caída visible al cliente (hay heartbeat; falta alerta "cámara sin señal").
- Seguridad/PII: video de personas → necesita política de privacidad, retención documentada (TTL 30d eventos / 7d frames ya implementado, bien), aviso legal (Colombia: Ley 1581 de habeas data; si hay clientes UE, RGPD), y cifrado/aislamiento de buckets por cliente. Los RTSP expuestos vía túnel (ngrok) son frágiles como método de onboarding.
- Pagos: PayU y Stripe conviven; activación manual de plan por admin (`subscriptions.mjs`) no escala → automatizar con webhook y estados `past_due/canceled` → corte automático de instancias.
- Rate limiting/abuso: no hay throttling por usuario en API Gateway ni cuota de `start`/`stop` más allá de `maxCameras`; añadir usage plans y WAF.
- Positivo: queries (no scans) con paginación, TTL en DynamoDB, DLQ en colas, auth server-side, idempotencia, errores clasificados.

## 5. Plan priorizado
1. **Semana 1 — dinero:** fijar precio/cámara por encima del coste; Fase B por defecto; AWS Budgets + tope de VLM por plan; pedir cuotas EC2.
2. **Semana 1–2 — red de seguridad:** GitHub Actions (lint + tests de Lambdas con mocks + `python -m compileall` y tests de `cascade/` con LocalQueue), despliegue de Lambdas por workflow, worker versionado (S3 con versión + tag en AMI/launch template), borrar código legado.
3. **Semana 2–4 — precisión real:** piloto con 2–3 clientes, etiquetar alertas, banco ≥200 clips, métricas por clase; YOLOv8-Pose para caídas; calibrar `persona`.
4. **Semana 3–4 — observabilidad:** dashboard CloudWatch + alarmas + status por cámara al cliente + SLA interno (detección→SMS < 30 s).
5. **Antes de vender en volumen:** política de privacidad/T&C, pagos automáticos con baja automática, onboarding de cámara sin ngrok (agente/VPN o ONVIF/cloud-camera).

## 6. Para coordinar con la tarea de adquisición de clientes
(No hay otra sesión alcanzable ahora; este documento es el canal — la otra tarea puede leerlo en la rama `claude/intelligent-meitner-vdd0uc`.)
- **No escalar adquisición pagada hasta corregir precio**: cada cliente a USD 50/5 cám pierde dinero.
- Nicho recomendable para pilotos: **negocios con pocas cámaras y alto dolor** (bodegas, tiendas, parqueaderos, obras) en Colombia, donde el eje "alerta con evidencia por SMS" vale más que "detectar pelea" (aún no fiable). Vender "alertas de persona fuera de horario + robos" primero; caídas/violencia como beta.
- Posicionamiento honesto: piloto de pago reducido con revisión humana; no prometer cifras de precisión hasta tener el banco real.
- Pedir a adquisición: lista de 10–20 prospectos piloto, precio objetivo que aceptan por cámara, y qué tipo de cámaras/cableado tienen (define el onboarding sin ngrok).

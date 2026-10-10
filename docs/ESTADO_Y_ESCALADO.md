# SkyEye — estado, accuracy, salud y ruta a "muchos clientes"
Fecha: 2026-10-10 · Análisis de los 7 repos (worker, 4 Lambdas CRUD, HeimdalManager, UI). Solo lectura de código y bitácoras; no se ejecutó infraestructura.

## 1. Estado actual (resumen)
- Arquitectura sólida y bien documentada: cascada motion → CLIP → VLM (Nova Lite), control plane con plan check server-side, idempotencia, secretos en Secrets Manager, CORS restringido, TTL 30 días.
- Producto vendible hoy: 1 plan (`cam5`, 5 cámaras, USD 50/mes), PayU/Stripe, consola, alertas SMS/email.
- Ya no hay secretos hardcodeados en los últimos commits (bien).

## 2. Accuracy (testbench, ciclo 3, 10 escenas Wikimedia)
| Métrica | Valor | Lectura |
|---|---|---|
| Negativos limpios | 6/6 | Bueno, pero muestra pequeña |
| Positivos detectados | 1/4 (25%) | **Insuficiente para vender "detección de incidentes"** |
| Falsos negativos | caída_escaleras, disturbios_saqueo, disturbios_calle (clasificada como fallo aunque alertó `caidas`: verdad de referencia rígida) | |
| Alerta `persona` | se dispara en casi todas las escenas | Es ruido si llega como notificación |

Problemas de método: n=10 escenas, clips de archivo (películas antiguas, motines), no CCTV real; sin intervalos de confianza; sin métricas de precisión/recall por clase ni por hora de cámara (falsas alarmas/cámara/día, la métrica que de verdad decide la retención de clientes).

## 3. Salud / riesgos operativos
1. **Sin CI ni tests automáticos** en ningún repo (no hay `.github/workflows`). Un deploy roto de Lambda = fallo para todos los clientes.
2. **Sin observabilidad visible**: heartbeat existe, pero no hay alarmas CloudWatch/SNS documentadas para: cola DLQ con mensajes, worker sin heartbeat, error rate de Lambdas, gasto Bedrock.
3. **Economía unitaria**: ingreso USD 10/cámara/mes (plan 5 cám = USD 50). Coste medido: dedicada USD 86/cám/mes; topología compartida (Fase B) ≈ USD 19–23/cám con 5–10 cámaras. **Solo es rentable en modo compartido y con >=5 cámaras activas por analysis box.** Con la dedicada se pierde dinero por cliente.
4. **Escalado**: una `analysis box` única para todos los clientes = punto único de fallo y cuello de botella (CLIP+VLM en una m7i-flex.large). Sin autoscaling por profundidad de cola.
5. **Estado en `localStorage`** para monitoreo (aceptado como UX, pero causa estados falsos multi-dispositivo y soporte).
6. **Multi-tenancy**: un solo plan, sin tiers, sin trial, sin onboarding guiado de cámara (RTSP detrás de NAT/túnel es la mayor fricción de venta).
7. **Privacidad/legal**: video de personas → faltan política de privacidad, retención configurable, DPA; relevante para clientes B2B (y GDPR/Ley 1581 en LatAm).
8. Límites de Bedrock/cuotas por cuenta: con muchos clientes, throttling del VLM; el rate limit por cámara existe pero no uno global ni backoff documentado.

## 4. Mejoras propuestas (priorizadas)
**P0 (antes de captar clientes de pago, 1–2 semanas)**
1. CI mínimo (GitHub Actions): lint + tests unitarios de auth/plan-check en HeimdalManager, `integration_test.py`/`camera_sim_test.py` en worker, `next build` en UI.
2. Alarmas: DLQ>0, sin heartbeat >5 min, Lambda 5xx, presupuesto AWS/Bedrock con alerta (AWS Budgets).
3. Dejar de notificar `persona` por defecto (solo evento "persona" como filtro/gate); notificar solo caídas/robos/violencia/arma.
4. Nuevo plan de entrada y precio por cámara (ver §6) y trial de 14 días con 1 cámara; ahora solo hay 1 plan visible.
**P1 (accuracy, 2–4 semanas)**
5. Caídas por pose (YOLOv8-Pose, 92–98% sin VLM) — ya identificado en bitácora; también baja coste VLM.
6. Banco de pruebas ≥100 clips con etiquetas multi-etiqueta aceptables; reportar precision/recall/F1 por clase + **falsas alarmas por cámara-día**. Meta comercial: ≥80% recall en caídas y <1 falsa alarma/cámara/día.
7. Piloto con 2–3 cámaras reales de clientes amigos para medir en CCTV real (el banco subestima: la puerta de personas ahorra mucho más en locales vacíos de noche).
8. Calibrar `clear_margin` de `persona` (curva medida antes de fijar).
**P2 (escala)**
9. Autoscaling de analysis box por profundidad de cola SQS (ASG o varias instancias con consumidores competitivos); idle-shutdown ya existe.
10. Onboarding de cámara: agente/túnel (WireGuard/Tailscale/cloudflared) con un comando; verificación de RTSP al registrar (test de conexión en StoreDevice).
11. Panel de salud para el cliente (cámara online/offline) y admin (costes por cliente) — HeimdalManager ya recibe heartbeats.
12. Retención configurable y exportación/borrado de datos del cliente.

## 5. Coordinación con la tarea de captación de clientes
Contexto que debe usar el otro agente al diseñar mensajes/ofertas — **qué se puede prometer HOY con honestidad**:
- OK prometer: monitoreo continuo con IA, alertas SMS/email con foto de evidencia, caídas y personas en zona restringida/horario, setup sin hardware nuevo si la cámara tiene RTSP.
- NO prometer aún: detección fiable de peleas/robos/armas (1/4 en banco); SLAs de uptime; "cero falsas alarmas".
- Mejor ICP inicial: locales/bodegas/obras **vacíos fuera de horario** (alto ahorro de coste, baja ambigüedad: "persona donde no debería haber nadie"), y adultos mayores/residencias para **caídas** (cuando esté pose). Evitar vía pública con multitudes.
- Oferta piloto sugerida: 30 días gratis, 1–2 cámaras, a cambio de feedback y permiso para medir precisión (alimenta el banco de pruebas reales y casos de éxito).
- Pregunta para la otra tarea: qué segmentos/canales responden mejor y qué precio toleran, para fijar tiers (§6). Datos de entrada que necesitamos: nº cámaras típico por prospecto, % con RTSP accesible, disposición a pagar.

## 6. Precio (propuesta a validar)
Costes: compartida ≈ USD 12–23/cám según uso. Plan actual 50/5 = USD 10/cám → margen nulo/negativo. Sugerido: Starter 1–2 cám USD 19/cám; Pro 5 cám USD 15/cám (=75); Business 20+ cám USD 11/cám + overage. Revisar tras medir coste real con cámaras de clientes.

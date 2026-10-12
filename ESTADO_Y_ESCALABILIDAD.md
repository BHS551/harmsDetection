# SkyEye — estado, accuracy, salud y plan para escalar a muchos clientes
Revisión automática, 2026-10-12. Basada en código y bitácora de los 7 repos (no se pudo medir producción en vivo: sin acceso a AWS/CloudWatch desde esta sesión).

## 1. Resumen ejecutivo
- **Arquitectura y seguridad: buenas** (auth server-side, secretos en Secrets Manager, CORS allow-list, TTL 30 d, idempotencia).
- **Accuracy: aún no vendible como "detecta caídas/peleas/robos"**. Banco de 10 escenas: negativos limpios 6/6, positivos reales 1/4 detectados por la matriz (2 verdaderos positivos verificados a mano en `pelea_calle`). Muestra minúscula.
- **Economía: el bloqueante nº1.** Plan único `cam5` = 50 USD/mes (10 USD/cámara). Coste medido por cámara 24/7 (ciclo 3): 85,97 USD dedicada; 19,29 USD en Fase B con 10 cám/caja; 115,62 en paquete de 5. **Cada cliente nuevo pierde dinero hoy**, y SMS no está ni incluido en esas cifras.
- **Operación/calidad: sin CI, sin tests automáticos en ningún repo, sin observabilidad ni alarmas.** Último commit funcional del worker: 2026-08-10.

## 2. Accuracy (datos de `testbench/`)
| Métrica | Valor |
|---|---|
| Negativos limpios (sin alerta de incidente) | 6/6 |
| Positivos detectados | 1/4 (caida_escaleras, disturbios_saqueo, disturbios_calle fallan) |
| Falsos positivos | `persona` salta en naturaleza vacía; `caidas` por alguien sentado (mitigado ciclo 3) |
| Referencia | solo 10 clips de Wikimedia; verdad rígida (penaliza aciertos legítimos) |

Problemas: (a) el VLM responde NO viendo indicios (puerta reventada, incendio) → preguntar por el rastro también en `robos`; (b) frame suelto no basta para actos; (c) no hay precision/recall con intervalos de confianza ni set etiquetado de cámaras reales (CCTV de baja luz, fisheye, IR nocturno).

## 3. Salud / riesgos operativos
1. **Worker no se auto-termina con cámara caída tras el arranque** (BITACORA/testbench): una m7i-flex.large puede facturar horas sin producir. Fix: contador de fallos consecutivos de lectura → `terminate_self` + notificar al cliente "cámara offline".
2. **Un EC2 dedicado por cámara** (~86 USD/mes) no escala ni en coste ni en límites de cuota EC2 (vCPU quota por región).
3. **Sin CI ni tests** en los 7 repos; `integration_test.py` tiene rutas de scratchpad hardcodeadas y cuenta AWS real en URLs.
4. `lib/monitoring.ts` usa localStorage (solo UX; aceptable, backend re-valida).
5. `yolov5s.pt`, `__pycache__`, scripts experimentales (`rtsp_*.py`) en el repo del worker: limpiar.
6. Sin alarmas: DLQ depth, errores 5xx de Lambdas, heartbeat perdido, gasto Bedrock diario, instancias huérfanas.
7. Un solo plan, sin trial, sin tabla de uso por cliente → no hay forma de medir coste real por cliente.

## 4. Mejoras propuestas (priorizadas)
**P0 — antes de captar clientes (bloquean escalar)**
1. **Arreglar márgenes**: Fase B (motion box + analysis box compartida) como topología por defecto; YOLOv8-Pose para caídas sin VLM (ya investigado, 92–98%); calibrar `clear_margin` de `persona`; SMS como add-on/límite por plan o solo email/push/WhatsApp. Objetivo: coste ≤ 30% del precio por cámara. Si no, subir precio (p. ej. 20–30 USD/cámara o planes 1/3/10 cámaras) — ver §5.
2. **Auto-terminación por cámara offline + alerta al cliente**.
3. **CI mínimo** (GitHub Actions: lint+typecheck en UI, `node --check`/tests de handlers con mocks, pytest de `vision`/`tiers` con colas locales) y **alarmas CloudWatch** (§3.6).
4. **Medición de uso y coste por cliente** (VLM calls, alertas, horas de instancia en DynamoDB) → base para facturar y detectar abuso.

**P1 — para 10–100 clientes**
5. Set de evaluación real: 100+ clips etiquetados (incl. nocturno/IR), reportar precision/recall por clase; verdad de referencia flexible; regresión automática en cada cambio de prompt/umbral.
6. Feedback del cliente en la consola ("falsa alarma" / "correcto") → dataset gratis + umbrales por cámara + métrica de FP/día por cámara visible al cliente.
7. Onboarding de cámara autoservicio: test de conexión RTSP, guía por marca (Hikvision/Dahua/Tapo), ONVIF discovery; hoy el túnel (ngrok) es fricción enorme para clientes no técnicos. Valorar agente/edge box pequeño (Raspberry/mini-PC) que haga tier 0 localmente y suba solo candidatos → baja ancho de banda, coste y expone menos la cámara.
8. Zonas de interés/enmascarado, horarios (armado/desarmado) → menos VLM y menos FP.
9. Alertas por WhatsApp/Telegram/push (mercado LatAm) en lugar de solo SMS.

**P2 — para 100+ clientes**
10. Multi-tenant real: cuotas EC2, autoscaling de analysis boxes (o migrar a ECS/Fargate o SageMaker/GPU batch), colas por tenant, límites de Bedrock.
11. Cumplimiento: aviso de privacidad/habeas data (Colombia, Ley 1581), retención configurable, DPA; base legal para video de terceros.
12. SLA/status page, backups DynamoDB (PITR), IaC (Terraform/CDK) — hoy todo parece configurado a mano.

## 5. Contexto para la tarea hermana (adquisición de clientes)
Para coordinar con la investigación de captación (no hay otra sesión activa en esta máquina; dejo esto como handoff):
- **Qué NO prometer aún**: "detección de robos/peleas fiable". **Qué sí**: detección de persona/intrusión fuera de horario, caída/persona en el suelo (tras Pose), con evidencia en foto. Segmentos con **cámaras sin gente la mayor parte del tiempo** son los ideales (bodegas, locales cerrados de noche, obras, parqueaderos, conjuntos residenciales): la puerta de personas hace que el coste caiga casi a cero allí.
- **Pricing**: hoy 50 USD/5 cám es incompatible con el coste. Recomiendo piloto de pago (no gratis ilimitado) con tope de 1–2 cámaras y precio por cámara; el canal PayU (COP) ya existe.
- **Capacidad para piloto**: con el estado actual, 3–5 pilotos manuales son seguros; >10 clientes simultáneos requieren P0.
- Pedir a los pilotos: permiso para usar clips etiquetados (mejora accuracy) y feedback de falsas alarmas.
- Landing: `skyeyeprotection.com`; verificar que no prometa más de lo medido.

## 6. Qué haría yo esta semana
1. Pose para caídas + auto-terminación + alarmas (2–3 días). 2. Re-calcular `costes.py` con Fase B y fijar precio. 3. CI mínimo. 4. Abrir 3–5 pilotos de pago en segmentos "poca gente".

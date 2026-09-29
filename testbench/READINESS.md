# SkyEye: preparación para escalar a muchos clientes (2026-09-29)

Revisión de las 7 repos (worker, LandingUi, HeimdalManager, Store/List Lambdas) y de la bitácora. Nada de esto se ha medido de nuevo hoy: son datos del ciclo 3 (2026-08-10) más lectura de código.

## Veredicto
**No está listo para muchos clientes.** Las tres razones, en orden de gravedad:

1. **Economía unitaria negativa.** Plan `cam5` = 50 USD/mes por 5 cámaras (10 USD/cámara). Coste medido por cámara: 85,97 USD (worker dedicado), Fase B 5 cámaras 115,62 USD/mes en total (~23 USD/cámara). Cada cliente pierde dinero incluso con la mejor topología. Solo a 10 cámaras/caja (19,29 USD/cámara) se acerca al precio, y ni así deja margen tras pasarela de pago + SMS.
2. **Precisión insuficiente para prometer "detección de daños".** Banco de 10 escenas: 1/4 positivos detectados, 6/6 negativos limpios. Robo y violencia en positivo: 0 verificados (disturbios no detectados). Accidente laboral: sin material positivo posible. La etiqueta `persona` genera casi todas las alertas (las 10 escenas alertan `persona`, incluida la montaña vacía): para el cliente es ruido.
3. **Sin observabilidad ni pruebas automáticas.** No hay CI (.github) en ninguna repo, ni alarmas, ni tests unitarios (solo integration_test/camera_sim_test manuales). Nadie se entera si un worker muere, si Bedrock falla o si el coste se dispara.

## Salud del sistema (lectura de código)
Bien: verificación de token en servidor, límite de plan y cámaras en HeimdalManager, start idempotente, credenciales RTSP en Secrets Manager, CORS restringido, TTL 30 días, DLQ, tres límites de gasto del VLM.
Riesgos:
- `lib/monitoring.ts` cuenta cámaras en localStorage (solo UX, el backend protege, ok).
- Un worker EC2 por cámara: no escala en coste ni en operación (limites de vCPU de la cuenta, cuota de EC2).
- Sin límite de gasto global ni alarma de Bedrock/SMS: un cliente con muchas alertas puede costar más que su suscripción.
- Sin reserved concurrency / rate limiting visible en los Lambdas ni WAF en la API.
- Sin política de reintentos/estado visible para el cliente cuando cae el túnel ngrok (existe reconexión, falta alerta "tu cámara está offline").
- Privacidad: vídeo de cámaras de terceros; no hay términos, DPA ni política de retención documentada más allá del TTL (Colombia: Ley 1581 de 2012, habeas data).

## Mejoras propuestas (por prioridad)
**P0, antes de captar clientes de pago**
1. Rehacer precio o coste: subir a ≥ 25–30 USD/cámara o mover a Fase B con ≥10 cámaras/caja y YOLOv8-Pose para caídas (quita una clase entera del VLM). Fijar objetivo: coste ≤ 30% del precio.
2. Presupuesto por cliente: tope diario de llamadas VLM y SMS por `OwnerUid`, con AWS Budgets + alarma.
3. Quitar `persona` de las alertas por defecto (o convertirla en evento silencioso/ conteo): es ruido y consume VLM.
4. Alarmas CloudWatch: worker sin heartbeat, DLQ no vacía, errores 5xx de Lambdas, gasto Bedrock/día. Notificar al operador.
5. CI mínimo en cada repo (lint + test + deploy manual controlado).

**P1, precisión**
6. Banco con material grabado a propósito (robo, pelea, caída, accidente) y métrica de precisión/recall por clase con umbral de aceptación (p.ej. recall ≥ 70% y ≤ 1 falso positivo/cámara/día) publicado antes de vender.
7. Verdad de referencia flexible (ciclo 4, paso 3) y calibrar `clear_margin` de `persona`.
8. Piloto gratuito con 2–3 clientes reales midiendo falsos positivos por cámara/día: es el dato comercial más valioso.

**P2, operación**
9. Estado de cámara en la consola (online/offline, última detección) y aviso al cliente si cae.
10. Autoservicio de alta de cámara (validar URL RTSP antes de cobrar), guía para túnel/VPN o agente local.
11. Términos de servicio, política de privacidad, contrato de tratamiento de datos, retención configurable.
12. Plan de escalado: cuotas EC2, Auto Scaling de la caja de análisis, pruebas de carga con la cámara mock (ya existe `mock_camera.py`).

## Puntos para coordinar con el agente de captación de clientes
- No vender aún "detección de robos/violencia" con garantía: ofrecer **piloto** con expectativas claras (caídas y presencia de personas son lo más fiable hoy).
- Segmento con mejor encaje técnico: locales cerrados de noche, almacenes, pasillos (la puerta de personas elimina casi todo el gasto VLM cuando no hay nadie; el banco lo subestima).
- Precio: cualquier oferta debe partir de ≥ 25 USD/cámara o compromisos de ≥10 cámaras; los 10 USD/cámara actuales no se sostienen.
- Capacidad: sin auto-escalado, limitar a unas 10–20 cámaras en el primer mes; piloto pequeño y medido.
- Necesita del agente de captación: lista de segmentos candidatos, disposición a pagar por cámara, y qué exigen (respaldo legal, SLA, cámaras compatibles).

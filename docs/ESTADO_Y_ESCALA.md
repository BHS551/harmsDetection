# SkyEye — estado, accuracy, health y plan para escalar a muchos clientes

Fecha: 2026-10-14. Análisis estático de los 7 repos + bitácora/resultados del
banco de pruebas (`testbench/`). No se pudo medir producción en vivo (sin
acceso a AWS/CloudWatch desde esta sesión).

## 1. Accuracy (banco de 10 escenas, ciclo 3)
- Negativos limpios 6/6; positivos 1/4 (sólo `pelea_calle`). Banco pequeño: no hay
  intervalos de confianza útiles, ni positivos de robo/accidente laboral.
- `persona` genera la mayor parte del tráfico hacia el VLM y alerta incluso en
  naturaleza vacía (1 alerta) -> ruido para el cliente si `persona` se notifica.
- Verdad de referencia rígida penaliza aciertos legítimos (`disturbios_calle`).
- Caídas: YOLOv8-Pose (92–98 % sin VLM) sigue pendiente (ciclo 4).
**Riesgo comercial:** 1/4 de recall no es vendible como "detección de robos/peleas"
sin matizar. Antes de captar clientes, medir con material propio/piloto.

## 2. Coste unitario (medido, `costes.py`)
Bedrock 120 -> 15 USD/cám/mes (ciclo 0 -> 3). Dedicada 85,97 USD/cám/mes;
paquete 5 cám 115,62; 10 cám 19,29/cám (fase B, motion box compartido).
=> El margen sólo existe con topología compartida (motion box + analysis box).
Los planes en `lib/plans.ts` deben fijarse por encima de ~20–25 USD/cám/mes
(+SMS 0,06 c/u) en topología compartida; la dedicada no es rentable a precios LATAM bajos.

## 3. Health / fiabilidad — hallazgos por repo
**Fortalezas:** plan check server-side, start idempotente, secretos RTSP por
referencia, CORS allow-list, TTL 30 d en detecciones, DLQs, heartbeats.

**Brechas para escalar (prioridad):**
1. **Sin CI ni tests automáticos** en ningún repo (no hay `.github/`; sólo scripts
   de integración manuales). Añadir GitHub Actions: lint + tests unitarios de
   Lambdas (mock DynamoDB/Firebase) + `integration_test.py` en humo.
2. **Observabilidad:** no hay alarmas. Crear CloudWatch alarms: Lambda errors/5xx,
   DLQ > 0, edad de cola, heartbeat ausente por cámara (-> notificar al cliente
   "tu cámara está caída"), gasto Bedrock/día, instancias EC2 huérfanas.
3. **`subscriptions.mjs` usa `ScanCommand`** (admin panel): O(N) con clientes ->
   paginar o GSI por `status`. Revisar que ningún Lambda de usuario haga Scan.
4. **Estado de monitoreo en `localStorage`** (por navegador): con muchos usuarios
   genera tickets de soporte; mover a backend (tag EC2/DynamoDB `devices.monitoring`).
5. **Cuotas AWS:** vCPU on-demand por región, Bedrock RPM/TPM de Nova Lite,
   límites SNS SMS (sandbox/gasto mensual), SES. Pedir aumentos antes del
   lanzamiento; con 1 instancia por cámara en modo dedicado se agota rápido.
6. **Escala del analysis box:** un único box compartido es SPOF y cuello de botella.
   Autoscaling por profundidad de SQS (ASG con target tracking) y 2+ AZ.
7. **Rate limiting / abuso:** API Gateway usage plans + throttling por uid;
   `X-Internal-Secret` estático -> rotación o IAM auth (SigV4) para M2M.
8. **Pagos:** PayU confirmation y Stripe webhook: verificar firma, idempotencia por
   `transaction_id`, y manejo de renovaciones/impagos (downgrade automático ->
   HeimdalManager debe parar instancias cuando `status != active`; hoy el check
   sólo ocurre al *iniciar*). Un cliente que deja de pagar sigue consumiendo.
9. **Privacidad/legal (video de terceros):** política de retención (30 d ok),
   consentimiento, Ley 1581 (Colombia)/GDPR según mercado, cifrado S3 + acceso
   firmado. Es requisito de ventas B2B.
10. **Higiene del repo:** `rtsp_*.py`, `heimdall-eye.py` (811 líneas),
    `yolov5s.pt` y `__pycache__` en el repo; archivar en `legacy/` y `.gitignore`.
11. **Onboarding de cámaras:** el mayor fricción real es RTSP detrás de NAT/túnel.
    Ofrecer un conector (contenedor/agente ligero o ONVIF + túnel gestionado)
    o soportar cámaras cloud (Hikvision/Dahua/Tapo vía API).

## 4. Plan propuesto
**Fase 0 (1–2 sem, bloqueante para vender):** CI mínimo, alarmas + heartbeat->aviso
al cliente, parada automática por impago, cuotas AWS, política de privacidad.
**Fase 1 (2–4 sem):** YOLOv8-Pose para caídas, calibrar `persona`, ampliar banco
(>=30 escenas, positivos propios del piloto), métricas de precisión/recall por clase
publicadas en `testbench/`. Piloto con 3–5 clientes gratis/pago reducido.
**Fase 2:** autoscaling analysis box, localStorage->backend, dashboard de uso
y coste por cliente, SLA y página de estado.

## 5. Coordinación con la tarea de captación de clientes
Mensaje para esa tarea (no hay sesión par alcanzable ahora; este archivo es el
canal compartido):
- **Qué se puede prometer hoy con honestidad:** detección de personas/presencia en
  zonas y horarios cerrados (alto valor, bajo coste por la puerta de personas),
  alertas de caídas con evidencia; peleas/robos como "beta".
- **ICP recomendado:** locales/almacenes/obras con ventanas sin gente (nocturno),
  donde coste/cám es mínimo y la precisión es mayor; 5–10 cámaras por cliente
  (economía de paquete: 19 USD/cám).
- **Piloto:** 3–5 clientes con cámaras IP accesibles; a cambio, clips etiquetados
  que alimentan el banco y cierran la brecha de accuracy.
- **Precios mínimos:** >= 25 USD/cám/mes compartida; evitar plan dedicado barato.
- **Necesito de ellos:** tipos de cámara/NAT de los prospectos, mercado
  (CO/MX/ES) para decidir cumplimiento y pasarela, y volumen esperado a 3 meses
  para dimensionar cuotas.

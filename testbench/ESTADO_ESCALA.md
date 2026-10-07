# Estado del sistema y preparación para escalar (2026-10-07)

Revisión automática de los 7 repos (harmsDetection, HeimdalManager, Store/List Detection/Device, LandingUi).
Fuente de las cifras: `testbench/BITACORA.md` (ciclo 3) y el código de cada repo. No se ejecutó nada contra AWS.

## 1. Veredicto
Arquitectura sólida (cascada motion→CLIP→VLM, plan check server-side, secretos por referencia). **No está listo para "muchos clientes"** por tres bloqueantes: economía unitaria, accuracy sin validar y credenciales en el repo.

## 2. Accuracy (medida, banco de 10 escenas)
- Negativos limpios 6/6 (bien: pocos falsos positivos). Pero `naturaleza_vacia` pasó de 0 a 1 alerta `persona` entre ciclos → falso positivo en cámara vacía.
- **Positivos 1/4** (solo `pelea_calle`). Recall sobre incidentes ≈ 25 %. Con 4 positivos el intervalo de confianza es enorme; no hay evidencia para prometer detección.
- `accidente laboral` no se puede evaluar en positivo (sin material).
- La verdad de referencia es rígida y penaliza aciertos (caídas en disturbios). Pendiente ciclo 4.
- Palanca grande pendiente: YOLOv8-Pose para caídas (92–98 % reportado, sin VLM).

## 3. Coste / economía unitaria (bloqueante comercial)
- Plan `cam5` = 50 USD/mes por 5 cámaras = **10 USD/cámara**.
- Coste medido por cámara/mes: Fase B 5 cámaras 23,12 USD; 10 cámaras 19,29 USD; dedicada 85,97 USD. Solo Bedrock ya es ~15 USD/cámara.
- **Margen negativo en todos los escenarios medidos.** Mitigante: el banco tiene escenas siempre con gente (sobreestima coste vs. locales vacíos de noche). Hay que medir en cámaras reales.
- Acciones: (a) medir coste en 2–3 cámaras reales una semana; (b) subir precio o por-cámara con mínimo; (c) YOLO-Pose + calibrar `clear_margin` de `persona` para sacar tráfico del VLM; (d) subir `VLM_MIN_INTERVAL`; (e) tope mensual de llamadas VLM por cliente.

## 4. Seguridad / higiene (bloqueante)
- `context.json`, `minimal.py`, `rtsp_*.py`, `multicore_detection_debug.py` contienen URLs RTSP con usuario/contraseña en texto plano (admin551:123456789, bhsentrance:mainSecurePass1…). Rotar esas credenciales, borrarlas del historial (git filter-repo) y dejarlas fuera del repo. `Dockerfile` aún empaqueta `multicore_detection_debug.py` (script legado con credencial).
- `yolov5s.pt` binario y scripts experimentales en la raíz: mover a `legacy/`.
- Falta `.gitignore` en HeimdalManager y similares revisar (ya tiene).

## 5. Fiabilidad / operación (falta para >10 clientes)
- **Sin tests ni CI** en los repos Lambda ni en la UI (solo `integration_test.py` manual). Añadir GitHub Actions: lint + tests unitarios de auth/plan-limit/idempotencia.
- **Sin observabilidad**: no hay dashboards ni alarmas. Mínimo: CloudWatch alarms en DLQ>0, edad de cola, heartbeat de worker ausente, 5xx de Lambdas, fallos de Bedrock/throttling.
- **Límites de escala**: motion box único t3.small (SPOF y techo de cámaras); analysis box único con apagado a 2 h idle (cold start de GPU/CPU en el primer incidente = latencia de alerta). Definir SLO de latencia alerta y probar con N cámaras simuladas (`mock_camera.py`).
- Cuotas Bedrock (Nova Lite) y límites de cuenta EC2 (vCPU) por región: solicitar aumentos antes de captar.
- Contador de cámaras en `localStorage` (UI) → solo UX, el servidor valida: OK.
- ListDetections usa Query por owner_uid con `Limit`: añadir paginación con cursor y TTL en DynamoDB (retención y coste de S3 de frames).
- Cámaras vía ngrok: frágil para clientes reales. Ofrecer agente/relay (túnel propio, o RTSP sobre WireGuard) o solo cámaras con IP pública/ONVIF-cloud.
- Onboarding: sin flujo guiado para probar la cámara/zonas, sin ajuste de sensibilidad por cliente, sin métrica de falsas alarmas por cliente (feedback "falsa alarma" en cada alerta alimentaría el accuracy real).
- Legal/privacidad: video de terceros → política de privacidad, retención, DPA; imprescindible para B2B.

## 6. Orden de prioridad propuesto
1. Rotar credenciales expuestas y limpiar historial (hoy).
2. Medir coste y accuracy en cámaras reales (piloto 2–3 clientes amigos) con botón "falsa alarma".
3. Corregir economía unitaria (precio y/o menos VLM).
4. CI + tests + alarmas.
5. Ciclo 4 de accuracy: Pose para caídas, calibrar persona, ampliar banco (>30 escenas, positivos reales grabados a propósito).
6. Pruebas de carga con cámaras simuladas; cuotas AWS.
7. Relay de cámaras, onboarding, legal.

## 7. Para la tarea de captación de clientes (coordinación)
Lo que sí se puede prometer hoy: detección de personas y alertas en cámaras de zonas con poca actividad (locales cerrados de noche, bodegas), piloto gratuito/pagado con pocos clientes. NO prometer: detección fiable de robos/accidentes laborales. Segmento recomendado: negocios pequeños con cámaras IP existentes que quieran alerta nocturna de intrusión; precio debe cubrir ≥ 25 USD/cámara hasta mejorar costes. Necesitamos de esa tarea: segmento objetivo, disposición a pagar, y 2–3 pilotos dispuestos a aportar video real/etiquetas.

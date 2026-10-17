# Estado del sistema y qué falta para tener muchos clientes (2026-10-17)

Revisión automática (rutina programada) de los 7 repos. Sin acceso a AWS en vivo:
todo sale del código, READMEs y la bitácora de `testbench/`. Pensado también como
contexto compartido para la rutina que investiga captación de clientes.

## 1. Accuracy (último dato medido: ciclo 3, 10 escenas)
- Negativos limpios 6/6 (falsos positivos controlados en el banco).
- Positivos 1/4 (solo `pelea_calle`; 2 alertas verificadas a mano). **Recall en incidentes reales sigue bajo.**
- Ciclo 0 era 0/4: hubo mejora, pero el banco es diminuto (10 clips, 4 positivos): no hay significancia estadística.
- La "verdad de referencia" es rígida y penaliza aciertos legítimos (caída en disturbio). Pendiente ciclo 4.
- Pendientes ya identificados: YOLOv8-Pose para caídas (92-98% sin VLM), calibrar `clear_margin` de `persona`, flexibilizar verdad, curva coste/latencia de `VLM_MIN_INTERVAL`.

## 2. Coste (medido, por cámara 24/7)
- Bedrock 120 -> 15 USD/mes; Fase B 10 cám/caja: 124.9 -> 19.3 USD/cám/mes; dedicada 86 USD.
- Un plan de venta debe quedar claramente por encima de ~20-25 USD/cám/mes (Fase B) para tener margen; con worker dedicado, ~86.

## 3. Health / riesgos de escala detectados en código
1. **Motion box único global** (`HeimdalManager startMotionBox`): una sola t3.small con roster fijo al arrancar; sin tope de cámaras por caja, sin sharding por cliente, y añadir cámara = relanzar. Es el primer límite al crecer (CPU de MOG2 en N streams).
2. **Una analysis box compartida** (CLIP+VLM, CPU m7i-flex) para todos: cola SQS crece sin autoscaling; latencia de alerta degrada con carga. Falta métrica de profundidad de cola/edad del mensaje + alarma.
3. **Límite de plan con carrera**: `countOwnerRunningInstances` + `RunInstances` no es atómico; dos starts concurrentes con `TaskId` distinto pueden superar `maxCameras`. Usar escritura condicional en DynamoDB.
4. **Cuotas AWS**: vCPU on-demand de EC2 por región y TPS de Bedrock (Nova Lite) limitan el modo "worker por cámara". Pedir aumento de cuota antes de vender.
5. **Sin CI ni tests automáticos** en ningún repo (solo harnesses manuales en testbench). Cada deploy es manual (zip a Lambda, `deploy-worker.sh`).
6. **Sin observabilidad de producto**: no hay dashboard/alarmas de latencia de alerta, tasa de falsos positivos reportados por usuarios, ni heartbeat caído. Falta feedback "alerta falsa/real" del usuario: es el dato que más mejora accuracy y no se captura.
7. `subscriptions.mjs` usa `ScanCommand` (admin only, sin paginar): se rompe pasado 1 MB de tabla.
8. Privacidad/legal (bloqueante para vender): video de cámaras con personas -> política de datos, retención (hoy 30 d de detecciones), consentimiento/aviso, y en Colombia Ley 1581 (habeas data).
9. Dockerfile base CUDA 11.6/Ubuntu 20.04 (fin de soporte) y `yolov5s.pt`/scripts `rtsp_*` legacy en el repo: limpiar.

## 4. Mejoras propuestas (orden sugerido)
P0 (antes de clientes de pago): feedback de falsos positivos en UI; alarmas CloudWatch (cola, heartbeat, errores Lambda); condicional DynamoDB para maxCameras; CI mínimo (lint + test de handlers); política de privacidad y términos; cuotas AWS.
P1: ampliar banco a >=50 clips con >=20 positivos y medir precisión/recall con IC; ciclo 4 (pose para caídas, verdad flexible); sharding de motion box por cliente/N cámaras; autoscaling de analysis box por cola.
P2: onboarding autoservicio de cámara (verificar RTSP al registrar), SLA/latencia publicada, plan por cámara con margen >= 3x del coste.

## 5. Para la rutina de captación de clientes
- Posicionar sin prometer recall: mensaje honesto = "alerta de persona/movimiento relevante + violencia/caídas en beta". Piloto con 3-5 negocios (local cerrado de noche, almacén, pasillo): ahí la compuerta de personas ahorra más y el ruido es menor.
- Los pilotos sirven para generar el dataset real etiquetado que hoy falta: ofrecer piloto gratis a cambio de feedback de alertas.
- Precio mínimo viable: >= ~60-70 USD/cámara/mes en Fase B (coste ~19) o decidir tarifa por paquete de 5 cámaras (~116 coste).
- Capacidad actual cómoda: unas pocas decenas de cámaras; antes de >50 hacer P0/P1.

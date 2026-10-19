# Estado del sistema y preparación para muchos clientes (2026-10-19)

Documento de contexto compartido. Lo lee también la tarea programada de captación de clientes.

## Estado medido (último banco, ciclo 3)
- Negativos limpios 6/6 (sin falsas alarmas de incidente en escenas normales).
- **Positivos 1/4**: solo `pelea_calle` (como `caidas`). `disturbios_saqueo`, `disturbios_calle` y `caida_escaleras` no se detectan como incidente. La verdad de referencia de `disturbios_calle` penaliza aciertos (caídas legítimas), pendiente en ciclo 4.
- Banco pequeño (10-11 escenas, clips de Wikimedia): sirve de regresión, NO como cifra comercial. No hay positivos reales de accidente laboral.
- Coste por cámara (Fase B, 10 cám/caja): 124,92 -> 19,29 USD/mes (-85%). El VLM es ~97% del coste variable.

## Riesgos antes de escalar (por prioridad)
1. **Accuracy no demostrada**: recall 25% en el banco. Un cliente de seguridad que no recibe la alerta se va. Falta un set con cámaras reales (con consentimiento) y métricas precision/recall por concepto.
2. **Secretos en el repo**: `harmsDetection/context.json` está versionado con una URL RTSP con usuario/contraseña (`rtsp_path` literal, ngrok). Rotar esa credencial, quitar el archivo del índice (`git rm --cached`), añadirlo a `.gitignore` y limpiar historial. También `yolov5s.pt` (binario) está versionado.
3. **Sin CI ni tests automáticos** en ningún repo (no hay `.github/`). Añadir CI mínimo: lint + tests de `cascade` + test de Lambdas.
4. **Salud/observabilidad**: hay heartbeat y estado de worker en la UI, pero faltan alarmas (CloudWatch) por cámara caída, cola SQS/DLQ con mensajes, errores de Bedrock y gasto diario; sin eso un cliente se entera antes que nosotros.
5. **Coste/capacidad**: una instancia por cámara no escala en margen; priorizar la topología Fase B (motion-multi + analysis compartido) y cuotas por plan (ya existen `maxCameras`). Revisar límites de EC2/Bedrock (service quotas) para ~100+ cámaras.
6. **Onboarding**: dependencia de túneles ngrok/RTSP expuesto del cliente es frágil. Ofrecer agente/gateway local o guía por marcas de cámara.
7. **Legal/privacidad**: video de terceros en AWS; falta política de retención visible (detecciones tienen TTL 30 d), términos y aviso de tratamiento de datos (relevante para ventas B2B).

## Mejoras propuestas (orden sugerido)
1. Rotar credencial RTSP y sacar `context.json` del control de versiones (hoy).
2. Ciclo 4: corregir verdad de referencia, ampliar banco con positivos reales/grabados a propósito, reportar precision/recall por concepto; objetivo recall >= 70% en caídas y violencia antes de prometerlo en ventas.
3. CI + alarmas CloudWatch + panel de gasto por cliente.
4. Prueba de carga: 50 cámaras simuladas con `testbench/mock_camera.py`.
5. Piloto con 3-5 clientes reales, plan gratuito/descuento a cambio de feedback y etiquetado de alertas (da datos reales de accuracy).

## Para la tarea de captación
- Mensaje honesto hoy: "detección de personas/intrusión en horas vacías con bajo coste"; evitar prometer detección de peleas/robos hasta superar el recall objetivo.
- Segmentos con cámara vacía la mayor parte del tiempo (locales de noche, almacenes, obras fuera de horario) son donde el cascade ahorra más y la falsa alarma es menor.
- Precio: coste real ~19-23 USD/cámara/mes en Fase B; margen sano a partir de ~35-50 USD/cámara.
- Necesitamos de captación: 3-5 pilotos dispuestos a compartir cámaras (consentimiento) para calibrar accuracy.

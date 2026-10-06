# SkyEye — preparación para escalar a muchos clientes

Evaluación 2026-10-06, hecha leyendo los 7 repos (harmsDetection, HeimdalManager,
StoreDetection/Device, List*, landing UI) y la BITACORA del banco de pruebas.
No se ejecutó nada contra AWS: todo es lectura de código y de resultados ya medidos.

Este documento es también el **canal de contexto compartido con la tarea programada
de adquisición de clientes**. Sección "Para go-to-market" al final.

## Veredicto

**No está listo para "muchos clientes". Sí está listo para un piloto cerrado de
3–5 clientes de confianza**, con las condiciones de abajo. Tres bloqueantes:
economía por cámara, evidencia de precisión, y fallos silenciosos en producción.

## 1. Economía (bloqueante n.º 1)

Números medidos en el ciclo 3 (BITACORA):

| escenario | coste/cámara/mes |
|---|---|
| Instancia dedicada | 85,97 USD |
| Fase B, 5 cámaras | 23,12 USD |
| Fase B, 10 cámaras | 19,29 USD |

Plan actual (`docs/stripe-setup.md`, landing): **cam5 = 50 USD/mes por 5 cámaras = 10 USD/cámara**.

- Con Fase B a 5 cámaras el coste (115,62 USD el paquete) es **más del doble** del ingreso (50 USD). Pérdida ≈ 65 USD por cliente/mes.
- Con 10 cámaras por caja sigue en 19,29 > 10 USD. Solo se equilibra con cajas más llenas o precio mayor.
- Un cliente en dedicada pierde ~76 USD/mes.
- El banco **sobreestima** el gasto en cámaras con poca gente (la puerta de personas lo recorta más en real) y **subestima** el de escenas con mucho tráfico. No hay medición en un cliente real: el coste real es desconocido.

Acciones: (a) subir precio o cobrar por cámara con mínimo (≥ 25–30 USD/cámara) hasta bajar coste; (b) medir 1–2 semanas de coste real en pilotos; (c) subir `clear_margin`/calibrar `persona` (ciclo 4, paso 2) y `HEIMDALL_VLM_MIN_INTERVAL`; (d) YOLOv8-Pose para caídas (quita una clase entera del VLM); (e) tope de gasto por cliente/mes en el worker (kill-switch de presupuesto VLM).

## 2. Precisión (bloqueante n.º 2)

Estado medido: 6/6 negativos limpios, **1/4 incidentes** detectados (solo `pelea_calle`) en 10 escenas
de Wikimedia. Un falso positivo verificado en caídas (persona sentada). 

Problemas de la evidencia, no solo del sistema:
- n = 10 clips: no se puede afirmar ninguna precisión/recall con significancia. No hay IC, ni curva PR, ni tasa de falsas alarmas por cámara-hora (la métrica que el cliente siente).
- Todo se juzga sobre **fotogramas sueltos**; caídas y peleas son temporales. Techo conocido.
- Clips de internet ≠ CCTV real (ángulo, baja luz, IR nocturno, compresión). No hay un solo clip nocturno.
- `accidente laboral` solo se evalúa en negativo; `robos` apenas validado; `cuchillo/arma` anunciado en el README pero sin ninguna escena de prueba.
- Verdad de referencia rígida (penaliza aciertos legítimos, p. ej. `caidas` en disturbios).
- Sin bucle de retroalimentación: el cliente no puede marcar "falsa alarma / correcta", así que no hay dato para mejorar con uso real.

Acciones: (a) subir a ≥ 100 clips etiquetados incl. noche/IR, y reportar falsas alarmas/cámara-día y recall por clase con IC; (b) botón 👍/👎 por detección en la consola + guardar el veredicto (es el dataset de mejora y la métrica de producto); (c) NO prometer en marketing clases no validadas (armas, accidente laboral) hasta medirlas; (d) CI que corra el banco en cada PR (hay ramas `benchmark-suite` y `ci-benchmark-quick` sin mergear a main).

## 3. Salud operativa (bloqueante n.º 3)

Hallazgos concretos del código:

1. **Cámara caída = instancia facturando sin producir** (documentado en BITACORA, sin corregir). `frames_from_rtsp` reintenta para siempre en modo 1-cámara. Falta auto-terminación tras N minutos sin frames + aviso al cliente "tu cámara está desconectada".
2. **Fallos silenciosos**: `raise_alert` envuelve todo en `try/except` y solo hace `print`. Si S3, `storeRegister` o la notificación fallan, la detección **se pierde sin reintento ni cola ni métrica**. `_post` no tiene reintentos ni backoff y pide un token Firebase en cada POST.
3. **Estado en memoria**: cooldowns (`_last_alert`, `_last_notify`) y `_ultimo_vlm` mueren con la instancia; un reinicio = ráfaga de SMS/llamadas VLM.
4. **Configuración hardcodeada**: bucket `detection-frames-tests` (nombre de test en producción), hosts de API Gateway por defecto en el código, región `us-east-1` fija. Un solo bucket y un solo prefijo para todos los clientes (`cameras/`): el aislamiento por cliente depende solo de la aplicación.
5. **Caja de análisis única y compartida** para todos los clientes: SPOF, sin equidad entre tenants (una cámara ruidosa degrada la latencia de las demás) y sin autoescalado. El arranque en frío tras 2 h apagada añade minutos de retraso a la **primera alerta** — para seguridad eso es inaceptable sin documentarlo o sin mínimo "caliente" en planes pagos.
6. **Cero observabilidad**: no hay métricas (latencia detección→notificación, profundidad de cola, errores VLM, gasto/h), alarmas CloudWatch ni dashboard de estado; la salud se infiere de un heartbeat. No hay página de estado para clientes.
7. **Sin límite de gasto ni rate-limit** en Lambdas/Bedrock (más allá del plan de cámaras): un bug o abuso puede generar factura abierta. Faltan presupuestos AWS + alarmas.
8. **Sin pruebas automáticas ni CI** en main (`test.py` de 178 bytes). El bug de `import os` faltante (ciclo 1) pasó `py_compile`. Higiene: `__pycache__/` y `yolov5s.pt` (14 MB) versionados; 8+ scripts `rtsp_*.py` obsoletos.
9. **Landing**: facturación en `MANUAL_ACTIVATION_MODE` (el admin activa a mano). No escala más allá de decenas de clientes; `localStorage` para el estado de monitoreo (por navegador). Un `.env.example` con PayU en sandbox por defecto.
10. **Infra manual**: Lambdas desplegadas con `zip` a mano; sin IaC (Terraform/CDK), sin entornos dev/prod separados, sin plan de recuperación (DynamoDB sin PITR documentado).

## 4. Legal / confianza (bloqueante para vender a empresas)

Video de personas = dato personal (y potencialmente biométrico-adyacente). Para clientes en Colombia/LatAm: Ley 1581 (Habeas Data) y política de tratamiento, aviso de videovigilancia, retención. Falta: política de privacidad, términos, **retención automática de frames en S3 (lifecycle)**, DPA para clientes B2B, y claridad de que las imágenes se envían a AWS Bedrock. Sin esto, cualquier cliente serio de seguridad lo pedirá antes de firmar.

## 5. Plan priorizado

**Antes de aceptar clientes de pago (1–2 semanas)**
1. Auto-terminación + aviso por cámara caída (#3.1). Reintentos/cola para `raise_alert` (#3.2).
2. Precio ≥ coste: replantear cam5; tope de gasto VLM por cliente; presupuestos y alarmas AWS.
3. Bucket/prefijo por cliente, lifecycle de retención (p. ej. 14–30 días), renombrar bucket.
4. Botón de feedback 👍/👎 en la consola.
5. Política de privacidad + términos.

**Antes de "muchos" (1–2 meses)**
6. Banco ≥ 100 clips con noche/IR + CI con benchmark; métricas de falsas alarmas/cámara-día.
7. YOLOv8-Pose para caídas; calibrar `persona`; evaluar armas por separado o retirarlas del discurso.
8. Observabilidad (CloudWatch métricas + dashboard + página de estado), SLO de latencia.
9. Caja de análisis con ≥ 2 instancias/autoescalado por profundidad de cola; mínimo caliente en planes pagos.
10. Pagos automáticos (PayU/Stripe en producción) + IaC + entornos separados.

## Para go-to-market (contexto a compartir con la tarea de adquisición)

- **A quién SÍ vender hoy**: cámaras en espacios que pasan mucho tiempo vacíos (local cerrado de noche, almacén, bodega, pasillo): ahí la puerta de personas baja el coste y hay menos falsas alarmas. Evitar vender a escenas con multitud permanente (ahí el coste sube y la precisión es la peor).
- **Qué prometer**: "alerta de persona en horario cerrado" y "persona en el suelo" (probadas). **No prometer** armas, robos ni accidentes laborales como funciones garantizadas todavía.
- **Formato**: piloto de 30 días con 3–5 clientes, cobrar algo desde el día 1 (aunque sea simbólico), instrumentar coste real y feedback 👍/👎. Eso resuelve a la vez los bloqueantes 1 y 2 y da casos de éxito.
- **Precio**: con el coste medido, cam5 a 50 USD pierde dinero. Cualquier campaña que ancle ese precio hay que alinearla antes.
- **Capacidad real hoy**: un solo analysis box compartido ⇒ máx. razonable ≈ unas decenas de cámaras antes de ver latencia. No cerrar volumen de ventas por encima de eso sin el punto 9.
- Pedir a la tarea de adquisición: segmentos y canales con mayor encaje con "local cerrado de noche", precios de la competencia (para fijar el precio por cámara), y qué quejas de falsas alarmas esperan los clientes en ese segmento.

# Estado del sistema y ruta a "listo para muchos clientes"

Informe de una sesión programada de investigación (2026-09-05), a través de
los 7 repositorios de SkyEye (`harmsDetection`, `harmsDetectionLandingUi`,
`HeimdalManager`, `StoreDevice`, `StoreDetection`, `ListDevices`,
`ListDetections`). Vive en el repo por la misma razón que `BITACORA.md`:
para que una sesión futura —de negocio o de ingeniería— parta de aquí.

## 1. Accuracy — el bloqueante real

El banco de pruebas (`testbench/`, ciclo 3, `BITACORA.md`) mide hoy:

```
negativos limpios    6 / 6
positivos detectados 1 / 4   (solo pelea_calle; caídas y disturbios sin confirmar)
```

Esto es una mejora real frente al ciclo 0 (0/4), y el coste bajó un 87%
(120,02 → 14,99 USD/cámara/mes de Bedrock). Pero el recall medido en
incidentes reales sigue siendo bajo, y el propio banco lo subestima poco:
10 escenas de Wikimedia Commons no son representativas de los clientes
reales, y no existe ni un solo positivo verificado de "accidente laboral"
(limitación ya documentada — no hay material libre que lo muestre).

**Vender el producto a muchos clientes hoy, con este nivel de evidencia,
es un riesgo reputacional y potencialmente legal**: un cliente que paga por
"detección de robos/violencia" y no recibe la alerta en un incidente real
puede considerarlo un incumplimiento, no una limitación técnica razonable.

El propio `BITACORA.md` ya tiene el ciclo 4 planeado y sin ejecutar:

1. YOLOv8-Pose para caídas (92–98% sin VLM, según la investigación citada) —
   sacaría una clase entera de las manos del VLM y del banco de pruebas actual.
2. Calibrar `clear_margin` para `persona` (la etiqueta con más tráfico al VLM).
3. Aplicar a `robos` el mismo principio de "rastro" que ya funcionó en
   `caidas` (preguntar por indicios visibles, no por el acto en curso).
4. Flexibilizar la verdad de referencia del banco, que hoy penaliza aciertos
   válidos (ver ciclo 3: dos alertas de `caidas` en `disturbios_calle`
   correctas pero contadas como fallo).

**Recomendación**: no escalar adquisición de clientes en paralelo a esto sin
comunicarlo. Ejecutar el ciclo 4 es la palanca de mayor impacto sobre si el
producto sostiene una base de clientes real.

## 2. Salud de ingeniería

Buena base de seguridad ya construida en los 5 Lambdas y en el frontend:
aislamiento por partition key (nunca scan+filter), CORS por allow-list,
credenciales RTSP en Secrets Manager (nunca en logs ni en la base), TTL de
30 días en eventos, tokens Firebase verificados server-side, arranque
idempotente en HeimdalManager, límites de plan aplicados en servidor (no en
el cliente).

Lo que falta y es común a los 7 repos:

- **Cero CI.** Ningún repo tiene `.github/workflows`. El despliegue de los
  Lambdas es manual (`zip` + `aws lambda update-function-code`), documentado
  así en cada README. Un solo despliegue erróneo no tiene red de seguridad.
- **Cero tests automatizados en CI.** `harmsDetection` tiene
  `cascade/integration_test.py`, `cascade/camera_sim_test.py` y el propio
  testbench, pero no corren en ningún pipeline. Los 5 Lambdas Node no tienen
  ningún test.
- **Cero observabilidad además de logs.** Ningún README menciona alarmas de
  CloudWatch, dashboards, ni tracking de tasa de error o latencia. Con un
  solo cliente esto se compensa mirando logs a mano; con muchos, no.
- **Sin rate limiting/WAF documentado** en las APIs detrás de API Gateway.

## 3. Escalabilidad de la arquitectura

El diseño ya apunta en la dirección correcta para muchos clientes:

- Cascada de 3 niveles + "motion box" compartido para muchas cámaras +
  "analysis box" bajo demanda que se autoapaga tras 2h de inactividad.
- HeimdalManager impone límites de plan del lado servidor y es idempotente.
- Coste proyectado en Fase B (10 cámaras compartidas): ~19,29 USD/mes total,
  frente a 124,92 USD en la línea base.

Lo que no está validado todavía:

- **Carga multi-tenant real**: nadie ha simulado N clientes × M cámaras
  simultáneas contra las colas SQS compartidas y Bedrock. El "vecino
  ruidoso" (una cámara con mucho movimiento saturando la cola de otras) no
  está descartado.
- **Throttling de Bedrock** a escala: `HEIMDALL_VLM_MIN_INTERVAL` limita por
  cámara, no hay límite agregado por cuenta/región documentado.
- **Fallos de red variados por cliente** (túneles RTSP inestables, cámaras
  domésticas vs. profesionales) — el testbench simula fallos controlados,
  pero no la diversidad real que traerá una base de clientes heterogénea.

## 4. Producto / negocio (para coordinar con la tarea de adquisición de clientes)

Ya construido: landing, checkout (PayU por defecto, Stripe alternativo),
panel de cuenta, panel de admin, catálogo de planes centralizado
(`lib/plans.ts`). Buena separación de responsabilidades (activación de plan
por webhook, no por redirect).

Lo que falta y afecta directamente a "conseguir clientes":

- **Nada visible de confianza/transparencia** para un cliente potencial:
  sin página de estado, sin cifras de latencia de alerta, sin casos de uso
  por vertical.
- **El mensaje de marketing no debería prometer detección infalible.** La
  fortaleza real y medible hoy es el **coste** (-87% desde la línea base) y
  la ausencia de falsos positivos en negativos limpios (6/6). La detección
  de incidentes abstractos (violencia, robos) sigue siendo débil.
- **Vertical más fuerte hoy**: presencia de personas + caídas (una vez se
  implemente pose-based) es más defendible que violencia/robos. Sugerir a
  la tarea de adquisición de clientes enfocar el mensaje inicial ahí
  (ej. adultos mayores, obra, residencias) en vez de seguridad genérica.
- **Legal/privacidad no revisado en esta sesión**: vigilancia por CCTV con
  clientes en Latinoamérica (PayU) implica normativas de protección de datos
  y consentimiento de terceros grabados que no aparecen documentadas en
  ningún repo.
- **Pilotos con clientes reales generarían el material de evaluación que
  hoy falta** (el banco actual es de clips libres de Wikimedia, no de
  cámaras de clientes).

## 5. Prioridades recomendadas, en orden

1. Ejecutar ciclo 4 de `BITACORA.md` (YOLOv8-Pose para caídas es la palanca
   de mayor impacto sobre accuracy).
2. Observabilidad mínima: alarmas CloudWatch en errores 5xx y latencia por
   Lambda, dashboard de coste/detecciones por cámara.
3. CI mínimo: correr los tests ya existentes de `harmsDetection` en cada
   push, y añadir lint + smoke test a los 5 Lambdas Node.
4. Prueba de carga multi-tenant simulada antes de abrir registro masivo.
5. Rate limiting/WAF en API Gateway.
6. Página de estado y transparencia de limitaciones para clientes.
7. Revisión legal de privacidad/videovigilancia antes de escalar ventas.
8. Ampliar el banco de pruebas con diversidad real (idealmente pilotos).

## Nota de coordinación

Existe (o existirá) una tarea programada separada investigando cómo
conseguir clientes para SkyEye. Este documento es el contexto técnico que
esa tarea debería tener antes de proponer mensajes de marketing o
compromisos de SLA: la fortaleza actual es el coste, no todavía la
cobertura de incidentes abstractos, y prometer lo segundo sin el ciclo 4
ejecutado es el riesgo más alto identificado en esta investigación.

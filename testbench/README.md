# Banco de pruebas: cámara mock

Cámara RTSP simulada para probar SkyEye de punta a punta con grabaciones reales,
sin depender de que un cliente tenga su túnel levantado.

## Por qué

Hasta ahora, probar la detección exigía una cámara real detrás de un túnel ngrok.
Eso hace las pruebas irrepetibles (el túnel se cae, la escena cambia) y arriesgadas
(las alertas van a un usuario real, con SMS incluidos).

Esta cámara sirve grabaciones libres por RTSP desde una EC2 propia. El worker **no
distingue esto de una cámara de cliente**: recibe una URL RTSP normal, la resuelve
desde Secrets Manager y ejerce el camino de producción completo — `storeDevice` →
`HeimdalManager` → worker EC2 → CLIP → VLM → S3 + DynamoDB + notificaciones.

## Uso

```bash
python3 mock_camera.py up          # crea o enciende la cámara
python3 mock_camera.py register    # registra el dispositivo y enciende el monitoreo
python3 mock_camera.py rate 6      # mide alertas/minuto durante 6 minutos
python3 mock_camera.py chaos down  # simula un túnel caído
python3 mock_camera.py chaos on    # restaura la emisión
python3 mock_camera.py stop        # apaga monitoreo y cámara (sin destruir)
```

## Apagar, no terminar

La instancia está pensada para **apagarse**, no destruirse. En una VPC la IP privada
se conserva al parar y arrancar, así que:

- la URL RTSP registrada en SkyEye sigue siendo válida entre pruebas,
- el secreto `heimdall/rtsp/<deviceId>` no hay que rehacerlo,
- los vídeos ya normalizados y MediaMTX siguen en el disco, así que arrancar de
  nuevo sirve stream en ~40 s en vez de 2-3 minutos.

Parada cuesta solo el EBS (12 GB gp3, bastante menos de 1 USD al mes).

## Sin SMS

Todo corre bajo el uid `skyeye-test-harness`, que **no tiene canales en
`userSettings`**, así que ninguna prueba envía SMS ni correos. `register` aborta si
detecta que ese uid tiene canales configurados. Se le crea una suscripción activa
porque HeimdalManager exige plan para encender el monitoreo.

## Cortes controlados (prueba de reconexión)

La instancia consulta `s3://detection-frames-tests/testcam/control.json` cada 15 s,
así que se puede provocar un fallo sin SSH ni SSM:

| Estado | Efecto | Simula |
|---|---|---|
| `on` | emisión normal | cámara sana |
| `off` | se para el publicador; el servidor sigue en pie pero sin vídeo | cámara muda |
| `down` | se paran servidor y publicador: *connection refused* | túnel ngrok caído |

## Grabaciones

Wikimedia Commons, licencias libres, con peatones bien visibles:

- *Scramble Crossing at Robinson Road* (Singapur) — CC BY 4.0
- *Kruciĝo de stratoj Respubliko kaj Orĝonikidze* (Tiumén) — CC BY-SA 4.0

Se normalizan a H.264 960x540@15fps con un keyframe por segundo (`-g 15`), para que
el worker abra el stream rápido; el bucle usa `-c copy`, así que la CPU en régimen
permanente es mínima. Añadir escenas es sumar entradas al array `CLIPS` de
`camera_userdata.sh`.

## Comportamiento ante cortes (medido)

Provocando `chaos down` sobre un worker en marcha:

| Hora | Evento |
|---|---|
| 17:36:27 | corte aplicado |
| 17:36:39 | primer `Connection refused` (cada ~3,5 s a partir de aquí) |
| 17:37:18 | emisión restaurada |
| 17:37:32 | primera detección nueva: **reconectado solo** |

Dos conclusiones:

- **El worker reconecta sin intervención**, unos 14 s después de que vuelva el
  stream. Una cámara que se recupera vuelve a detectar sola.
- **El worker NO se auto-termina** mientras la cámara está caída: sigue vivo
  reintentando, así que una `m7i-flex.large` puede facturar horas sin producir
  nada. Conviene revisar el umbral de auto-terminación por cámara inalcanzable.

## Para qué sirve

Referencia medida con este banco: antes del cooldown, una escena transitada generaba
**83,6 alertas/minuto** (~5.000/hora por cámara), cada una con su SMS. El techo
teórico con `ALERT_COOLDOWN=20s` es 3/min por (cámara, etiqueta).

Úsalo para validar cambios de detección, de umbrales o de caudal antes de que
lleguen a una cámara de cliente.

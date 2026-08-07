# harmsDetection — Worker "Heimdall Eye"

Worker de detección en tiempo real de **SkyEye**: el proceso Python que corre
en una instancia EC2 por cámara, lee el stream RTSP, detecta eventos con IA
(OpenAI CLIP) y dispara las alertas.

> 📚 Contexto completo del proyecto: `docs/SKYEYE_PROJECT.md` en el repo
> `harmsDetectionLandingUi`.

## Cómo funciona (`heimdall-eye.py`)

Pipeline en dos etapas para que la IA solo corra cuando hace falta:

1. **Captura RTSP resiliente** — transporte TCP forzado (los túneles ngrok no
   reenvían UDP), reintentos con espera de keyframe (GOPs largos de TP-Link) y
   tolerancia a lecturas fallidas antes de reconectar (reconectar en cada
   fallo agota el pool de sesiones de la cámara).
2. **Etapa 1 (barata): movimiento** — sustracción de fondo MOG2 sobre el frame
   reducido al 50%; produce ROIs (regiones con movimiento), máximo 10.
3. **Etapa 2 (cara): CLIP** — ViT-B/32 (FP16 en GPU) clasifica los ROIs en
   batch contra los prompts de la lista de detección. Cada ~3 s corre además
   un **barrido de frame completo** (sliding window) como red de seguridad
   para objetivos estáticos que el modelo de fondo ya "aprendió".
4. **Alerta** — con score coseno > 0.27, 3 frames positivos consecutivos y
   cooldown de 10 s: en un pool de I/O (nunca bloquea la captura) sube el
   frame a S3 (`detection-frames-tests/cameras/`), registra el evento vía el
   endpoint `storeRegister` (Lambda `StoreDetection`) y el usuario recibe la
   notificación por sus canales configurados.

Las palabras de detección llegan en español desde la UI
(`detection_blacklist` en `context.json`: caidas, robos, violencia, persona,
cuchillo…) y se expanden a variantes de prompt en inglés (CLIP fue entrenado
con alt-text en inglés), reportando siempre la palabra original del usuario.

Parámetros ajustables por variable de entorno: `HEIMDALL_THRESHOLD`,
`HEIMDALL_MIN_INTERVAL`, `HEIMDALL_ALERT_THRESHOLD`, `HEIMDALL_ALERT_COOLDOWN`,
`HEIMDALL_WARMUP_FRAMES`, `HEIMDALL_MAX_READ_FAILURES`,
`HEIMDALL_OPEN_TIMEOUT`, `HEIMDALL_FULL_SCAN_INTERVAL`.

## Archivos

| Archivo | Rol |
|---|---|
| `heimdall-eye.py` | **Worker vigente** (el que corre en producción) |
| `firebase_auth.py` | Autenticación del worker: custom token Firebase (uid `heimdall`) → ID token, con caché y renovación; el secreto sale de `heimdall/firebase` (Secrets Manager) |
| `context.json` | Configuración de ejemplo por cámara (`camera_name`, `detection_blacklist`, `rtsp_path`/secreto, `client_id`) — en producción la escribe el UserData de `HeimdalManager` |
| `deploy-worker.sh` | Publica el worker en `s3://detection-frames-tests/worker/`; cada instancia EC2 sincroniza esa versión al arrancar |
| `requirements.txt` | Dependencias (torch, opencv-headless, CLIP, boto3, …) |
| `Dockerfile` | Imagen CUDA para pruebas del prototipo multicore |
| `multicore_detection*.py`, `rtsp_*.py`, `minimal.py`, `recordCamera.bat` | **Prototipos/legacy** (versiones antiguas con Twilio); no se usan en producción |
| `listDevices.mjs`, `listDetections.mjs` | Copias antiguas de las Lambdas; las versiones vigentes viven en los repos `ListDevices` y `ListDetections` |

## Ejecución local

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

python3 heimdall-eye.py context.json
```

Requiere `ffmpeg` instalado en el host (`apt install ffmpeg`) y credenciales
AWS con acceso a S3 y Secrets Manager.

## Despliegue

```bash
./deploy-worker.sh
```

Sube `heimdall-eye.py` y `firebase_auth.py` a S3; las **próximas** instancias
EC2 lanzadas por `HeimdalManager` usarán esa versión (descarga atómica: si
falla, la instancia usa la copia horneada en la AMI). En la instancia el
worker corre bajo systemd (`heimdall-worker.service`) con reinicio automático.

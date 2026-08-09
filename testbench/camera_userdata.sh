#!/bin/bash
# Cámara RTSP simulada para probar SkyEye de punta a punta.
# Sirve en bucle grabaciones libres (Wikimedia Commons) por RTSP, igual que una
# cámara IP real: MediaMTX escucha en :8554 y ffmpeg le empuja el vídeo.
#
# Pensada para APAGAR/ENCENDER, no para recrear: los vídeos ya normalizados viven
# en el disco EBS y todos los servicios quedan habilitados, así que un arranque
# posterior sirve el stream en ~40 s sin volver a descargar ni transcodificar.
# La IP privada se conserva al parar/arrancar, así que la URL RTSP registrada en
# SkyEye sigue siendo válida entre pruebas.
set -x
exec > >(tee -a /var/log/skyeye-cam.log) 2>&1

S3_BUCKET="detection-frames-tests"
S3_PREFIX="testcam"
WORKDIR="/opt/stream"

say() { echo "[skyeye-cam $(date -Is)] $*"; }

say "=== arranque de la cámara de prueba ==="
export DEBIAN_FRONTEND=noninteractive
APT="apt-get -o DPkg::Lock::Timeout=600 -y"

$APT update
# Instalaciones por separado: un paquete inexistente no debe tumbar al resto
# (en 24.04 'awscli' no existe en el archivo, y arrastraba a ffmpeg consigo).
$APT install ffmpeg curl || say "WARN: fallo instalando ffmpeg/curl"
$APT install python3-boto3 || pip3 install --break-system-packages boto3 || say "WARN: sin boto3"

command -v ffmpeg >/dev/null || { say "FATAL: ffmpeg no disponible"; exit 1; }
say "ffmpeg OK: $(ffmpeg -version | head -1)"

# Subida a S3 vía boto3 (sin AWS CLI, que no está empaquetado en 24.04).
cat > /usr/local/bin/s3put <<'PYEOF'
#!/usr/bin/env python3
import sys, boto3
boto3.client("s3", region_name="us-east-1").upload_file(sys.argv[1], "detection-frames-tests", sys.argv[2])
PYEOF
chmod +x /usr/local/bin/s3put
push_log() { s3put /var/log/skyeye-cam.log "${S3_PREFIX}/status.log" || true; }
push_log

cat > /etc/systemd/system/skyeye-camlog.service <<'UNIT'
[Unit]
Description=Sube el log de la camara de prueba a S3
[Service]
ExecStart=/bin/bash -c 'while true; do /usr/local/bin/s3put /var/log/skyeye-cam.log testcam/status.log || true; sleep 20; done'
Restart=always
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now skyeye-camlog.service

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# --- 1. Descargar grabaciones libres -----------------------------------------
# Wikimedia Commons, licencias libres, con peatones bien visibles:
#  - Scramble Crossing, Robinson Road (Singapur)          CC BY 4.0
#  - Kruciĝo de stratoj Respubliko/Orĝonikidze (Tiumén)   CC BY-SA 4.0
# Para añadir escenas basta con sumar entradas a CLIPS (url|nombre).
UA="SkyEyeTestHarness/1.0 (prueba de monitoreo)"
CLIPS=(
  "https://upload.wikimedia.org/wikipedia/commons/f/f2/Scramble_Crossing_at_Robinson_Road_in_Singapore_-_September_2022.webm|v1"
  "https://upload.wikimedia.org/wikipedia/commons/8/8f/Kruci%C4%9Do_de_stratoj_Respubliko_kaj_Or%C4%9Donikidze_%28Tjumeno%29.webm|v2"
)

# El worker abre el stream con OpenCV/FFmpeg y espera un keyframe: con -g 15
# (uno por segundo a 15 fps) la apertura es rápida. Luego el bucle usa -c copy,
# así que la CPU en régimen permanente es mínima.
transcode() {
  ffmpeg -y -loglevel warning -i "$1" \
    -vf "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:-1:-1,fps=15" \
    -c:v libx264 -preset veryfast -profile:v main -pix_fmt yuv420p -g 15 -an "$2"
}

: > list.txt
for entry in "${CLIPS[@]}"; do
  url="${entry%%|*}"; name="${entry##*|}"
  if [ -s "${name}.mp4" ]; then
    say "${name}.mp4 ya está en disco, se reutiliza"
  else
    say "descargando ${name}..."
    curl -fsSL -A "$UA" -o "${name}.src" "$url" || { say "ERROR descargando ${name}"; continue; }
    say "${name} descargado ($(du -h ${name}.src | cut -f1)), normalizando..."
    transcode "${name}.src" "${name}.mp4" || { say "ERROR transcodificando ${name}"; continue; }
    rm -f "${name}.src"
    say "${name}.mp4 listo ($(du -h ${name}.mp4|cut -f1))"
  fi
  echo "file '${WORKDIR}/${name}.mp4'" >> list.txt
  push_log
done

if [ ! -s list.txt ]; then
  say "FATAL: no hay ningún vídeo utilizable"; push_log; exit 1
fi
ffmpeg -y -loglevel warning -f concat -safe 0 -i list.txt -c copy loop.mp4
say "loop.mp4 listo: $(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 loop.mp4)s"
push_log

# --- 2. Servidor RTSP --------------------------------------------------------
# MediaMTX se comporta como una cámara IP real (RTSP sobre TCP, multi-cliente,
# reconexión). Si la descarga fallara, se cae a ffmpeg en modo listen.
MTX_OK=0
[ -x "${WORKDIR}/mediamtx" ] && MTX_OK=1 && say "MediaMTX ya instalado en disco"
if [ "$MTX_OK" = "0" ]; then
  LATEST="$(curl -fsSL https://api.github.com/repos/bluenviron/mediamtx/releases/latest \
            | grep -oP '"tag_name":\s*"\K[^"]+' || true)"
  for V in "$LATEST" v1.15.1 v1.11.3 v1.9.3; do
    [ -z "$V" ] && continue
    if curl -fsSL -o mediamtx.tar.gz \
        "https://github.com/bluenviron/mediamtx/releases/download/${V}/mediamtx_${V}_linux_amd64.tar.gz"; then
      tar xzf mediamtx.tar.gz && [ -x ./mediamtx ] && MTX_OK=1 && say "MediaMTX ${V} instalado" && break
    fi
    say "no se pudo bajar MediaMTX ${V}, probando siguiente"
  done
fi
push_log

if [ "$MTX_OK" = "1" ]; then
  cat > /etc/systemd/system/skyeye-rtspserver.service <<UNIT
[Unit]
Description=Servidor RTSP (MediaMTX) de la camara de prueba
After=network-online.target
Wants=network-online.target
[Service]
ExecStart=${WORKDIR}/mediamtx
WorkingDirectory=${WORKDIR}
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT

  cat > /etc/systemd/system/skyeye-publisher.service <<UNIT
[Unit]
Description=Publica el bucle de video en el servidor RTSP
After=skyeye-rtspserver.service
Requires=skyeye-rtspserver.service
[Service]
ExecStartPre=/bin/sleep 5
ExecStart=/usr/bin/ffmpeg -nostdin -re -stream_loop -1 -i ${WORKDIR}/loop.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now skyeye-rtspserver.service skyeye-publisher.service
else
  say "MediaMTX no disponible; usando ffmpeg en modo listen"
  cat > /etc/systemd/system/skyeye-publisher.service <<UNIT
[Unit]
Description=Servidor RTSP de respaldo (ffmpeg listen)
After=network-online.target
[Service]
ExecStart=/usr/bin/ffmpeg -nostdin -re -stream_loop -1 -i ${WORKDIR}/loop.mp4 -c copy -f rtsp -rtsp_flags listen rtsp://0.0.0.0:8554/cam1
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now skyeye-publisher.service
fi

# --- 3. Control remoto de cortes (prueba de reconexión) ----------------------
# Sin SSH ni SSM: la instancia consulta un fichero de control en S3 cada 15 s.
#   {"stream":"on"}   -> emisión normal
#   {"stream":"off"}  -> se para el publicador: el servidor sigue en pie pero sin
#                        vídeo (la cámara "se queda muda")
#   {"stream":"down"} -> se paran servidor y publicador: connection refused, que
#                        es exactamente el síntoma de un túnel ngrok caído
cat > /usr/local/bin/skyeye-control <<'PYEOF'
#!/usr/bin/env python3
import json, subprocess, time, boto3, botocore
s3 = boto3.client("s3", region_name="us-east-1")
KEY = "testcam/control.json"
SRV = "skyeye-rtspserver.service"
PUB = "skyeye-publisher.service"

def sysctl(action, *units):
    subprocess.run(["systemctl", action, *units], check=False)

applied = None
while True:
    try:
        body = s3.get_object(Bucket="detection-frames-tests", Key=KEY)["Body"].read()
        want = json.loads(body).get("stream", "on")
    except botocore.exceptions.ClientError:
        want = "on"          # sin fichero de control -> emisión normal
    except Exception:
        want = applied or "on"
    if want != applied:
        print(f"[control] estado solicitado: {want}", flush=True)
        if want == "on":
            sysctl("start", SRV); time.sleep(2); sysctl("start", PUB)
        elif want == "off":
            sysctl("stop", PUB)
        elif want == "down":
            sysctl("stop", PUB, SRV)
        applied = want
    time.sleep(15)
PYEOF
chmod +x /usr/local/bin/skyeye-control

cat > /etc/systemd/system/skyeye-control.service <<'UNIT'
[Unit]
Description=Aplica cortes de emision segun el fichero de control en S3
After=network-online.target
[Service]
ExecStart=/usr/local/bin/skyeye-control
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now skyeye-control.service

sleep 15
say "publisher: $(systemctl is-active skyeye-publisher.service) / servidor: $(systemctl is-active skyeye-rtspserver.service 2>/dev/null) / control: $(systemctl is-active skyeye-control.service)"
push_log

# --- 4. Autocomprobación: capturar un fotograma del propio stream ------------
say "autocomprobación: leyendo un fotograma de rtsp://127.0.0.1:8554/cam1"
OK=0
for i in 1 2 3 4 5 6; do
  if ffmpeg -y -loglevel warning -rtsp_transport tcp -i rtsp://127.0.0.1:8554/cam1 \
       -frames:v 1 /tmp/selftest.jpg && [ -s /tmp/selftest.jpg ]; then
    s3put /tmp/selftest.jpg "${S3_PREFIX}/selftest.jpg" && OK=1
    say "OK: stream servido correctamente"
    break
  fi
  say "intento $i fallido, reintentando en 5s"
  sleep 5
done
[ "$OK" = "1" ] || say "FATAL: el stream RTSP no responde"
say "=== camara lista ==="
push_log

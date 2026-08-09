#!/bin/bash
# Cámara RTSP simulada multi-escena para evaluar SkyEye.
#
# Sirve por RTSP una escena seleccionable en caliente, para poder medir la
# detección sobre situaciones distintas (normales y de incidente) sin depender
# de ninguna cámara real. Todo el material es de Wikimedia Commons con licencia
# libre verificada; la atribución vive en testbench/CLIPS.md.
set -x
exec > >(tee -a /var/log/skyeye-cam.log) 2>&1

S3_BUCKET="detection-frames-tests"
S3_PREFIX="testcam"
WORKDIR="/opt/stream"
say() { echo "[skyeye-cam $(date -Is)] $*"; }

say "=== arranque de la camara multi-escena ==="
export DEBIAN_FRONTEND=noninteractive
APT="apt-get -o DPkg::Lock::Timeout=600 -y"
$APT update
$APT install ffmpeg curl || say "WARN: fallo instalando ffmpeg/curl"
$APT install python3-boto3 || pip3 install --break-system-packages boto3 || say "WARN: sin boto3"
command -v ffmpeg >/dev/null || { say "FATAL: ffmpeg no disponible"; exit 1; }

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

mkdir -p "$WORKDIR"; cd "$WORKDIR"
UA="SkyEyeTestHarness/1.0 (evaluacion de deteccion)"

# escena|url|inicio(s)|duracion(s)
# --- NEGATIVAS: situacion normal, NO deberia alertar ---
# --- POSITIVAS: incidente real, SI deberia alertar ---
ESCENAS=(
  "calle_peatones|https://upload.wikimedia.org/wikipedia/commons/f/f2/Scramble_Crossing_at_Robinson_Road_in_Singapore_-_September_2022.webm|10|40"
  "naturaleza_vacia|https://upload.wikimedia.org/wikipedia/commons/7/75/Mount_Rainier_Weather_Timelapse.webm|0|40"
  "obra_normal|https://upload.wikimedia.org/wikipedia/commons/a/ab/Building_construction_Moira_Close_Broadwater_Farm_Haringey_2025_21.webm|60|40"
  "caida_escaleras|https://upload.wikimedia.org/wikipedia/commons/1/19/Bits_%26_Pieces_-_BP152_Falling_down_the_stairs_-_EYE_FLM7636_-_OB_105118.ogv|0|60"
  "caida_judo|https://upload.wikimedia.org/wikipedia/commons/4/44/Tai-otoshi_in_detail_by_Laszlo_Horvath_edited_0.webm|0|40"
  "disturbios_saqueo|https://upload.wikimedia.org/wikipedia/commons/1/17/Jacked_at_London_riots_-_8th_August.webm|0|40"
  "disturbios_calle|https://upload.wikimedia.org/wikipedia/commons/4/41/Medan-Indonesia_omnibus_law_riots.webm|0|40"
  "accidente_laboral|https://upload.wikimedia.org/wikipedia/commons/5/5a/Las_Ca%C3%ADdas_Cuestan_-_La_Historia_de_un_Safety_Man.webm|30|60"
)

# Normaliza a H.264 960x540@15fps con keyframe por segundo: el worker abre el
# stream rapido y el bucle luego usa -c copy (CPU minima en regimen permanente).
for e in "${ESCENAS[@]}"; do
  IFS='|' read -r name url ss dur <<< "$e"
  out="scene_${name}.mp4"
  if [ -s "$out" ]; then say "${name}: ya en disco"; continue; fi
  say "descargando ${name}..."
  curl -fsSL -A "$UA" -o "src_${name}" "$url" || { say "ERROR descarga ${name}"; continue; }
  say "normalizando ${name} (desde ${ss}s, ${dur}s)..."
  ffmpeg -y -loglevel warning -ss "$ss" -t "$dur" -i "src_${name}" \
    -vf "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:-1:-1,fps=15" \
    -c:v libx264 -preset veryfast -profile:v main -pix_fmt yuv420p -g 15 -an "$out" \
    || { say "ERROR transcodificando ${name}"; rm -f "src_${name}"; continue; }
  rm -f "src_${name}"
  # Hoja de contactos (4 fotogramas) para verificar VISUALMENTE el contenido.
  ffmpeg -y -loglevel error -i "$out" -vf "fps=1/8,scale=320:-1,tile=2x2" -frames:v 1 "thumb_${name}.jpg" || true
  [ -s "thumb_${name}.jpg" ] && s3put "thumb_${name}.jpg" "${S3_PREFIX}/thumbs/${name}.jpg"
  say "${name} listo ($(du -h $out|cut -f1))"
  push_log
done

ls -1 scene_*.mp4 > escenas.txt || true
say "escenas disponibles: $(tr '\n' ' ' < escenas.txt)"
DEFECTO="$(head -1 escenas.txt)"
ln -sf "${WORKDIR}/${DEFECTO}" "${WORKDIR}/current.mp4"
say "escena por defecto: ${DEFECTO}"
push_log

# --- Servidor RTSP ---
if [ ! -x "${WORKDIR}/mediamtx" ]; then
  LATEST="$(curl -fsSL https://api.github.com/repos/bluenviron/mediamtx/releases/latest | grep -oP '"tag_name":\s*"\K[^"]+' || true)"
  for V in "$LATEST" v1.15.1 v1.11.3; do
    [ -z "$V" ] && continue
    curl -fsSL -o mediamtx.tar.gz "https://github.com/bluenviron/mediamtx/releases/download/${V}/mediamtx_${V}_linux_amd64.tar.gz" \
      && tar xzf mediamtx.tar.gz && [ -x ./mediamtx ] && say "MediaMTX ${V} instalado" && break
  done
fi

cat > /etc/systemd/system/skyeye-rtspserver.service <<UNIT
[Unit]
Description=Servidor RTSP (MediaMTX)
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
Description=Publica la escena actual en el servidor RTSP
After=skyeye-rtspserver.service
Requires=skyeye-rtspserver.service
[Service]
ExecStartPre=/bin/sleep 3
ExecStart=/usr/bin/ffmpeg -nostdin -re -stream_loop -1 -i ${WORKDIR}/current.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cam1
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now skyeye-rtspserver.service skyeye-publisher.service

# --- Control remoto: escena + cortes, via fichero en S3 (sin SSH ni SSM) ---
cat > /usr/local/bin/skyeye-control <<'PYEOF'
#!/usr/bin/env python3
import json, os, subprocess, time, boto3, botocore
s3 = boto3.client("s3", region_name="us-east-1")
WORK, KEY = "/opt/stream", "testcam/control.json"
SRV, PUB = "skyeye-rtspserver.service", "skyeye-publisher.service"

def sysctl(action, *units):
    subprocess.run(["systemctl", action, *units], check=False)

def publicar_estado(escena, stream):
    s3.put_object(Bucket="detection-frames-tests", Key="testcam/estado.json",
                  Body=json.dumps({"escena": escena, "stream": stream,
                                   "ts": time.time()}).encode())

escena_actual, stream_actual = None, None
while True:
    try:
        cfg = json.loads(s3.get_object(Bucket="detection-frames-tests", Key=KEY)["Body"].read())
    except botocore.exceptions.ClientError:
        cfg = {}
    except Exception:
        time.sleep(5); continue
    escena = cfg.get("escena")
    stream = cfg.get("stream", "on")

    if escena and escena != escena_actual:
        destino = f"{WORK}/scene_{escena}.mp4"
        if os.path.exists(destino):
            subprocess.run(["ln", "-sf", destino, f"{WORK}/current.mp4"], check=False)
            sysctl("restart", PUB)          # el publicador recoge la escena nueva
            escena_actual = escena
            print(f"[control] escena -> {escena}", flush=True)
            publicar_estado(escena_actual, stream_actual or "on")
        else:
            print(f"[control] escena inexistente: {escena}", flush=True)

    if stream != stream_actual:
        print(f"[control] stream -> {stream}", flush=True)
        if stream == "on":
            sysctl("start", SRV); time.sleep(2); sysctl("start", PUB)
        elif stream == "off":
            sysctl("stop", PUB)
        elif stream == "down":
            sysctl("stop", PUB, SRV)
        stream_actual = stream
        publicar_estado(escena_actual or "?", stream_actual)
    time.sleep(5)
PYEOF
chmod +x /usr/local/bin/skyeye-control

cat > /etc/systemd/system/skyeye-control.service <<'UNIT'
[Unit]
Description=Selector de escena y cortes de emision
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

sleep 12
say "publisher: $(systemctl is-active skyeye-publisher.service) / servidor: $(systemctl is-active skyeye-rtspserver.service) / control: $(systemctl is-active skyeye-control.service)"
for i in 1 2 3 4 5 6; do
  if ffmpeg -y -loglevel warning -rtsp_transport tcp -i rtsp://127.0.0.1:8554/cam1 -frames:v 1 /tmp/selftest.jpg && [ -s /tmp/selftest.jpg ]; then
    s3put /tmp/selftest.jpg "${S3_PREFIX}/selftest.jpg"; say "OK: stream servido"; break
  fi
  say "autocomprobacion intento $i fallido"; sleep 5
done
say "=== camara lista ==="
push_log

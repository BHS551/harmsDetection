#!/bin/bash
# Publica la versión actual del worker para que las próximas instancias EC2
# la descarguen al arrancar (el UserData de HeimdalManager las sincroniza
# desde este bucket antes de lanzar el proceso).
#
# Uso: ./deploy-worker.sh
set -euo pipefail

BUCKET="detection-frames-tests"
PREFIX="worker"

aws s3 cp heimdall-eye.py "s3://${BUCKET}/${PREFIX}/heimdall-eye.py"
aws s3 cp firebase_auth.py "s3://${BUCKET}/${PREFIX}/firebase_auth.py"

echo "Worker publicado en s3://${BUCKET}/${PREFIX}/ — las próximas instancias usarán esta versión."

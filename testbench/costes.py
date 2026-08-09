#!/usr/bin/env python3
"""
Coste de monitorizar UNA cámara 24/7, a partir de consumo medido, no estimado.

Las tasas (llamadas al VLM por hora, alertas por hora) se sacan de los logs del
worker durante una evaluación real; los precios son de us-east-1.

Uso:  python3 costes.py <instance_id_del_worker> [minutos_observados]
"""
import sys
import time
from collections import Counter

import boto3

REGION = "us-east-1"
HORAS_MES = 730

# --- Precios us-east-1 (bajo demanda), USD ---
P_M7I_FLEX_LARGE = 0.0966      # worker dedicado por cámara
P_T3_SMALL = 0.0208            # motion box compartido
P_S3_GB_MES = 0.023
P_S3_PUT_1000 = 0.005
P_DDB_WRITE_MILLON = 1.25      # escritura bajo demanda (1 KB)
P_LAMBDA_INVOC_MILLON = 0.20
P_SECRETO_MES = 0.40           # un secreto RTSP por cámara
P_SMS = 0.06                   # SNS a Colombia, punto medio del rango 0.035-0.09
# Nova Lite: 0.06 USD/1M tokens de entrada, 0.24 USD/1M de salida.
# Un fotograma ~1300 tokens de entrada y ~40 de salida por juicio.
P_VLM_LLAMADA = (1300 * 0.06 + 40 * 0.24) / 1_000_000
TAM_FRAME_MB = 0.33            # medido sobre los frames subidos en las pruebas

logs = boto3.client("logs", region_name=REGION)


def tasas_medidas(instance_id, minutos):
    """Cuenta llamadas al VLM y alertas en los últimos `minutos` de log."""
    desde = int((time.time() - minutos * 60) * 1000)
    c = Counter()
    tok = None
    while True:
        kw = dict(logGroupName="/ec2/heimdall-eye", logStreamNames=[instance_id],
                  startTime=desde, limit=10000)
        if tok:
            kw["nextToken"] = tok
        r = logs.filter_log_events(**kw)
        for e in r.get("events", []):
            m = e["message"]
            if m.startswith("[vlm]"):
                c["vlm"] += 1
            elif m.startswith("[clip]"):
                c["clip"] += 1
            elif m.startswith("ALERTA"):
                c["alerta"] += 1
                if "notificado=si" in m:
                    c["notificacion"] += 1
        tok = r.get("nextToken")
        if not tok:
            break
    return {k: v / (minutos / 60.0) for k, v in c.items()}   # por hora


def coste(por_hora, sms_activo=False):
    vlm_h = por_hora.get("vlm", 0)
    alertas_h = por_hora.get("alerta", 0)
    notif_h = por_hora.get("notificacion", 0)

    filas = []
    # Los frames de alerta caducan a los 7 días -> el almacenamiento se estabiliza.
    gb_estables = alertas_h * 24 * 7 * TAM_FRAME_MB / 1024
    filas.append(("Bedrock Nova Lite (capa VLM)", vlm_h * HORAS_MES * P_VLM_LLAMADA))
    filas.append(("S3 PUT de frames", alertas_h * HORAS_MES / 1000 * P_S3_PUT_1000))
    filas.append(("S3 almacenamiento (TTL 7 días)", gb_estables * P_S3_GB_MES))
    filas.append(("DynamoDB (1 escritura/alerta)", alertas_h * HORAS_MES / 1e6 * P_DDB_WRITE_MILLON))
    # storeRegister + workerEvents(notify) + heartbeat cada 30 s
    invoc_h = alertas_h + notif_h + 120
    filas.append(("Lambda (registro, aviso, heartbeat)", invoc_h * HORAS_MES / 1e6 * P_LAMBDA_INVOC_MILLON))
    filas.append(("Secrets Manager (1 secreto RTSP)", P_SECRETO_MES))
    if sms_activo:
        filas.append(("SNS SMS", notif_h * HORAS_MES * P_SMS))
    return filas


def main():
    inst = sys.argv[1]
    minutos = float(sys.argv[2]) if len(sys.argv) > 2 else 25.0
    ph = tasas_medidas(inst, minutos)
    print(f"=== consumo medido ({minutos:.0f} min de {inst}) ===")
    for k in ("clip", "vlm", "alerta", "notificacion"):
        print(f"  {k:14} {ph.get(k,0):8.1f} / hora")

    variable = coste(ph)
    var_total = sum(v for _, v in variable)

    print("\n=== coste mensual por cámara, 24/7 ===")
    print("\n  Parte variable (igual en cualquier topología):")
    for n, v in variable:
        print(f"    {n:36} {v:8.2f}")
    print(f"    {'subtotal variable':36} {var_total:8.2f}")

    dedicada = P_M7I_FLEX_LARGE * HORAS_MES
    print(f"\n  A) Instancia dedicada por cámara (modelo actual)")
    print(f"    {'m7i-flex.large 24/7':36} {dedicada:8.2f}")
    print(f"    {'TOTAL':36} {dedicada + var_total:8.2f}")

    for n_cam in (5, 10, 20):
        motion = P_T3_SMALL * HORAS_MES / n_cam
        # La analysis box es compartida y se apaga sola tras 2 h ociosa; se
        # supone encendida ~8 h/día en un despliegue con actividad diurna.
        analysis = P_M7I_FLEX_LARGE * 8 * 30 / n_cam
        print(f"\n  B) Fase B compartida, {n_cam} cámaras por caja")
        print(f"    {'motion box t3.small (prorrateado)':36} {motion:8.2f}")
        print(f"    {'analysis box ~8 h/día (prorrateado)':36} {analysis:8.2f}")
        print(f"    {'TOTAL':36} {motion + analysis + var_total:8.2f}")


if __name__ == "__main__":
    main()

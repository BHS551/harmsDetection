#!/usr/bin/env python3
"""
Evaluación de SkyEye sobre escenas reales.

Reproduce una a una las escenas de la cámara mock (normales e incidentes) contra
un worker configurado para detectar varias situaciones a la vez, y construye la
matriz de confusión: qué etiqueta salta en qué escena.

Configurar TODAS las palabras en un solo worker permite medir la matriz completa
con un único arranque, en vez de uno por combinación.

Uso:
    python3 evaluar.py <device_id> [minutos_por_escena]
"""
import json
import sys
import time
from collections import defaultdict

import boto3

REGION = "us-east-1"
BUCKET = "detection-frames-tests"
UID = "skyeye-test-harness"

# escena -> etiquetas que un humano consideraría correctas ahí.
# Las negativas tienen conjunto vacío: cualquier alerta es un falso positivo.
# Verdad de referencia establecida VIENDO cada escena (hojas de contactos en
# testcam/thumbs/), no asumiendo por el título del fichero. Dos clips no eran lo
# que su nombre prometía y se reclasificaron:
#   - accidente_laboral: es una charla de seguridad (hombre con casco hablando,
#     ajuste de arnés). NO muestra ningún accidente -> es una escena NORMAL.
#   - caida_escaleras: cine mudo muy oscuro con bandas negras. La caída existe,
#     pero el material no representa a una cámara de seguridad; se mide aparte.
VERDAD = {
    "naturaleza_vacia":  set(),
    "calle_peatones":    {"persona"},
    "obra_normal":       {"persona"},
    "accidente_laboral": {"persona"},
    # Reclasificado en el ciclo 2: el VLM respondía "es una demostración de
    # técnica de artes marciales, no violencia" y TIENE RAZÓN. Un randori no es
    # una agresión. Etiquetarlo como violencia penalizaba al sistema por acertar.
    # Como negativo es valiosísimo: movimiento brusco entre dos personas que NO
    # debe alertar, justo el falso positivo que arruina un producto de seguridad.
    "caida_judo":        {"persona"},
    "caida_escaleras":   {"caidas"},
    "disturbios_saqueo": {"robos", "violencia"},
    "disturbios_calle":  {"robos", "violencia"},
}
NEGATIVAS = {"naturaleza_vacia", "calle_peatones", "obra_normal", "accidente_laboral",
             "caida_judo"}
# Material no representativo de CCTV: se informa, pero no cuenta en el titular.
NO_REPRESENTATIVAS = {"caida_escaleras"}

s3 = boto3.client("s3", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)


# Segundos que tarda el worker en reconectar tras reiniciarse el publicador.
# Medido en la prueba de cortes: ~14 s desde que vuelve el stream.
GRACIA_RECONEXION = 20


def set_escena(nombre):
    s3.put_object(Bucket=BUCKET, Key="testcam/control.json",
                  Body=json.dumps({"stream": "on", "escena": nombre}).encode(),
                  ContentType="application/json")


def esperar_escena(nombre, timeout=60):
    """Espera a que la cámara publique en estado.json que sirve `nombre`."""
    limite = time.time() + timeout
    while time.time() < limite:
        try:
            est = json.loads(s3.get_object(Bucket=BUCKET, Key="testcam/estado.json")["Body"].read())
            if est.get("escena") == nombre:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def detecciones_desde(ts_iso):
    """Detecciones del uid de pruebas creadas después de ts_iso."""
    out, lek = [], None
    while True:
        kw = dict(TableName="detections", IndexName="owner-index",
                  KeyConditionExpression="#o = :o AND created_at > :t",
                  FilterExpression="#ty = :ty",
                  ExpressionAttributeNames={"#o": "owner_uid", "#ty": "type"},
                  ExpressionAttributeValues={":o": {"S": UID}, ":t": {"S": ts_iso},
                                             ":ty": {"S": "event"}})
        if lek:
            kw["ExclusiveStartKey"] = lek
        r = ddb.query(**kw)
        for it in r["Items"]:
            try:
                raw = json.loads(it["raw"]["S"])
            except Exception:
                raw = {}
            out.append({"ts": it["created_at"]["S"],
                        "etiqueta": raw.get("event_type"),
                        "score": raw.get("cosine_sim"),
                        "via": raw.get("confirmed_by")})
        lek = r.get("LastEvaluatedKey")
        if not lek:
            return out


def ahora_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def main():
    minutos = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    # Margen tras cambiar de escena: el control tarda <=5 s en aplicarla, el
    # publicador reinicia y el worker reconecta. Lo anterior a esto no cuenta.

    resultados = {}
    for escena in VERDAD:
        print(f"\n=== escena: {escena} ===", flush=True)
        set_escena(escena)
        # Esperar a que la CÁMARA confirme el cambio (estado.json) en vez de dormir
        # un tiempo fijo, y sumar solo la gracia de reconexión del worker. En el
        # ciclo 1 se perdieron 10 de 30 detecciones —incluido el único verdadero
        # positivo— porque caían en una ventana muerta de 45 s.
        aplicada = esperar_escena(escena, timeout=60)
        if not aplicada:
            print("  aviso: la cámara no confirmó el cambio; se mide igualmente", flush=True)
        time.sleep(GRACIA_RECONEXION)
        t0 = ahora_iso()
        time.sleep(minutos * 60)
        dets = detecciones_desde(t0)
        etiquetas = defaultdict(int)
        for d in dets:
            etiquetas[d["etiqueta"]] += 1
        resultados[escena] = {"total": len(dets), "por_etiqueta": dict(etiquetas)}
        print(f"  alertas: {len(dets)}  {dict(etiquetas)}", flush=True)

    print("\n\n########## MATRIZ ##########")
    vp = fp = vn = fn = 0
    for escena, r in resultados.items():
        esperado = VERDAD[escena]
        disparadas = set(r["por_etiqueta"])
        if escena in NEGATIVAS:
            # En una escena normal, cualquier alerta de incidente es falso positivo.
            incidentes = disparadas - {"persona"}
            if incidentes:
                fp += 1
                veredicto = f"FALSO POSITIVO {sorted(incidentes)}"
            else:
                vn += 1
                veredicto = "correcto (sin alerta de incidente)"
        else:
            aciertos = disparadas & esperado
            if aciertos:
                vp += 1
                veredicto = f"DETECTADO {sorted(aciertos)}"
            else:
                fn += 1
                veredicto = f"NO DETECTADO (esperado {sorted(esperado)})"
        print(f"  {escena:20} alertas={r['total']:4}  {str(r['por_etiqueta'])[:44]:46} {veredicto}")

    print(f"\n  positivos detectados: {vp}/{vp+fn}   negativos limpios: {vn}/{vn+fp}")
    json.dump(resultados, open("resultados_evaluacion.json", "w"), indent=1, ensure_ascii=False)
    print("  detalle -> resultados_evaluacion.json")


if __name__ == "__main__":
    main()

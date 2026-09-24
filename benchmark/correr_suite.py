"""
correr_suite.py — Suite Manual de Heimdall (fase 1, sin CI todavía).

Corre TODO el banco de videos (ground_truth.json) de una sola vez y junta el
resultado en un reporte único -- en vez de invocar evaluar_video_v3.py y
calcular_metricas.py video por video, a mano, como se hizo durante toda la
sesión en la que se armó este banco.

No dispara nada solo: alguien lo corre antes de abrir un PR que toque
vision.py / tiers.py / vlm.py, y pega la tabla de salida en la descripción.
Ver benchmark/README-cascada.md y ground_truth.json para el detalle de cada
video y por qué está incluido (casos exitosos Y fallas conocidas -- ambos
sirven para detectar regresiones).

Uso:
    python correr_suite.py                 # sin Mímir (rápido, gratis)
    python correr_suite.py --vlm           # con Mímir real (Bedrock, ~$0.01 total)
    python correr_suite.py --vlm --solo caidas   # solo los videos de un concepto
"""
import argparse
import csv
import json
import os
import time

import evaluar_video_v3 as ev
import calcular_metricas as cm

MANIFEST_DEFAULT = "ground_truth.json"
RESULTADOS_DIR_DEFAULT = "resultados_suite"


def cargar_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def correr_uno(entry, use_vlm, resultados_dir):
    """Corre evaluar_video_v3.evaluar() sobre un video del manifiesto y
    calcula sus métricas. Devuelve un dict con el resumen de esa corrida."""
    video = entry["video"]
    label = entry["label"]
    fps = entry.get("fps", 30.0)
    anomaly = entry.get("anomaly")  # [inicio, fin] o null

    base = os.path.splitext(os.path.basename(video))[0]
    out_csv = os.path.join(resultados_dir, f"{base}_{label}.csv")

    t0 = time.time()
    ev.evaluar(video, label, sample_every=5, clear_margin=ev.CLEAR_MARGIN_DEFAULT,
               out_csv=out_csv, use_vlm=use_vlm, vlm_budget=1000)
    elapsed = time.time() - t0

    filas = cm.cargar_scores(out_csv)
    vlm_calls = sum(1 for r in csv.DictReader(open(out_csv, encoding="utf-8"))
                    if r.get("vlm_called") == "1")

    if anomaly:
        ranges = [tuple(anomaly)]
        labels, scores = cm.etiquetar(filas, ranges)
        auc = cm.auc_seguro(labels, scores)
        lat = cm.calcular_latencia(filas, ranges, threshold=0.5, fps=fps)
    else:
        auc, lat = None, None

    detectado_alguna_vez = any(d == 1 for _, _, d in filas if d is not None)

    return {
        "video": video,
        "concepto": label,
        "auc": round(auc, 4) if auc is not None else None,
        "latencia_s": round(lat, 2) if lat is not None else None,
        "detecto_alguna_vez": detectado_alguna_vez,
        "llamadas_vlm": vlm_calls,
        "tiempo_corrida_s": round(elapsed, 1),
        "nota": entry.get("nota", ""),
    }


def imprimir_resumen(filas_resumen):
    print("\n=== Resumen de la suite ===")
    ancho_video = max(len(r["video"]) for r in filas_resumen) + 2
    encabezado = (f"{'Video':<{ancho_video}} {'Concepto':<11} {'AUC':>7} "
                  f"{'Latencia':>9} {'Detectó':>8} {'Llam.VLM':>9}")
    print(encabezado)
    print("-" * len(encabezado))
    for r in filas_resumen:
        if "error" in r:
            print(f"{r['video']:<{ancho_video}} {r['concepto']:<11} ERROR: {r['error']}")
            continue
        auc_s = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
        lat_s = f"{r['latencia_s']:.2f}s" if r["latencia_s"] is not None else "n/a"
        print(f"{r['video']:<{ancho_video}} {r['concepto']:<11} {auc_s:>7} "
              f"{lat_s:>9} {str(r['detecto_alguna_vez']):>8} {r['llamadas_vlm']:>9}")


def main():
    ap = argparse.ArgumentParser(description="Corre el banco completo de videos de Heimdall")
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT)
    ap.add_argument("--vlm", action="store_true",
                    help="Corre con Mímir real (Bedrock). Necesita AWS_PROFILE activo.")
    ap.add_argument("--solo", default=None,
                    help="Filtra el manifiesto a un solo concepto (ej: caidas)")
    ap.add_argument("--resultados-dir", default=RESULTADOS_DIR_DEFAULT)
    ap.add_argument("--out", default=None, help="JSON de salida (default: <resultados-dir>/<fecha>.json)")
    args = ap.parse_args()

    os.makedirs(args.resultados_dir, exist_ok=True)
    manifest = cargar_manifest(args.manifest)
    if args.solo:
        manifest = [e for e in manifest if e["label"] == args.solo]
        if not manifest:
            raise SystemExit(f"Ningún video en el manifiesto tiene concepto '{args.solo}'")

    print(f"Suite de {len(manifest)} videos | Mímir: {'ON (Bedrock real)' if args.vlm else 'OFF'}\n")

    filas_resumen = []
    t_inicio = time.time()
    for i, entry in enumerate(manifest, 1):
        print(f"[{i}/{len(manifest)}] {entry['video']} ({entry['label']})...")
        if not os.path.exists(entry["video"]):
            print(f"    -> SALTADO: no se encontró el archivo")
            filas_resumen.append({"video": entry["video"], "concepto": entry["label"],
                                  "error": "archivo no encontrado"})
            continue
        try:
            r = correr_uno(entry, args.vlm, args.resultados_dir)
        except Exception as e:
            r = {"video": entry["video"], "concepto": entry["label"], "error": str(e)}
        filas_resumen.append(r)
        print(f"    -> {r}")

    fecha = time.strftime("%Y-%m-%d_%H%M")
    out_path = args.out or os.path.join(args.resultados_dir, f"{fecha}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filas_resumen, f, ensure_ascii=False, indent=2)

    imprimir_resumen(filas_resumen)
    print(f"\nTiempo total: {(time.time() - t_inicio)/60:.1f} min")
    print(f"Guardado en: {out_path}")


if __name__ == "__main__":
    main()

"""
calcular_metricas.py — Fase 3 del benchmark

Cruza los scores generados por evaluar_video.py contra las etiquetas reales
del dataset (ground-truth) y calcula:

  - AUC-Micro : el AUC estándar de la literatura (todos los frames juntos)
  - AUC-Macro : el AUC promediado por video (más fiel al caso de uso de SkyEye,
                donde importa localizar el evento DENTRO de una misma cámara)
  - Latencia  : segundos desde que empieza el evento hasta la primera detección

Incluye un AUTO-TEST (--selftest) que valida el cálculo con datos inventados
donde ya se sabe la respuesta correcta. SIEMPRE correr --selftest primero,
antes de confiar en cualquier número real. (Esto evita construir conclusiones
sobre una métrica mal calculada, que es un error real y documentado en la
literatura de VAD.)

Uso:
    # 1) Validar que el cálculo funciona (hazlo SIEMPRE primero):
    python calcular_metricas.py --selftest

    # 2) Calcular sobre un video real:
    #    necesita el CSV de scores + el rango de frames anómalos del ground-truth
    python calcular_metricas.py --scores scores_Fighting003.csv \
        --anomaly-start 90 --anomaly-end 240 --fps 30

    # Si el video es NORMAL (sin anomalía), no pasar --anomaly-start/end.
"""

import csv
import argparse
import sys

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Falta scikit-learn. Instala con: pip install scikit-learn")
    sys.exit(1)


CASCADE_SCORE_COLUMNS = ["score_clip", "score_clip_yolo", "score_cascade"]


def detectar_columnas(csv_path):
    """Devuelve la lista de columnas de score disponibles en el CSV.

    CSV básico: solo 'score'. CSV de cascada (evaluar_video.py --cascade):
    'score_clip', 'score_clip_yolo', 'score_cascade'.
    """
    with open(csv_path) as f:
        header = next(csv.reader(f))
    if "score" in header:
        return ["score"]
    return [c for c in CASCADE_SCORE_COLUMNS if c in header]


def cargar_scores(csv_path, score_col="score"):
    """Lee el CSV -> lista de (frame_idx, score, detected).

    'detected' viene de la columna 'detected' si existe (CSV básico); en un CSV
    de cascada no hay esa columna, así que queda en None (calcular_latencia cae
    de vuelta a comparar score >= threshold, que sí tiene sentido ahí porque los
    tres score_* de cascada.py ya vienen normalizados a 0..1).
    """
    filas = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            detected = int(r["detected"]) if "detected" in r and r["detected"] != "" else None
            filas.append((int(r["frame_idx"]), float(r[score_col]), detected))
    return filas


def etiquetar(filas, anomaly_ranges):
    """Asigna 1 (anómalo) o 0 (normal) a cada frame según los rangos de ground-truth.

    anomaly_ranges: lista de tuplas (inicio, fin) en número de frame.
    """
    labels = []
    scores = []
    for frame_idx, score, *_ in filas:
        es_anomalo = any(ini <= frame_idx <= fin for ini, fin in anomaly_ranges)
        labels.append(1 if es_anomalo else 0)
        scores.append(score)
    return labels, scores


def calcular_latencia(filas, anomaly_ranges, threshold, fps):
    """Segundos desde el inicio del primer evento hasta la primera detección.

    Usa la columna 'detected' del CSV si está disponible (ya calculada con el
    umbral REAL de producción para ese concepto). Si el CSV es viejo y no la
    trae, cae de vuelta a comparar contra --threshold (menos confiable, porque
    el umbral correcto varía por concepto: 0.02 para violencia/persona/cuchillo
    en el sistema nuevo, no el 0.268 del CLIP original).
    """
    if not anomaly_ranges:
        return None
    inicio_evento = min(ini for ini, _ in anomaly_ranges)
    for frame_idx, score, detected in filas:
        if frame_idx < inicio_evento:
            continue
        es_deteccion = bool(detected) if detected is not None else (score > threshold)
        if es_deteccion:
            return round((frame_idx - inicio_evento) / fps, 3)
    return None  # nunca detectó


def auc_seguro(labels, scores):
    """roc_auc_score falla si solo hay una clase; lo manejamos con gracia."""
    if len(set(labels)) < 2:
        return None  # video con una sola clase: AUC no está definido
    return roc_auc_score(labels, scores)


# ---------------------------------------------------------------------------
# UN video
# ---------------------------------------------------------------------------
def metricas_un_video(scores_csv, anomaly_ranges, fps, threshold, score_col="score"):
    filas = cargar_scores(scores_csv, score_col)
    labels, scores = etiquetar(filas, anomaly_ranges)

    auc = auc_seguro(labels, scores)
    lat = calcular_latencia(filas, anomaly_ranges, threshold, fps)

    print(f"\n=== Métricas para {scores_csv} (columna: {score_col}) ===")
    print(f"  Frames evaluados: {len(filas)}")
    print(f"  Frames anómalos:  {sum(labels)}  |  normales: {len(labels)-sum(labels)}")
    if auc is None:
        print("  AUC: no definido (el video tiene una sola clase de frames)")
    else:
        print(f"  AUC (este video): {auc:.4f}")
    if lat is None:
        if anomaly_ranges:
            print("  Latencia: el sistema NUNCA detectó el evento")
        else:
            print("  Latencia: n/a (video normal, sin evento)")
    else:
        print(f"  Latencia de detección: {lat} s")
    return auc, labels, scores


# ---------------------------------------------------------------------------
# AUTO-TEST: validar el cálculo con datos donde ya sabemos la respuesta
# ---------------------------------------------------------------------------
def selftest():
    print("=== AUTO-TEST del cálculo de métricas ===\n")
    ok = True

    # Caso 1: separación perfecta -> AUC debe ser 1.0
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    auc = roc_auc_score(labels, scores)
    print(f"Caso 1 (separación perfecta): AUC={auc:.3f}  (esperado 1.000)")
    ok &= abs(auc - 1.0) < 1e-9

    # Caso 2: predicción invertida -> AUC debe ser 0.0
    scores_inv = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    auc = roc_auc_score(labels, scores_inv)
    print(f"Caso 2 (totalmente invertido): AUC={auc:.3f}  (esperado 0.000)")
    ok &= abs(auc - 0.0) < 1e-9

    # Caso 3: azar puro (scores iguales) -> AUC debe ser 0.5
    scores_azar = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    auc = roc_auc_score(labels, scores_azar)
    print(f"Caso 3 (azar / scores iguales): AUC={auc:.3f}  (esperado 0.500)")
    ok &= abs(auc - 0.5) < 1e-9

    # Caso 4: latencia — evento empieza en frame 100, se detecta en 130, fps 30
    #         -> (130-100)/30 = 1.0 s. Usamos la columna 'detected' (como haría
    #         un CSV real de evaluar_video_v2.py), no un threshold reconstruido.
    filas = [(90, 0.1, 0), (100, 0.2, 0), (120, 0.25, 0), (130, 0.9, 1), (140, 0.95, 1)]
    lat = calcular_latencia(filas, [(100, 200)], threshold=0.5, fps=30)
    print(f"Caso 4 (latencia): {lat} s  (esperado 1.0)")
    ok &= (lat == 1.0)

    # Caso 4b: sin columna 'detected' (CSV viejo) -> debe caer al threshold
    filas_sin_detected = [(90, 0.1, None), (100, 0.2, None), (130, 0.9, None)]
    lat2 = calcular_latencia(filas_sin_detected, [(100, 200)], threshold=0.5, fps=30)
    print(f"Caso 4b (latencia, fallback a threshold): {lat2} s  (esperado 1.0)")
    ok &= (lat2 == 1.0)

    # Caso 5: etiquetado por rangos
    filas = [(10, 0.1, 0), (50, 0.9, 1), (150, 0.8, 1), (300, 0.1, 0)]
    labels, _ = etiquetar(filas, [(40, 200)])
    print(f"Caso 5 (etiquetado): {labels}  (esperado [0, 1, 1, 0])")
    ok &= (labels == [0, 1, 1, 0])

    print("\n" + ("TODOS LOS TESTS PASARON ✓" if ok else "FALLÓ ALGÚN TEST ✗"))
    print("El cálculo es confiable." if ok else "NO usar hasta arreglar.")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calcula AUC y latencia de SkyEye")
    ap.add_argument("--selftest", action="store_true",
                    help="Valida el cálculo con datos de juguete (hazlo primero)")
    ap.add_argument("--scores", help="CSV de scores generado por evaluar_video.py")
    ap.add_argument("--anomaly-start", type=int, action="append", default=[],
                    help="Frame de inicio del evento (se puede repetir)")
    ap.add_argument("--anomaly-end", type=int, action="append", default=[],
                    help="Frame de fin del evento (se puede repetir)")
    ap.add_argument("--fps", type=float, default=30.0, help="FPS del video (default 30)")
    ap.add_argument("--threshold", type=float, default=0.268,
                    help="Umbral. En CSVs de cascada (score 0..1) usa algo como 0.5")
    ap.add_argument("--score-column", default=None,
                    help="Forzar una columna (score_clip/score_clip_yolo/score_cascade). "
                         "Si no se indica, en un CSV de cascada se comparan las tres.")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if not args.scores:
        ap.error("Falta --scores (o usa --selftest). Ver --help.")

    ranges = list(zip(args.anomaly_start, args.anomaly_end))
    columnas = [args.score_column] if args.score_column else detectar_columnas(args.scores)

    if not columnas:
        ap.error("No encontré ninguna columna de score reconocible en ese CSV "
                 f"(busqué: score, {', '.join(CASCADE_SCORE_COLUMNS)}).")

    if len(columnas) > 1:
        print(f"CSV de cascada detectado -> comparando las {len(columnas)} variantes:\n")

    for col in columnas:
        metricas_un_video(args.scores, ranges, args.fps, args.threshold, score_col=col)

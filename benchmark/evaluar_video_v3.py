"""
evaluar_video_v3.py — Capas 0+1+2 completas, CON puerta de personas, decisión
clear/ambiguous/none real, y Mímir real (Bedrock) cerrando lo ambiguo.

Importa ABSTRACT_EVENTS y _hay_persona directamente de tiers.py, y judge()
directo de vlm.py — los mismos archivos de producción — así que la decisión
es idéntica a la real, no una aproximación.

Para un concepto ABSTRACTO (caidas, robos, violencia):
  - "none"      -> no pasa el umbral, o no hay persona en el frame (puerta)
  - "ambiguous" -> pasa el umbral y hay persona; con --vlm se le pregunta a
                   Mímir (Nova Lite/Bedrock) y la respuesta decide "clear"
                   (confirma) o "none" (rechaza). Sin --vlm, se queda ambiguous.
  - "clear"     -> nunca ocurre DIRECTO para conceptos abstractos (así es en
                   producción) — solo llega ahí vía confirmación de Mímir.

Para un concepto CONCRETO (persona, cuchillo):
  - "clear" si el margen es alto (>= clear_margin); si no, "ambiguous", y
    --vlm aplica igual que arriba.

Uso:
    python evaluar_video_v3.py --video Fighting003_x264A.mp4 --label violencia
    python evaluar_video_v3.py --video Fighting003_x264A.mp4 --label violencia --vlm
"""

import cv2
import csv
import time
import argparse
import os
from collections import deque

import sys
_AQUI = os.path.dirname(os.path.abspath(__file__))
# Orden importa: primero benchmark/ (vision.py y vlm.py candidatos, con las mejoras
# a medir), luego cascade/ (tiers, common, transport: los mismos de producción).
sys.path[:0] = [_AQUI, os.path.join(_AQUI, '..', 'cascade')]

from vision import MotionDetector, ClipScorer, normalize_word
from tiers import ABSTRACT_EVENTS, _hay_persona, VLM_MIN_INTERVAL
import vlm as vlm_mod

CLEAR_MARGIN_DEFAULT = 0.15  # mismo default que run_clip() en tiers.py
VLM_COST_PER_CALL = 0.000076  # de la BITÁCORA del repo

# Ráfaga que se le manda a Mímir en vez de un solo frame (ver vlm.py v4): un
# solo fotograma no alcanza para distinguir un golpe real de un empujón en
# video granulado -- medido con 0/24 confirmaciones en dos videos distintos.
# BURST_SPAN_SECONDS de historia, BURST_FRAMES muestras parejas dentro de ella.
BURST_SPAN_SECONDS = 1.5
BURST_FRAMES = 4


def resolver_con_vlm(burst_jpgs, label, decision, ahora_seg, ultimo_vlm_seg, vlm_calls, vlm_budget):
    """Si decision=='ambiguous', le pregunta a Mímir (con una ráfaga de frames,
    no uno solo) respetando el mismo rate-limit de producción (VLM_MIN_INTERVAL,
    en SEGUNDOS DE VIDEO, no en frames muestreados) y el presupuesto del benchmark.

    Devuelve (decision_final, ultimo_vlm_seg, vlm_calls, llamado, confirmed, reason).
    """
    if decision != "ambiguous":
        return decision, ultimo_vlm_seg, vlm_calls, False, "", ""
    if vlm_calls >= vlm_budget or ahora_seg - ultimo_vlm_seg < VLM_MIN_INTERVAL:
        return decision, ultimo_vlm_seg, vlm_calls, False, "", ""

    confirmed, reason = vlm_mod.judge(burst_jpgs, label)
    ultimo_vlm_seg = ahora_seg
    vlm_calls += 1
    if confirmed is True:
        return "clear", ultimo_vlm_seg, vlm_calls, True, 1, reason
    if confirmed is False:
        return "none", ultimo_vlm_seg, vlm_calls, True, 0, reason
    return decision, ultimo_vlm_seg, vlm_calls, True, "", reason  # VLM falló: sigue ambiguous


def armar_rafaga(buffer_frames, n_muestras):
    """buffer_frames: deque de frames BGR recientes, en orden. Toma n_muestras
    parejas a lo largo del buffer (incluye siempre el más reciente) y las
    codifica a JPEG. Devuelve lista de bytes, en orden temporal."""
    total = len(buffer_frames)
    if total == 0:
        return []
    if total <= n_muestras:
        indices = range(total)
    else:
        indices = [round(i * (total - 1) / (n_muestras - 1)) for i in range(n_muestras)]
    salida = []
    for i in indices:
        ok, buf = cv2.imencode(".jpg", buffer_frames[i])
        if ok:
            salida.append(buf.tobytes())
    return salida


def decidir(label, score, por_etiqueta, scorer, clear_margin):
    """Reproduce la rama de decisión de tiers.run_clip(), sin el rate-limit de VLM."""
    if label is None or score < scorer.threshold_for(label):
        return "none"
    if normalize_word(label) in ABSTRACT_EVENTS and not _hay_persona(por_etiqueta, scorer):
        return "none"  # puerta de personas
    if normalize_word(label) in ABSTRACT_EVENTS or score < clear_margin:
        return "ambiguous"  # necesitaría Mímir (no medido en este script)
    return "clear"


def evaluar(video_path: str, label: str, sample_every: int, clear_margin: float, out_csv: str,
            use_vlm: bool = False, vlm_budget: int = 1000):
    # Cargamos también "persona" para que la puerta de personas tenga con qué
    # trabajar (igual que en producción, si la cámara monitorea eventos
    # abstractos, "persona" siempre está entre los conceptos puntuados).
    concepts = [label] if normalize_word(label) in ("persona", "person") else [label, "persona"]
    scorer = ClipScorer(blacklist=concepts)
    motion = MotionDetector()
    umbral = scorer.threshold_for(label)
    es_abstracto = normalize_word(label) in ABSTRACT_EVENTS

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.2f} | Frames totales: {total_frames}")
    print(f"  Concepto: '{label}' | abstracto: {es_abstracto} | umbral: {umbral} | clear_margin: {clear_margin}")
    if es_abstracto:
        print(f"  (concepto abstracto -> 'clear' directo nunca ocurre; lo ambiguo "
              f"{'se le pregunta a Mímir' if use_vlm else 'se queda sin confirmar (usa --vlm para cerrarlo)'})")
    if use_vlm:
        print(f"  Mímir: ON (Nova Lite/Bedrock) | ráfaga: {BURST_FRAMES} frames en {BURST_SPAN_SECONDS}s | "
              f"rate-limit: 1 cada {VLM_MIN_INTERVAL}s de video | presupuesto: {vlm_budget} llamadas")

    rows = []
    frame_idx = 0
    processed = 0
    conteo = {"none": 0, "ambiguous": 0, "clear": 0}
    ultimo_vlm_seg = -1e9
    vlm_calls = 0
    buffer_frames = deque(maxlen=max(1, int(BURST_SPAN_SECONDS * fps)))
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        buffer_frames.append(frame)  # historia reciente, para la ráfaga del VLM

        rois = motion.rois(frame)  # Capa 0: SIEMPRE con todos los frames

        if not rois:
            score, decision = 0.0, "none"
        else:
            _, _, _, por_etiqueta = scorer.score_detallado(frame, rois)
            score = por_etiqueta.get(label, 0.0)
            decision = decidir(label, score, por_etiqueta, scorer, clear_margin)

        vlm_called, vlm_confirmed, vlm_reason = False, "", ""
        if use_vlm and decision == "ambiguous":
            ahora_seg = frame_idx / fps
            rafaga = armar_rafaga(buffer_frames, BURST_FRAMES)
            if rafaga:
                decision, ultimo_vlm_seg, vlm_calls, vlm_called, vlm_confirmed, vlm_reason = resolver_con_vlm(
                    rafaga, label, decision, ahora_seg, ultimo_vlm_seg, vlm_calls, vlm_budget)

        conteo[decision] += 1

        if frame_idx % sample_every == 0 or decision != "none" or vlm_called:
            time_seconds = frame_idx / fps
            rows.append((frame_idx, round(time_seconds, 3), round(score, 6), decision,
                         int(vlm_called), vlm_confirmed, vlm_reason.replace("\n", " ")[:160]))
            processed += 1
            if processed % 100 == 0:
                print(f"  ...{processed} filas registradas (última decisión: {decision})")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - t_start

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "time_seconds", "score", "decision", "detected",
                          "vlm_called", "vlm_confirmed", "vlm_reason"])
        for frame_idx, ts, score, decision, vlm_called, vlm_confirmed, vlm_reason in rows:
            # "detected" = alerta CONFIRMADA (decision == "clear"), no "ambiguous".
            # "ambiguous" significa "hace falta preguntarle a Mímir", no es una
            # alerta todavía -- contarlo como detección infla la latencia medida
            # a ~0s en cualquier concepto abstracto, donde "clear" nunca ocurre
            # sin el VLM (ver decidir()). Con --vlm, "clear" sí puede llegar vía
            # confirmación real, así que la latencia queda medible de verdad.
            writer.writerow([frame_idx, ts, score, decision, int(decision == "clear"),
                             vlm_called, vlm_confirmed, vlm_reason])

    video_seconds = (total_frames / fps) if total_frames else 0
    speed = (video_seconds / elapsed) if elapsed > 0 else 0
    print(f"\nListo. {frame_idx} frames de video ({processed} filas guardadas) en {elapsed:.1f}s")
    print(f"  Decisiones -> none: {conteo['none']} | ambiguous: {conteo['ambiguous']} | clear: {conteo['clear']}")
    if use_vlm:
        costo = vlm_calls * VLM_COST_PER_CALL
        print(f"  Llamadas reales a Mímir: {vlm_calls} (costo: ${costo:.6f})")
    elif es_abstracto and conteo["ambiguous"] > 0:
        print(f"  De esos {conteo['ambiguous']} 'ambiguous', en producción cada uno costaría "
              f"~${VLM_COST_PER_CALL} preguntarle a Mímir (con límite de 1 cada {VLM_MIN_INTERVAL}s por cámara+etiqueta) "
              f"-- corre con --vlm para preguntarle de verdad.")
    if video_seconds:
        print(f"  Duración del video: {video_seconds:.1f}s | Velocidad: {speed:.1f}x tiempo real")
    print(f"  CSV guardado: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evalúa Capas 0+1 completas (con puerta de personas)")
    ap.add_argument("--video", required=True)
    ap.add_argument("--label", required=True, help="persona, cuchillo, violencia, robos, caidas")
    ap.add_argument("--sample-every", type=int, default=5,
                    help="Cada cuántos frames SIN detección se registra igual una fila de contexto")
    ap.add_argument("--clear-margin", type=float, default=CLEAR_MARGIN_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--vlm", action="store_true",
                    help="Cierra 'ambiguous' preguntándole al Mímir real (Nova Lite/Bedrock). "
                         "Necesita credenciales AWS activas (AWS_PROFILE) con permiso bedrock:InvokeModel.")
    ap.add_argument("--vlm-budget", type=int, default=1000,
                    help="Tope de llamadas reales a Mímir por corrida (control de costo)")
    args = ap.parse_args()

    out = args.out or f"scores_v3_{os.path.splitext(os.path.basename(args.video))[0]}_{args.label}.csv"
    evaluar(args.video, args.label, args.sample_every, args.clear_margin, out,
            use_vlm=args.vlm, vlm_budget=args.vlm_budget)

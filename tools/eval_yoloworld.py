#!/usr/bin/env python3
"""
Evalúa YOLO-World (detector open-vocabulary) sobre el MISMO set etiquetado que
eval_detection.py, para comparar contra CLIP con números objetivos.

Score por imagen = confianza máxima de una caja de la clase pedida (0 si ninguna).
Mide además el tiempo de inferencia por imagen (para estimar cámaras/instancia).

Uso: python3 eval_yoloworld.py --data knife_eval --concept knife
"""
import argparse, glob, os, time
import numpy as np
from ultralytics import YOLOWorld

CONCEPT_CLASSES = {
    "knife": ["knife"],
    "cuchillo": ["knife"],
    "persona": ["person"],
    "person": ["person"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--concept", default="knife")
    ap.add_argument("--weights", default="yolov8s-worldv2.pt")
    args = ap.parse_args()

    model = YOLOWorld(args.weights)
    classes = CONCEPT_CLASSES[args.concept]
    model.set_classes(classes)

    groups = {}
    for sub in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, sub)
        if not os.path.isdir(d):
            continue
        label = 1 if sub.startswith("pos") else 0
        files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        groups[sub] = (label, files)

    all_sl = []
    per_group = {}
    times = []
    for sub, (label, files) in groups.items():
        scores = []
        for f in files:
            t0 = time.time()
            r = model.predict(f, conf=0.001, verbose=False)[0]
            times.append(time.time() - t0)
            conf = float(r.boxes.conf.max().item()) if len(r.boxes) else 0.0
            scores.append(conf)
            all_sl.append((conf, label))
        per_group[sub] = (label, scores)
        print(f"  {sub}: {len(scores)} imágenes | confianza media {np.mean(scores):.3f}")

    pos_s = [s for s, y in all_sl if y == 1]
    neg_s = [s for s, y in all_sl if y == 0]
    wins = sum((s > n) + 0.5 * (s == n) for s in pos_s for n in neg_s)
    auc = wins / (len(pos_s) * len(neg_s))

    def metrics_at(thr):
        tp = sum(1 for s, y in all_sl if y == 1 and s >= thr)
        fp = sum(1 for s, y in all_sl if y == 0 and s >= thr)
        fn = sum(1 for s, y in all_sl if y == 1 and s < thr)
        tn = sum(1 for s, y in all_sl if y == 0 and s < thr)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return prec, rec, f1, tp, fp, fn, tn

    best = max((metrics_at(t) + (t,) for t in np.round(np.arange(0.02, 0.6, 0.01), 3)), key=lambda r: r[2])
    bp, br, bf1, btp, bfp, bfn, btn, bthr = best

    print(f"\n=== YOLO-World | concepto: {args.concept} | {len(pos_s)} pos / {len(neg_s)} neg ===")
    print(f"AUC (fiabilidad global): {auc:.3f}")
    print(f"score medio  positivos={np.mean(pos_s):.3f}  negativos={np.mean(neg_s):.3f}  (separación {np.mean(pos_s)-np.mean(neg_s):+.3f})")
    print(f"Mejor umbral por F1 ({bthr}): precisión {bp:.2f} | recall {br:.2f} | F1 {bf1:.2f} (TP {btp} FP {bfp} FN {bfn} TN {btn})")
    # operar con precisión alta (menos falsas alarmas): mejor umbral con prec>=0.9
    hp = [(metrics_at(t), t) for t in np.round(np.arange(0.02, 0.9, 0.01), 3)]
    hp = [(m, t) for m, t in hp if m[0] >= 0.9]
    if hp:
        (p, r, f, tp, fp, fn, tn), t = max(hp, key=lambda x: x[0][1])  # mayor recall con prec>=0.9
        print(f"A precisión>=0.90 (umbral {t}): recall {r:.2f} (atrapa {tp}/{tp+fn} cuchillos con solo {fp} falsas alarmas)")
    print("\nFalsos positivos por tipo de negativo (al mejor umbral F1):")
    for sub, (label, scores) in per_group.items():
        if label == 0 and scores:
            n = sum(1 for s in scores if s >= bthr)
            print(f"  {sub}: {n/len(scores)*100:.0f}% ({n}/{len(scores)})")
    print(f"\nVelocidad: {np.mean(times)*1000:.0f} ms/imagen en CPU (mediana {np.median(times)*1000:.0f} ms)")

if __name__ == "__main__":
    main()

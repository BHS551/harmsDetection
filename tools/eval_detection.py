#!/usr/bin/env python3
"""
Banco de pruebas offline para medir la fiabilidad del detector (CLIP) SIN cámaras.

Reproduce la misma detección del worker (heimdall-eye.py): realce CLAHE + CLIP
ViT-B/32 + similitud coseno contra los prompts del concepto, tomando el máximo
sobre parches (igual que el barrido de ventana deslizante del worker).

Uso:
  python3 eval_detection.py --data <carpeta> --concept knife [--mode sliding|full]

La carpeta debe tener subcarpetas cuyo nombre empiece por "pos" (positivos) o
"neg" (negativos); las que empiezan por "neg_hard" se reportan aparte como
negativos difíciles (objetos confundibles).

Salida: precisión / recall / F1 barriendo umbral, el umbral de mejor F1, y las
métricas al umbral por defecto del worker (0.27). No inventa nada: cada número
sale de correr CLIP sobre las imágenes etiquetadas.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import clip
from PIL import Image

# Mismos prompts que el worker (heimdall-eye.py PROMPT_MAP).
PROMPT_MAP = {
    "knife": ["a photo of a knife"],
    "cuchillo": ["a photo of a knife"],
    "persona": ["a photo of a person"],
    "person": ["a photo of a person"],
    "caidas": [
        "a person fallen on the floor",
        "a person falling down",
        "a person collapsed on the ground",
    ],
    "robos": [
        "a robbery in progress",
        "a person stealing from someone",
        "a burglar breaking into a building",
    ],
    "violencia": [
        "people fighting violently",
        "a person hitting another person",
        "a violent physical assault",
    ],
}


def enhance_frame(bgr):
    """CLAHE en espacio LAB, idéntico al worker."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def sliding_windows(w, h, patch=224, stride=192):
    last_top, last_left = max(0, h - patch), max(0, w - patch)
    tops = list(range(0, last_top + 1, stride)) or [0]
    lefts = list(range(0, last_left + 1, stride)) or [0]
    if tops[-1] != last_top:
        tops.append(last_top)
    if lefts[-1] != last_left:
        lefts.append(last_left)
    return [(l, t, min(l + patch, w), min(t + patch, h)) for t in tops for l in lefts]


def score_image(path, model, preprocess, text_emb, device, mode):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    enhanced = enhance_frame(bgr)
    pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    h, w = enhanced.shape[:2]
    if mode == "full":
        crops = [pil]
    else:  # sliding: máximo sobre parches, como el worker
        crops = [pil.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in sliding_windows(w, h)]
    tensors = torch.stack([preprocess(c) for c in crops]).to(device)
    with torch.no_grad():
        emb = model.encode_image(tensors)
        emb /= emb.norm(dim=-1, keepdim=True)
        sims = emb @ text_emb.T  # [parches, prompts]
    return float(sims.max().item())  # mejor coincidencia sobre parches y prompts


def metrics_at(scores_labels, thr):
    tp = sum(1 for s, y in scores_labels if y == 1 and s >= thr)
    fp = sum(1 for s, y in scores_labels if y == 0 and s >= thr)
    fn = sum(1 for s, y in scores_labels if y == 1 and s < thr)
    tn = sum(1 for s, y in scores_labels if y == 0 and s < thr)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1, tp, fp, fn, tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--concept", default="knife")
    ap.add_argument("--mode", default="sliding", choices=["sliding", "full"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    prompts = PROMPT_MAP[args.concept]
    with torch.no_grad():
        text_emb = model.encode_text(clip.tokenize(prompts).to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)

    groups = {}
    for sub in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, sub)
        if not os.path.isdir(d):
            continue
        label = 1 if sub.startswith("pos") else 0
        files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png")))
        groups[sub] = (label, files)

    all_sl = []          # (score, label)
    per_group = {}       # sub -> list of scores
    for sub, (label, files) in groups.items():
        scores = []
        for i, f in enumerate(files):
            s = score_image(f, model, preprocess, text_emb, device, args.mode)
            if s is None:
                continue
            scores.append(s)
            all_sl.append((s, label))
        per_group[sub] = (label, scores)
        print(f"  {sub}: {len(scores)} imágenes | score medio {np.mean(scores):.3f}")

    # barrido de umbral -> mejor F1
    best = max(
        (metrics_at(all_sl, t) + (t,) for t in np.round(np.arange(0.18, 0.36, 0.005), 3)),
        key=lambda r: r[2],
    )
    bp, br, bf1, btp, bfp, bfn, btn, bthr = best
    dp, dr, df1, dtp, dfp, dfn, dtn = metrics_at(all_sl, 0.27)

    # ROC-AUC (Mann-Whitney): probabilidad de que un positivo puntúe más alto
    # que un negativo al azar. 0.5 = azar (inútil), 1.0 = separación perfecta.
    pos_s = [s for s, y in all_sl if y == 1]
    neg_s = [s for s, y in all_sl if y == 0]
    wins = sum((s > n) + 0.5 * (s == n) for s in pos_s for n in neg_s)
    auc = wins / (len(pos_s) * len(neg_s)) if pos_s and neg_s else 0.0
    print(f"\nAUC (fiabilidad global, 0.5=azar): {auc:.3f}")
    print(f"score medio  positivos={np.mean(pos_s):.3f}  negativos={np.mean(neg_s):.3f}  "
          f"(separación {np.mean(pos_s)-np.mean(neg_s):+.3f})")

    npos = sum(1 for _, y in all_sl if y == 1)
    nneg = sum(1 for _, y in all_sl if y == 0)
    print(f"\n=== Concepto: {args.concept} | modo: {args.mode} | {npos} pos / {nneg} neg ===")
    print(f"Umbral por defecto del worker (0.27): precisión {dp:.2f} | recall {dr:.2f} | F1 {df1:.2f} "
          f"(TP {dtp} FP {dfp} FN {dfn} TN {dtn})")
    print(f"Mejor umbral por F1 ({bthr}):        precisión {bp:.2f} | recall {br:.2f} | F1 {bf1:.2f} "
          f"(TP {btp} FP {bfp} FN {bfn} TN {btn})")

    # tasa de falsos positivos por grupo de negativos (al mejor umbral)
    print("\nFalsos positivos por tipo de negativo (al mejor umbral):")
    for sub, (label, scores) in per_group.items():
        if label == 0 and scores:
            fp_rate = sum(1 for s in scores if s >= bthr) / len(scores)
            print(f"  {sub}: {fp_rate*100:.0f}% ({sum(1 for s in scores if s>=bthr)}/{len(scores)})")


if __name__ == "__main__":
    main()

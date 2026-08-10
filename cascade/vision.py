"""
vision.py — lógica de visión compartida por las capas de la cascada.

Reúne en un solo lugar lo validado en el worker monolítico (heimdall-eye.py):
prompts por concepto, distractores conscientes del concepto, realce CLAHE,
region-proposal por movimiento (MOG2 + pad/cuadrado/min-size + merge) y el
scoring contrastivo de CLIP. Las capas lo importan; ninguna re-implementa CV.
"""
import unicodedata
import cv2
import numpy as np
import torch
import clip
from PIL import Image

# --- Concepto -> prompts (ES->EN, igual que el worker) ---
PROMPT_MAP = {
    "caidas": ["a person fallen on the floor", "a person falling down", "a person collapsed on the ground"],
    "robos": ["a robbery in progress", "a person stealing from someone", "a burglar breaking into a building"],
    "violencia": ["people fighting violently", "a person hitting another person", "a violent physical assault"],
    # Prompts descriptivos (CuPL): medido +0.083 de margen y 100% recall en personas reales.
    "persona": ["a photo of a person", "a person standing", "a man standing in a room",
                "a human figure", "someone walking", "a person seen by a security camera",
                "a person in the background of a room"],
    "person": ["a photo of a person", "a person standing", "a human figure", "someone walking"],
    # Hoja/metal (no "sostener"): +recall sin dispararse con objetos de mano.
    "cuchillo": ["a photo of a knife", "a sharp knife blade", "a metal knife blade",
                 "the blade of a knife", "a kitchen knife"],
    "knife": ["a photo of a knife", "a sharp knife blade", "a metal knife blade",
              "the blade of a knife", "a kitchen knife"],
}

DEFAULT_DISTRACTORS = [
    "a photo of a smartphone", "a photo of a wallet", "a photo of a hand",
    "a person standing normally", "an empty room", "furniture",
]  # nota: sin "food on a table" (colisionaba con cuchillos de cocina)

DEFAULT_PROMPT_THRESHOLDS = {
    "persona": 0.03, "person": 0.03, "cuchillo": 0.03, "knife": 0.03,
    "pistola": 0.02, "pistol": 0.02, "caidas": 0.02, "robos": 0.02, "violencia": 0.02,
}

# Conceptos que solo tienen sentido CON movimiento (eventos y persona): en escena
# estática producen falsos positivos que se solapan con los verdaderos.
MOTION_ONLY_LABELS = {"caidas", "robos", "violencia", "persona", "person"}

# Universo de conceptos que la caja de análisis COMPARTIDA puntúa siempre. Como una
# sola caja sirve a muchas cámaras (cada una con su propia blacklist), carga todos los
# prompts una vez y luego cada candidato se filtra a los conceptos de su cámara.
UNIVERSAL_CONCEPTS = ["persona", "cuchillo", "caidas", "robos", "violencia"]

# Distractores que colisionan con un concepto-objetivo (no restarlo a sí mismo).
CONFLICTING_DISTRACTORS = {
    "persona": {"a person standing normally", "a photo of a hand"},
    "person": {"a person standing normally", "a photo of a hand"},
}

# Region proposal
ROI_PADDING = 0.0
MIN_ROI_SIZE = 96
MOTION_DOWNSCALE = 0.5
MIN_MOTION_AREA = 500
MAX_ROIS = 10


def normalize_word(word):
    word = str(word).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")


def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def pad_square_roi(box, w, h, pad=ROI_PADDING, min_size=MIN_ROI_SIZE):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(bw * (1 + 2 * pad) / 2.0, bh * (1 + 2 * pad) / 2.0, min_size / 2.0)
    return (int(max(0, cx - half)), int(max(0, cy - half)),
            int(min(w, cx + half)), int(min(h, cy + half)))


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua else 0.0


def merge_boxes(boxes, iou_thr=0.3):
    merged = []
    for box in boxes:
        placed = False
        for i, m in enumerate(merged):
            if _iou(box, m) > iou_thr:
                merged[i] = (min(box[0], m[0]), min(box[1], m[1]), max(box[2], m[2]), max(box[3], m[3]))
                placed = True
                break
        if not placed:
            merged.append(box)
    return merged


class MotionDetector:
    """MOG2 con estado propio (una instancia por cámara/proceso)."""
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def rois(self, frame):
        small = cv2.resize(frame, None, fx=MOTION_DOWNSCALE, fy=MOTION_DOWNSCALE)
        mask = self.bg.apply(small)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        inv = 1.0 / MOTION_DOWNSCALE
        h, w = frame.shape[:2]
        found = []
        for c in contours:
            if cv2.contourArea(c) < MIN_MOTION_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            found.append((cv2.contourArea(c),
                          (max(0, int(x*inv)), max(0, int(y*inv)),
                           min(w, int((x+bw)*inv)), min(h, int((y+bh)*inv)))))
        found.sort(key=lambda r: r[0], reverse=True)
        boxes = [pad_square_roi(b, w, h) for _, b in found[:MAX_ROIS]]
        return merge_boxes(boxes)


class ClipScorer:
    """Carga CLIP una vez y puntúa ROIs con score contrastivo por concepto."""
    def __init__(self, blacklist, distractor_prompts=None, thresholds=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        self.use_half = self.device == "cuda"
        if self.use_half:
            self.model = self.model.half()

        cleaned = [w for w in blacklist if str(w).strip()] or ["person"]
        self.prompts, self.labels = [], []
        for word in cleaned:
            for p in PROMPT_MAP.get(normalize_word(word), [str(word)]):
                self.prompts.append(p)
                self.labels.append(word)

        distractors = list(distractor_prompts or DEFAULT_DISTRACTORS)
        drop = set()
        for w in cleaned:
            drop |= CONFLICTING_DISTRACTORS.get(normalize_word(w), set())
        distractors = [d for d in distractors if d not in drop]
        self.distractor_prompts = distractors

        self.thresholds = dict(DEFAULT_PROMPT_THRESHOLDS)
        if thresholds:
            for k, v in thresholds.items():
                try:
                    self.thresholds[normalize_word(k)] = float(v)
                except (TypeError, ValueError):
                    pass

        self.text_emb = self._embed(self.prompts)
        self.distr_emb = self._embed(distractors)

    def _embed(self, prompts):
        with torch.no_grad():
            e = self.model.encode_text(clip.tokenize(prompts, truncate=True).to(self.device))
            e /= e.norm(dim=-1, keepdim=True)
        return e

    def threshold_for(self, label):
        return self.thresholds.get(normalize_word(label), 0.02)

    def score(self, frame_bgr, rois):
        """Devuelve (best_score, best_label, best_coords) por MARGEN contrastivo."""
        s, l, c, _ = self.score_detallado(frame_bgr, rois)
        return s, l, c

    def score_detallado(self, frame_bgr, rois):
        """Como score(), más un dict {etiqueta: mejor margen} con TODAS las etiquetas.

        Lo necesita la puerta de personas de la capa 1: para decidir si vale la pena
        preguntarle al VLM por un robo o una pelea hay que saber si además hay
        alguien en el fotograma, no solo cuál fue la etiqueta ganadora.
        """
        enhanced = enhance_frame(frame_bgr)
        image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        tensors, coords = [], []
        for (x1, y1, x2, y2) in rois:
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            tensors.append(self.preprocess(image.crop((x1, y1, x2, y2))))
            coords.append((x1, y1, x2, y2))
        if not tensors:
            return 0.0, None, None, {}
        batch = torch.stack(tensors).to(self.device)
        if self.use_half:
            batch = batch.half()
        with torch.no_grad():
            emb = self.model.encode_image(batch)
            emb /= emb.norm(dim=-1, keepdim=True)
            sims = emb @ self.text_emb.T
            dsims = emb @ self.distr_emb.T
            margins = sims - dsims.max(dim=1, keepdim=True).values
        flat = int(torch.argmax(margins).item())
        npr = margins.shape[1]
        pi, qi = flat // npr, flat % npr

        # Mejor margen por ETIQUETA (varios prompts pueden mapear a la misma).
        por_etiqueta = {}
        col_max = margins.max(dim=0).values          # mejor ROI para cada prompt
        for j, etiqueta in enumerate(self.labels):
            v = float(col_max[j].item())
            if v > por_etiqueta.get(etiqueta, float("-inf")):
                por_etiqueta[etiqueta] = v

        return float(margins[pi, qi].item()), self.labels[qi], coords[pi], por_etiqueta

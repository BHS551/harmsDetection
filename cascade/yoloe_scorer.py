"""
yoloe_scorer.py — Capa 1 con YOLOE (detección de vocabulario abierto) en el
puesto que ocupaba CLIP.

POR QUÉ SE CAMBIA. Medido sobre el test completo de UCF-Crime (290 vídeos,
1.111.808 fotogramas):

    movimiento (MOG2) solo ....... 58,35% AUC
    CLIP (margen contrastivo) .... 55,49% AUC   <- peor que el movimiento
    cascada capa 0 + capa 1 ...... 55,67% AUC

CLIP no aportaba información: restaba. Refinarlo en sondas de atributo concretas
("¿hay personas?", "¿hay alguien en el suelo?") solo dio +1,08 ± 1,16 puntos
sobre 12 particiones — dentro del ruido. Y las sondas que mejor puntuaban
resultaron redundantes con "¿hay personas?" (r=0,69 y r=0,75): CLIP responde a
los sustantivos del prompt, no al verbo, que es el efecto bolsa de palabras
documentado (ARO: 63% en atributos, 59% en relaciones).

POR QUÉ YOLOE. Usa MobileCLIP por dentro, así que hereda la misma semántica
abierta, pero está OBLIGADO a localizar: no puede afirmar "persona" sin dibujar
una caja y dar una confianza. Ese es justo el fallo que se midió en CLIP, que
daba margen 0,089 a "persona" sobre una montaña nevada vacía.

Coste, medido en `m7i-flex.large` con fotogramas reales de 1920x1080:
    yoloe-11s (texto) ....... 242,5 ms/frame   4,12 fps
    CLIP ViT-B/32 (3 ROIs) .. 304,5 ms/frame   3,28 fps
El cambio AHORRA un 20% de CPU.

VOCABULARIO DIRIGIDO, NUNCA ABIERTO. El modo sin prompt de YOLOE (4.585 clases)
devuelve ruido de escena sobre CCTV: medido en fotogramas reales de UCF-Crime
respondía "bamboo forest", "pilgrim", "chicken coop", "magician". El modo con
lista corta de clases devuelve person/car/motorcycle de forma limpia y va al
doble de velocidad. Solo se usa el segundo.

Interfaz: es un reemplazo directo de `vision.ClipScorer` — mismos métodos
(`score_detallado`, `threshold_for`) y mismo atributo `labels`, para que
`tiers.py` no note la diferencia.
"""
import os

# Concepto de SkyEye -> términos del vocabulario de YOLOE que lo evidencian.
# Solo objetos: YOLOE detecta cosas, no situaciones. Los eventos abstractos se
# derivan más abajo a partir de estas evidencias.
VOCABULARIO = {
    "persona": ["person"],
    "person": ["person"],
    "cuchillo": ["knife", "kitchen knife", "pocketknife"],
    "knife": ["knife", "kitchen knife", "pocketknife"],
    "pistola": ["handgun", "gun"],
    "pistol": ["handgun", "gun"],
}
# Términos que se piden SIEMPRE al detector, aunque la cámara no los vigile:
# alimentan las reglas de evento abstracto de abajo.
EVIDENCIA_EXTRA = ["knife", "handgun", "backpack", "hoodie"]

# Eventos que YOLOE no puede ver directamente. No se inventan: se derivan de
# evidencia geométrica, y su papel es DISPARAR LA CONSULTA al VLM, no la alerta.
# Sigue el patrón de Paza (arXiv 2604.14846), cuyo filtro es geométrico y sin
# semántica: precisión 89,5% y especificidad 92,8% sobre DCSASS.
EVENTOS_ABSTRACTOS = {"robos", "violencia", "caidas"}

# Umbrales. OJO: son confianzas de detección (0-1), NO márgenes contrastivos de
# CLIP (0,02-0,15). Reutilizar los umbrales de CLIP aquí dispararía con todo.
UMBRAL_POR_DEFECTO = float(os.environ.get("HEIMDALL_YOLOE_CONF", "0.35"))
UMBRALES = {
    "persona": UMBRAL_POR_DEFECTO,
    "person": UMBRAL_POR_DEFECTO,
    # Las armas son pequeñas y salen mal en CCTV: se pide menos confianza porque
    # el coste de perderlas es alto y el VLM confirma después.
    "cuchillo": 0.20, "knife": 0.20, "pistola": 0.20, "pistol": 0.20,
    # Los eventos abstractos no se puntúan por confianza sino por evidencia; el
    # umbral es bajo a propósito porque quien decide de verdad es Mimir.
    "robos": 0.30, "violencia": 0.30, "caidas": 0.30,
}

# Proximidad entre personas para sospechar interacción física, como fracción de
# la diagonal de la caja. Es el parámetro `rho` de Paza, con su mismo valor.
PROXIMIDAD = float(os.environ.get("HEIMDALL_YOLOE_PROXIMIDAD", "0.3"))

MODELO_S3 = ("detection-frames-tests", "worker/models/yoloe-11s-seg.pt")
RUTA_LOCAL = os.environ.get("HEIMDALL_YOLOE_MODEL", "/home/ubuntu/app/cascade/yoloe-11s-seg.pt")


def _norm(t):
    import unicodedata
    t = str(t).strip().lower()
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")


class YoloeScorer:
    """Reemplazo directo de ClipScorer basado en detección con vocabulario dirigido."""

    def __init__(self, blacklist, distractor_prompts=None, thresholds=None, device=None):
        self.labels = [w for w in (blacklist or []) if str(w).strip()] or ["persona"]
        self.umbrales = dict(UMBRALES)
        for k, v in (thresholds or {}).items():
            try:
                self.umbrales[_norm(k)] = float(v)
            except (TypeError, ValueError):
                pass

        # Términos a pedirle al detector: los de los conceptos vigilados más la
        # evidencia que necesitan las reglas de evento abstracto.
        terminos = []
        for etiqueta in self.labels:
            terminos += VOCABULARIO.get(_norm(etiqueta), [])
        terminos += EVIDENCIA_EXTRA
        if "person" not in terminos:
            terminos.append("person")     # sin personas no hay evento que juzgar
        self.terminos = sorted(set(terminos))

        self._modelo = None
        self._roto = False

    # -- carga perezosa ----------------------------------------------------
    def _cargar(self):
        if self._modelo is not None or self._roto:
            return self._modelo
        try:
            if not os.path.exists(RUTA_LOCAL):
                import boto3
                os.makedirs(os.path.dirname(RUTA_LOCAL), exist_ok=True)
                boto3.client("s3", region_name="us-east-1").download_file(
                    MODELO_S3[0], MODELO_S3[1], RUTA_LOCAL)
            from ultralytics import YOLOE
            m = YOLOE(RUTA_LOCAL)
            m.set_classes(self.terminos, m.get_text_pe(self.terminos))
            self._modelo = m
            print(f"[yoloe] cargado con vocabulario {self.terminos}")
        except Exception as e:
            self._roto = True
            print(f"[yoloe] NO disponible ({type(e).__name__}: {e})")
        return self._modelo

    def threshold_for(self, label):
        return self.umbrales.get(_norm(label), UMBRAL_POR_DEFECTO)

    # -- reglas de evento abstracto ---------------------------------------
    @staticmethod
    def _cercanas(cajas):
        """¿Hay dos personas al alcance del brazo? Proxy de interacción física.

        No afirma que haya violencia: afirma que vale la pena preguntárselo al
        VLM, que es una decisión mucho más barata de acertar.
        """
        for i in range(len(cajas)):
            for j in range(i + 1, len(cajas)):
                (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = cajas[i], cajas[j]
                acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
                bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                diag = max(((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5, 1.0)
                if ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 <= PROXIMIDAD * diag * 2:
                    return True
        return False

    def _eventos(self, detecciones, cajas_persona):
        """Evidencia -> puntuación de evento abstracto. Conservador a propósito."""
        out = {}
        conf_persona = max([c for t, c, _ in detecciones if t == "person"], default=0.0)
        arma = max([c for t, c, _ in detecciones
                    if t in ("knife", "handgun", "gun", "kitchen knife", "pocketknife")],
                   default=0.0)
        if conf_persona <= 0:
            return out            # sin personas no hay robo, pelea ni caída
        # ROBO: persona + objeto de interés. Sin arma la evidencia es débil y se
        # deja por debajo del umbral, para no convertir a cada transeúnte en un
        # candidato y disparar el gasto de Mimir.
        out["robos"] = max(arma, 0.15 if conf_persona > 0 else 0.0)
        # VIOLENCIA: hacen falta DOS personas y que estén cerca.
        out["violencia"] = (min(conf_persona, 0.9) if len(cajas_persona) >= 2
                            and self._cercanas(cajas_persona) else 0.10)
        # CAÍDA: la resuelve `pose.py` por geometría corporal, que es mejor que
        # cualquier detector de objetos para esto. Aquí no se puntúa.
        out["caidas"] = 0.0
        return out

    # -- interfaz de ClipScorer -------------------------------------------
    def score_detallado(self, frame_bgr, rois):
        """(mejor_score, mejor_etiqueta, coords, {etiqueta: score})."""
        vacio = (0.0, None, None, {})
        modelo = self._cargar()
        if modelo is None:
            return vacio
        try:
            r = modelo.predict(frame_bgr, verbose=False,
                               conf=min(self.umbrales.values()) * 0.8)[0]
        except Exception as e:
            print(f"[yoloe] error de inferencia: {type(e).__name__}: {e}")
            return vacio
        cajas = getattr(r, "boxes", None)
        detecciones, cajas_persona = [], []
        if cajas is not None:
            for i in range(len(cajas)):
                try:
                    termino = r.names[int(cajas.cls[i])]
                    conf = float(cajas.conf[i])
                    xy = tuple(float(v) for v in cajas.xyxy[i][:4])
                except Exception:
                    continue
                detecciones.append((termino, conf, xy))
                if termino == "person":
                    cajas_persona.append(xy)

        # Objeto -> concepto: el mejor score de cualquier término que lo evidencie.
        por_etiqueta, coords_de = {}, {}
        for etiqueta in self.labels:
            terminos = VOCABULARIO.get(_norm(etiqueta))
            if not terminos:
                continue
            mejores = [(c, xy) for t, c, xy in detecciones if t in terminos]
            if mejores:
                c, xy = max(mejores, key=lambda p: p[0])
                por_etiqueta[etiqueta] = c
                coords_de[etiqueta] = tuple(int(v) for v in xy)

        for concepto, score in self._eventos(detecciones, cajas_persona).items():
            if any(_norm(e) == concepto for e in self.labels):
                por_etiqueta[concepto] = score
                if cajas_persona:
                    coords_de[concepto] = tuple(int(v) for v in cajas_persona[0])

        if not por_etiqueta:
            return vacio
        etiqueta = max(por_etiqueta, key=lambda k: por_etiqueta[k])
        return por_etiqueta[etiqueta], etiqueta, coords_de.get(etiqueta), por_etiqueta

"""
Cámara falsa end-to-end: "streamea" grabaciones (clips) hacia la cascada tal como
lo haría una cámara en vivo (frames a fps real), corre motion -> CLIP -> VLM
(Bedrock real) y MIDE accuracy contra el ground-truth de cada clip.

Clips:
  - persona_ctrl : imagen REAL de una persona que se mueve sobre un cuarto  -> GT: persona
  - cuchillo_ctrl: imagen REAL de un cuchillo (Sohas) que entra y se mueve   -> GT: cuchillo
  - vacio        : cuarto estático, sin movimiento                          -> GT: (nada)
  - parque_real  : grabación REAL de un parque (árboles/autos, benigno)      -> GT: (nada)

Las alertas se CAPTURAN (no se escribe a la tabla ni se notifica). VLM real.
"""
import os, sys, time, threading, glob, zipfile, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import common, vlm as vlm_mod, tiers
from vision import ClipScorer

SCR = "/tmp/claude-0/-home-user/4ba2934f-f96f-56e2-aa3c-44d0857a4cd9/scratchpad"
E2E = SCR + "/e2e"

# --- capturar alertas + contar VLM (sin efectos reales) ---
CAPTURED = []
def fake_alert(jpg, meta, score, coords, label, source):
    CAPTURED.append({"label": label, "source": source, "score": round(float(score), 3)})
    return "test"
common.raise_alert = fake_alert
_orig_judge = vlm_mod.judge
VLM_LOG = []
def judge2(jpg, label):
    c, r = _orig_judge(jpg, label); VLM_LOG.append((label, c)); return c, r
vlm_mod.judge = judge2

# --- construir clips desde imágenes reales ---
def load_person():
    fs = sorted(glob.glob(SCR + "/brayham_frames/01_*.jpg"))
    img = cv2.imread(fs[0]) if fs else None
    if img is None: img = np.full((480, 320, 3), 190, np.uint8)
    return cv2.resize(img, (300, 450))

def load_knife():
    z = zipfile.ZipFile(SCR + "/granada_weapons.zip")
    lab = json.load(open(SCR + "/sohas_det_labels.json"))
    idx = {n.split('/')[-1]: n for n in z.namelist() if ('/images/' in n or '/images_test/' in n) and n.lower().endswith('.jpg')}
    for fn in lab["pos"]:
        if fn in idx:
            a = cv2.imdecode(np.frombuffer(z.read(idx[fn]), np.uint8), cv2.IMREAD_COLOR)
            if a is not None: return cv2.resize(a, (360, 270))
    return np.full((270, 360, 3), 160, np.uint8)

def moving_clip(obj, warmup=24, active=60, W=1280, H=720):
    bg = np.full((H, W, 3), 105, np.uint8)
    cv2.rectangle(bg, (0, H-90), (W, H), (90, 90, 95), -1)  # "piso"
    frames = [bg.copy() for _ in range(warmup)]
    oh, ow = obj.shape[:2]
    for i in range(active):
        f = bg.copy()
        x = int(120 + i * (W - ow - 240) / active)
        y = H - 100 - oh
        f[max(0,y):y+oh, x:x+ow] = obj
        frames.append(f)
    return frames

def static_clip(warmup=60, W=1280, H=720):
    bg = np.full((H, W, 3), 105, np.uint8)
    cv2.rectangle(bg, (0, H-90), (W, H), (90, 90, 95), -1)
    return [bg.copy() for _ in range(warmup)]

def real_clip(path, max_frames=140):
    cap = cv2.VideoCapture(path); out = []
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release()
    return out

# --- "cámara": entrega frames a fps real ---
def camera(frames, fps=12, stop=None):
    dt = 1.0 / fps
    for f in frames:
        if stop is not None and stop.is_set(): return
        yield f
        time.sleep(dt)

def run_one(name, frames, blacklist, expected, clear_margin=0.15):
    CAPTURED.clear(); VLM_LOG.clear()
    scorer = ClipScorer(blacklist=blacklist)
    from transport import LocalQueue
    cq, vq = LocalQueue(), LocalQueue()
    stop = threading.Event()
    threading.Thread(target=tiers.run_clip, args=(cq, vq, scorer),
                     kwargs={"stop_event": stop, "clear_margin": clear_margin}, daemon=True).start()
    threading.Thread(target=tiers.run_vlm, args=(vq,), kwargs={"stop_event": stop}, daemon=True).start()
    meta = {"device_id": name, "owner_uid": "sim", "camera_name": name, "client_id": 0}
    t0 = time.time()
    tiers.run_motion(camera(frames, stop=stop), cq, meta, distributed=False, window_seconds=60, emit_interval=1.0)
    # drenar
    dt0 = time.time()
    while time.time() - dt0 < 12 and (cq._q.qsize() or vq._q.qsize()):
        time.sleep(1)
    time.sleep(3); stop.set()
    fired = sorted({a["label"] for a in CAPTURED})
    hit = (expected in fired) if expected else (len(fired) == 0)
    dur = time.time() - t0
    print(f"\n### {name} | GT={expected or '(nada)'} | frames={len(frames)} | {dur:.0f}s")
    print(f"    disparó: {fired or '(nada)'} | VLM llamadas: {len(VLM_LOG)} | alertas: {len(CAPTURED)}")
    print(f"    RESULTADO: {'✅ correcto' if hit else '❌ incorrecto'}")
    return {"clip": name, "gt": expected, "fired": fired, "ok": hit, "alerts": list(CAPTURED)}

def main():
    print("== CÁMARA FALSA E2E: grabaciones -> cascada (motion->CLIP->VLM) -> accuracy ==")
    suite = [
        ("persona_ctrl", moving_clip(load_person()), ["persona"], "persona"),
        ("cuchillo_ctrl", moving_clip(load_knife()), ["cuchillo"], "cuchillo"),
        ("vacio", static_clip(), ["persona", "cuchillo"], None),
        ("parque_real", real_clip(E2E + "/sample-10s.mp4"), ["persona", "cuchillo", "violencia"], None),
    ]
    results = [run_one(n, f, bl, exp) for n, f, bl, exp in suite]
    ok = sum(r["ok"] for r in results)
    print("\n" + "=" * 60)
    print(f"ACCURACY end-to-end: {ok}/{len(results)} clips correctos")
    for r in results:
        print(f"  {r['clip']:<14} GT={str(r['gt']):<10} disparó={r['fired'] or '(nada)'}  {'OK' if r['ok'] else 'FALLO'}")

if __name__ == "__main__":
    main()

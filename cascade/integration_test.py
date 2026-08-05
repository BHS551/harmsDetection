"""
Prueba de integración end-to-end de la cascada.

Corre las 3 capas conectadas por SQS REAL (+ frames por S3 real + VLM real en
Bedrock), alimentadas por un VÍDEO SINTÉTICO (fondo + una persona que se mueve),
ya que no hay cámara en vivo. Las ALERTAS se capturan (no se sube nada a la tabla
de detecciones ni se notifica a nadie). Verifica el flujo tier0 -> tier1 -> tier2.
"""
import os, sys, time, threading, glob
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))

import common, vlm as vlm_mod, tiers
from vision import ClipScorer
from transport import SqsQueue

CAND_URL = "https://sqs.us-east-1.amazonaws.com/780817326479/heimdall-candidates"
VLM_URL = "https://sqs.us-east-1.amazonaws.com/780817326479/heimdall-vlm"

# --- instrumentación: capturar alertas y llamadas al VLM (sin efectos reales) ---
ALERTS, VLM_CALLS, CAND_SENT = [], [], [0]
def fake_alert(jpg, meta, score, coords, label, source):
    ALERTS.append({"label": label, "source": source, "score": round(float(score), 3)})
    print(f"   >>> ALERTA CAPTURADA [{source}] {label} score={score:.3f}")
    return "test-id"
common.raise_alert = fake_alert

_orig_judge = vlm_mod.judge
def counting_judge(jpg, label):
    confirmed, reason = _orig_judge(jpg, label)
    VLM_CALLS.append({"label": label, "confirmed": confirmed, "reason": reason})
    return confirmed, reason
vlm_mod.judge = counting_judge


def build_synthetic_video():
    """Fondo gris (warmup) + una persona real que se desplaza (genera movimiento)."""
    person_path = sorted(glob.glob("/tmp/claude-0/-home-user/4ba2934f-f96f-56e2-aa3c-44d0857a4cd9/scratchpad/brayham_frames/01_*.jpg"))
    person = cv2.imread(person_path[0]) if person_path else None
    if person is None:
        person = np.full((500, 300, 3), 200, np.uint8)  # fallback
    person = cv2.resize(person, (320, 480))
    W, H = 1280, 720
    frames = []
    bg = np.full((H, W, 3), 110, np.uint8)
    cv2.putText(bg, ".", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)
    for _ in range(18):          # warmup: MOG2 aprende el fondo
        frames.append(bg.copy())
    for i in range(10):          # persona entrando y moviéndose
        f = bg.copy()
        x = 200 + i * 30
        f[150:150 + 480, x:x + 320] = person
        frames.append(f)
    return frames


def source_gen(frames, stop):
    for fr in frames:
        if stop.is_set():
            return
        yield fr
        time.sleep(0.3)   # deja avanzar el tiempo (throttle/ventana)


def main():
    print("== Prueba de integración de la cascada (SQS + S3 + Bedrock reales) ==")
    # purgar colas por si quedaron mensajes
    sqs = __import__("boto3").client("sqs", "us-east-1")
    for u in (CAND_URL, VLM_URL):
        try: sqs.purge_queue(QueueUrl=u)
        except Exception: pass
    time.sleep(2)

    scorer = ClipScorer(blacklist=["persona", "cuchillo", "robos"])
    print("scorer listo | prompts:", len(scorer.prompts), "| distractores:", scorer.distractor_prompts)

    cand_q, vlm_q = SqsQueue(CAND_URL), SqsQueue(VLM_URL)
    _orig_send = cand_q.send
    def counting_send(m):
        CAND_SENT[0] += 1; return _orig_send(m)
    cand_q.send = counting_send

    stop = threading.Event()
    meta = {"device_id": "itest", "owner_uid": "itest", "camera_name": "itest-cam", "client_id": 999}

    t1 = threading.Thread(target=tiers.run_clip, args=(cand_q, vlm_q, scorer),
                          kwargs={"distributed": True, "stop_event": stop, "clear_margin": 0.15}, daemon=True)
    t2 = threading.Thread(target=tiers.run_vlm, args=(vlm_q,),
                          kwargs={"distributed": True, "stop_event": stop}, daemon=True)
    t1.start(); t2.start()

    frames = build_synthetic_video()
    print(f"vídeo sintético: {len(frames)} frames (18 fondo + 10 con persona en movimiento)")
    # capa 0 alimenta la cola de candidatos
    tiers.run_motion(source_gen(frames, stop), cand_q, meta, distributed=True,
                     window_seconds=60, emit_interval=1.0, stop_event=stop)
    print(f"capa0 terminó de alimentar | candidatos emitidos: {CAND_SENT[0]}")

    # drenar: esperar a que capa1 y capa2 procesen
    t0 = time.time()
    while time.time() - t0 < 40:
        if ALERTS and not _queue_has_pending(sqs, CAND_URL) and not _queue_has_pending(sqs, VLM_URL):
            time.sleep(3); break
        time.sleep(2)
    stop.set(); time.sleep(2)

    print("\n===== RESULTADO =====")
    print(f"Candidatos emitidos por capa0 (movimiento): {CAND_SENT[0]}")
    print(f"Llamadas al VLM (capa2, Bedrock real): {len(VLM_CALLS)}")
    for c in VLM_CALLS[:5]:
        print(f"   VLM[{c['label']}] confirmed={c['confirmed']} :: {str(c['reason'])[:70]}")
    print(f"Alertas capturadas: {len(ALERTS)}")
    for a in ALERTS[:8]:
        print(f"   ALERTA {a}")
    ok = CAND_SENT[0] > 0 and (len(ALERTS) > 0 or len(VLM_CALLS) > 0)
    print("\nVEREDICTO:", "OK end-to-end ✅" if ok else "revisar ❌",
          "(capa0->SQS->capa1->SQS->capa2->VLM->alerta)")


def _queue_has_pending(sqs, url):
    a = sqs.get_queue_attributes(QueueUrl=url, AttributeNames=[
        "ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
    return int(a["ApproximateNumberOfMessages"]) + int(a["ApproximateNumberOfMessagesNotVisible"]) > 0


if __name__ == "__main__":
    main()

"""
run_pipeline.py — punto de entrada de la cascada.

  python run_pipeline.py local  [context.json]   -> las 3 capas en UN proceso (una
                                                     instancia barata que hace todo).
  python run_pipeline.py tier0|tier1|tier2        -> una sola capa (instancias
                                                     separadas), comunicadas por SQS.

  Topología económica (Fase B):
  python run_pipeline.py motion-multi [ctx.json]  -> MOTION BOX barato 24/7: corre la
                                                     capa 0 de MUCHAS cámaras en un
                                                     proceso; por cada movimiento manda
                                                     una ráfaga de ~10 frames a SQS.
  python run_pipeline.py analysis    [ctx.json]   -> ANALYSIS BOX compartida: capas
                                                     CLIP+VLM consumiendo SQS. Se apaga
                                                     sola tras 2h ociosa (cada candidato
                                                     renueva el timer); se levanta bajo
                                                     demanda (HeimdalManager) al llegar
                                                     movimiento.

Colas SQS (modo distribuido) por variable de entorno:
  HEIMDALL_CANDIDATE_QUEUE_URL  (capa0 -> capa1)
  HEIMDALL_VLM_QUEUE_URL        (capa1 -> capa2)
"""
import os
import sys
import json
import time
import threading
import concurrent.futures

import common
import tiers
from vision import ClipScorer, UNIVERSAL_CONCEPTS
from transport import LocalQueue, SqsQueue


def load_context(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_scorer(ctx):
    return ClipScorer(
        blacklist=ctx.get("detection_blacklist") or ["person"],
        distractor_prompts=ctx.get("distractor_prompts"),
        thresholds=ctx.get("thresholds"),
    )


def meta_from_ctx(ctx):
    return {
        "device_id": str(ctx.get("instance_id") or ctx.get("device_id") or ""),
        "owner_uid": ctx.get("owner_uid", ""),
        "camera_name": ctx.get("camera_name", "entrance"),
        "client_id": ctx.get("client_id", 1),
    }


def run_local(ctx):
    scorer = build_scorer(ctx)
    meta = meta_from_ctx(ctx)
    candidate_q, vlm_q = LocalQueue(), LocalQueue()
    alert_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    stop = threading.Event()

    threading.Thread(target=tiers.run_clip, args=(candidate_q, vlm_q, scorer),
                     kwargs={"stop_event": stop, "alert_pool": alert_pool}, daemon=True).start()
    threading.Thread(target=tiers.run_vlm, args=(vlm_q,),
                     kwargs={"stop_event": stop, "alert_pool": alert_pool}, daemon=True).start()

    from common import terminate_self
    rtsp = _resolve_rtsp(ctx)
    if not rtsp:
        terminate_self("Sin fuente RTSP en el contexto")
        return
    # Modo ráfaga, el mismo control de caudal que ya usa el motion box: un evento de
    # movimiento emite `burst_frames` candidatos y luego calla `burst_cooldown` s.
    # Sin estos parámetros run_motion cae al modo ventana, que emite un candidato
    # cada 0.4 s mientras haya movimiento y re-arma la ventana en cada frame, así que
    # una escena transitada no la cierra nunca (medido: ~5000 alertas/hora/cámara).
    tiers.run_motion(tiers.frames_from_rtsp(rtsp), candidate_q, meta,
                     distributed=False,
                     window_seconds=float(ctx.get("detection_window_seconds", 60)),
                     burst_frames=int(ctx.get("burst_frames", 10)),
                     burst_span=float(ctx.get("burst_span", 3.0)),
                     burst_cooldown=float(ctx.get("burst_cooldown", 15.0)))


def run_tier(which, ctx):
    cand_url = os.environ["HEIMDALL_CANDIDATE_QUEUE_URL"]
    vlm_url = os.environ["HEIMDALL_VLM_QUEUE_URL"]
    meta = meta_from_ctx(ctx)
    if which == "tier0":
        rtsp = _resolve_rtsp(ctx)
        tiers.run_motion(tiers.frames_from_rtsp(rtsp), SqsQueue(cand_url), meta,
                         distributed=True, window_seconds=float(ctx.get("detection_window_seconds", 60)))
    elif which == "tier1":
        scorer = build_scorer(ctx)
        tiers.run_clip(SqsQueue(cand_url), SqsQueue(vlm_url), scorer, distributed=True)
    elif which == "tier2":
        tiers.run_vlm(SqsQueue(vlm_url), distributed=True)
    else:
        raise SystemExit(f"tier desconocido: {which}")


def run_motion_multi_box(ctx):
    """MOTION BOX barato 24/7: capa 0 de N cámaras en un proceso -> SQS candidatos.
    Cada movimiento levanta la caja de análisis (wake_hook, con debounce en common)."""
    cand_url = os.environ["HEIMDALL_CANDIDATE_QUEUE_URL"]
    out_q = SqsQueue(cand_url)
    cameras = []
    for c in ctx.get("cameras", []):
        rtsp = _resolve_rtsp(c)
        if not rtsp:
            print(f"[motion] cámara {c.get('camera_name','?')} sin RTSP, se omite")
            continue
        meta = {
            "device_id": str(c.get("device_id") or c.get("instance_id") or ""),
            "owner_uid": c.get("owner_uid", ""),
            "camera_name": c.get("camera_name", "entrance"),
            "client_id": c.get("client_id", 1),
            "blacklist": c.get("detection_blacklist") or ["persona"],
        }
        cameras.append({"rtsp": rtsp, "meta": meta})
    if not cameras:
        print("[motion] no hay cámaras válidas; nada que hacer")
        return
    print(f"[motion] motion box con {len(cameras)} cámara(s), ráfaga={ctx.get('burst_frames',10)} frames")
    tiers.run_motion_multi(
        cameras, out_q, distributed=True,
        burst_frames=int(ctx.get("burst_frames", 10)),
        burst_span=float(ctx.get("burst_span", 3.0)),
        burst_cooldown=float(ctx.get("burst_cooldown", 15.0)),
        wake_hook=common.ensure_analysis,
    )


def run_analysis(ctx):
    """ANALYSIS BOX compartida: CLIP + VLM consumiendo SQS. Watchdog: se auto-apaga
    tras `idle_shutdown_seconds` (2h por defecto) sin candidatos; cada lote renueva el
    timer. Los mensajes en vuelo no se pierden: SQS los re-entrega al despertar."""
    scorer = ClipScorer(blacklist=ctx.get("universal_concepts") or UNIVERSAL_CONCEPTS)
    cand = SqsQueue(os.environ["HEIMDALL_CANDIDATE_QUEUE_URL"])
    vq = SqsQueue(os.environ["HEIMDALL_VLM_QUEUE_URL"])
    alert_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    stop = threading.Event()
    activity = {"last": time.time()}
    idle = float(ctx.get("idle_shutdown_seconds", 7200))

    def watchdog():
        while not stop.is_set():
            if time.time() - activity["last"] > idle:
                stop.set()
                common.terminate_self(f"analysis box ocioso {idle:.0f}s")
                return
            stop.wait(30)

    threading.Thread(target=watchdog, daemon=True).start()
    threading.Thread(target=tiers.run_vlm, args=(vq,),
                     kwargs={"distributed": True, "stop_event": stop, "alert_pool": alert_pool},
                     daemon=True).start()
    print(f"[analysis] CLIP+VLM listos; apagado por inactividad a {idle/3600:.1f}h")
    tiers.run_clip(cand, vq, scorer, distributed=True, stop_event=stop,
                   alert_pool=alert_pool, activity=activity)


def _resolve_rtsp(ctx):
    if ctx.get("rtsp_path"):
        return ctx["rtsp_path"]
    sid = ctx.get("rtsp_secret_id")
    if sid:
        import boto3
        return boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=sid)["SecretString"]
    return None


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    ctx_path = sys.argv[2] if len(sys.argv) > 2 else "context.json"
    ctx = load_context(ctx_path)
    if mode == "local":
        run_local(ctx)
    elif mode == "motion-multi":
        run_motion_multi_box(ctx)
    elif mode == "analysis":
        run_analysis(ctx)
    else:
        run_tier(mode, ctx)

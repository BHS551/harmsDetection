"""
run_pipeline.py — punto de entrada de la cascada.

  python run_pipeline.py local  [context.json]   -> las 3 capas en UN proceso (una
                                                     instancia barata que hace todo).
  python run_pipeline.py tier0|tier1|tier2        -> una sola capa (instancias
                                                     separadas), comunicadas por SQS.

Colas SQS (modo distribuido) por variable de entorno:
  HEIMDALL_CANDIDATE_QUEUE_URL  (capa0 -> capa1)
  HEIMDALL_VLM_QUEUE_URL        (capa1 -> capa2)
"""
import os
import sys
import json
import threading
import concurrent.futures

import tiers
from vision import ClipScorer
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
    tiers.run_motion(tiers.frames_from_rtsp(rtsp), candidate_q, meta,
                     distributed=False, window_seconds=float(ctx.get("detection_window_seconds", 60)))


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
    else:
        run_tier(mode, ctx)

# SkyEye — scale readiness review (2026-10-09)

Automated review across harmsDetection, HeimdalManager, Store/List Lambdas and the landing UI.
Intended as shared context for the go-to-market routine: what can honestly be promised to customers today.

## 1. Accuracy (source: testbench/RESULTADOS.txt, BITACORA.md cycle 3)
- Negatives clean: 6/6 (no false incident alerts on empty nature, pedestrians, construction, judo, intersection).
- Positives detected: 1/4 (only `pelea_calle`; `caida_escaleras`, `disturbios_saqueo`, `disturbios_calle` missed).
- Sample is tiny (10 scenes, ~8 min windows), single environment, no precision/recall with confidence intervals.
- Ground truth is too rigid (legit `caidas` in a riot counted as a miss).
- `persona` alerts fire on nearly every scene -> alert noise; it is the main VLM traffic driver.
- Implication: **do not market "detects fights/robberies/falls" as reliable.** Position as "person-on-ground / intrusion / presence alerts, pilot phase" until recall is measured on >=100 labelled clips per class.

## 2. Cost / unit economics (cycle 3)
- VLM calls/hour fell 1877 -> 234; Bedrock ~USD 15/camera/month; dedicated EC2 ~USD 86/camera/month; shared "Phase B" 10 cams ~USD 19/cam/month (5 cams ~USD 23).
- Pricing must stay well above ~USD 25-30/cam/month for shared topology; dedicated-per-camera only works at premium prices.
- Real deployments (empty premises at night) should be cheaper than the bench suggests (people gate).

## 3. Health / reliability gaps
1. **No CI, no unit tests** in any repo (only ad-hoc integration scripts). Lambdas have no workflows.
2. Heartbeat exists (common.py) but no alarm on missing heartbeats / DLQ depth / Bedrock throttling visible in repo -> add CloudWatch alarms + a status page.
3. CORS on Store/List Lambdas falls back to `Access-Control-Allow-Origin: *` for unknown origins (HeimdalManager is stricter) — unify.
4. Single region (us-east-1), single-process worker per camera; no autoscaling story for tier-1/2 box beyond "start on demand".
5. Camera ingress relies on customer tunnels (ngrok TCP) — fragile and a support burden at scale. Offer an on-prem/edge forwarder or a documented RTSP-over-VPN option.
6. No rate limiting / quota on SMS/email notifications beyond cooldowns; per-tenant budget caps for Bedrock absent.

## 4. Security hygiene (act first)
- `harmsDetection/context.json` and `minimal.py`, `multicore_detection.py` commit literal RTSP URLs with credentials (admin551:123456789@4.tcp.ngrok.io…). Rotate those, remove from history, keep `rtsp_secret_id` only.
- Large artifacts in git (yolov5s.pt, __pycache__ .pyc) — add to .gitignore.
- Legacy `rtsp_*.py` experiments and `heimdall-eye.py` duplicate the cascade; archive to reduce confusion for contributors.

## 5. Proposed roadmap to "many customers"
P0 (before onboarding paying customers)
- Rotate/remove committed credentials; add secret scanning.
- Build a labelled eval set (>=100 clips/class, diverse lighting/cameras) + CI job that reports precision/recall per class; flexible ground truth.
- CloudWatch alarms: missing heartbeat, DLQ>0, Bedrock throttle, Lambda errors; public status page.
- Per-tenant Bedrock/SMS budget caps; load test with 50-100 simulated cameras (camera_sim already exists).
P1
- YOLOv8-Pose for falls (planned in cycle 4), calibrate `persona` clear_margin to cut VLM traffic.
- Alert-quality loop: thumbs up/down in console feeding labelled data (also a marketing-proof asset).
- Unit tests + GitHub Actions for Lambdas and UI; staging environment.
- Edge forwarder/VPN option replacing ngrok.
P2
- Multi-region, infra-as-code (CDK/Terraform), SOC2-lite docs, privacy/retention policy for frames (S3 lifecycle) — needed for B2B sales.

## 6. Suggestions for the customer-acquisition routine
- Target segments where "person present / on ground / after-hours intrusion" is valuable and tolerance for false alarms is higher: closed premises at night, warehouses, construction sites, parking.
- Run free 30-day pilots with 3-5 design partners to produce labelled data + case studies; position as pilot, not guarantee.
- Prices must clear >=USD 30-40/cam/month shared; LatAm pricing (PayU) needs to be checked against that floor.
- Don't promise violence/robbery detection in copy until P0 eval shows recall numbers.

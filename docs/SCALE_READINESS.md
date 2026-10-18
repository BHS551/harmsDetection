# SkyEye — scale-readiness assessment (2026-10-18)

Scheduled review of accuracy, health and what blocks "many customers".
Also the hand-off note for the customer-acquisition routine: section 5 lists
what the product can and cannot honestly promise today.

Sources: README/BITACORA/RESULTADOS in this repo, `cascade/*`, HeimdalManager,
the four Lambdas and the landing UI. Nothing was run against live AWS.

## 1. Accuracy (measured, testbench cycle 3)

| Metric | Result | Caveat |
|---|---|---|
| Clean negatives | 6/6 | `persona` still fires on every scene (alerts for plain people) |
| Incident positives | 1/4 detected | `caida_escaleras`, `disturbios_saqueo`, `disturbios_calle` missed |
| Verified true positives | 2 (both `pelea_calle`) | Only one real fight clip in the bench |
| Sample | 10 clips, ~8-32 min each | Far too small for any claim |

Takeaway: ~25% recall on incident clips. Not sellable as "detects robberies /
fights" without caveats. Falls are the strongest class (and the pose-model
route in cycle 4 reports 92-98%).

Gaps:
- No precision/recall per class, no confidence intervals, no false-alarms/camera/day figure — the number a buyer actually cares about.
- Ground truth is too rigid (a fall during a riot counts as a miss).
- Bench is YouTube-style footage, not real CCTV (night, IR, fisheye, low bitrate).
- `robos` (theft) has no working signal at all; a single frame cannot show it.

## 2. Health / reliability risks (ordered)

1. **Committed credentials.** `harmsDetection/context.json` is tracked and
   contains `rtsp://admin551:123456789@...ngrok.io`. Rotate that camera
   password and remove the file from history/`git rm --cached`. (Related fixes
   already landed: Secrets Manager for RTSP, Firebase creds.)
2. **No CI, no unit tests** in any of the 7 repos (only `integration_test.py` /
   `camera_sim_test.py`, run by hand). Every deploy is a manual Lambda paste.
3. **Hardcoded infra IDs** in `cascade/common.py` / HeimdalManager (account id
   `780817326479`, API Gateway hosts, bucket `detection-frames-tests` — a
   *test* bucket in production path). Move to env/IaC.
4. **Wildcard CORS remains in HeimdalManager** (`Access-Control-Allow-Origin: "*"`);
   other Lambdas were restricted, this one was not.
5. **No observability.** Prints only. No CloudWatch metrics/alarms for: queue
   depth/DLQ, VLM error rate (`judge` returns `None` on error and the frame is
   silently dropped → missed alerts during a Bedrock throttle), camera offline,
   heartbeat gaps, analysis box down.
6. **Single analysis box = single point of failure and throughput cap.**
   One `m7i-flex.large` runs CLIP+VLM for every camera; cold start on demand
   adds latency to the first alert after idle; no autoscaling by queue depth.
7. **Camera transport is ngrok tunnels** to customer cameras. Fragile, a TOS
   issue at volume, and customer-side setup friction. Needs an edge agent
   (outbound-only RTSP relay/WireGuard) or ONVIF/cloud-camera integrations.
8. **Per-camera EC2 in "local" mode** + a launch-per-camera model hits EC2
   vCPU quotas quickly; Phase B (shared boxes) is the right direction but
   plan caps are counted via running instances only.
9. Notification: SMS-only cost channel, 300 s cooldown, no delivery
   tracking, no escalation, no push/WhatsApp (the dominant channel in LatAm).
10. `Dockerfile`, stale `__pycache__`, `yolov5s.pt` (weights) tracked in git.

## 3. Cost (measured, cycle 3)

Bedrock ~$15/camera/month, dedicated $86, Phase B $19-23/camera at 5-10
cameras. Unit economics only work if price/camera > ~$40 and cameras are packed
(Phase B). Need an explicit margin model per plan in `lib/plans.ts`.

## 4. Proposed roadmap to "ready for many customers"

P0 (this week, cheap):
- Rotate/remove committed RTSP credential; add secret scanning (`run_secret_scanning`).
- Restrict CORS on HeimdalManager.
- GitHub Actions: lint + `node --check`/unit tests for Lambdas, `pytest` for cascade (vision/tiers with mocks).
- CloudWatch alarms: DLQ>0, analysis queue age>60 s, VLM errors>5%, heartbeat gap>2 min; SNS to owner.
- Retry/backoff on VLM `None` instead of dropping.

P1 (2-4 weeks, accuracy + trust):
- Build a labelled eval set of >=200 clips (50+/class incl. night/IR), report per-class precision/recall and **false alerts per camera per day**; gate releases on it.
- Pose-based fall detection (YOLOv8-Pose) in the motion/CLIP box; calibrate `persona` clear margin (cycle 4 items 1-2).
- Drop/rebrand `robos` until it has a real signal (dwell/zone/rule-based: door forced, after-hours entry, object removal).
- Per-camera sensitivity + schedule + ROI masks in the console (biggest lever on false alarms).
- Alert UX: push/WhatsApp, snapshot + 5 s clip, one-tap "false alarm" feedback stored as training signal.

P2 (1-3 months, scale):
- Autoscale analysis workers on SQS depth (ASG/ECS), keep warm pool of 1; multi-region optional.
- Replace ngrok with an outbound edge agent or ONVIF/cloud relays; onboarding wizard that tests the stream.
- IaC (CDK/Terraform) for queues, Lambdas, tables, launch template; staging env; no test bucket in prod.
- Multi-tenant data hygiene: per-tenant retention (30-day TTL exists), export/delete, ToS/privacy (video of people → GDPR/Habeas-Data style consent, signage obligations).
- Billing: usage metering vs plan, dunning, per-plan margin dashboard.

## 5. Hand-off to the customer-acquisition routine

Honest claims today: 24/7 motion-gated monitoring; strong on "person on the
ground / fall" and obvious street fights; ~$20-25 variable cost/camera; setup
via console with plan enforcement.
Do not promise: reliable theft/robbery detection, low false alarms (unmeasured),
night/IR performance (untested), uptime SLA (no monitoring).
Best early-customer profile: small sites with a few existing RTSP cameras
and a clear after-hours/fall use case (elder care, warehouses, closed shops at
night) where the person-gate makes cost near zero when empty. Offer a
paid pilot with 1-3 cameras and measure false-alerts/day together — that data
fills the biggest gap above and becomes the sales proof.
Suggested sequencing: finish P0 + the eval set (P1 first bullet) before
onboarding more than ~10 cameras.

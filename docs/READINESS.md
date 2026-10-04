# SkyEye — readiness assessment for scale (2026-10-04)

Automated review (accuracy, health, cost). Shared as context for the
client-acquisition routine: **what we can honestly promise customers today, and
what must be fixed before volume.**

## 1. Accuracy (source: testbench/BITACORA.md, RESULTADOS.txt, cycle 3)

| Metric | Value | Caveat |
|---|---|---|
| Clean negatives | 6/6 | 10-scene bench, small sample |
| Incident scenes detected | 1/4 (`pelea_calle`) | after manual frame review; 2 TPs verified |
| Known failure | VLM sees broken door / fire / riot police, answers "NO" | question asks for the act, not the trace (`robos`, `violencia`) |
| `caidas` | FP on seated person | needs explicit horizontal-posture requirement |
| `persona` | most frequent alert (every scene) | cheap, but noisy as a *product alert* |

Bench is 10 public clips: no statistical weight. No precision/recall on real
customer cameras exists. Reference truth is too rigid (penalises legit alerts).

**Honest claim today:** reliable person/knife detection with low false alarms
on empty scenes; fall/violence/robbery are *beta* (~25% scene recall).
Do not market "detects robberies/fights" as guaranteed.

## 2. Cost (measured, cycle 3)

- VLM calls/hour/camera: 1,877 -> 234 (-88%). Bedrock ~15 USD/cam/month.
- Dedicated worker per camera ~86 USD/mo; shared "Phase B" ~19 USD/cam/mo at 10 cams.
- Plans in code: cam1 = 1.00 USD (priceCents 100, looks like test price), cam5 = 50 USD.
  At cam5 (10 USD/cam) revenue < cost in dedicated mode; margin only works with Phase B
  and ~10+ cams per analysis box. **Pricing vs. cost must be reviewed before selling.**

## 3. Health / scale risks (code review)

P0 — before any paid volume
1. **Dedicated EC2 per camera** (HeimdalManager `RunInstances` per device) does not
   scale or pay; make Phase B (motion-multi + shared analysis box) the default.
2. **No CI, no tests, no IaC** in any of the 7 repos (no `.github`); lambdas deployed
   by hand; hardcoded ACCOUNT_ID default. One bad deploy = outage. Add GitHub Actions
   (lint+tests) and IaC (SAM/CDK/Terraform).
3. **No observability/SLOs**: no alarms on DLQ depth, worker heartbeat loss, VLM
   error rate, Bedrock throttling, EC2 launch failures, per-tenant cost. Add
   CloudWatch alarms + dashboard + budget alerts.
4. **Quotas**: EC2 vCPU limit, Bedrock TPM/RPM, Lambda concurrency are
   account defaults; request increases before onboarding >~20 cameras.
5. **CORS `*` in HeimdalManager and subscriptions** (other lambdas already restricted).
6. **cam1 priced 1 USD** — confirm intentional.

P1
7. Plan limit check counts running instances then launches: race (two simultaneous
   starts bypass maxCameras). Use conditional write / DynamoDB counter.
8. Subscriptions are activated manually by an admin panel (`PLAN_CAMERAS` cam1/cam5
   duplicated in 2 repos); automate via PayU/Stripe webhook -> DynamoDB, add renewal,
   cancellation and dunning handling.
9. `monitoring.ts` camera count per-browser (known, safe, but confusing UX).
10. Single region (us-east-1), single analysis box (SPOF; 2h idle shutdown means
    cold start latency of minutes on first event) — add warm-pool or SLA wording.
11. Detection TTL 30 days: define retention/privacy policy (video of people!):
    GDPR/LatAm data-protection notice, consent, DPA — prerequisite for B2B.
12. RTSP over ngrok tunnels is fragile for customers; offer a lightweight on-prem
    agent/connector (outbound only) or ONVIF/Cloud camera integrations.

P2
13. Repo hygiene: stale experiments (`rtsp_*.py`, `yolov5s.pt` binary committed).
14. Notifications: SMS is costly; add WhatsApp/push/webhook; alert fatigue controls
    (per-user quiet hours, per-label thresholds, daily digest).

## 4. Improvements to raise accuracy (ordered by ROI)

1. YOLOv8-Pose for falls (92-98% reported, no VLM) -> removes a whole class from VLM.
2. Apply the "trace not act" prompt to `robos`/`violencia` (broken door, fire,
   scattered goods, crowd + police).
3. Multi-frame (burst) VLM input / short clip instead of a single frame.
4. Build a labelled evaluation set of >=200 clips (incl. real customer pilots
   with consent); report precision/recall per label; gate releases on it.
5. Per-customer feedback button (TP/FP) in console -> labelled data + threshold tuning.
6. Calibrate `clear_margin` for `persona`.

## 5. Suggestions for the client-acquisition routine

- Position first as **"person-presence / intrusion after hours + knife"** (strongest),
  with falls (elderly care, warehouses) as pilot feature once pose lands.
- Target segments where the cascade saves most: closed premises at night (shops,
  warehouses, parking, construction sites) — the person gate zeroes cost when empty.
- Offer paid **pilots of 1-5 cameras (30 days)** to collect labelled data and
  testimonials; do not sign SLA contracts before items P0-1..4 are done.
- Capacity guide: until P0 is done, cap onboarding at ~10-20 cameras total.
- Need from acquisition: top 3 segments, price sensitivity (USD/cam/month), whether
  they have RTSP/ONVIF cameras, data-privacy requirements by country. Reply by
  committing to `docs/ACQUISITION.md` in this repo; this routine will read it.

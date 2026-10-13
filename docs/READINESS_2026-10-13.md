# SkyEye — readiness review for scale (2026-10-13)

Automated review across harmsDetection, HeimdalManager, List/Store Lambdas and the landing UI.
Shared with the "get clients" routine: section 5 lists what is safe to promise customers today.

## 1. Accuracy (the main blocker)
Source: `testbench/BITACORA.md`, `RESULTADOS.txt` (cycle 3, 10 real scenes).
- Clean negatives: 6/6. Positive incidents detected: **1/4** (only `pelea_calle`). Falls on stairs, looting and riots missed.
- Of 3 alerts inspected on riot footage: 1 true violence, 1 false "fall" (a seated person).
- VLM answers NO on frames showing forced doors/fire/riot lines: question asks about the act, not the trace (fixed for falls, not yet for robberies).
- Benchmark is tiny (10 clips), ground truth too rigid (penalises legit alerts), no precision/recall per class, no real-customer footage.
- Every non-incident scene still emits `persona` alerts (1-5 per clip): a "person detected" alert on every walk-by will be noise for customers.

## 2. Cost / unit economics
- Plan cam5 = $50/mo for 5 cameras ($10/camera). Measured compute: ~$86/camera/mo dedicated (one EC2 per camera, `mode: local`), ~$19/camera/mo on shared "Phase B" at 10 cameras ($115 for 5 cameras).
- => **Every customer on the current plan loses money** unless Phase B (motion box + on-demand analysis box) is the default path. HeimdalManager still launches one instance per camera (`index.mjs` ~l.320, `mode: "local"`).
- Bedrock VLM was 97% of variable cost; now ~$15/cam/mo. SMS is the other uncapped cost (per-customer cap missing).
- Idle empty-site cost is much lower than benchmark suggests (person gate), but unmeasured on real deployments.

## 3. Health / operability
- No CI, no unit tests in any repo (only integration scripts needing AWS). No lint gate on Lambdas.
- No metrics/alarms: heartbeat exists but nothing alerts on dead camera, SQS DLQ depth, analysis box stuck, EC2 launch failures, Bedrock throttling.
- Camera outage: tier 0 gives up after 30 retries (~2 min) and the thread dies for the multi-camera box; customer is not told their camera is blind.
- Single analysis box for all cameras = single point of failure and a latency cliff with many simultaneous events.
- Lambdas default `Access-Control-Allow-Origin: *` (overridden by allowlist function; verify allowlist env is set in prod).
- `ListDetections` pagination uses client-supplied `startKey` parsed with `JSON.parse` (fine, owner-scoped via GSI) but has no error handling for malformed key -> 500.
- `subscriptions` admin list uses `Scan`: OK for admin, fine at hundreds of users.
- Hygiene: `harmsDetection/context.json` (tracked in git) contains a literal RTSP URL with credentials (`admin551:...@ngrok`) — rotate and remove from history; also `yolov5s.pt` and legacy `rtsp_*.py` tracked; Dockerfile references stale CUDA 11.6/py script.
- Customer cameras reached via ngrok tunnels: not viable for many customers (needs an on-prem connector / WireGuard / P2P relay).

## 4. Prioritised improvements
P0 (before any paid customers beyond pilots)
1. Build a real eval set: 100+ clips per class, 50% from pilot customers (with consent), report precision/recall per class in CI. Replace rigid ground truth with "any plausible incident label".
2. Make shared motion/analysis boxes the default; remove per-camera EC2. Re-price or add per-plan alert/SMS caps so margin > 0.
3. Pose-based falls (YOLOv8-Pose, 92-98% reported) to remove a whole class from the VLM; apply "trace not act" prompting to `robos`.
4. Stop emitting bare `persona` alerts by default; make it a per-camera opt-in/"after hours only" schedule — this is the biggest user-facing noise source.
5. Rotate leaked RTSP credentials; add secret scanning in GitHub.
P1
6. CloudWatch dashboard + alarms: camera offline >5 min (notify customer), DLQ depth, analysis-box idle/stuck, Bedrock throttles, EC2 launch errors, per-customer spend.
7. Customer-side camera connector (small container/agent doing outbound tunnel) replacing ngrok.
8. GitHub Actions: lint + node --check + pytest of vision/tier logic with synthetic frames; deploy-worker via CI.
9. Alert feedback buttons ("false alarm"/"real") stored in DynamoDB -> continuous labelled data and live precision metric.
P2
10. Multiple analysis boxes / SQS-driven autoscaling, per-tenant quotas, status page, onboarding wizard with a camera test-stream check.

## 5. For the acquisition routine (what to say / who to target)
- Honest positioning today: **low-cost "someone is on site after hours" + person-on-ground alerts with photo evidence**, NOT "detects robberies and fights" (1/4 on the bench).
- Best-fit early customers: small businesses with existing IP/RTSP cameras that are empty at night (stores, warehouses, workshops, car lots, gyms off-hours), LatAm first (PayU, Spanish UI). Offer paid pilots of 2-5 cameras for 30 days in exchange for consented footage + feedback -> that is exactly the missing eval data.
- Do NOT scale ad spend until P0 items 1, 2 and 4 are done; each customer currently costs more than the $10/camera it pays.
- Pricing idea: per-camera $12-20 with SMS cap, or an after-hours-only cheaper tier (matches the cheap person-gate path).

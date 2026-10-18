# Demo guide

Everything runs from `src/`. Needs Python with `src/requirements.txt` installed, the final model in `artifacts/`, and the sample calls in `dataset_calls/flac/`.
Sample pair used below: fake `E_0001322931.flac` (verify, score about 0.78) and genuine `E_0000502309.flac` (allow, score about 0.005).

## 1. One call from the command line (30 s)
```bash
cd src
python -m audiodf predict ../dataset_calls/flac/E_0001322931.flac     # verdict JSON: action, score, latency
python -m audiodf predict ../dataset_calls/flac/E_0000502309.flac
```
Say: the fake lands at *verify* (ask for an extra check), the genuine call at *allow*. The score is a ranking, not a probability.

## 2. Why: the explanation (1 min)
```bash
python -m audiodf explain ../dataset_calls/flac/E_0001322931.flac --plot fake.png
```
Shows which 2-second, 4-band regions pushed the score up, with a faithfulness check, then a plain-language reason. Ready-made outputs are in `results/explain_demo/` (`*.png`, `*.json`).
Optional cloud wording (Groq): set the key in your own terminal, `export AUDIODF_LLM_API_KEY=...`, add `--provider groq`. Without a key it uses the built-in template.

## 3. Scam intent plus voice (1 min)
```bash
python -m audiodf intent --audio ../dataset_calls/flac/E_0000502309.flac     # local Whisper transcript -> intent -> one action
python -m audiodf intent --text "caller: this is the fraud department of your bank, your account is compromised, you must transfer your money to a safe account immediately and read me the one time password. receiver: ok"   # -> intent high (credential request, payment redirect, urgency)
```
Say: intent can only raise the check, never lower it, and it is never a decision on its own.

## 4. The full stack in Docker (3 min)
```bash
cd src
docker compose -f deploy/docker-compose.yml up -d              # Kafka, API, worker, Prometheus, Grafana (CPU)
# with an NVIDIA GPU:  docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up -d --build
python deploy/smoke_test.py ../dataset_calls/flac/E_0001322931.flac ../dataset_calls/flac/E_0000502309.flac --kafka    # add --image audiodf-gpu for the GPU image
docker compose -f deploy/docker-compose.yml down
```
The smoke test prints 12 checks, all should say `[ok]`: API health, `/predict`, `/explain`, Kafka verdicts for both calls, Prometheus targets, Grafana dashboard.
Open Grafana at http://localhost:3000 (dashboard "Deepfake voice detection", no login in demo mode) and Prometheus at http://localhost:9090. API docs: http://localhost:8000/docs.
First start downloads the speech model weights (internet needed once). The Kafka step needs a broker: the compose stack starts one; a separate `kafka` container on port 9092 must be stopped first.

## 5. Live call stream
```bash
python -m audiodf serve --port 8000          # then connect a WebSocket to /stream/<call-id>
python -m audiodf produce ../dataset_calls/flac/E_0001322931.flac --realtime   # Kafka: needs a broker
```
Send 16 kHz mono PCM16 frames; a verdict comes back every second after the first 2 s; send the text frame `end` for the final one.

## What to say, and not to say
- Say: it gives the agent a reason to add a check, in about 60 ms on a GPU, with a visible reason.
- Say: fakes missed at the check level are the number that matters (83 per 1000 on real phone channels, as few as 6 on simulated calls), and genuine callers asked to verify are about 1 in 10.
- Do **not** say it has been validated on live bank calls, that it is a probability, or that *escalate* means "reject". At low fraud rates most escalated callers are genuine people.

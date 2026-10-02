# Replan: Real-Time Deepfake Voice Detection

Base: the existing prototype (`AudioDeepfakeModel.ipynb`: SVM + CNN-LSTM, weighted-average ensemble on ASVspoof2019 LA) and the architecture in `BEing-Techies_PSB_HACKATHON_2026.pptx`.

## 1. Where we are vs. the original plan

| Pipeline stage (pptx) | Status in repo | Gap |
|---|---|---|
| 1. Ingestion (VoIP/SIP + Kafka) | Not built | Entire stage |
| 2. Preprocessing (16 kHz, denoise, VAD, 2 s segments) | Only loading ASVspoof wavs + fixed-length pad/crop | Denoise, VAD, segmenting/buffering |
| 3. Features (Log-Mel + MFCC/spectral) | MFCC, LFCC, MGDCC and Log-Mel extractors exist | Unify; pick the set that earns its place |
| 4a. SVM + MFCC | Built (RBF SVC, 5-fold CV) | Trained on a random 80/20 split of train; needs speaker/attack-aware validation |
| 4b. CNN + LSTM | Placeholder (1 conv layer, hardcoded LSTM input size, lr=0.01, 100 epochs, no checkpointing) | Real architecture, shape-safe, proper training loop |
| 5. Ensemble | Fixed 0.7/0.3 weighted average | Stacking / learned weights, calibrated probabilities |
| 6. 10 s window aggregation | Not built | Chunk-score aggregation logic |
| 7. Risk engine | Not built | Thresholds (slide 5: >=0.80 Fake, 0.50-0.80 Suspicious, <0.50 Real) |
| 8. FastAPI / Docker / K8s / CI-CD | Not built | Entire stage |
| 9. Prometheus / Grafana | Not built | Entire stage |

## 2. Principles for the replan

1. **Make the numbers trustworthy before building around them.** The 99.59% accuracy / 0.43% EER on the slides must be reproduced from a clean, leak-free evaluation. Until then they are targets, not results.
2. **Convert the notebook into a package first.** Everything downstream (API, streaming, tests) needs importable modules, not notebook cells.
3. **Build a vertical slice, then widen it.** A working file-in, verdict-out service beats nine half-built boxes. Kafka/K8s come after the slice works.
4. **Keep the hybrid idea; make each branch defensible.** The classical branch is the fast, explainable one; the deep branch is the pattern learner.

## 3. Phases

### Phase 0: Audit and baseline (do first)
- Re-run the notebook end to end; record the real SVM, CNN-LSTM and ensemble metrics (EER, AUC, FPR, recall, per-segment latency) on **dev and eval** protocols.
- Resolve the leakage signals in the notebook: the train/test and train/dev overlap checks, and the random `train_test_split` over the train CSV. Evaluate only on the official ASVspoof2019 dev/eval splits; tune on dev, report on eval.
- Decide which of MFCC / LFCC / MGDCC the SVM actually uses and drop the rest if they don't help (ablation table).
- Output: `docs/baseline_results.md` with honest numbers. Update slide/README claims to match.

### Phase 1: Refactor into a package
```
audiodf/
  data/        protocol parsing, datasets, segmenter
  preprocess/  resample, denoise, VAD, framing
  features/    logmel, mfcc (+ lfcc if kept)
  models/      svm.py, cnn_lstm.py, ensemble.py
  train/       train_svm.py, train_cnn_lstm.py, train_ensemble.py
  eval/        metrics.py (EER, AUC, FPR, recall, latency)
  risk/        thresholds + Allow/Verify/Block mapping
configs/       yaml configs, fixed seeds
tests/
```
- Pin dependencies (`requirements.txt`); fix seeds; save model artifacts (`.pt`, `.joblib`) with a version tag.

### Phase 2: Strengthen the models
- **CNN-LSTM:** replace the placeholder with a shape-safe stack (2-3 conv blocks with BatchNorm and pooling, then a BiLSTM, then a classifier). Compute the LSTM input size from a dummy forward pass. Use lr around 1e-3 with a scheduler, early stopping on dev EER, and checkpointing. Optionally try a Transformer encoder as the slide's alternative.
- **SVM:** keep as the fast baseline; tune C/gamma on dev; calibrate probabilities (Platt/isotonic).
- **Ensemble:** compare (a) the current fixed 0.7/0.3 average, (b) weights tuned on dev, (c) logistic-regression stacking on dev predictions. Keep whichever wins on eval EER. Stacking must be trained on held-out predictions only.
- **Robustness:** augmentation (noise, codec/telephony band-limiting 8 kHz, compression) since the target is phone calls, plus a separate evaluation on degraded audio. ASVspoof2019 LA is clean studio-style speech, so real-call robustness is the main open risk.

### Phase 3: Real-time inference core (no Kafka yet)
- Implement the streaming logic as a plain Python class: ring buffer, denoise + VAD, 2 s segments, per-segment ensemble score, aggregate over the 10 s window (mean/median/max-of-top-k, chosen on dev), then risk engine mapping.
- Benchmark per-segment latency on CPU; verify the 10 s end-to-end budget including buffering time.
- Output contract: `{label, fake_probability, risk_level, action, segments_scored}`.

### Phase 4: Serving and containerisation
- FastAPI service: `POST /predict` (file/bytes), `WS /stream` (chunked audio), `GET /health`, `GET /metrics`.
- Dockerfile for the service; CPU image first.
- Unit tests for preprocessing, features, metrics, risk mapping; one integration test with a fixture wav.

### Phase 5: Streaming backbone
- Kafka (docker-compose locally): producer simulates a call by chunking a wav into 0.5-1 s PCM messages on topic `Deepfake-Detection-Audio-Chunks`; consumer feeds the Phase 3 inference core and publishes verdicts to a results topic.
- Key chunks by call/session ID so one call's chunks stay ordered on one partition.
- VoIP/SIP capture (Asterisk/Twilio) is the last, optional integration; a simulator is enough for the demo.

### Phase 6: CI/CD and monitoring
- GitHub Actions: lint, tests, build image, model validation gate (fail if dev EER regresses past a threshold), push image.
- Kubernetes manifests (Deployment, Service, Ingress; HPA on consumer lag), starting with a local cluster (kind/minikube); AWS/GCP only if time allows.
- Prometheus metrics (request count, latency histogram, fake-rate, risk-level counts); Grafana dashboard; basic drift signal (rolling mean fake probability, feature distribution shift).

## 4. Suggested order and cut line

1. Phase 0 and 1 (foundation)
2. Phase 2 and 3 (model quality and the real-time core)
3. Phase 4 (shippable demo: Dockerised API)
4. Phase 5 (Kafka simulator)
5. Phase 6 (CI/CD, monitoring)

If time is short, ship through Phase 4 plus a Kafka simulator and a Grafana dashboard, and describe Kubernetes and live SIP as the production path rather than claiming them as done.

## 5. Open questions to settle

- Training data beyond ASVspoof2019 LA (e.g. ASVspoof 2021 DF/LA with codecs, WaveFake, In-the-Wild) for real-call generalisation?
- Compute budget: is a GPU available for the CNN-LSTM, or CPU-only?
- Demo target: live microphone demo, replayed call recordings, or both?
- Is the hackathon timeline still active (what is the deadline), and who owns which phase (current README lists three team members)?

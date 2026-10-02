# audiodf: real-time deepfake voice detection (SVM + RCNN ensemble)

Production-shaped version of the notebook prototype. Two independent branches, each with its own
audio processing, fused into one P(fake) and mapped to Allow / Verify / Block.

```
audio chunks (0.5 s, PCM16)
   |  Kafka topic, keyed by call_id        or      FastAPI  POST /predict | WS /stream/{call_id}
   v
CallSession (per call): rolling 10 s buffer
   |--> SVM branch : whole buffer (up to 10 s) -> ONE 318-d vector (MFCC+LFCC+MGDCC, mean/std) -> SVM
   '--> RCNN branch: every completed 2 s window (1 s hop) -> Log-Mel 64x200 -> CNN -> BiLSTM
                     (scores averaged over the windows inside the last 10 s)
   v
fuse: 0.7 * P_svm + 0.3 * P_rcnn  ->  risk engine (>=0.80 block, >=0.50 verify, else allow)
   v
Verdict JSON  (+ Prometheus metrics)
```

## The two branches are processed differently on purpose

| | SVM branch | RCNN branch |
|---|---|---|
| Module | `features/svm_features.py` | `features/rcnn_features.py` |
| Input | the whole buffer, any length (up to 10 s) | fixed 2 s windows, 1 s hop |
| Representation | one fixed vector: mean and std over time of MFCC, LFCC, MGDCC and their deltas | Log-Mel spectrogram, standardised per window |
| Normalisation | `StandardScaler` inside the SVM pipeline | per-window zero mean / unit variance |
| Training data | one row per utterance | one row per 2 s segment |

Do not feed the SVM 2 s segments: its eval EER doubled (8.6% to 16.9%) in our tests.
The BiLSTM only looks across one fixed window, so its look-ahead is the 2 s the buffer already waits for.

## Layout

```
audiodf/
  config.py          settings (defaults + YAML + env overrides)
  data/              audio I/O, protocol parsing, segmenter, on-disk feature cache
  features/          svm_features.py, rcnn_features.py
  models/            svm.py, rcnn.py, ensemble.py        risk.py
  training/          train_svm.py, train_rcnn.py, pipeline.py
  evaluation/        metrics.py (EER, AUC, per-attack), benchmark.py (latency)
  inference/         engine.py (scoring + fusion), session.py (per-call streaming state)
  serving/api.py     FastAPI        streaming/  processor.py + kafka_io.py
  monitoring/        Prometheus metrics        artifacts.py (versioned model bundle)
configs/default.yaml   tests/   deploy/ (Dockerfile, compose, Prometheus, Grafana, k8s)
```

## Usage

```bash
cd src
pip install -r requirements-dev.txt          # CPU/GPU torch: install the build you want first
export AUDIODF_DATA=../dataset/LA/LA         # ASVspoof2019 LA root

python run.py                                # ONE command: features -> train SVM + RCNN -> evaluate -> artifacts/
python run.py --epochs 30 --rnn lstm         # optional flags (--limit 300 for a quick smoke test)

# `python -m audiodf <command>` is the multi-command CLI (needs a subcommand); run.py is the one-click training.
python -m audiodf train --epochs 20          # same as run.py
python -m audiodf train --rcnn-checkpoint X  # reuse an RCNN checkpoint instead of training it
python -m audiodf evaluate                   # re-score saved artifacts on dev/eval
python -m audiodf predict some.flac          # one file -> verdict JSON
python -m audiodf benchmark                  # per-stage latency
python -m audiodf serve --port 8000          # API: /predict, /stream/{id}, /health, /metrics
python -m audiodf consume                    # Kafka worker (needs a broker + confluent-kafka)
python -m audiodf produce call.wav --realtime
python -m pytest                             # 32 tests, no dataset or GPU needed
```

Stream over a WebSocket: send binary frames of 16 kHz mono PCM16, receive a JSON verdict each time a
2 s window completes (from 2 s on, then every 1 s); send the text frame `end` for the final verdict.

## Results (ASVspoof2019 LA, utterance level, spoof = positive)

Dev shares its attacks (A01-A06) with training; **eval has 13 unseen attacks (A07-A19)** and is the honest number.

| | Dev EER | Eval EER |
|---|---|---|
| SVM | 0.24% | 8.58% |
| RCNN (CNN-BiLSTM, 5 epochs) | 2.98% | 15.24% |
| **Ensemble 0.7 / 0.3** | **0.14%** (acc 99.87%) | **5.67%** (acc 94.3%) |

- The refactored code reproduces the experiment exactly, and streaming 1,500 random eval files in 0.5 s
  chunks gives 5.66% EER, so the serving path matches the offline evaluation.
- The branches fail on different attacks (SVM is weak on A10, A12, A17, A18; RCNN on A10, A13-A15), which is why fusing helps.
- Fusion weights tuned on dev made eval worse (10.3% EER), so the weight is fixed at 0.7.
- Latency on an RTX 3050 laptop GPU, batch of one: first verdict about 22 ms of compute once the 2 s window completes;
  SVM features over a 10 s buffer (41 ms) are the largest cost, the RCNN forward pass is 6 ms.

## Known limitations (read before relying on the risk levels)

1. **Risk thresholds are not calibrated for unseen attacks.** With the slide thresholds (0.80 / 0.50) on eval,
   only 51% of spoofs are blocked and 72% reach verify-or-block, while 28% pass as "allow" (only 0.2% of
   bonafide are flagged). The EER operating point is near P(fake) = 0.03. Choose thresholds from the cost of a
   missed fake versus a false alarm, ideally validated on held-out attack types.
2. **Trained on clean ASVspoof2019 LA.** Phone-call codecs, background noise and unseen voice-cloning systems are untested.
   The pptx's noise reduction and VAD are not included: the models were not trained with them, so adding them only at
   inference would create a mismatch. Add them to both training and serving together.
3. **RCNN trained for 5 epochs only**; dev EER was still falling. Run `train --epochs 20`.
4. **Kafka, Docker, Kubernetes and Grafana files are untested scaffolds** (no broker or cluster was available).
   The chunk-routing logic (`streaming/processor.py`) is unit-tested; the Kafka glue is not.
5. The SVM is recomputed on each verdict (about 41 ms per 10 s buffer). Fine for now; cache or subsample if you
   need many concurrent calls per GPU.
6. ViT is not integrated. If tried, benchmark it against the RCNN on the same eval set and per-attack table.

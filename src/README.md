# audiodf: real-time deepfake voice detection (SVM + RCNN ensemble)

Production-shaped version of the notebook prototype. Two independent branches, each with its own
audio processing, fused into one P(fake) and mapped to Allow / Verify / Block. Trained on ASVspoof5.

```
audio chunks (0.5 s, PCM16)
   |  Kafka topic, keyed by call_id        or      FastAPI  POST /predict | WS /stream/{call_id}
   v
CallSession (per call): skip silence before speech -> rolling 10 s buffer
   |--> SVM branch : the buffer so far (grows 2 -> 10 s) -> ONE 318-d vector (MFCC+LFCC+MGDCC) -> SVM
   '--> RCNN branch: every completed 2 s window (1 s hop) -> Log-Mel 64x200 -> CNN -> BiLSTM
                     (scores averaged over the windows inside the last 10 s)
   v
fuse: w * P_svm + (1-w) * P_rcnn  ->  risk engine (verify / block thresholds)
   v                                  w and thresholds are tuned on dev and stored in the model bundle
Verdict JSON  (+ Prometheus metrics)
```

## The two branches are processed differently on purpose

| | SVM branch | RCNN branch |
|---|---|---|
| Module | `features/svm_features.py` | `features/rcnn_features.py` |
| Input | the buffer so far, any length up to 10 s | fixed 2 s windows, 1 s hop |
| Representation | one fixed vector: mean and std over time of MFCC, LFCC, MGDCC and their deltas | Log-Mel spectrogram, standardised per window |
| Training rows | **growing-buffer snapshots** of each clip: its first 2, 4, 6, 8, 10 s | **random 2 s windows** inside the first 10 s of speech, a fixed number per clip |
| Why | a live call scores the SVM on a growing buffer, so it must be accurate at every length | windows mirror the Kafka chunk buffer and keep long clips from dominating |

Never train or score the SVM on isolated 2 s crops: that doubled its error in our ASVspoof2019 tests
(eval EER 8.6% to 16.9%). The BiLSTM only looks across one fixed window, so its look-ahead is the 2 s
the buffer already waits for.

## Data preprocessing (identical for both classes, mirrored in serving)

`load -> fill digital silence -> VAD trim -> [training only: label-blind codec augmentation] -> branch features`

- **VAD trim** (`data/vad.py`, -45 dBFS): ASV5 train bonafide clips carry 4-10x more edge silence than spoof
  (and ASV2019 up to 15x), a shortcut unrelated to the voice. Training trims both ends of every clip;
  `CallSession` skips pre-speech audio with the same rule.
- **Digital silence** (`data/audio.py`): exact-zero samples become +-1 LSB noise. Some TTS systems emit
  exact zeros (72% of ASV5 dev attack A11), which no microphone produces and which broke the SVM's log features.
- **Codec augmentation** (`data/augment.py`): train and dev have no codec audio, while ~75% of eval (and every
  real call) is codec-processed, equally for both classes. The augmenter takes no label, so it cannot create a
  "codec means real/fake" shortcut. Covers Opus, MP3, Vorbis, G.711, narrowband; not AMR, Speex, EnCodec, AAC
  or Bluetooth channels (no in-process encoder), which the per-codec eval breakdown will expose.
- **Audit** (`python -m audiodf audit`): integrity checks and shortcut detection; training refuses to start on
  hard errors (missing audio, duplicate IDs, speakers shared across splits, wrong format).

## Training and evaluation discipline

Models fit on **ASV5 train**. Everything tuned (RCNN epoch, SVM calibration, fusion weight, risk thresholds)
uses a stratified subset of **ASV5 dev**, whose attacks (A09-A16) differ from train's (A01-A08). **ASV5 eval**
(A17-A32, 11 codec conditions) and ASV2019 eval (cross-dataset) are scored once at the end and never tuned on.
Reports: EER at the 10 s decision, EER vs seconds of speech (2/4/6/8/10 s), per-attack and per-codec EER.
Attack IDs are separate namespaces per dataset ("A01" differs between ASV2019 and ASV5).

## Layout

```
audiodf/
  config.py          settings (defaults + YAML + env overrides)
  data/              audio.py, protocol.py (ASV2019+ASV5), vad.py, segmenter.py, augment.py,
                     integrity.py (audit), prepare.py (index, SVM snapshots, RCNN window loader)
  features/          svm_features.py, rcnn_features.py (both carry a FEATURE_VERSION)
  models/            svm.py (calibrated), rcnn.py, ensemble.py        risk.py
  training/          train_svm.py, train_rcnn.py, pipeline.py
  evaluation/        metrics.py, stream_eval.py (live-call scoring), benchmark.py (latency)
  inference/         engine.py (scoring + fusion), session.py (per-call streaming state)
  serving/api.py     FastAPI        streaming/  processor.py + kafka_io.py
  monitoring/        Prometheus metrics        artifacts.py (versioned model bundle)
configs/default.yaml   tests/   deploy/ (Dockerfile, compose, Prometheus, Grafana, k8s)
```

## Usage

```bash
cd src
pip install -r requirements-dev.txt          # install the torch build you want first
export AUDIODF_ASV5=../dataset5              # ASVspoof5: ASVspoof5.*.tsv, flac_T/, flac_D/ (flac_E_eval/ for the test)
export AUDIODF_DATA=../dataset/LA/LA         # optional: ASVspoof2019 LA, used as a cross-dataset test

python run.py                                # ONE command: audit -> index -> SVM -> RCNN -> tune on dev -> test -> artifacts/
python run.py --epochs 12 --eval-utts 0      # flags (--eval-utts 0 = whole eval split, hours; --limit 800 = smoke test)

# `python -m audiodf <command>` is the multi-command CLI (needs a subcommand); run.py is the one-click training.
python -m audiodf audit --dataset asv5       # integrity audit, exit 1 on hard errors
python -m audiodf prepare                    # index clips (decode once; cached)
python -m audiodf evaluate --dataset asv5 --split eval   # score saved artifacts
python -m audiodf predict some.flac          # one file -> verdict JSON
python -m audiodf benchmark                  # per-stage latency
python -m audiodf serve --port 8000          # API: /predict, /stream/{id}, /health, /metrics
python -m audiodf consume                    # Kafka worker (needs a broker + confluent-kafka)
python -m audiodf produce call.wav --realtime
python -m pytest                             # 65 tests, no dataset or GPU needed
```

Stream over a WebSocket: send binary frames of 16 kHz mono PCM16, receive a JSON verdict each time a
2 s window completes (from 2 s of speech on, then every 1 s); send the text frame `end` for the final verdict.
The fusion weight and risk thresholds in `artifacts/bundle.json` override the config at serving time.

## Results

Run 1 (ASV5 train only; details in `results/training_report.json`, `results/evaluate_asv5_eval.json`):

| EER at the 10 s decision | dev (unseen attacks, codec-aug) | **ASV5 eval** (30k clips, 16 unseen attacks, real codecs) | ASV2019 eval (cross-dataset) |
|---|---|---|---|
| SVM | 21.8% | 33.6% | 41.4% |
| RCNN | 17.6% | 37.4% | 33.3% |
| Fused (weight 0.2) | 15.9% | **33.0%** | 37.8% |

**This model is not production-ready.** It generalises poorly to unseen attacks (22.8% EER even on codec-free
eval audio) and to real codecs (26-41%), and the dev-tuned fusion weight did not hold on eval. See
`new_plan.md` section 7.3d for the analysis and the next experiments.
The earlier ASVspoof2019 prototype (5.7% eval EER) used feature version 1 and an easier benchmark; it is not
reproducible with this code and not comparable to these numbers.

## Known limitations (read before relying on the risk levels)

1. **Risk thresholds are derived on dev, not proven on unseen attacks.** They are set so 1% / 10% of dev
   bonafide clips score at or above block / verify (`risk.block_fpr`, `risk.verify_fpr`). Check the report's
   per-attack and per-codec numbers on eval before trusting them; adjust the FPR targets to your cost of a
   missed fake versus a blocked genuine call.
2. **Codec coverage is partial** (see above), and ASV5 audio is crowdsourced speech, not live VoIP: expect
   a gap on real calls (packet loss, DTX, echo). Collect labelled production audio when possible.
3. **Eval scoring is a stratified subset by default** (60k of 680k clips; the SVM runs at ~35 clips/s).
   Run `--eval-utts 0` once for the final number.
4. **Kafka, Docker, Kubernetes and Grafana files are untested scaffolds** (no broker or cluster was available).
   The chunk-routing logic (`streaming/processor.py`) is unit-tested; the Kafka glue is not.
5. The SVM is recomputed on each verdict (about 41 ms per 10 s buffer). Fine for now; cache or share the
   STFT across prefixes if you need many concurrent calls per GPU.
6. ViT is not integrated. If tried, benchmark it against the RCNN on the same eval set and per-attack table.

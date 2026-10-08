# audiodf: real-time deepfake voice detection (SVM + RCNN ensemble)

Production-shaped version of the notebook prototype. Independent branches (SVM, RCNN, WavLM), each with its own
audio processing, fused into one P(fake) and mapped to Allow / Verify / Block. Trained on ASVspoof5 + ASVspoof2019.

**Current default: WavLM alone** (`ensemble.branches = [wavlm]`). The served model in `artifacts/` is **run 7's WavLM**
(trained with noise, room echo, bursty packet loss and G.711; swapped in 2026-10-07; run 4's WavLM, served before, is kept in
`artifacts_run4_wavlm_only/`). The SVM and RCNN branches described below
are still in the code (`run.py --branches svm rcnn wavlm`) but add nothing on eval once WavLM is present; the diagram
and branch table show all three. Results and the telephony check are at the end of this file.

```
audio chunks (0.5 s, PCM16)
   |  Kafka topic, keyed by call_id        or      FastAPI  POST /predict | WS /stream/{call_id}
   v
CallSession (per call): skip silence before speech -> rolling 10 s buffer
   |--> SVM branch : the buffer so far (grows 2 -> 10 s) -> ONE 318-d vector (MFCC+LFCC+MGDCC) -> SVM
   '--> RCNN branch: every completed 2 s window (1 s hop) -> Log-Mel 64x200 -> CNN -> BiLSTM
                     (scores averaged over the windows inside the last 10 s)
   v
fuse: w * P_svm + (1-w) * P_rcnn  ->  risk engine (allow / verify / escalate thresholds)
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

The default training pool is **ASV5 train + dev and ASVspoof2019 train + dev** (22 attack systems).
Three ASV5 dev attacks (A10, A12, A15) and every clip of 25% of ASV5 dev's bonafide speakers are removed
from the pool and form the **tuning set**: everything tuned (RCNN epoch, SVM calibration, fusion weight,
risk thresholds) is chosen on attacks and voices the models never trained on. **ASV5 eval** (A17-A32, 11 codec
conditions) and ASVspoof2019 eval (cross-dataset) are scored once at the end and never trained or tuned on.
Run 1 instead trained on ASV5 train and tuned on ASV5 dev (`--train-splits asv5:train --holdout-attacks`).
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
export AUDIODF_DATA=../dataset19/LA/LA         # optional: ASVspoof2019 LA, used as a cross-dataset test

python run.py                                # ONE command: audit -> index -> SVM -> RCNN -> tune -> test -> artifacts/
python run.py --config configs/run6.yaml --eval-utts 30000   # run 6 recipe: needs dataset_noise/ (MUSAN noise, RIRS_NOISES, DEMAND subset)
python run.py --no-impairments               # the runs 3-5 recipe: no noise, echo or packet loss
python run.py --epochs 12 --eval-utts 0      # flags (--eval-utts 0 = whole eval split, hours; --limit 800 = smoke test)
python run.py --train-splits asv5:train --holdout-attacks   # run-1 setup: ASV5 train only, tune on ASV5 dev

# `python -m audiodf <command>` is the multi-command CLI (needs a subcommand); run.py is the one-click training.
python -m audiodf audit --dataset asv5       # integrity audit, exit 1 on hard errors
python -m audiodf prepare                    # index clips (decode once; cached)
python -m audiodf evaluate --dataset asv5 --split eval   # score saved artifacts
# options: --bundle DIR (another model folder), --branches wavlm (one branch of a fused bundle), --tag T, --save-scores
python -m audiodf prepare21                  # unpack the ASVspoof 2021 LA eval parquet download (real phone channels)
python -m audiodf evaluate --dataset asv21 --eval-utts 30000 --save-scores
python -m audiodf render-calls               # simulated VoIP calls from clean ASV5 eval clips (noise + codec + packet loss)
python -m audiodf evaluate --dataset calls --eval-utts 0 --save-scores
python -m audiodf calibrate --scores S.csv --protocol P.tsv --tag T --repeat 20   # thresholds on speaker halves
python -m audiodf live-check --scores S.csv --bundle DIR --tag T   # stream through CallSession vs batched scores
python -m audiodf.evaluation.channel_report --protocol ../dataset21/ASVspoof2021.LA.eval.tsv --runs a=... b=...
python -m audiodf predict some.flac          # one file -> verdict JSON
python -m audiodf benchmark                  # per-stage latency
python -m audiodf serve --port 8000          # API: /predict, /stream/{id}, /health, /metrics
python -m audiodf consume                    # Kafka worker (needs a broker + confluent-kafka)
python -m audiodf produce call.wav --realtime
python -m pytest                             # 176 tests, no dataset or GPU needed
python -m audiodf explain call.wav --plot out.png   # verdict + SHAP regions + plain-language reasons (--provider groq|gemini|openrouter with AUDIODF_LLM_API_KEY)
```

Stream over a WebSocket: send binary frames of 16 kHz mono PCM16, receive a JSON verdict each time a
2 s window completes (from 2 s of speech on, then every 1 s); send the text frame `end` for the final verdict.
The fusion weight and risk thresholds in `artifacts/bundle.json` override the config at serving time.

## Results

**ASV5 eval** (same 30k clips, 16 unseen attacks, real codecs), EER at the 10 s decision:

| | Run 1: ASV5 train only | Run 2: pooled ASV5 + ASVspoof2019, held-out-attack tuning | Run 3: run 2 + real-codec augmentation | Run 4: run 3 + WavLM-Base+ branch | Run 5 (current `artifacts/`): WavLM alone + EnCodec augmentation |
|---|---|---|---|---|---|
| SVM | 33.6% | 31.1% | 29.6% | 29.6% (reused) | - |
| RCNN | 37.4% | 31.8% | 29.4% | 29.4% (reused) | - |
| WavLM | - | - | - | 5.4% | **5.5%** |
| **Fused / served** | **33.0%** | **31.6%** | **29.4%** (SVM weight 0) | **5.5%** (svm 0.10, rcnn 0.15, wavlm 0.75) | **5.5%** (WavLM alone) |
| Codec-free eval clips | 22.8% | 16.2% | 26.3% | 0.7% | 1.8% |
| EnCodec C04 / MP3+EnCodec C07 | - | - | 36.0% / 37.0% | 15.8% / 19.3% | 9.4% / 10.5% |
| Worst codec condition | - | - | 37.0% | 19.3% | 10.5% |

The model now served is WavLM-Base+ alone (run 5), 65 ms to the first verdict on a laptop GPU. Adding EnCodec to the
codec augmentation fixed the two neural-codec conditions but cost 0.1-2.3 points on every other condition, including
narrowband telephony (C08 8.3%), so run 4's WavLM is kept for comparison on real call audio. WavLM's pretraining data
includes the audiobook source of ASV5's real speech, and nothing has been tested on phone-call audio yet. Details:
`results/training_report.json` (run 5), `training_report_run{1,2,3,4}.json`, `new_plan.md` 7.3d-7.3l, `audit.md` phases 12-14.

**Telephony check** (ASVspoof 2021 LA eval: the 2019 LA eval utterances sent over real VoIP and public-phone-network
channels; 29,994 clips, WavLM alone, EER at 10 s):

| | Run 4's WavLM | Run 5's WavLM |
|---|---|---|
| All clips | **8.8%** | 10.0% |
| No channel / a-law / mu-law / G.722 / Opus | 6.0 / 8.6 / 8.4 / 7.8 / 7.5% | 7.1 / 9.6 / 8.6 / 8.4 / 8.6% |
| GSM / public phone network (Spain) | 10.6% / 11.5% | 13.0% / 12.6% |

Run 4's WavLM is the better call model (better in every channel). Run 5's stored verify/block thresholds were too loose on
phone audio (designed for 10% / 1% of genuine clips flagged; measured 14.6% / 6.2%), so run 4's bundle has thresholds set on
phone-channel data instead.

**Simulated live calls** (9,600 calls: clean ASV5 eval clips -> background noise -> real VoIP codec -> packet loss; EER at
10 s, WavLM alone): **run 4 4.96% vs run 5 7.45%** (difference 2.49 points, 95% interval 2.01-2.83), so run 4's WavLM is the
model now in `artifacts/`. **Known weak spots:** packet loss (genuine calls flagged at the verify level: 6% with no loss,
42% at 5% loss in the simulation), white noise (EER 2.9% -> 7.5%), the hardest attacks (A28 15.6%), and the block level
(catches 39% of fakes at a 1% false-alarm budget). Details: `audit.md` phases 15-16, `new_plan.md` 7.3m-7.3n.

**Runs 6 and 7 (noise, room echo, bursty packet loss, G.711 in the training audio; run 6 with 7% EnCodec copies, run 7 without;
`configs/run6.yaml`, `configs/run7.yaml`). Run 7 is the served model since 2026-10-07:**

| EER at 10 s, WavLM alone | run 4 (served until 2026-10-07) | run 6 | **run 7 (served)** |
|---|---|---|---|
| ASV5 eval (30k) | 5.4% | **4.5%** | 5.7% |
| ASVspoof 2019 eval (30k) / ASVspoof 2021 real phone channels (30k) | **4.9% / 8.8%** | 5.8% / 9.9% | 5.3% / 9.1% |
| Simulated calls: as trained (9,600) / babble (4,800) | 5.0% / 15.5% | 3.2% / 7.8% | **3.1% / 7.0%** |
| Simulated calls with held-out noise, real room echo and unseen loss (6,400) | 16.9% | 11.4% | **10.8%** |
| Genuine calls flagged at the verify level, 0% -> 5% packet loss | 6% -> 42% | 10% -> 14% | 12% -> 18% |

Run 7 gives up 0.3-0.5 points on clean-speech phone audio against run 4 for 2-8.5 points on noisy, babbling, echoey and lossy calls
(your decision; it missed the rule fixed before training by 0.18 / 0.03 points, `audit.md` phase 18). Thresholds were set on all 67
ASVspoof 2021 speakers (verify >= 0.0060); the high level was lowered to >= 0.90 on 2026-10-07 from a call-audio threshold study (banking
policy: verify kept strict because a missed fake costs far more than a verification; `audit.md` phase 19). The live serving path matches the batched scores (400 calls, p95
difference 0.0013) with a 47 ms median verdict time. A classical denoising front end was tested and rejected: it made every call set worse.

**Run 8 (more echo, measured rooms, low-SNR babble; `configs/run8.yaml`; not served):** ASV5 5.73%, ASVspoof 2019 5.32%, ASVspoof 2021 9.42%,
babble 6.29% (run 7: 7.00%), held-out calls v2 10.47% (10.78%). It missed all five targets fixed before training, so run 7 stays served
(`audit.md` phase 20). Training used the MIT Impulse Response Survey (270 measured rooms, CC-BY 4.0): J. Traer and J. H. McDermott,
"Statistics of natural reverberation enable perceptual separation of sound and space", PNAS 2016.
The earlier ASVspoof2019 prototype (5.7% eval EER) used feature version 1 and an easier benchmark; it is not
reproducible with this code and not comparable to these numbers.

**Actions (since 2026-10-09):** `allow`; `verify` (a cheap step-up check such as an OTP); `escalate` (the strongest check, e.g. an agent
callback on the number on file). The high level never rejects a call by itself: on real phone lines, if 1 call in 1,000 were a deepfake, about 96%
of calls at that level would be genuine customers (`audit.md` phase 26). `escalate` replaces the old `block` action.

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

# Audit log: Real-Time Deepfake Voice Detection

Running record of what was done, what was found, what turned out wrong, and what is open.
Last updated: 2026-10-06 (after the telephony check). Companion docs: `new_plan.md` (plan and analysis), `src/README.md` (usage).

## 1. Current state at a glance

| Item | State |
|---|---|
| Codebase | `src/audiodf/` (35 modules), 108 passing tests. Pushed through `38fe200`; CI green. The telephony-test code (phase 15) is not yet pushed. |
| Trained model | `artifacts/` = run 5 (**WavLM-Base+ alone**, EnCodec in the codec augmentation). Copy in `artifacts_run5_encodec/`; run 4 (3-branch) in `artifacts_run4_wavlm/`; runs 1-3 kept. |
| Previous model | `artifacts_prev_asv2019/` (ASV2019 prototype, feature v1, cannot load in current code). Kept, git-ignored. |
| Best honest number | ASV5 eval EER **5.51%**, worst codec condition 10.5% (run 5, WavLM alone; run 4 WavLM alone 5.43% with a worst condition of 18.5%; run 3 29.4%). ASV2019 eval 5.82% (in-domain). |
| Deployment | FastAPI tested for real (65 ms to first verdict on GPU, WavLM alone); **Kafka, Docker runtime, Kubernetes, Grafana untested**. No real phone-call audio. |
| Disk | Data: `dataset/LA` 7.1 GB, `dataset5/` ~75 GB (train, dev, 2 of 10 eval tars); codec renders in the cache (catalogue 1 ~35 GB, catalogue 2 adds ~7 GB, the rest hard links). |
| Telephony check | ASVspoof 2021 LA eval (real VoIP/PSTN channels), 30k clips, WavLM alone: **run 4 8.81%, run 5 9.96%**; run 4 better in every channel. Stored thresholds are too loose for phone audio (phase 15). |
| Next | Decide the call model (run 4's WavLM recommended) and re-calibrate its thresholds on speaker-disjoint phone-channel data (section 9). |

## 2. Timeline

### Phase 1: understand the prototype
- Parsed the 10-slide pptx (workflow, Kafka, hybrid-model and MLOps diagrams), README and notebook.
  The notebook was an offline SVM + CNN-LSTM prototype on ASVspoof2019 LA; everything else was architecture only.
- Wrote `new_plan.md` (phased replan).

### Phase 2: re-run the prototype honestly (ASVspoof2019 LA)
- Kaggle download via token auth was too slow; you supplied the dataset locally (`dataset/`).
- `experiments/segmented_baseline.py` (clean re-implementation, 2 s windows, CNN-BiLSTM):

| EER, utterance level | dev (seen attacks) | eval (13 unseen attacks) |
|---|---|---|
| SVM on whole clips | 0.24% | 8.58% |
| SVM on 2 s segments | 0.36% | 16.86% |
| RCNN (CNN-BiLSTM), 5 epochs | 2.98% | 15.24% |
| Ensemble 0.7/0.3 | 0.14% | **5.67%** |
| Weights tuned on dev / LR stacking | 0.06% | 10.25% / 12.30% |
| RCNN 20 epochs (best epoch 6) / ensemble | 2.11% / 0.113% | 12.76% / 5.64% |

- Findings: slide metrics (99.59% accuracy, 0.43% EER) were dev-only, where attacks match training; learned fusion
  tuned on dev generalised worse than a fixed weight; the notebook read the attack ID from the wrong column.

### Phase 3: production codebase v1 (`src/`)
- Separate feature paths per branch (SVM: 318-d vector per buffer; RCNN: Log-Mel per 2 s window), `CallSession`
  (rolling 10 s buffer, 2 s windows / 1 s hop), fusion, risk engine, FastAPI (`/predict`, WebSocket, `/metrics`),
  Kafka producer/consumer, Dockerfile, compose, k8s, CI workflow, `run.py`, CLI.
- Verified: new extractors matched experiment features (SVM ~1e-8); trained through the new code gave identical metrics;
  1,500 eval files streamed in 0.5 s chunks gave 5.66% EER (offline 5.67%); real HTTP server tested; latency ~22 ms/verdict on GPU.
- Committed by you as `a839261`.

### Phase 4: dataset research and download
- "ASVspoof 2022" does not exist; options were ASVspoof 2021 (no new train data) and **ASVspoof5** (2024, chosen).
- Zenodo ran at ~300 KB/s (about 2 days for train+dev); the Hugging Face mirror ran at ~12 MB/s. Used HF.
- Downloaded ASV5 train (37.5 GB) and dev (20 GB); later 2 of 10 eval tars (17 GB, 136,376 clips, 20%), one tar at a
  time, deleting each after extraction. Subset verified representative (spoof share 79.4% vs 79.6%; all 16 attacks, 12 codec
  conditions, 737 speakers; every share within 0.5 pp).
- Freed 40.3 GB: deleted `dataset/PA` (replay attacks, a different problem, never read) and the redundant kagglehub archive.

### Phase 5: data integrity system (`audiodf audit|prepare`)
Hard checks (training refuses to start): missing audio, duplicate IDs, speakers shared across splits,
bonafide/attack inconsistency, wrong format. Findings and actions:

| Finding | Action |
|---|---|
| ASV5 train bonafide clips have 4-10x more edge silence than spoof (full data: 0.33/0.40 s vs 0.03/0.06 s); attacks A07/A08 look like bonafide on this | VAD trim (-45 dBFS) of every training clip; `CallSession` skips pre-speech audio with the same rule |
| ASV2019 has the same silence shortcut, up to 15x | one reason ASV2019 was not used for training in run 1 |
| 72% of ASV5 dev attack A11 clips contain exact digital silence | exact zeros replaced by +-1 LSB noise at every audio entry point |
| SVM log features exploded on digital zeros (log 1e-10) | magnitude floor 1e-6; SVM features version 2 |
| Train clips long (bonafide 14 s vs spoof 11 s), dev/eval ~7 s | training windows limited to the first 10 s of speech (the live decision horizon) |
| Train/dev have no codec audio; eval is ~75% codec-processed, equally for both classes | label-blind codec augmentation (Opus, MP3, Vorbis, G.711, narrowband) |

Also built: growing-buffer SVM snapshots (2/4/6/8/10 s), RCNN windows read straight from FLAC, compact index (one
decode per clip), feature-version stamping (bundle refuses to load on mismatch), multi-dataset index with dataset-prefixed
attack IDs, partial-test-split support, per-attack / per-codec / time-to-decision evaluation.

### Phase 6: training run 1 (ASV5 train only)
Config: SVM on 19,994 clips (98,527 snapshot vectors, codec aug 0.5), RCNN 10 epochs x ~22 min, tuned on 12,000 codec-augmented dev clips.

| EER at 10 s | dev (tuned on) | ASV5 eval (30k clips) | ASV2019 eval (15k) |
|---|---|---|---|
| SVM | 21.8% | 33.6% | 41.4% |
| RCNN (epoch 6) | 17.6% | 37.4% | 33.3% |
| Fused (SVM weight 0.2) | 15.9% | **33.0%** | 37.8% |

- Operating point stored in the bundle: verify >= 0.169, block >= 0.372 (dev: 1%/10% of bonafide flagged; catches 67%/82% of spoofs).
- ASV5 eval: codec-free audio 22.8% EER; real codec conditions 26-41% (worst: narrowband Opus 40.9%, device channels 39%).
  Per attack: A29 4.7%, A21 7.8%, A24 10.1%, A17 11.0% ... A19 59%, A20 58%, A32 47%. Dev attack A12 was inverted (EER 69%).
- RCNN stopped improving on unseen attacks after ~3 epochs while training loss fell 4x (overfits the 8 training attacks).

### Phase 7: run 2 setup (approved 2026-10-03)
- Why run 1 failed on ASV5 eval (33.0%): unseen attacks ~23 points (codec-free eval audio alone 22.8%), real codecs
  ~10 more, and tuning on a set unlike eval (dev picked SVM weight 0.2; on eval the SVM is the stronger branch).
- Run 2: pool ASV5 train+dev + ASV2019 train+dev (22 attack systems). ASV5 dev attacks A10, A12, A15 and every clip of
  25% of ASV5 dev bonafide speakers are held out as the tuning set, leaving 19 attack systems for training.
  1 RCNN window per clip per epoch keeps the work near run 1. Tests unchanged: ASV5 eval (same 30k subset) + ASV2019 eval.
- Decision rule agreed in advance: run 2 replaces run 1 if its fused EER on the same ASV5 eval subset is lower;
  if it stays above ~25%, move to a pretrained speech front end.
- Code: training pool from `data.train_splits` (test splits refused), `holdout_split` (tuning attacks and voices
  never trained on), `SplitIndex.subset` + per-clip `source`, SVM snapshot cache keyed by clip IDs (subsets
  re-number clips), `run.py --train-splits / --holdout-attacks`. 72 tests.
- Codec check (in parallel): bundled ffmpeg 7.1 (`imageio-ffmpeg`) has AMR-NB, AMR-WB, Speex, AAC, Opus, MP3, SBC,
  G.722, G.726, GSM encoders, covering every eval codec family except the neural EnCodec. Round trips cost
  ~170-250 ms per 10 s clip, so realistic codec augmentation must be rendered offline (~20 min per 50k clips).
  AAC through a pipe failed; needs a temp file. Follow-up, not part of run 2.
- Run 2 progress before the crash (RCNN tuning EER on held-out attacks): epochs 1-6 = 29.0, 31.6, 20.6, 23.1, **20.2**, 23.0%.
  On the identical tuning subset, run 1's RCNN scores **34.0%** (A10 2.9%, A12 75.2% inverted, A15 16.5%):
  pooling more attack systems cut RCNN error on never-seen attacks by ~14 points.
- Run 2 crashed at epoch 7 (see section 4, item 15). Finished from the epoch-5 checkpoint (your choice): SVM refit from
  cached features, tuning, both tests (~1.3 h).

### Phase 8: run 2 results
| EER at 10 s | run 1 | run 2 |
|---|---|---|
| Held-out tuning set, RCNN | 34.0% | 20.2% |
| ASV5 eval fused (same 30k subset) | 33.0% | **31.6%** |
| ASV5 eval, share-weighted per-codec | 32.0% | 26.2% |
| ASV5 eval, codec-free clips | 22.8% | 16.2% |
| ASV2019 eval | 37.8% (cross-dataset) | 14.3% (in-domain now: ASV2019 train/dev were trained on) |

- Run 2 replaces run 1 by the agreed rule; still above the ~25% line. Every codec condition improved, but scores now
  shift with the codec: pooled EER is 5.4 points worse than per-codec (run 1: 0.9). Tuning picked SVM weight 0.05,
  yet on eval the SVM alone (31.1%) beats the fused score. Per attack: A19 -17, A20 -18, A26 -10; A18 +13 (50.8%).
- Operating point stored: verify >= 0.107, block >= 0.352 (tuning set: 10% / 1% of bonafide flagged; catches 73% / 56%
  of spoofs). Models: `artifacts/` (run 2), `artifacts_run1_asv5/` (run 1), `artifacts/rcnn_run2_epoch5.pt` (backup).

### Phase 9: run 3 setup (realistic codec augmentation)
- Your call: try realistic codecs first; if there is no significant gain on unseen attacks, move to WavLM.
  Rule set before the run: significant = ASV5 eval fused EER <= 28.6% (3 points below run 2's 31.6%).
- Built `data/ffmpeg_codecs.py`: 11 real codecs via ffmpeg 7.1 at ASV5 eval bitrates; codec delay measured per clip
  and removed (residual <= 8 samples in tests); copies rendered once to `~/.cache/audiodf/render_ff1/` (resumable);
  label-blind, reproducible selection from clip ID + seed. ffmpeg's built-in Speex decoder mis-decodes wideband
  (double length); the libspeex decoder is used. AAC decodes with the `aac` demuxer (the earlier failure used `adts`).
- Index tracks rendered clips; rendered clips get no simulated codec on top; SVM snapshot caches are keyed by the
  render set. 88 tests pass.

### Phase 10: run 3 results (2026-10-03/04, ~6.5 h)
- Rendering: 154,209 training clips (50.2%) and 8,967 of 11,998 tuning clips (74.7%) got real-codec copies, 151,573 new
  in 77 min (~31/s). SVM fit 33 min. RCNN tuning EER fell every epoch: 29.9 -> 12.9% (epoch 10 kept). Tuning set
  picked SVM weight 0.00 (RCNN only); verify >= 0.032, block >= 0.258 (catches 85% / 65% of tuning spoofs at
  10% / 1% bonafide flagged).
- ASV5 eval fused **29.4%** (run 2: 31.6%): **not significant** by the agreed rule (<= 28.6%). Next: WavLM.
- Codec score-shift gap 5.4 -> 1.0 points (target met), but codec-free eval clips 16.2 -> 26.3%, ASV2019 eval
  14.3 -> 16.0%; AUC 0.775 -> 0.765. Gains on 10 of 16 eval attacks; A18 and A30 worse and inverted (~65%).
- Models: `artifacts/` = run 3; run 2 in `artifacts_run2_pooled/`, run 1 in `artifacts_run1_asv5/`.
  Reports: `results/training_report.json` (run 3), `training_report_run2.json`, `training_report_run1.json`.

### Phase 11: run 4 setup, WavLM-Base+ as a third branch (2026-10-04)
- Your choices: third branch (SVM + RCNN + WavLM fused); fine-tune the top 4 of 12 transformer layers + learned
  layer mix; success = ASV5 eval fused EER <= 24.4% on the same 30k clips (5 points below run 3).
- Code: `models/wavlm.py` (torchaudio WavLM-Base+, lower 8 layers and CNN frozen and kept in eval mode, softmax mix of
  the 12 layer outputs, attentive statistics pooling, small head; checkpoint stores only the trained part, ~29M
  values, and refuses to load if any trained tensor is missing). `training/window_trainer.py`: one training loop
  for RCNN and WavLM (per-group learning rates: backbone 2e-5, head 1e-3). `WaveWindowDataset` feeds raw windows
  through the same reader and codec augmentation as the RCNN.
- Fusion is now over any set of branches: weights on a 0.05 simplex grid (231 points for 3 branches), ties within
  0.05 EER points broken toward equal weights. The bundle stores `branches` and `fusion_weights`; bundles from runs
  1-3 (`svm_weight`) still load. Serving runs only branches with non-zero weight; `Verdict` gains `wavlm_probability`.
- `run.py --branches`, `--wavlm-epochs`, `--svm-checkpoint / --rcnn-checkpoint / --wavlm-checkpoint` (reuse a
  trained branch). Run 4 reuses run 3's SVM (re-calibrated on the same tuning set) and RCNN, so only WavLM trains.
- 95 tests pass (WavLM tests build the architecture without downloading weights).
- Run 3 preserved in `artifacts_run3_codecs/` (hashes verified) and `results/training_report_run3.json`. The duplicate
  `artifacts/rcnn_run2_epoch5.pt` was removed after confirming it is byte-identical to `artifacts_run2_pooled/rcnn.pt`.
- Smoke test (400 clips per split, 1 WavLM epoch) ran every stage; serving benchmark 77 ms to first verdict on GPU.

### Phase 12: run 4 results (2026-10-04 02:08-09:17, ~7.1 h)
- WavLM tuning EER by epoch: 1.20, 0.92, 0.95, 0.60, 0.55, **0.41** (epoch 6 kept), 0.47, 0.43%; ~47-50 min per epoch
  including tuning-set scoring (estimate was 35 min).
- Fusion tuned on the held-out set: **svm 0.10, rcnn 0.15, wavlm 0.75** (the most balanced weighting within 0.05 EER
  points of the best). Tuning set EER: svm 26.97%, rcnn 12.95%, wavlm 0.41%, fused 0.34%. Operating point: verify
  >= 0.0835, block >= 0.1784 (tuning set: 10% / 1% of bonafide flagged; catches 99.95% / 99.8% of spoofs).

| EER at 10 s, same test subsets | run 3 | run 4 |
|---|---|---|
| ASV5 eval (30k clips), fused | 29.36% (AUC 0.765) | **5.51%** (AUC 0.979) |
| ASV5 eval, WavLM alone | - | 5.43% (AUC 0.985) |
| ASV5 eval, codec-free clips | 26.30% | 0.72% |
| ASV2019 eval (30k), fused | 15.99% | 4.85% |
| ASV5 eval fused, by seconds of speech | - | 2 s 8.02%, 4 s 6.23%, 6 s 5.62%, 10 s 5.51% |

- **Success criterion met** (<= 24.4%), by 19 points. SVM and RCNN scores are identical to run 3 (29.58% / 29.36%),
  confirming the comparison uses the same clips and scoring.
- Every eval attack improved; run 3's inverted attacks are fixed (A18 65.7 -> 6.0%, A30 64.8 -> 10.8%, A32 51.9 ->
  3.9%, A31 48.7 -> 8.8%). Worst now: A30 10.8%, A28 10.7%, A31 8.8%.
- Every codec condition improved. Two remain weak, both the **neural codec EnCodec**, the one family ffmpeg cannot
  render for augmentation: C04 EnCodec 36.0 -> 15.8%, C07 MP3+EnCodec 37.0 -> 19.3%. Next worst: C08 narrowband Opus
  6.3%, C10 narrowband Speex 5.9%. All others 1.5-3.7%.
- On eval, fusion adds nothing over WavLM alone (5.51% vs 5.43%, AUC 0.979 vs 0.985): the SVM and RCNN are ~29%
  branches. The weights were chosen on the tuning set as agreed, so they are not changed after seeing eval.
- Caveat: WavLM-Base+ was pretrained (without labels) on 94k h including Libri-Light, which is LibriVox audiobooks;
  ASV5 bonafide speech is also LibriVox (MLS). In ASVspoof5 terms this is an open-condition result, in line with
  published SSL-front-end systems; it says less about live phone audio, which remains untested.
- Serving with the run 4 bundle: 78 ms to first verdict on GPU (WavLM 48 ms per window, SVM features 41 ms, RCNN 7 ms).
- Run 4 bundle copied to `artifacts_run4_wavlm/` (hashes verified); report `results/training_report_run4.json`
  (also `training_report.json`), log `results/train_run4.log`. 97 tests pass after the CI fix.
- CI failure on `573fe30` fixed in `bef9ad7` (section 4, item 19); run 4 results pushed as `756e79e`, CI green.

### Phase 13: your decisions after run 4, and run 5 setup (2026-10-04)
- Your calls: **WavLM alone** from now on; **EnCodec treated like the ffmpeg codecs**, retrain WavLM only; then real
  call audio; GitHub cleanup only at the very end (push as we go until then). Note: the WavLM-alone choice was made
  after seeing eval numbers; the difference there was small (5.43% vs 5.51%), and latency (78 -> 65 ms) supports it.
- Remaining ML work listed in `new_plan.md` 7.3k (9 items: out-of-domain test, telephony robustness, calibration on
  harder data, second seed, probability calibration, early decisions, serving cost, full eval).
- Code: codec catalogue 2 = 11 ffmpeg codecs + `encodec` (1.5-24 kbps, like eval C04) + `mp3_encodec` (MP3 then
  EnCodec, 25 pairings, like C07), EnCodec 24 kHz via Meta's `encodec` package, batched on the GPU (batch 8). 18% of
  copies get a neural codec (eval: 18.7% of codec-processed clips), decided by hash bytes catalogue 1 never used, so
  every other clip keeps exactly its catalogue-1 codec and bitrate and its copy is hard-linked, not re-rendered.
  Batched EnCodec output is not sample-identical across batch compositions (GPU arithmetic changes with batch size and
  the quantiser can pick other codes; same quality, SNR within 0.1 dB); documented and tested.
- Default config: `ensemble.branches = (wavlm,)`. SVM/RCNN code kept (`--branches svm rcnn wavlm`). 103 tests.
- Decision rule set before the run (baseline run 4 WavLM alone: 5.43% overall, C04 14.56%, C07 18.51%): adopt run 5
  if C04 <= 11.6% and C07 <= 15.5% and overall <= 5.73%.
- Smoke test passed (616 + 100 copies linked, 143 EnCodec rendered, no GPU memory clash, 65 ms first verdict).

### Phase 14: run 5 results (2026-10-04 11:06-19:00, ~7.9 h)
- Rendering: 126,852 training + 7,300 tuning copies reused (hard links), 27,357 + 1,667 EnCodec copies rendered in
  54 min (~8.5/s; MP3 stage and file reads keep it below the 15.8/s GPU-only rate). Same 154,209 training / 8,967
  tuning clips with a copy as runs 3-4; 17.7% of them now EnCodec.
- WavLM tuning EER by epoch: 1.99, 1.28, 1.07, **0.81** (epoch 4 kept), 1.10, 1.03, 0.83, 0.95% (tuning set now
  includes EnCodec copies, so not comparable with run 4's 0.41%). Operating point: verify >= 0.0155, block >= 0.2250.

| WavLM alone, EER at 10 s, same subsets | run 4 | run 5 | change |
|---|---|---|---|
| ASV5 eval overall (30k) | 5.43% (AUC 0.9847) | **5.51%** (AUC 0.9904) | +0.08 |
| C04 EnCodec | 14.56% | **9.40%** | -5.16 |
| C07 MP3 + EnCodec | 18.51% | **10.54%** | -7.97 |
| Every other codec condition (C01-C03, C05, C06, C08-C11) | 1.45-6.07% | 2.15-8.31% | +0.1 to +2.3, all worse |
| Codec-free eval clips | 1.01% | 1.76% | +0.75 |
| ASV2019 eval (30k, no codecs) | 4.85% | 5.82% | +0.97 |
| ASV5 eval at 2 s of speech | 6.88% | 6.79% | -0.09 |

- **Decision rule met on all three parts; run 5 replaces run 4.** But it is a trade, not a free gain: the two EnCodec
  conditions improved by 5-8 points, while every other codec condition, clean audio and ASV2019 got 0.1-2.3 points
  worse. Overall EER is flat and AUC is better; the worst condition is far better (10.5% vs 18.5%).
- Per attack: 12 of 16 eval attacks improved (A17 2.4 -> 0.6%, A19 2.6 -> 0.7%); the four hardest got 0.5-1.7 points
  worse (A28 11.1%, A30 10.0%, A31 9.5%, A18 6.6%).
- Telephony matters for calls: narrowband Opus (C08) 6.07 -> 8.31% and AMR-NB (C09) 3.24 -> 4.64% regressed. The real
  call audio check should score **both** run 4's and run 5's WavLM (both bundles kept), since calls carry classical
  telephony codecs far more often than EnCodec.
- Possible causes of the regressions (not yet separated): epoch 4 was kept (run 4 kept epoch 6) because the tuning set
  now weighs EnCodec; and 18% of copies moved from classical codecs to EnCodec. Run-to-run noise is unmeasured
  (per-condition subsets are ~2,100 clips, roughly +-1 point), but 10 of 10 non-EnCodec conditions moving the same way
  points to a real shift.
- Saved: `artifacts/` and `artifacts_run5_encodec/` (hashes verified), `results/training_report_run5.json` (also
  `training_report.json`), `results/train_run5.log`.
- `.gitignore`: you replaced the per-run `artifacts_run*` lines with one `artifacts_*/` pattern during the run; kept as
  is (a duplicate line I appended was removed).
- Pushed as `34467a8` (your yes); CI green: Linux tests, including the EnCodec tests (CPU, model downloaded in CI), and
  the Docker build. Free disk afterwards: 130 GB.
- Paused here at your request (2026-10-04 evening); resumed 2026-10-06 with section 9 step 1 (phase 15).

### Phase 15: telephony check on ASVspoof 2021 LA eval (2026-10-06)
- **Why this set, not ASV2019 PA** (your question): PA is replay (a genuine voice played through a loudspeaker; its
  "spoof" label is not synthetic speech) and simulates rooms, not phone channels; we had also deleted it on 2026-10-03.
  ASVspoof 2021 LA eval is the organisers' own version of "play it through a real phone/VoIP channel": the 2019 LA eval
  utterances transmitted over a real Asterisk VoIP exchange (some routed France -> Italy / Singapore) and the public
  phone network in Spain, with codecs a-law, mu-law, GSM (8 kHz), G.722, Opus (16 kHz) and an untouched reference.
- **Download:** Zenodo ran at 320 KB/s (~6.7 h); the Hugging Face packaging `SpeechAntiSpoofingBenchmarks/
  ASVspoof2021_LA` ran at 16 MB/s (8 min, 24 parquet files, 7.6 GB). Verified: 24 files, 181,566 rows, 18,452 genuine +
  163,114 fake (the published counts). Its README says the audio was decoded from the official FLACs (which libsndfile
  often cannot read) and re-encoded as clean 16 kHz FLAC with the samples unchanged; **I did not verify that against the
  official files**. Unpacked to `dataset21/` (181,566 FLAC + `ASVspoof2021.LA.eval.tsv` with codec, transmission route,
  phase and trim per clip; git-ignored via `dataset*/`). Indexing (decode + VAD) took 496 s, no clip without speech.
- **Code:** `data/asv21.py` (unpack, resumable, checks label vs attack), `asv21` registered in `data/protocol.py`
  (test-only: training refuses it), `paths.asv21_root`, `audiodf prepare21`, `evaluate --dataset asv21 --bundle DIR
  --branches wavlm --tag T --save-scores`, `operating_point_report` (genuine flagged / fakes caught at the stored
  thresholds, per codec), `write_scores`, and `evaluation/channel_report.py` (breakdown by codec, route, phase, trim,
  attack; runs side by side; flagged/caught at the single global threshold). `pyarrow` added to the dev requirements.
  108 tests (5 new), also green with GSM hidden.
- **Test set:** the same 29,994 clips for both models (10,000 genuine + 19,994 fake, 13 attacks A07-A19 x 1,538), WavLM
  alone, EER at 10 s of speech. Run 4's WavLM was taken out of its 3-branch bundle with `--branches wavlm`.

| EER at 10 s, WavLM alone, same clips | run 4 | run 5 |
|---|---|---|
| **All clips** (AUC 0.969 / 0.961) | **8.81%** | 9.96% |
| No channel (reference) | 5.98% | 7.11% |
| a-law / mu-law | 8.60% / 8.36% | 9.64% / 8.55% |
| G.722 / Opus | 7.82% / 7.48% | 8.41% / 8.57% |
| GSM | 10.64% | 13.03% |
| Public phone network (Madrid, PSTN) | 11.49% | 12.57% |
| Routes: local / Italy / Singapore | 9.07 / 8.75 / 7.75% | 10.38 / 9.71 / 8.75% |
| Silence-trimmed "hidden" clips (2,833) | 13.14% | 14.70% |
| After only 2 s of speech | 8.91% | 10.02% |

- **Run 4's WavLM is better than run 5's on real telephony in every codec (7 of 7), every route (5 of 5) and 11 of 13
  attacks** (the other two within 0.35 points), including the untouched reference (5.98 vs 7.11%). This is the same
  direction as on ASV5 eval (run 5 lost 0.1-2.3 points on non-EnCodec conditions): the EnCodec augmentation buys
  nothing on phone channels and costs about one point. Same clips for both, so the comparison is paired; the
  breakdowns overlap, so they are not independent evidence, and seed noise is still unmeasured.
- The phone channel costs about 2-5.5 points of EER over the reference (VoIP codecs 7.5-8.6%, GSM 10.6%, public phone
  network 11.5% for run 4). Attacks matter more than channels: A10 (24% EER, 44% caught) and A11 (15%) are the
  hardest, 3-8x worse than the easy ones (A09 2.2%, A17 2.4%, A19 3.1%). A16 and A19 repeat the systems of 2019's A04 and
  A06 (as I recall from the 2019 paper, not re-checked), which fits their low error.
- **The stored thresholds are too loose for phone audio** (run 5, whose thresholds are for WavLM alone): at the verify
  threshold (designed for 10% of genuine flagged) 14.6% of genuine clips were flagged and 93.3% of fakes caught; at the
  block threshold (designed for 1%) 6.2% of genuine clips were blocked and 84.8% of fakes caught. GSM: 26% / 12% of
  genuine clips flagged; the public phone network: only 87% / 73% of fakes caught. This confirms limitation "the tuning
  set is far easier than eval". Run 4's WavLM has no valid thresholds yet (its bundle's belong to the 3-branch fusion).
- Scores shift with the channel at one global threshold: genuine clips flagged 5.5-6.2% with no channel vs 15-19% on GSM.
- Silence-trimmed clips (hidden phase, speech only) are harder (13-14.7% EER; ~29% of genuine flagged at the global
  threshold): relevant if a call pipeline cuts audio down to speech only.
- Caveat: these are the 2019 LA eval utterances (same speakers and attack families) sent through real channels: a real
  telephony test, but not a new-speaker or new-attack test, and not live call audio (no noise, echo, packet loss
  concealment, or real phones).
- Files: `results/evaluate_asv21_eval_run{5,4_wavlm}.json`, per-clip `..._scores.csv` (2.9 MB each),
  `results/channel_report_asv21.json`, logs `eval_asv21_run{5,4}.log`.

## 3. Decisions

- **Train on ASV5 only, drop ASV2019 (run 1).** Chosen because ASV5 dev gives attack-disjoint tuning and ASV2019 has a silence
  shortcut. The cross-dataset result (37.8%) now weakens this: mixed training is coded and tested but not run.
- **Codec augmentation is label-blind** (augmenter takes no label), so it cannot create a "codec means real/fake" shortcut.
- **Fusion weight tie-break:** among weights within 0.05 EER points of the best, pick the most balanced one
  (closest to 0.5 with two branches, to equal weights with three).
- **Run 4 reused run 3's SVM and RCNN** so the only change was the WavLM branch (identical pool, renders, tuning set, tests).
- **Rendering refuses an incomplete ffmpeg build** rather than skipping codecs (codec choice is fixed per clip ID).
- **WavLM alone** from run 5 on (your call): same eval EER as the 3-branch fusion, 13 ms less per verdict, one model.
- **EnCodec added without disturbing earlier picks**: catalogue 2 keeps every catalogue-1 assignment that stays
  classical, so copies are reused and runs 4 and 5 differ only in the EnCodec share.
- **ASVspoof 2021 LA eval is test-only**, and no choice (checkpoint, threshold) is to be tuned on the 30k scored clips
  and then reported on them: threshold calibration uses a speaker-disjoint split of them (see section 9).
- **Decision rules are fixed before each run** and applied as written (run 5 adopted although it trades other
  conditions for EnCodec; the trade is reported, not hidden).
- **Risk thresholds from bonafide quantiles** (1% block, 10% verify) instead of the slide's fixed 0.8/0.5.
- **Bundle carries its operating point** (fusion weight, thresholds) and overrides the config at serving time.
- **Eval scored on a stratified subset** (SVM runs ~35-80 clips/s), not the whole split.

## 4. Mistakes and corrections (mine unless noted)

1. Told you an ASV5 "codec perfectly predicts spoof" shortcut existed. Wrong: I read `ATTACK_TAG` (AC1-3) as the codec. The real `CODEC`
   column is empty in train/dev and balanced across classes in eval. Corrected; the real issue is a train/test codec gap.
2. Said the shortcut finding was in `new_plan.md` before writing it; it was added later.
3. Stated 64 passing tests when it was 58 at that point.
4. Predicted fusion would lean toward the SVM after a 500-clip smoke test; at full scale the SVM was weaker (weight 0.2). On eval the SVM was the stronger one.
5. Hypothesis that the models rely on loudness/noise-floor/bandwidth cues (from A12's acoustic profile): tested and refuted (those 8 statistics alone gave 40% dev EER and a different per-attack pattern).
6. Time estimates were off: indexing ran ~4x faster than estimated, but RCNN epochs took ~22 min (not 10) and the SVM fit 15 min (not 3-8).
7. `.gitignore` append glued onto the previous line; fixed.
8. Download scripts used `bc`, which is not installed, so their free-disk guard never worked (disk was ample both times).
9. A scripted edit to `pipeline.py` silently matched nothing; a test caught it and it was redone.
10. The first baseline eval crashed because I edited `SplitIndex` while its DataLoader workers were spawning; re-run after the code was stable.
11. A leftover API server and a stuck smoke test were left running at one point; both were stopped.
12. Proposed run 2 as "about 30 attack systems"; the real count is 22 (19 trained, 3 held out).
13. Scripted edits turned `\n` inside strings into real line breaks twice (a test file and the snapshot cache key);
    both caught before use. Strings containing `\n` are now edited with the editor, not scripts.
14. Was asked to delete the 8.6 GB cache a second time after already deleting it; confirmed it was gone instead of re-running.
15. **Crashed run 2** (2026-10-03 15:47, end of epoch 7, `MemoryError`): I ran a run-1 comparison script with 4 extra data
    workers while run 2 was spawning its per-epoch evaluation workers, on a machine with ~3-5 GB free RAM. I had called it
    "light load"; memory was the binding constraint. Lost: epochs 7-10 and the in-memory SVM. Kept: best RCNN checkpoint
    (epoch 5, tuning EER 20.2%) as `artifacts/rcnn_run2_epoch5.pt`; SVM snapshot features are cached on disk.
    Fixes: training data workers are no longer kept alive during evaluation (halves peak worker count);
    `run.py --rcnn-checkpoint` reuses a trained RCNN and runs every other stage. Rule: never start a second data-loading
    job while a training run is active.
16. My log watcher expired at the moment of the crash and I re-armed it to read only new lines, so the traceback went
    unnoticed for ~30 min. Rule: when re-arming, check the log's recent lines too.
17. The render disk guard used GB where it meant MB per copy (estimated 604 GB for 2,745 copies; real ~0.6 GB). It
    refused to run, so nothing was harmed; fixed, with a test that checks the estimate at run-3 scale (~33 GB).
18. Before launching run 3, the command meant to set run 2's model aside was blocked by a safety check and did not run;
    I launched anyway without reading its result. Caught within minutes, before run 3 wrote anything: run 2 was copied
    to `artifacts_run2_pooled/` (verified: same files, report shows 31.58%). Rule: confirm each prep step's output
    before launching a long run.
19. **CI failed on push `573fe30`** (you spotted it): I checked that the push landed but not the GitHub Actions result.
    Cause: CI runs on Linux, where imageio-ffmpeg ships ffmpeg 7.0.2 **without the libgsm encoder** (Windows ships 7.1
    with it), so the GSM codec tests failed. The CI log needs admin rights; the cause was confirmed by inspecting the
    Linux binary (every other catalogue codec is present). Fix: `missing_codecs()` probes the ffmpeg build;
    rendering refuses to start if a codec it must encode is missing (skipping it would silently change which copies
    exist, since codec choice is fixed by clip ID); tests skip only what the build lacks; pipeline tests render only
    with the full catalogue. Verified locally in both modes (real build, and one with libgsm hidden). Rule: after a
    push, check the CI result.
20. Small slips while building the telephony test (2026-10-06), all caught before they mattered: I estimated the
    unpacking of 181,566 files at about an hour from an early file count (it took ~6 min); a `du` over those files in the
    OneDrive folder blocked for minutes; two of my own test assertions were loose (a stray `or True`, an `or` that made a
    check vacuous) and one expectation was numerically wrong (3% of simulated genuine scores fall below 0.1). Fixed
    before running or on the first run. Rule: write each assertion so it can fail; do not size jobs from early partial counts.

## 5. Known limitations (current)

- Codec trade-off (run 5): EnCodec conditions improved to 9.4% / 10.5%, but the other conditions, clean audio and
  ASV2019 lost 0.1-2.3 points; narrowband telephony codecs (C08 8.3%, C10 6.2%) are now the weakest after EnCodec.
- Run-to-run (seed) noise is unmeasured, so differences under ~1 point per condition are uncertain.
- The tuning set is far easier than eval (run 4: 0.34% vs 5.51%; run 3: 12.9% vs 29.4%); it ranks candidates but its
  absolute numbers and thresholds do not transfer. **Measured on phone audio (phase 15): thresholds designed for 10% / 1%
  of genuine clips flagged flag 14.6% / 6.2%.** Re-calibrate before any user-facing use.
- ASV5 clips are read audiobook speech, not live VoIP audio (packet loss, DTX, echo untested). WavLM's pretraining
  includes the same audiobook source (open condition).
- Only 2 of 10 ASV5 eval tars (20%, verified representative); the 30k-clip subset is used for every comparison.
- WavLM needs a GPU to serve many calls (48 ms per window on GPU, ~234 ms on CPU). SVM features cost ~41 ms per verdict
  for a 0.10 weight.
- ASV5 eval has been used to compare candidates; never tune on it.
- No true cross-dataset test since run 2 (ASV2019 train/dev are in the pool).

## 6. Inventory

- Code: `src/audiodf/` (data, features, models, training, evaluation, inference, serving, streaming, monitoring), `src/run.py`, `src/tests/` (6 test files), `src/deploy/`, `.github/workflows/ci.yml`.
- Docs: `new_plan.md`, `src/README.md`, `audit.md`.
- Models (git-ignored): `artifacts/` = run 5; `artifacts_run5_encodec/`, `artifacts_run4_wavlm/`, `artifacts_run3_codecs/`,
  `artifacts_run2_pooled/`, `artifacts_run1_asv5/`, `artifacts_prev_asv2019/`.
- Results: `results/training_report.json` (run 5), `training_report_run{1,2,3,4,5}.json`, `train_run{2,3,4,5}.log`, `evaluate_asv5_eval.json`, `data_integrity_asv5.json`, `data_integrity_asv19.json`, `segmented_baseline_bilstm.json`, logs (`train_asv5.log`, `asvspoof5_download.log`, ...).
- Cache (outside the repo): `~/.cache/audiodf/prep_vad-45_0.025_0.05_svmv2_melv1/` (indexes + SVM snapshots, 0.38 GB). The obsolete
  ASV2019 v1 feature caches (`seg2s_hop1s*`, 8.60 GB) were **deleted on 2026-10-03** after checking that nothing in `src/` reads them (only
  `experiments/segmented_baseline.py` did; it rebuilds them in ~28 min). Their one model file, the prototype's 5-epoch RCNN, was kept as
  `artifacts_prev_asv2019/experiment_5epoch_cnn_bilstm.pt`. Free disk after cleanup: 87.7 GB.

## 7. Reproduce

```
cd src
python run.py --eval-utts 30000           # run 5: WavLM alone, codec catalogue 2 with EnCodec (~8 h; pip install encodec)
python run.py --eval-utts 30000 --branches svm rcnn wavlm --svm-checkpoint ../artifacts_run3_codecs/svm.joblib \
              --rcnn-checkpoint ../artifacts_run3_codecs/rcnn.pt     # run 4's branches; exact run 4 = commit 756e79e (catalogue 1)
python -m audiodf audit --dataset asv5     # integrity audit
python -m audiodf evaluate --dataset asv5 --split eval --eval-utts 30000
python -m audiodf benchmark                # serving latency of artifacts/
python -m audiodf serve --port 8000        # API
# telephony test (pip install pyarrow; download SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA parquet files into dataset21/data/)
python -m audiodf prepare21                # unpack to dataset21/flac_eval + protocol (~6 min)
python -m audiodf evaluate --dataset asv21 --eval-utts 30000 --tag run5 --save-scores
python -m audiodf evaluate --dataset asv21 --eval-utts 30000 --bundle ../artifacts_run4_wavlm --branches wavlm --tag run4_wavlm --save-scores
python -m audiodf.evaluation.channel_report --protocol ../dataset21/ASVspoof2021.LA.eval.tsv --runs run4=... run5=... --column wavlm_10s
python -m pytest                           # 108 tests
```

## 8. Open items

| Item | Status |
|---|---|
| Run 2: pooled data + held-out-attack tuning (options 1+2 combined) | done: ASV5 eval 31.6% (run 1 33.0%) |
| Real-codec augmenter via ffmpeg (encoders confirmed available) | done: run 3, 29.4% (not significant) |
| Pretrained speech front end (WavLM) | done: run 4, **5.51%** (target <= 24.4%) |
| EnCodec (neural codec) augmentation for C04/C07 | done: run 5, C04 9.4%, C07 10.5% (rule met; trade-off on other conditions) |
| Serve WavLM alone | done: your call, default from run 5 |
| Telephony check on real phone channels (ASVspoof 2021 LA eval), run 4 and run 5 WavLM | done: run 4 8.81% vs run 5 9.96% (phase 15) |
| Pick the call model + re-calibrate thresholds on speaker-disjoint phone data | **next** (section 9 step 1a; run 4's WavLM recommended) |
| Live call audio check | open (section 9 step 1b); needs your choice of data source |
| Remaining ML list (`new_plan.md` 7.3k): out-of-domain test, telephony robustness, calibration, second seed, ... | open |
| Investigate A12 inversion and the cross-dataset collapse | superseded: run 4 has no inverted eval attack |
| Download the remaining 8 ASV5 eval tars (~68 GB; needs space) for the final number | optional |
| Delete the obsolete ASV2019 cache | done (8.60 GB freed) |
| Kafka / Docker runtime / Kubernetes / Grafana verification | not started (Docker image builds in CI) |
| `configs/default.yaml` paths are relative to the working directory (`artifacts` from `src/` misses the repo's folder) | open, minor |
| Commits | `573fe30` (code), `bef9ad7` (CI fix), `756e79e` (run 4), `34467a8` (run 5), `38fe200` (docs) pushed; all CI green; the telephony-test code and results are uncommitted |
| GitHub cleanup | at the very end, after everything is final (your decision); push as we go until then |

## 9. Next steps (written 2026-10-04 after run 5; updated 2026-10-06 after the telephony check)

**Done 2026-10-06: telephony check on ASVspoof 2021 LA eval (phase 15).** Result: run 4's WavLM beats run 5's on every
real phone channel (8.81% vs 9.96%); thresholds are too loose for phone audio. Next, in order:

**Step 1a: pick the call model and re-calibrate its thresholds (recommended next, no retraining).**
1. Call model = **run 4's WavLM alone** (recommended; revisit if the second seed or live-call data says otherwise).
2. Build a WavLM-only bundle from run 4 (`wavlm.pt`, `branches: [wavlm]`, weights `{wavlm: 1.0}`), kept separate from
   `artifacts/` (run 5) until you approve swapping.
3. Calibrate verify / block thresholds on **phone-channel data that is speaker-disjoint from the data used to report
   them**: split the 30k scored ASVspoof 2021 clips by speaker (the per-clip scores are saved, so no new scoring), set the
   thresholds on one half (1% / 10% of genuine clips flagged, per the existing policy, ideally per the worst channel or
   pooled over channels), report false-alarm and catch rates on the other half. Check whether one global threshold is
   acceptable or the channel-dependent score shift (GSM, PSTN) needs a channel-aware offset.
4. Same for run 5, so the two bundles are compared on equal terms.

**Step 1b: live call audio (still open, still wanted).** The 2021 set is a real-telephony test but not live calls.

1. **Choose the call data (your decision).** Options, best first:
   - *Our own recordings*: genuine calls (several speakers, phones, networks: mobile, VoIP app, landline) plus fake
     calls made by playing TTS / voice-clone output into a real call. Most realistic; needs consent from speakers.
   - *Replay through a real channel*: play a balanced sample of ASV5 eval clips (bonafide and spoof) into a phone or
     VoIP call and record the far end. Labels come for free; tests the channel, not new attacks.
   - *Public data*: ASVspoof 2019 PA was ruled out (replay, not synthetic; no phone channel); ASVspoof 2021 LA is done
     (above); In-the-Wild (real-world deepfakes of public figures) is the closest out-of-domain set, though not phone
     audio (see step 2).
   - Aim for at least ~200 genuine and ~200 fake calls, so EER and false-alarm rates are not dominated by noise.
2. **Tooling to build** (small): `audiodf evaluate-calls --dir <folder> --labels <csv>` that streams each recording
   through `CallSession` exactly as live audio (0.5 s chunks, VAD gate). (The `--bundle` and `--branches wavlm` options
   for scoring one branch of a multi-branch bundle already exist.) Reuse the existing metrics.
3. **Report for both models:** EER; false-alarm rate on genuine calls at the stored verify / block thresholds (the
   number users would feel); detection rate on fake calls; time to decision (2 / 4 / 6 / 10 s); per-channel results;
   latency per verdict.
4. **Decide:** which WavLM goes forward for calls; whether thresholds must be re-set on call-like audio (likely,
   since the tuning set is easier than eval).

**Step 2: remaining ML work** (`new_plan.md` 7.3k), order to revisit after step 1's results:
1. Out-of-domain test (In-the-Wild and/or ASVspoof 2021 DF): no retraining, checks the audiobook-overlap caveat.
2. Telephony robustness: label-blind renders with 8 kHz G.711, packet loss with concealment, background noise,
   reverberation; also rebalance the codec mix (run 5 showed EnCodec gains cost narrowband accuracy). Retrain WavLM.
3. Thresholds calibrated on harder, call-like data; probability calibration of the WavLM score.
4. Second training seed, to measure run-to-run noise before the next decision rule.
5. Early-decision accuracy (first 2 s), serving cost (layer mix, FP16/INT8, distillation), full ASV5 eval.

**Step 3: deployment work** (after the model is final): Kafka end-to-end with real chunked audio, Docker runtime
test, Kubernetes manifests, monitoring (Grafana), CPU vs GPU serving capacity; fix the relative paths in
`configs/default.yaml`.

**Step 4: finalise** documentation, then the GitHub cleanup you planned.

# Audit log: Real-Time Deepfake Voice Detection

Running record of what was done, what was found, what turned out wrong, and what is open.
Last updated: 2026-10-06 (after the live-call check and the swap to run 4's WavLM). Companion docs: `new_plan.md` (plan and analysis), `src/README.md` (usage).

## 1. Current state at a glance

| Item | State |
|---|---|
| Codebase | `src/audiodf/` (39 modules), 126 passing tests. Pushed through `4a69e80`; CI green. The call-test code (phase 16) is not yet pushed. |
| Trained model | `artifacts/` = **run 4's WavLM-Base+ alone** (swapped in 2026-10-06 after beating run 5 on simulated live calls, 4.96% vs 7.45%); thresholds set on 67 phone-channel speakers. Run 5 in `artifacts_run5_encodec/`; run 4's 3-branch bundle in `artifacts_run4_wavlm/`; WavLM-only copy in `artifacts_run4_wavlm_only/`; runs 1-3 kept. |
| Previous model | `artifacts_prev_asv2019/` (ASV2019 prototype, feature v1, cannot load in current code). Kept, git-ignored. |
| Best honest number | Served model (run 4's WavLM): ASV5 eval EER **5.43%** (worst codec condition, EnCodec, 18.5%), ASVspoof 2021 real-phone-channel EER **8.81%**, simulated live calls **4.96%**. Run 5: 5.51% / 9.96% / 7.45%. |
| Telephony check | ASVspoof 2021 LA eval (real VoIP/PSTN channels), 30k clips: run 4 8.81%, run 5 9.96% (phase 15). Simulated live calls (VoIP codecs + noise + packet loss from clean ASV5 eval clips), 9,600 calls: run 4 4.96%, run 5 7.45% (phase 16). |
| Deployment | FastAPI tested for real (53 ms to first verdict on GPU, WavLM alone; live path matches batched scoring); **Kafka, Docker runtime, Kubernetes, Grafana untested**. No real phone-call audio (you cannot provide any; simulated calls used instead). |
| Disk | Data: `dataset/LA` 7.1 GB, `dataset5/` ~75 GB (train, dev, 2 of 10 eval tars); codec renders in the cache (catalogue 1 ~35 GB, catalogue 2 adds ~7 GB, the rest hard links). |
| Next | Retrain WavLM with packet loss, more noise types and G.711 in the codec augmentation (run 6), then a second threshold pass on call-like data set and reported on different speakers (section 9). |

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

### Phase 16: run 4 vs run 5 on simulated live calls (2026-10-06; plan and decision rule written before any result)
- Your decisions: use **run 4's WavLM weights and thresholds**; check whether it beats run 5 on live calls; **if yes, swap
  it into `artifacts/`**. You cannot provide recordings, so live calls are built from **ASV5 eval clips sent through a
  VoIP chain with a little artificial noise**.
- I cannot place real VoIP calls from this machine. The "live call" is therefore a software VoIP channel built from
  real codecs, streamed through the same code a live call uses. Not simulated: echo, jitter-buffer behaviour, real
  phones and networks. Packet loss is crude (see below).
- **Thresholds for run 4's WavLM** (its bundle's own belong to the 3-branch fusion): the 30k scored ASVspoof 2021
  phone-channel clips are split **by speaker** into two halves; verify/block are set on half A with the existing policy
  (10% / 1% of genuine clips flagged, pooled over channels) and checked on half B. Run 5 gets the same procedure so the
  comparison is on equal terms. Bundle: `artifacts_run4_wavlm_only/` (run 4's `wavlm.pt`, `branches [wavlm]`).
- **Live-call set:** clean ASV5 eval clips only (CODEC "-"; a real call starts from clean speech, and eval's own codec
  copies, EnCodec included, would stack an unrealistic second codec): 4,800 genuine + 300 per attack x 16 attacks.
  Per call, chosen from the clip ID only (label-blind, same distribution for genuine and fake): VoIP codec profile
  (Opus wideband 30%, Opus narrowband 8%, AMR-WB 17%, AMR-NB 10%, G.722 10%, G.711 a-law 8% / mu-law 7%, GSM 5%, none
  5%; these mix weights are my assumption), background noise before encoding (75% of calls; white, pink or brown;
  SNR 15-35 dB against the speech level), and packet loss after decoding (50% of calls none, else 1 / 3 / 5% of 20 ms
  frames with crude repeat-and-fade concealment).
- **Decision rule (fixed now):** run 4 replaces run 5 in `artifacts/` if its EER on the call set is lower than run 5's
  **and** the 95% interval of the paired bootstrap difference (run 5 minus run 4, resampling clips) excludes zero. If
  lower but the interval includes zero, I ask you; if run 5 is lower or equal, no swap. The bootstrap covers
  test-sample noise only, not training-seed noise (unmeasured). Secondary, reported but not blocking: per-profile and
  per-noise breakdowns, genuine clips flagged and fakes caught at the calibrated thresholds, time to decision.
- A streaming check (400 calls through `CallSession` in 0.5 s chunks) confirms the batched scoring matches the live path.
- ASV5 eval clips were used earlier to compare candidates; here they only supply clean speech, as you chose. Nothing is
  tuned on the call set.

**Threshold calibration (run 4's WavLM and run 5, scored ASVspoof 2021 clips, 67 speakers):**
- One speaker split (seed 0; 33 / 34 speakers): thresholds set on half A hit 10% / 1% there by construction, and on half B
  flagged **16.2% / 1.2%** of genuine clips (run 4) and 17.8% / 2.0% (run 5). That first split was the worst of 20: over
  **20 random speaker splits** the held-out half gave 10.6% +- 2.2% (range 7.1-16.2%) at verify and 1.1% +- 0.2% at block for
  run 4 (run 5: 10.6% +- 2.6%, 1.1% +- 0.4%): the procedure is on target on average, but a 33-speaker calibration half
  moves the verify rate by about +-2 points.
- **Deviation from what you approved:** because of that spread, the shipped bundle's thresholds are set on **all 67
  speakers** (verify >= 0.00767, block >= 0.98767), not on half A (0.00278 / 0.98234). The half / half and repeated
  splits are the held-out evidence for the procedure. The bundle manifest records this.
- At those thresholds on held-out speakers, fakes caught: verify ~92% (run 4) / 90% (run 5); block ~62% (run 4) / 52%
  (run 5): the 1% block level needs a very high threshold because some genuine phone clips look strongly fake.
- Run 4's WavLM-only bundle: `artifacts_run4_wavlm_only/` (run 4's `wavlm.pt`, branches `[wavlm]`, fusion weight 1).
  Reports: `results/calibration_run4_wavlm_asv21.json`, `calibration_run5_asv21.json`.

**Live-call set** (`dataset_calls/`, git-ignored): 9,600 calls = 4,800 genuine + 300 per attack x 16 attacks, from clean ASV5
eval clips; rendered in 418 s (23 calls/s); indexing 65 s, no call without speech. The set itself confirms label-blindness
(per-class mix of codec, noise and loss within ~1 point, e.g. Opus-WB 29.4% genuine vs 30.3% fake).

**Result, WavLM alone, EER at 10 s on the same 9,600 calls:**

| | run 4 (WavLM-only bundle) | run 5 |
|---|---|---|
| **All calls** (AUC 0.991 / 0.983) | **4.96%** | 7.45% |
| After 2 s / 4 s / 6 s of speech | 6.50 / 5.85 / 5.28% | 8.20 / 7.76 / 7.65% |
| Codec profile: a-law / mu-law / GSM | 4.16 / 4.45 / 3.89% | 6.63 / 5.48 / 6.55% |
| Opus WB / Opus NB / AMR-WB / AMR-NB / G.722 | 4.40 / 5.56 / 5.02 / 5.02 / 4.96% | 7.93 / 5.56 / 8.53 / 6.42 / 8.21% |
| No codec (471 calls) | 6.17% | 9.98% |
| Noise: none / brown / pink / white | 2.89 / 2.81 / 5.88 / 7.46% | 5.53 / 5.99 / 7.93 / 9.68% |
| Packet loss 0 / 1 / 3 / 5% | 3.20 / 4.55 / 5.31 / 6.03% | 3.94 / 6.36 / 7.64 / 9.46% |
| Hardest attacks A28 / A31 / A30 / A18 | 15.6 / 9.0 / 8.7 / 5.4% | 17.3 / 11.6 / 13.6 / 9.5% |

- **Decision rule (fixed beforehand): met.** Run 4 is lower, and the paired bootstrap (2,000 resamples) gives EER(run 5) -
  EER(run 4) = **+2.49 points, 95% interval 2.01 to 2.83**, run 4 better in 100% of resamples. Run 4 is also better in 8
  of 9 codec profiles (Opus-NB identical), every noise type and level and packet-loss level, and 13 of 16 attacks (A22 a
  tie; run 5 better on A21 and A24 by under 1 point). The bootstrap covers test-sample noise only, not training-seed noise.
- **Swap done:** `artifacts/` = run 4's WavLM-only bundle (files byte-identical to `artifacts_run4_wavlm_only/`, checked by
  hash); before copying, run 5's files in `artifacts/` were confirmed identical to `artifacts_run5_encodec/`. Serving check
  after the swap: 53 ms to the first verdict on the GPU; `predict` on a genuine call gives allow (P=0.00005), on an A28 fake
  verify (P=0.062, "real" at 0.5 but above the verify threshold).
- **Streaming check** (400 calls through `CallSession`, 0.5 s chunks, run 4 bundle): median |live - batched| score
  difference 0.0, 95th percentile 0.001, max 0.106; 99.5% within 0.01; same side of 0.5 for all; EER 4.25% live vs
  4.50% batched on those clips; verdict latency p50 52.6 ms, p95 62.1 ms, max 112.6 ms; median 6 verdicts per call.

**What the call test exposed (not blocking the swap, which the rule decided on EER):**
- **Packet loss raises false alarms sharply.** Genuine calls flagged at the verify threshold (run 4 bundle): 6.4% with
  no loss, 12.5% / 26.3% / 42.1% at 1 / 3 / 5% loss (run 5 with its stored thresholds: 7.1 / 16.6 / 33.2 / 49.1%). My
  concealment is crude (repeat the last 20 ms at 0.6 gain), harsher than real codec concealment, so this is probably
  exaggerated; but neither model was trained with packet loss.
- **Thresholds on the call set:** run 4 bundle verify >= 0.0077 flags 16.5% of genuine calls and catches 98.7% of fakes;
  block >= 0.9877 flags 0.0% but catches only 39.2% (the 2021 set's genuine tail is heavy, so its 1% threshold sits where
  few call fakes reach). Run 5's stored thresholds: verify 19.8% / 98.7%, block 3.9% / 89.1%. Run 5 re-calibrated like
  run 4: verify 12.7% / 96.3%, block 0.0% / 27.0%. The thresholds need a second pass on call-like data with loss, set
  and reported on different speakers (not on this call set).
- White noise hurts most (7.46% vs 2.89% with none; brown noise is harmless); A28 (15.6%, catches 65% at the global
  threshold) and A30 / A31 are the weak attacks.
- Caveats: software VoIP chain, not real calls; the codec mix, noise levels and loss rates are my assumptions; no echo,
  jitter buffer or real phones; clean ASV5 read speech as the source.
- Files: `results/evaluate_calls_eval_run{5,4_wavlm_only}.json` (+ per-clip `_scores.csv`), `channel_report_calls.json`,
  `compare_calls_run4_vs_run5.json`, `live_check_run4_wavlm_only.json`, `render_calls.log`, `eval_calls_run{5,4}.log`.
- Code: `data/voip_sim.py` (call chain), `calls` dataset in `data/protocol.py`, `calibrate.py`, `evaluation/compare_runs.py`,
  `evaluation/live_check.py`, `channel_report --kind calls`; CLI `render-calls`, `calibrate`, `live-check`. 126 tests
  (18 new), also green with GSM hidden.

## 3. Decisions and the reasoning behind them

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
- **Live calls are simulated** (you have no recordings and I cannot place real calls): clean ASV5 eval clips -> noise ->
  real VoIP codec -> packet loss, every choice from the clip ID only. Swap rule (EER lower and a paired-bootstrap interval
  excluding zero) written in `audit.md` before any result; applied as written: run 4's WavLM replaced run 5.
- **Thresholds of the served bundle from all 67 phone-channel speakers**, not from half A (the first half / half split was
  the worst of 20 and a 33-speaker half moves the verify rate by about +-2 points); the splits stay as the validation.
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
21. While adding phase 15 I replaced the heading "## 3. Decisions and the reasoning behind them" with "## 3. Decisions"
    (pushed in `4a69e80`); noticed when a later edit would not match, restored in phase 16. In the call test the vacuous
    `or` assertion from mistake 20 reappeared once in the packet-loss test; replaced with a check that fails (the test
    still passes, now meaningfully).

## 5. Known limitations (current)

- **Packet loss and noise (served model):** genuine calls flagged at the verify threshold rise from 6.4% (no loss) to
  42% (5% simulated loss); white noise doubles the EER (2.9% -> 7.5%). Neither was in training; the concealment in the
  test is cruder than real codecs', so the size of the loss effect is uncertain. Hardest attacks: A28 (15.6% EER, 65%
  caught), A30, A31.
- **Thresholds:** verify holds roughly (10.6% +- 2.2% of genuine flagged on unseen speakers; 6.4% on loss-free calls), but
  block (1% budget) catches only 39% of fakes on simulated calls (62% on held-out 2021 speakers). Not final.
- Codec trade-off (run 5, not served): EnCodec conditions improved to 9.4% / 10.5%, but the other conditions, clean audio
  and ASV2019 lost 0.1-2.3 points, and run 5 lost to run 4 on every real phone channel.
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
python -m audiodf render-calls             # simulated VoIP calls from clean ASV5 eval clips (~7 min, dataset_calls/)
python -m audiodf evaluate --dataset calls --eval-utts 0 --tag run5 --save-scores   # also --bundle DIR
python -m audiodf calibrate --scores ... --protocol ... --tag T --repeat 20 [--src-bundle DIR --out-bundle DIR]
python -m audiodf live-check --scores ... --bundle DIR --tag T     # stream through CallSession
python -m audiodf.evaluation.compare_runs --first a=... --second b=... --column wavlm_10s   # paired bootstrap
python -m pytest                           # 126 tests
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
| Pick the call model, thresholds on phone-channel data, simulated live-call test, swap | done: run 4's WavLM-only now in `artifacts/` (4.96% vs 7.45% on 9,600 calls, +2.49 [2.01, 2.83]; phase 16) |
| Telephony-robust retrain (packet loss, more noise, G.711), run 6 | **proposed next** (section 9 step 1) |
| Second threshold pass on call-like data with loss, other speakers | open (section 9 step 2) |
| Live call audio from real recordings | not possible now (no recordings); simulated calls used |
| Remaining ML list (`new_plan.md` 7.3k): out-of-domain test, second seed, probability calibration, serving cost, ... | open |
| Investigate A12 inversion and the cross-dataset collapse | superseded: run 4 has no inverted eval attack |
| Download the remaining 8 ASV5 eval tars (~68 GB; needs space) for the final number | optional |
| Delete the obsolete ASV2019 cache | done (8.60 GB freed) |
| Kafka / Docker runtime / Kubernetes / Grafana verification | not started (Docker image builds in CI) |
| `configs/default.yaml` paths are relative to the working directory (`artifacts` from `src/` misses the repo's folder) | open, minor |
| Commits | `573fe30`, `bef9ad7`, `756e79e`, `34467a8`, `38fe200`, `4a69e80` (telephony test) pushed; all CI green; the live-call test code and results are uncommitted |
| GitHub cleanup | at the very end, after everything is final (your decision); push as we go until then |

## 9. Next steps (updated 2026-10-06, after the live-call check and the swap to run 4's WavLM)

**Done:** telephony check on real phone channels (phase 15); simulated live-call test, thresholds on phone-channel data and
the swap to run 4's WavLM-only bundle (phase 16). Live calls cannot be recorded (you have no recordings to provide), so
the call test is simulated; a small set of real recordings would still be the best final check if one ever becomes available.

**Step 1: telephony-robust retrain (run 6, WavLM only; proposed next).** The call test showed what the model lacks:
packet loss (false alarms 6% -> 42% of genuine calls at 5% loss), white noise (2.9% -> 7.5% EER), and the hardest attacks.
- Add to the label-blind codec renders (a new catalogue, reusing existing copies where nothing changes): packet loss with
  several concealment styles including smoother, codec-like ones; white / pink / brown noise at 10-35 dB; G.711 (a-law,
  mu-law); optional room reverberation. Re-think the EnCodec share (run 5's 18% cost about a point elsewhere).
- Keep test and train apart: the simulated call set uses my chain, so a model trained on the same chain would look better
  than it is. Judge run 6 on **ASV5 eval and ASVspoof 2021 (real channels)** first, and use the call set only as a secondary
  stress test with a differently-seeded, differently-parameterised variant (e.g. another concealment style).
- Decision rule to fix before training: adopt run 6 only if it is no worse than run 4 on ASV5 eval and ASVspoof 2021
  (within the noise margin) and clearly better under packet loss and noise.
- Cost: ~8 h of training (like runs 4-5) plus rendering.

**Step 2: second threshold pass (no retraining; can run before step 1 on run 4).** Calibrate verify / block on call-like
data **with loss**, on one set of speakers, and report on other speakers (a fresh call render from different source clips
and speakers, not the call set already used for the swap). The block level at a 1% false-alarm budget currently catches
only 39% of fakes; the product decision is how many false blocks are acceptable, so I would report the catch-vs-false-alarm
curve per channel and loss level and let you choose the operating points, rather than fix 1% / 10% blindly.

**Step 3: remaining ML work** (`new_plan.md` 7.3k): out-of-domain test (In-the-Wild and/or ASVspoof 2021 DF; no retraining);
a second training seed (measures run-to-run noise, which every decision so far lacks); probability calibration of the WavLM
score; early-decision accuracy (first 2 s); serving cost (drop unused layers, FP16/INT8, distillation); the full ASV5 eval.

**Step 4: deployment work** (after the model is final): Kafka end-to-end with real chunked audio, Docker runtime test,
Kubernetes manifests, monitoring (Grafana), CPU vs GPU serving capacity; fix the relative paths in `configs/default.yaml`.

**Step 5: finalise** documentation, then the GitHub cleanup you planned.

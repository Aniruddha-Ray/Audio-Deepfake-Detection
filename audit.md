# Audit log: Real-Time Deepfake Voice Detection

Running record of what was done, what was found, what turned out wrong, and what is open.
Last updated: 2026-10-09 (end of the session of 2026-10-06 to 2026-10-09: model final, explanation agent, scam intent). Companion docs: `new_plan.md` (start at section 0, RESUME HERE), `src/README.md` (usage).

## 1. Current state at a glance

| Item | State |
|---|---|
| Codebase | `src/audiodf/` (+ `explain/`, `intent/`, `gating/`), 195 passing tests. Pushed with this session's last commit; `1db0270` CI green, later CI results to be read (API rate limit). Run 8's own files and the DVC pointer files are kept local on purpose. |
| **Final model** | **Run 7 seed 0, WavLM-Base+ alone** in `artifacts/` (your decision, 2026-10-08): verify >= 0.0060, **escalate** >= 0.90 (the high level escalates to the strongest check; it never rejects a call by itself). Other bundles kept for reference: runs 1-9, run 7 seed 1, run 9 (WavLM + Whisper). |
| Best honest numbers (final model) | EER at 10 s / fakes missed per 1,000 at the verify level: ASVspoof 2021 real phone lines 9.14% / 83; simulated calls v1 3.06% / 9, babble 7.00% / 8, held-out v2 10.78% / 6; **In-the-Wild real-world deepfakes 4.47% / 16**; ASV5 eval 5.74%; ASVspoof 2019 5.33%. |
| Deciding measure | Fakes missed at the verify level (each system's verify level flags 10% of genuine ASVspoof 2021 callers), then genuine callers asked to verify; EER is reported, never decides (`new_plan.md` 7.3x). |
| Explanations | `audiodf explain` / `POST /explain`: SHAP over time x frequency regions, faithfulness check, numbers-only facts, free-tier LLM (Groq / Gemini / OpenRouter) with number checking and a template fallback. The LLM demo needs your API key. |
| Scam intent | `audiodf intent`: local Whisper speech-to-text, 8 intent patterns, optional bounded LLM, a policy that only raises the check. Rules: 0.6% of normal calls flagged at high on BothBosu, **37.8% on the independent set** (legitimate calls on the same topics); the LLM path is unmeasured. |
| Serving | GPU float16: 53 ms per window alone, ~235 simultaneous calls per laptop GPU; CPU ~5 calls; int8 rejected. Verify after ~6 s of speech is as good as at 10 s; no escalation before ~4-6 s. |
| Deployment | FastAPI tested (live path matches batched scoring). **Kafka, Docker runtime, Kubernetes, Grafana untested**; Docker Desktop installed, not running; no cluster. Waiting for your go. |
| Backup | DVC initialised, 13 model folders in the local DVC cache, **no remote** (DagsHub pending you); the weights exist only on this disk. |
| Next | `new_plan.md` section 0: your go for deployment, a free LLM key, the LLM intent measurement (rule fixed), the DVC remote, final docs and cleanup. |

## 1b. Session of 2026-10-06 to 2026-10-09: findings and improvements (details in phases 15-31)

**Findings**
- Telephony (phase 15): on ASVspoof 2021 real phone channels run 4's WavLM beat run 5 (8.81 vs 9.96%); thresholds tuned on the tuning set were far too loose on phone audio.
- Simulated live calls (16): run 4 beat run 5 (4.96 vs 7.45%) and was served; the served model was fragile to babble (15.5%), real echo (18%) and packet loss (genuine calls flagged 6% -> 42%).
- Denoising (17): ffmpeg afftdn made every set worse: rejected. Noise / echo / loss training (run 6) gained 2-8 points on noisy calls but cost ~1 point on clean phone audio.
- EnCodec (18): removing it (run 7) halved that cost; run 7 was adopted by your override (the pre-set rule missed by 0.18 / 0.03 points).
- Call-audio thresholds (19): phone-speech thresholds asked 15-45% of genuine callers to verify on noisy calls; room echo became the main driver of false alarms.
- More echo of the same kind (run 8, 20) does not fix very long reverberation; the held-out rooms are far more reverberant than real call rooms; more epochs do not help (epochs 5-8 equivalent).
- Whisper (21): the better encoder under echo, the worse one on clean audio; fixed and gated fusion (22) both lose on real phone lines (+0.72 / +0.32) and win on noisy calls.
- Run-to-run spread (23): 0.00-0.15 points on clean sets, 0.36-0.56 on noisy ones (up to 1.35 in one condition): differences under ~0.5 on noisy sets are noise.
- At the operating point run 7 alone misses the fewest fakes; every Whisper ensemble that won on EER misses more (`new_plan.md` 7.3x): EER can mislead for banking.
- Out of domain (27): In-the-Wild 4.47% EER, 16 fakes missed per 1,000; one voice holds 31% of the misses; short clips are harder.
- Calibration (26): the raw score is not a probability; at 1 deepfake in 1,000 calls ~96% of high-level phone-line calls would be genuine, so the high level must escalate, not block.
- Serving (28): a GPU is needed (~235 calls vs ~5 on CPU); int8 changes a quarter of the decisions.
- Scam intent (30): phrase rules find scam patterns but cannot tell a scam from a legitimate call on the same subject (37.8% of legitimate calls flagged on the independent set).

**Improvements built**
- Noise bank, impairment kit (echo, noise, babble, bursty loss with five concealment styles), measured MIT rooms with a time stretch, call simulators v1 / v2, calibration
  call sets with a fixed speaker split, `call-thresholds`, `calibrate`, `live-check`, `compare_runs` (paired bootstrap), `channel_report`.
- Reuse of renders across runs (EnCodec share, backward-compatible configuration tags), per-pass checkpoints, the Whisper branch end to end, the channel-quality gate.
- The final model's operating policy: verify 0.0060 (banking: missed fakes outrank verification load), escalate 0.90, intent raises the check.
- Explanation agent and scam-intent module (local speech-to-text, a bounded LLM with verbatim-quote checks, an injection-safe floor).
- In-the-Wild loader and the index's per-folder file type (a latent `.flac` assumption), the serving-cost benchmark, DVC tracking.
- Process: decision rules fixed before every run and applied as written; a clean-checkout test before partial pushes; slower CI polling. Mistakes 20-26 are in section 4.

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

### Phase 17: denoising test, then a noise- and loss-robust retrain (plan written 2026-10-06 before any result)
- **Your plan, in order:** (1) test denoising on the call sets without retraining; (2) train on noise instead of filtering it:
  real noise recordings, babble from other speakers' clips, room echo, SNR 5-35 dB; (3) run 6 = noise + bursty packet loss +
  G.711 + a smaller EnCodec share, judged first on ASV5 eval and ASVspoof 2021 (real phone channels) and on a call set with
  differently built noise and loss. Then compare with the current state and decide.
- **Step 1 design (fixed now):** the served model (run 4's WavLM) is scored, unchanged, on (a) the 9,600 existing calls and (b) a
  new babble call set (4,800 calls = 2,400 genuine + 150 per attack, seed 1; same builder and codec / loss plan; background
  talkers = 4-6 other speakers' genuine ASV5 eval clips that are not call sources, replacing the synthetic colours; babble at
  SNR 5-25 dB because babble at 15-35 dB would be too mild to test a denoiser), each with and without denoising. Primary denoiser: ffmpeg `afftdn` (FFT denoiser, no new
  dependency) at `nr=12:nf=-45:tn=1` (12 dB reduction, noise tracking on), applied to every call, as it would have to be in
  a live system that cannot know which calls are noisy. A stronger setting (`nr=24`) is a sensitivity check only.
- **Rule for adopting a denoiser (fixed now), all three required:** (i) EER improves on **both** call sets with a paired
  bootstrap interval excluding zero; (ii) the calls without added noise (control) get no worse than +0.5 points; (iii) latency is
  acceptable (measured per 10 s of audio and judged against the model's 53 ms per window).
- Caveats written down in advance: offline whole-file denoising is optimistic compared with streaming chunks (noise tracking
  needs warm-up); the synthetic noise in set (a) is stationary, the easy case, which is why set (b) exists.
- **Step 1 result: denoising is rejected; the rule fails on all counts.** Served model (run 4's WavLM), EER at 10 s,
  original -> denoised (difference denoised minus original, 95% interval of the paired bootstrap, 1,000 resamples):

| Call set | no denoising | afftdn nr=12 (primary) | afftdn nr=24 (strong) |
|---|---|---|---|
| 9,600 calls, synthetic noise | **4.96%** | 5.91% (+0.95 [+0.55, +1.34]) | 6.56% (+1.60 [+1.18, +2.09]) |
| 4,800 calls, babble (5-25 dB) | **15.54%** | 17.62% (+2.08 [+1.35, +2.73]) | 18.71% (+3.17 [+2.35, +3.94]) |
| Control: calls with no added noise (synthetic set) | **2.89%** | 3.96% (+1.07) | 4.21% (+1.32) |
| Control: calls with no added noise (babble set) | **3.58%** | 4.91% (+1.33) | 5.25% (+1.67) |
| AUC (synthetic set / babble set) | 0.991 / 0.924 | 0.988 / 0.913 | 0.984 / 0.905 |

  - (i) EER improves on both sets: **no, it gets worse on both, and the interval excludes zero in the wrong direction** (the
    original is better in 100% of resamples); (ii) the no-noise control may not get worse by more than 0.5: **it gets worse
    by 1.1-1.7**; (iii) latency: ~187-200 ms per 10 s of audio for the offline filter including ffmpeg process start-up (not
    the deciding factor). A stronger setting is worse than the mild one, so more denoising = more damage.
  - Where it helps a little: moderate babble (SNR 15-25 dB: 13.99% -> 13.21%, 9.10% -> 7.91%) and white noise (7.46% ->
    7.25%, within noise). Where it hurts: low-SNR babble (10-15 dB: 18.3% -> 21.5%), pink and brown noise, and clean audio.
    It also shifts the score scale (the global EER threshold moves from 0.119 to 0.024 / 0.010), so thresholds calibrated
    without it would be wrong with it.
  - Reading: the filter removes some noise but also the fine detail and artifacts the detector reads, and adds its own; so
    it hurts clean calls too, and a front end cannot know which calls are noisy. Not tested: neural denoisers (RNNoise,
    DeepFilterNet) and streaming operation; classical offline FFT denoising is the only thing rejected here.
  - **The finding that matters more:** the served model is fragile to **background speech**: on the babble set 19.1% EER on
    babble calls against 3.6% on the clean calls of the same set; at 5-10 dB SNR, 61% of genuine calls are flagged at the
    global threshold. White noise (7.5%) was the mild case. This is the main target of step 2.
  - Files: `results/evaluate_calls_eval_{babble,calls_dn12,calls_dn24,babble_dn12,babble_dn24}.json` (+ per-clip
    `_scores.csv`), `denoise_test_compare.json`, `denoise_*.json`, `denoise_test.log`. Code: `data/denoise.py`,
    `BabblePool` and a per-folder index key in `data/voip_sim.py` / `data/prepare.py` (the call-set index used to be shared
    between folders: a latent bug, fixed with a test), CLI `render-calls --noise babble`, `denoise-calls`.
- **Noise data for steps 2-3** (downloaded to `dataset_noise/`, git-ignored): training noise = MUSAN noise + a DEMAND subset;
  room echo = real and simulated impulse responses (RIRS_NOISES); held-out for testing only = ESC-50 environmental sounds and
  babble from held-out speakers, so the run 6 call set uses noise the model never trained on.

- **Run 6 recipe (fixed before training; WavLM alone, from the pretrained start, same pool / tuning set / epochs / seed / WavLM
  settings as run 4: only the data recipe changes):**
  - Codec copies (catalogue 3): the catalogue-2 ffmpeg codecs + **G.711** (a-law, mu-law); **EnCodec share 7%** (run 5: 18%);
    60% of training clips get a copy (was 50%) because noise and echo live in the copies.
  - Before the codec, per copy (label-blind, from clip ID): **room echo** on 30% (RIRS_NOISES *simulated* impulse
    responses) then **noise** on 65% at **5-35 dB**: MUSAN 35% / point-source 15% / DEMAND subset 5% (28 clips) / **babble** 20%
    (4-6 other ASV5 *train* speakers' genuine clips) / white-pink-brown 25%.
  - After the codec, per training window, on 35% of windows: **bursty packet loss** (Gilbert-Elliott chain, average loss 1-8%,
    mean burst 1-4 packets of 20 ms) with one of **four concealment styles** (repeat-fade, zero, crossfade, comfort noise),
    drawn fresh each epoch.
  - Held out of training, used only to test: ESC-50 noise, real measured room impulse responses, babble from ASV5 *eval*
    speakers, and the concealment style `ola_repeat` with longer bursts (5-12 packets).
  - Not isolated: which part of the recipe helps (noise, echo, loss, G.711, EnCodec share changed together).
- **Run 6 test sets (EER at 10 s, same clips for every model; baseline = the served model, run 4's WavLM):**
  ASV5 eval 30k (5.43%), ASVspoof2019 eval 30k (4.85%), **ASVspoof 2021 real phone channels 30k (8.81%)**, call set v1 9,600
  (4.96%; synthetic noise and iid repeat-fade loss, which the recipe also trains on, so *in-distribution*), babble set 4,800
  (15.54%), **call set v2 6,400 (held-out noise / echo / loss; baseline measured before training)**. Judged first on ASV5 eval
  and ASVspoof 2021, as you said; the call sets are the stress tests.
- **Recommended decision framework (you decide after seeing the comparison):** recommend adopting run 6 only if (a) **no real
  regression**: ASV5 eval <= 5.43% + 0.5, ASVspoof2019 eval <= 4.85% + 0.5, ASVspoof 2021 <= 8.81% + 0.3 (same noise caveat
  as before: seed noise is unmeasured); and (b) **real robustness gain**: v2 EER <= 0.8 x the baseline's, babble-set EER
  <= 0.75 x 15.54% (= 11.7%), and call set v1 no worse than 4.96% + 0.3; and (c) **fewer loss-driven false alarms**: with
  thresholds set the same way for both models (all 67 ASVspoof 2021 speakers), the rise in genuine calls flagged at the verify
  level from 0% to 5% simulated loss on call set v1 shrinks from the baseline's. Every number, per-condition tables and paired
  intervals are reported either way. If (a) fails, I recommend against adopting it whatever (b) shows.
- **Baseline on call set v2 (served model, measured before training): 16.95% EER (AUC 0.911), 6,400 calls.** By condition: no
  echo and no noise 3.98%; no noise but real room echo **18.34%**; ESC-50 noise 8.64% (with echo 26.24%); babble 17.05% (with
  echo 33.34%); white 12.51% (with echo 31.34%); loss at 5-12 packet bursts barely matters (16.10% with no loss, 14.9-18.8%
  with loss). So **real room echo is the largest weakness** (4.6x worse with echo alone), then babble. The framework's target for
  run 6 on v2 is therefore <= 0.8 x 16.95% = 13.6%. Caveat: run 6 trains on *simulated* echo and is tested on *real* echo.
- Run 6 outputs go to `artifacts_run6/` and `results/run6/` (config `src/configs/run6.yaml`); `artifacts/` (the served model) is
  not touched. Log: `results/train_run6.log`.
- Expected duration: rendering ~2.5-3 h, 8 training epochs ~50 min each, scoring ~1 h; about 11 h.

- **Run 6 happened (2026-10-06, 13:08-22:00 incl. a restart).** The first attempt crashed after 34,412 copies because **you renamed
  `dataset/` to `dataset19/` while it was reading ASVspoof2019** (log: `results/train_run6_crash_dataset_rename.log`); I changed the
  default path to `dataset19/LA/LA` (config, default.yaml, README) and restarted; the render resumed from the copies already made.
  Rendered 184,407 training clips (60%) and 8,967 of 11,998 tuning clips (same tuning clips as runs 3-5); 7% EnCodec. WavLM tuning EER
  (a much harder set than run 4's, with noise, echo and codecs, so not comparable with it): 6.05, 5.66, 4.03, 4.54, 4.04, 3.73,
  **3.37** (epoch 7 kept), 3.48%; epoch 8 took 65 min (low free RAM) instead of ~48. Bundle `artifacts_run6/`, thresholds verify >=
  0.149 / block >= 0.512 (from that hard tuning set; recomputed on phone data below). Report `results/run6/training_report.json`.
- **Run 6 against the served model (run 4's WavLM), WavLM alone, EER at 10 s, same clips; 95% interval of the paired bootstrap
  (run 6 minus served) for the sets with per-clip scores:**

| Test set | served | run 6 | run 6 minus served (95% interval) |
|---|---|---|---|
| **ASV5 eval (30k)** (AUC 0.985 / 0.992) | 5.43% | **4.46%** | -0.97 (no per-clip scores kept; sample noise is small at 30k) |
| ASVspoof2019 eval (30k) | **4.85%** | 5.82% | +0.97 (same remark) |
| **ASVspoof 2021 real phone channels (30k)** (AUC 0.969 / 0.962) | **8.81%** | 9.86% | **+1.05 [+0.77, +1.39]** |
| Call set v1, synthetic noise and loss (9,600; *in-distribution* for run 6) | 4.96% | **3.18%** | **-1.78 [-2.23, -1.29]** |
| Babble set (4,800) | 15.54% | **7.75%** | **-7.79 [-8.79, -6.71]** |
| **Call set v2, held-out noise / echo / loss (6,400)** | 16.95% | **11.41%** | **-5.55 [-6.48, -4.53]** |
| ASV5 eval after only 2 s of speech | 6.88% | **5.92%** | -0.96 |

  (Correction: the intervals printed on screen for this comparison had the sign of the bracket flipped; the figures above are the
  right ones, and the point estimates and the share of resamples were right.)
  - ASV5 eval detail: the EnCodec conditions improved (C04 14.56 -> 9.08%, C07 18.51 -> 11.49%), C08 6.07 -> 4.59%, the others within
    +-0.6 (clean 1.01 -> 1.29%); **14 of 16 attacks better** (A23 4.58 -> 1.27%, A30 8.96 -> 6.01%, A18 5.51 -> 2.87%; worse: A28
    10.54 -> 13.43%, A32 3.52 -> 4.32%).
  - ASVspoof 2021 detail: worse in **every** channel by 0.5-1.6 points, including the untouched reference (5.98 -> 7.60%), GSM
    10.64 -> 12.29%, public phone network 11.49 -> 12.71%. These are the 2019 evaluation utterances (clean read speech).
  - **Run 5 had the same ~1 point regression on exactly these 2019-derived sets** (ASVspoof2019 eval 5.82% for both, ASVspoof 2021
    9.96% / 9.86%), and run 6 shares with run 5 the neural-codec (EnCodec) copies and an EnCodec-containing tuning set. Runs 5 and 6
    differ in everything else (noise, echo, loss, G.711, EnCodec share 18% vs 7%), so EnCodec is a candidate cause, **not a proven
    one**; training-seed noise is unmeasured and could account for part of it.
  - Call sets: run 6 is better under every noise type and every packet-loss level of v1; babble at 10-25 dB halves the error
    (e.g. 15-20 dB: 13.99 -> 5.87%); on v2, clean calls 3.98 -> 3.03%, babble 17.05 -> 7.51%, ESC-50 8.64 -> 5.61%, white 12.51 -> 7.99%.
  - **Still weak:** real room echo (run 6 trained on *simulated* echo): no noise + real echo 18.34 -> 13.18% (3.03% without echo), echo
    + ESC-50 20.78%, echo + white 22.68%, echo + babble 22.03%; babble at 5-10 dB 24.51 -> 16.52%; attack A28 13.4%.
- **Loss-driven false alarms (framework item c), thresholds set the same way for both (all 67 ASVspoof 2021 speakers), call set v1:**
  genuine calls flagged at the verify level at 0 / 1 / 3 / 5% simulated loss: served 6.4 / 12.5 / 26.3 / 42.1% (rise **+35.7**
  points); run 6 9.8 / 10.2 / 12.3 / 13.6% (rise **+3.8**). Run 6's thresholds: verify >= 0.0148, block >= 0.9992; at the block level
  neither model blocks any genuine call and run 6 catches 25.5% of fakes (served 39.2%): the 1% block budget is not usable with
  either model on phone audio.
- **Offline model average (served + run 6, equal weights, nothing fitted; not a candidate):** ASVspoof 2021 9.09% (served 8.81), v1
  3.05%, babble 9.00% (run 6 alone 7.75), v2 10.95%: no clean win, and twice the compute.
- **Verdict by the framework fixed before training:** (a) no real regression: ASV5 eval 4.46% <= 5.93% **met**; ASVspoof2019 eval
  5.82% <= 5.35% **not met** (+0.97); ASVspoof 2021 9.86% <= 9.11% **not met** (+1.05). (b) robustness gain: v2 11.41% <= 13.56%
  **met**; babble 7.75% <= 11.66% **met**; v1 3.18% <= 5.26% **met**. (c) fewer loss-driven false alarms: **met** (+3.8 vs +35.7
  points). **Because (a) fails on two sets, the rule as written recommends against adopting run 6**, whatever (b) and (c) show. The
  rule was written to protect real-channel accuracy, and the same two sets sank run 5. Not swapped: `artifacts/` is still run 4's WavLM.
- **The trade-off in one line:** run 6 gives up about 1 point on clean-speech phone audio (the 2019-derived sets) to cut the error by
  2-8 points on noisy, echoey, babbling and lossy calls and to nearly remove loss-driven false alarms. Which is better for a product
  depends on how noisy real calls are, which we have not measured.
- Files: `results/run6/training_report.json`, `evaluate_asv21_eval_run6.json`, `evaluate_calls_eval_run6_{calls,babble,v2}.json`
  (+ per-clip `_scores.csv`), `evaluate_calls_eval_v2_run4.json`, `run6_vs_served_compare.json`, `run6_channel_{calls_v1,babble,v2}.json`,
  `calibration_run6_asv21.json`, `eval_run6_battery.log`, `train_run6.log`.
- Code added this stretch: `data/noise_bank.py`, `data/impairments.py` (reverb, noise mixing, Gilbert-Elliott loss, five concealment
  styles, per-clip plans, `ImpairKit`), `data/calls_v2.py`, `data/denoise.py`, `ImpairKit` hook in `render_copies` (catalogue 3 with
  G.711), packet-loss augmentation in the window datasets, `BabblePool`, a per-folder index key for call sets (a latent bug), CLI
  `prepare-noise`, `render-calls-v2`, `denoise-calls`, `render-calls --noise babble`; `run.py --no-impairments --noise-root`;
  `configs/run6.yaml`. 147 tests (all green, including a guard that no training draw can name a held-out corpus).

### Phase 18: run 7, the run 6 recipe without EnCodec (2026-10-06 23:38 to 2026-10-07 06:44; rule written before training)

- **Your decision:** option C (run 7) first; your noise-gated ensemble of run 4 and run 6 as plan B (`new_plan.md` 7.3p, with the
  offline ceiling and five conditions). Rule fixed before training (7.3p): ASVspoof2019 <= 5.15%, ASVspoof 2021 <= 9.11% (run 4 + 0.3),
  ASV5 eval <= 5.93%, babble <= 11.7%, v2 <= 13.6%, loss-driven false-alarm rise <= +10 points.
- **Setup:** `configs/run7.yaml` = EnCodec share 0, everything else run 6. New: `data.reuse_neural_share` hard-links copies of an earlier
  share whose codec is unchanged (171,482 of 184,407 training and 8,312 of 8,967 tuning copies from run 6; only the 12,925 + 655 former
  EnCodec clips rendered again, now with classical codecs: 8 min instead of ~3 h). `wavlm.keep_epochs` saves every pass
  (`artifacts_run7/epochs/wavlm_e1..8.pt`). Two tests: the reuse links exactly the unchanged copies, and a copy that is classical under
  both shares is sample-identical when rendered from scratch under either. 149 tests green.
- **Training:** tuning EER by pass 5.56, 5.15, 3.64, 4.78, 3.39, 3.28, 2.95, **2.94** (pass 8 kept); ~48 min per pass. Lower than run 6
  at every pass, but the tuning set lost its EnCodec copies, so it is easier: not comparable.
- **Results, WavLM alone, EER at 10 s, same clips for all three runs; 95% paired-bootstrap intervals where per-clip scores exist:**

| Test set | run 4 (served) | run 6 | run 7 | run 7 minus run 4 | run 7 minus run 6 |
|---|---|---|---|---|---|
| ASV5 eval (30k) | 5.43% | **4.46%** | 5.74% | +0.31 | +1.28 |
| ASVspoof2019 eval (30k) | **4.85%** | 5.82% | 5.33% | +0.48 | -0.49 |
| ASVspoof 2021 real phone channels (30k) | **8.81%** | 9.86% | 9.14% | +0.32 [+0.03, +0.63] | **-0.73 [-0.94, -0.53]** |
| Call set v1 (9,600) | 4.96% | 3.18% | **3.06%** | **-1.90 [-2.40, -1.45]** | -0.12 [-0.38, +0.06] |
| Babble set (4,800) | 15.54% | 7.75% | **7.00%** | **-8.54 [-9.56, -7.52]** | **-0.75 [-1.21, -0.35]** |
| Call set v2, held-out noise / echo / loss (6,400) | 16.95% | 11.41% | **10.78%** | **-6.17 [-7.08, -5.25]** | **-0.63 [-1.16, -0.14]** |
| Genuine calls flagged at verify, 0% -> 5% loss | 6.4 -> 42.1% (+35.7) | 9.8 -> 13.6% (+3.8) | 12.4 -> 17.8% (+5.4) | | |
| Block level: fakes caught at ~0% false blocks (v1) | 39.2% | 25.5% | 34.8% | | |

  (Thresholds for run 7 set on all 67 ASVspoof 2021 speakers, as for runs 4 and 6: verify >= 0.0060, block >= 0.9958; 20 speaker
  splits: verify 10.6% +- 2.5% genuine flagged / 91.8% caught, block 1.0% / 60.2%.)
  - ASV5 detail: removing EnCodec cost much more than the neural-codec conditions (C04 9.08 -> 15.34%, C07 11.49 -> 19.61%, back to run
    4's 14.56 / 18.51): run 6's broad gain across ASV5 attacks is mostly gone too (A17 0.64 -> 2.82, A23 1.27 -> 3.20, A21 0.43 ->
    1.99; run 7 is near run 4 on most attacks). EnCodec copies helped attack generalisation on ASV5, not only neural-codec audio.
  - ASVspoof2019 detail: run 7 is near run 4 on most attacks; the remaining gap is mostly A18 (5.78 / 10.11 / 8.16 for runs 4 / 6 / 7)
    and A13, A14, A19.
  - Offline, run 4 + run 7 (no training): plain average 2021 8.77, v1 3.10, babble 9.06, v2 10.75; perfect noise routing 8.81, 3.23,
    7.02, 10.84. The ensemble again only recovers the clean-phone gap.
- **Verdict by the rule fixed before training:** ASV5 eval 5.74 <= 5.93 **met**; babble 7.00 <= 11.7 **met**; v2 10.78 <= 13.6 **met**;
  loss-driven rise +5.4 <= +10 **met**; ASVspoof2019 5.33 <= 5.15 **not met** (by 0.18); ASVspoof 2021 9.14 <= 9.11 **not met** (by
  0.03). As written, the rule does not recommend adopting run 7. Not swapped: `artifacts/` is still run 4's WavLM.
  **Caveat (my mistake 23):** run 6's framework allowed +0.5 on ASVspoof2019 (limit 5.35%); for run 7 I wrote +0.3 (5.15%) without saying
  I was tightening it. Under run 6's limits run 7 would miss only ASVspoof 2021, by 0.03 points.
- **What run 7 tells us:** EnCodec explains about half of run 6's clean-phone regression (ASVspoof2019 +0.97 -> +0.48, ASVspoof 2021
  +1.05 -> +0.32 points), not all of it. Run 7 beats run 6 everywhere except ASV5 eval (the EnCodec gain), and is the most robust model on
  every call set. The rest of the gap to run 4 (0.3-0.5 points) is either a real cost of the noise recipe or run-to-run spread, which
  has never been measured (the 2021 interval [+0.03, +0.63] covers test-clip sampling only, not training randomness).
- **Your decision (2026-10-07): adopt run 7**, accepting the 0.3-0.5 point cost on clean phone audio for the gains on noisy calls
  (an explicit override of the pre-set rule, after seeing all 8 passes). Swap: `calibrate --src-bundle ../artifacts_run7 --out-bundle
  ../artifacts_run7_wavlm_only --thresholds-from all` (thresholds on all 67 ASVspoof 2021 speakers: verify >= 0.0060, block >= 0.9958),
  then its `wavlm.pt` and `bundle.json` copied into `artifacts/`. Run 4's served bundle stays in `artifacts_run4_wavlm_only/` (checked
  byte-identical before the swap). Live check through the streaming path (400 calls of set v1): EER 3.75% live vs 3.50% batched, score
  difference median 0.00001 / p95 0.0013, verdict latency p50 47 ms (`results/live_check_run7_served.json`).
- Remaining weak spots of run 7 on held-out set v2: real room echo (no noise 2.7% -> with echo 12.6%; noise + echo 19.9%), babble 13.2%,
  5-10 dB SNR 16.6%; neural-codec audio (ASV5 C04 / C07 15.3 / 19.6%).
- Files: `results/run7/training_report.json`, `evaluate_asv21_eval_run7.json`, `evaluate_calls_eval_run7_{calls,babble,v2}.json` (+
  per-clip `_scores.csv`), `calibration_run7_asv21.json`, `run7_vs_run4_run6_compare.json`, `eval_run7_battery.log`, `train_run7.log`.

### Phase 19: second threshold pass for run 7 on call audio (plan written 2026-10-07 before any result)

- **Why:** the served verify / block levels were set on clean-speech phone audio (ASVspoof 2021). On simulated calls they ask 12.4% of
  genuine callers to verify with no loss (17.8% at 5% loss), and the block level catches 35% of fakes.
- **Design (no retraining, nothing tuned on a test set):** a fixed split of the 737 ASV5 eval speakers by a hash of the speaker name:
  half A (361 speakers) sets thresholds, half B (376) only reports. Every earlier call set used both halves (all 616 speakers with unused
  clips already appear in one), and only 777 clean genuine clips were unused, so a set from wholly new speakers is impossible.
  - Calibration audio: two new call sets from half-A speakers only, new seed 5 (fresh noise, echo, loss and codec draws): `cal_v1` (v1
    chain: synthetic noise, independent loss) and `cal_v2` (v2 chain: ESC-50 / babble / colours, real room echo, bursty loss). Each:
    all 3,519 clean genuine clips of half A + 100 per attack. Babble talkers come from half A only.
  - Thresholds: from the pooled half-A genuine calls at genuine-flag budgets 0.5, 1, 2, 5, 10, 15, 20%.
  - Report on half B of call sets v1, babble and v2 (clips never used to set anything), per loss level, noise and codec, and on
    ASVspoof 2021 (other speakers entirely), side by side with today's thresholds (verify >= 0.0060, block >= 0.9958).
  - The served bundle's thresholds change only after you pick the operating points.
- **Run (2026-10-07):** `cal_v1` and `cal_v2` rendered (5,119 calls each: 3,519 genuine of half A + 100 per attack; 272 s / 195 s) and
  scored with the served run 7: EER 3.06% / 10.69% (originals v1 3.06%, v2 10.78%: the half-A sets behave like them). First `cal_v2` render
  failed (with every genuine clip of half A a source, babble had no talkers left); fixed: in half mode babble uses the half's clips,
  never the call's own speaker (test added). Analysis: `audiodf call-thresholds` (`evaluation/call_thresholds.py`), report
  `results/call_thresholds_run7.json`.
- **Results (genuine flagged / fakes caught; call sets = half B only, ASVspoof 2021 = all 67 speakers):**

| threshold | v1 | babble | v2 (held-out noise / echo / loss) | ASVspoof 2021 |
|---|---|---|---|---|
| today's verify 0.0060 | 14.8% / 98.9% | 37.1% / 99.1% | 44.6% / 99.4% | 10.0% / 91.7% |
| verify 0.161 (10% of half-A genuine calls) | 1.8% / 95.6% | 10.7% / 94.3% | 19.2% / 95.9% | 4.6% / 84.6% |
| 0.30 | 0.5% / 92.0% | 5.4% / 88.8% | 9.4% / 88.8% | 3.7% / 82.5% |
| block 0.600 (1% of half-A genuine calls) | 0.1% / 80.2% | 0.8% / 71.0% | 1.6% / 69.5% | 2.7% / 78.1% |
| 0.90 | 0.0% / 56.9% | 0.0% / 42.2% | 0.1% / 39.2% | 1.9% / 71.9% |
| today's block 0.9958 | 0.0% / 35.1% | 0.0% / 21.6% | 0.0% / 19.5% | 1.0% / 60.0% |

  - Today's levels (set on clean phone speech) ask 15-45% of genuine callers to verify on noisy calls and block only 20-35% of fakes.
  - At call-based levels packet loss no longer drives false alarms (v1, 0-5% loss: 1.2-2.7% flagged at verify 0.161, vs 12.6-20.9% today).
  - **Room echo now drives them:** on v2 at verify 0.161, calls without echo flag 0-9% of genuine callers, calls with echo 29-58%
    (false blocks at 0.600: 2-7% with echo, ~0 without). Babble: 13.6% flagged (1.4% without babble).
  - ASVspoof 2021 genuine audio has a high-score tail: no block level above 0.6 gets it below ~1.6% except today's 0.9958 (1.0%).
  - Only the 0.161 / 0.600 rows come purely from half A; the other rows are points on the curve shown for choosing (reading an
    operating point off a reported curve is a product choice, not a fit, but those rows' rates are not independent estimates).
- **Your decision (2026-10-07): verify 0.0060 (unchanged), block 0.90 (was 0.9958)**, applied to `artifacts/bundle.json` (recorded under
  `metrics.threshold_policy`; the previous file is kept in the session scratchpad). Engine check: it loads 0.90 / 0.006032 and maps
  0.006 -> allow, 0.0061 -> verify, 0.899 -> verify, 0.90 -> block.
  - **Why verify stays strict (banking):** a missed fake can empty an account; a verification costs a genuine customer a step-up check.
    Moving verify 0.161 -> 0.0060 cuts fakes passing verify per 1,000 fakes from 44 -> 11 (v1), 57 -> 9 (babble), 41 -> 6 (v2) and
    154 -> 83 (ASVspoof 2021), at 130-260 extra verifications per 1,000 genuine calls. With 1 fake call in 1,000 (assumed), the strict
    level costs ~3,900-7,300 verifications per fraud stopped (~$2,000-3,600 at $0.50 a check), well under the cost of one drained account.
  - **Why block 0.90:** about twice today's catch rate on calls (39-57% vs 20-35%; 72% vs 60% on ASVspoof 2021) at ~0% false blocks on
    simulated calls and 1.9% (vs 1.0%) on real phone lines.
  - **Stated limits:** on real phone lines 8% of fakes still score below the verify level (the model is confident they are genuine:
    no threshold fixes that, only a better model); the numbers come from benchmark and simulated audio, and real voice-cloning tools are
    newer. Recommended deployment: money movement (transfers, payee changes, resets) always gets a step-up check whatever the score;
    the detector decides how strong it is. The verification load on noisy calls (37-45% of genuine callers) is to fall through better
    models (run 8: echo, babble), not looser thresholds. An option for later: per-action levels (strict for money movement, 0.161 for
    enquiries), which needs a small serving change.

### Phase 20: run 8, echo and low-SNR babble robustness (plan and rule written 2026-10-07 before any result)

- **Why:** run 7's weakest spots on held-out call set v2 are real room echo (2.7% without echo, 12.6% with it, 19.9% with noise too),
  babble (13.2%) and 5-10 dB noise (16.6%), and echo is what now drives genuine-caller flags at the call-based thresholds (phase 19).
  Banking policy: keep the strict verify level and cut the verification load through the model.
- **Recipe, rule and flow:** `new_plan.md` 7.3r (the rule is there, with run 7's numbers). Config `src/configs/run8.yaml`; run 7 and run
  8 differ only in echo share (.30 -> .50), echo sources (+ MIT IR Survey, 270 measured spaces, CC-BY 4.0, stretched in time by 0.8-3.0
  for half of them), the noise floor (5 -> 0 dB) and the mix (babble .20 -> .30, colours .25 -> .20, MUSAN .35 -> .30).
- **Code:** `NoiseBank.rir(stretch_p, stretch_range, stretch_corpora)` (random stream untouched when off), `ImpairConfig.rir_stretch*`,
  `DataConfig.noise_mix / rir_corpora / rir_stretch_p / rir_stretch`, `mit_rir` in the bank (`dataset_noise/mit_rir`, from the Hugging
  Face parquet `benjamin-paine/mit-impulse-response-survey-16khz`), `build_kit` refuses the held-out corpora. 155 tests green; a test pins
  run 8's difference to run 7 to exactly those settings.
- **Disk:** the re-render needs ~24 GB (73 GB free); caches of runs 3-7 are kept.
- **Plan after run 8 (your order):** Whisper-encoder branch, then transcript-based intent as a separate module, then LLM explanations
  (`new_plan.md` 7.3r).
- **Training (2026-10-07 ~14:15 to 22:10; render 7,059 s + 322 s, then 8 passes of ~48 min):** tuning EER by pass 7.95, 7.70, 5.88, 6.36,
  4.67, 4.56, **4.50** (pass 7 kept), 4.53%. The tuning set holds more echo and noise than run 7's, so it is not comparable with it. All
  passes saved in `artifacts_run8/epochs/`. Bundle `artifacts_run8/` (not served), report `results/run8/training_report.json`.
- **Results (EER at 10 s, WavLM alone, same clips; 95% paired-bootstrap interval of run 8 minus run 7 where per-clip scores exist):**

| Test set | run 4 | run 7 (served) | run 8 | run 8 minus run 7 | rule for run 8 |
|---|---|---|---|---|---|
| ASV5 eval (30k) | 5.43% | 5.74% | 5.73% | -0.01 | <= 6.24 met |
| ASVspoof2019 eval (30k) | 4.85% | 5.33% | 5.32% | -0.01 | <= 5.63 met |
| ASVspoof 2021 real phone channels (30k) | 8.81% | 9.14% | 9.42% | **+0.28 [+0.06, +0.51]** | <= 9.44 met (by 0.02) |
| Call set v1 | 4.96% | 3.06% | 3.03% | -0.03 [-0.22, +0.25] | <= 3.36 met |
| Babble set | 15.54% | 7.00% | 6.29% | **-0.71 [-1.13, -0.08]** | target <= 6.0 **not met** |
| Call set v2 overall | 16.95% | 10.78% | 10.47% | -0.31 [-0.81, +0.17] | target <= 9.5 **not met** |
| v2 no noise, no echo | 3.98% | 2.72% | 2.41% | | <= 3.2 met |
| v2 echo, no noise | 18.34% | 12.58% | 12.27% | | target <= 9.5 **not met** |
| v2 echo + noise | 30.47% | 19.95% | 19.16% | | target <= 16.5 **not met** |
| v2 SNR 5-10 dB | 26.58% | 16.56% | 15.90% | | target <= 14.5 **not met** |
| Fakes caught with verify at 10% of genuine ASVspoof 2021 calls | | 91.7% | 91.1% | -0.6 | >= 90.7 met |

- **Verdict by the rule fixed before training: 0 of 5 targets met, every regression limit met (ASVspoof 2021 by 0.02 points).** Not
  proposed for serving; `artifacts/` is still run 7. The gains are real but small: babble -0.71 (interval excludes zero), v2 -0.31
  (interval includes zero), at +0.28 on real phone lines (interval excludes zero) and 4.4 points fewer fakes caught at a 1% false-alarm
  budget there (run 7 60.0%, run 8 55.7%).
- **Why the echo did not move (analysis after the result, not a rule change):** the held-out echo rooms are very reverberant (218 real
  rooms, rough RT60 percentiles 10/50/90 = 0.16 / 0.92 / 1.93 s, cut at 1 s), far above the 0.2-0.6 s of typical homes and offices
  (my 270 measured MIT rooms: median 0.3 s). By room reverberation, echo calls without noise (run 4 / 7 / 8): RT60 < 0.3 s 1.8 / 2.5 / 2.5%
  (n=118); 0.3-0.6 s 24.1 / 7.6 / **4.9%** (n=79); 0.6-1.0 s 24.8 / 15.0 / 14.5% (n=186); > 1.0 s 19.8 / 13.4 / 14.8% (n=277).
  With noise: 8.3 / 16.8 / 20.8 / 20.5% for run 7 and 8.9 / 16.5 / 20.8 / 18.7% for run 8. So the training helps rooms of plausible
  size (0.3-0.6 s, small samples) and does nothing for halls; 60% of v2's echo calls are in rooms of RT60 > 0.6 s.
- **Mistake 24:** I fixed the echo targets (12.6 -> 9.5 and 19.9 -> 16.5) without first checking what the held-out rooms look like;
  the RT60 analysis above was done after the result. The targets were guesses that the test set's long rooms made unreachable by
  this recipe. The rule stands as written (run 8 fails it); a better test of "realistic call rooms" would hold out measured rooms
  of 0.2-0.8 s.
- **Epoch diagnostic (your hypothesis: would more epochs of run 8 help? run 2026-10-08 after run 9; diagnostic only, never used to pick a
  checkpoint):** run 8's saved epochs 5, 6, 8 scored next to epoch 7 (the bundle; weights verified identical):

| Run 8 epoch | tuning EER | ASVspoof 2021 | Babble | Held-out v2 |
|---|---|---|---|---|
| 5 | 4.67% | 9.42% | 6.00% | 9.97% |
| 6 | 4.56% | 9.21% | 6.37% | 10.25% |
| **7 (kept)** | 4.50% | 9.42% | 6.29% | 10.47% |
| 8 | 4.53% | 9.33% | 6.33% | 10.16% |

  Range across epochs 5-8: 0.21 / 0.37 / 0.50 points, in no consistent direction. Paired bootstrap, epoch 7 minus epoch 5: ASVspoof 2021
  -0.00 [-0.18, +0.24]; babble +0.29 [-0.10, +0.71]; v2 +0.50 [+0.05, +1.06]; epoch 8 minus 7: -0.09 [-0.24, -0.01], +0.04 [-0.33, +0.21],
  -0.31 [-0.52, -0.02]. **Epochs 5 to 8 are equivalent within a few tenths of a point: the extra passes did not improve test
  performance** (epoch 7 is no better than epoch 5 and slightly worse on v2), which does not support a longer schedule. Choosing epoch 5 or 6 from
  this table would be tuning on test sets and is not done.
- **What run 8 tells us:** more training data of the same kind does not fix echo; the two weak spots are very long reverberation
  (where speech is smeared for a second or more) and echo combined with noise. Whether those matter for real calls is unknown;
  they matter less than typical-room echo, which run 7 already handles at 2.5-7.6%. The next lever is a different model (the Whisper
  branch: different pretraining data, may keep traces WavLM loses), not more of this recipe.

### Phase 21: run 9, the Whisper-encoder branch (plan and rule written 2026-10-07 before training)

- **Why:** step 2 of your flow; run 8 showed more data of the same kind does not close run 7's weak spots (very long reverberation,
  echo + noise, 8% of fakes on real phone lines scored as confidently genuine). A second encoder trained on different audio is the next
  lever. The Whisper-small encoder is as fast as WavLM (measured: 2.4 vs 4.5 ms per window, 1.2 vs 1.9 GB VRAM).
- **Code (162 tests, 6 new):** `models/whisper.py` (encoder over 2 s windows without the 30 s pad; GPU log-mel equal to Whisper's
  feature extractor), `training/train_whisper.py`, branch registered everywhere WavLM is (config `WhisperConfig`, `BRANCHES`, bundle files
  and manifest, `stream_eval.WINDOW_BRANCHES`, serving engine and verdict `whisper_probability`, benchmark, CLI, `run.py
  --whisper-checkpoint`); `transformers` added to `requirements-dev.txt` only (the Docker image needs it only if the branch is served).
  A test caught that the checkpoint did not store the head width (a model trained with another `hidden` could not be loaded): fixed.
  The render-folder tag was made backward-compatible (run 6 / 7 / 8 tags verified: cafd37e9 / a7fd0e14 / 93c5eff0), so run 7's cached
  copies are reused.
- **Recipe and rule:** `new_plan.md` 7.3t; config `src/configs/run9.yaml`.
- **Smoke test (real weights, 150 clips per split, 1 pass, 10.7 min) passed before the real run.** Training 2026-10-07 22:50 to 2026-10-08
  ~04:20, 8 passes of 31-37 min (faster than WavLM's 48): Whisper tuning EER by pass 6.45, 5.93, 5.33, 4.83, 4.96, **4.72** (pass 6 kept),
  4.87, 4.89%. Fusion tuned on the tuning set: WavLM 0.65, Whisper 0.35 (tuning set: WavLM 2.94, Whisper 4.72, fused 2.57%). Run 9's WavLM
  is run 7's, verified weight for weight against `artifacts_run7/wavlm.pt` and `artifacts/wavlm.pt`. Bundle `artifacts_run9/` (not served),
  report `results/run9/training_report.json`.
- **Results (EER at 10 s, same clips; WavLM column = run 7; 95% paired-bootstrap interval of fused minus WavLM):**

| Test set | WavLM (run 7) | Whisper alone | Fused 65/35 | fused - WavLM | rule |
|---|---|---|---|---|---|
| ASV5 eval (30k) | 5.74% | 7.75% | 5.72% | -0.02 | <= 6.24 met |
| ASVspoof2019 eval (30k) | 5.33% | 8.46% | 5.40% | +0.07 | <= 5.63 met |
| ASVspoof 2021 real phone lines (30k) | 9.14% | 13.86% | 9.83% | **+0.72 [+0.39, +1.00]** | <= 8.64 **not met** |
| Call set v1 | 3.06% | 5.47% | 2.96% | -0.10 [-0.30, +0.16] | <= 3.36 met |
| Babble set | 7.00% | 8.31% | 6.21% | **-0.79 [-1.19, -0.33]** | <= 7.30 met |
| Call set v2 | 10.78% | 10.48% | 9.62% | **-1.16 [-1.62, -0.69]** | <= 11.08 met |
| Fakes caught, verify at 10% / 5% / 1% of genuine ASVspoof 2021 calls | 91.7 / 85.3 / 60.0% | 83.1 / 75.9 / 48.1% | 90.3 / 84.3 / 64.0% | | 10%: >= 93.0 **not met** |
| First-verdict latency, batch 1, GPU (`benchmark`) | 51.1 ms | Whisper window 40.3 ms | 91.4 ms | **1.79x** | <= 1.7x **not met** |

  (The fused ASVspoof 2021 figure is 9.83% in the evaluator and 9.86% in the comparison helper, from the rounding of the scores file.)
- **Verdict by the rule fixed before training: not adopted.** The phone-line gain is missed (the fused model is 0.69 points worse than
  WavLM, and catches 1.4 points fewer fakes at the 10% level), and the cost limit is missed (1.79x). Every no-regression limit is met.
  Whisper stays a documented branch; `artifacts/` is still run 7.
- **Where Whisper helps and where it does not (v2, EER %, run 7 / run 8 / Whisper alone / fused):** no noise, no echo 2.72 / 2.41 / 4.60 /
  2.62; noise, no echo 5.94 / 5.45 / 7.54 / 4.96; **echo, no noise 12.58 / 12.27 / 10.91 / 11.82; echo + noise 19.95 / 19.16 / 17.50 /
  17.71**; babble 13.19 / 12.48 / 13.61 / 12.18; 5-10 dB 16.56 / 15.90 / 15.26 / 15.03. Whisper alone is the better encoder wherever there
  is echo (better than run 8, which trained on 40% more echo and measured rooms, without any echo-specific training of its own) and the
  worse one on clean and noise-only audio. Its pretraining on reverberant web audio is the likely reason; this is a hypothesis.
- **Is it complementary on real phone lines?** WavLM passes 8.3% of fakes at its verify level (0.0060); Whisper, at its own
  10%-of-genuine level, flags 24.4% of those. But an "either model flags" verify rule flags 15.9% of genuine calls and catches 93.7%,
  while WavLM alone with its level loosened to flag the same 15.9% catches 94.8%: no gain over simply loosening WavLM. On the call sets
  the either-model rule flags 10 more points of genuine calls for +0.2-0.3 points of fakes caught. Offline, thresholds from ASVspoof 2021.
  Log-score correlation WavLM vs Whisper on ASVspoof 2021: 0.83.

### Phase 22: channel-quality-gated fusion of run 7 and Whisper (plan and rule written 2026-10-08 before any test)

- **Your decisions (2026-10-08):** keep run 7, do not adopt run 8; fuse run 7 with Whisper where Whisper is good; push (without run 8's own files);
  DVC for the weights (remote: DagsHub, pending your account and token). Order: gated fusion, second seed of run 7, transcript intent, LLM explanations.
- **Mistake 25:** the push of 2026-10-08 failed CI (`d561fbe`): two tests read `configs/run8.yaml`, which I had not pushed. Found with a clean
  checkout of the pushed commit, fixed (`1db0270`: recipe inline in the tests, YAML check skipped where the file is absent), clean checkout 162 passed.
  I should have run the suite in a clean checkout before pushing a subset of the files.
- **Design and rule:** `new_plan.md` 7.3v (written before any test).
- **Build (`audiodf/gating/`, 6 tests):** `channel_quality.py` (35 hand-computed features of the first 10 s of speech; two gradient-boosted classifiers, echo and
  noisy; q = the larger probability; fitted on 30,000 training clips with echo / noise known from the impairment plan, a label unrelated to real / fake),
  `fusion.py` (Whisper weight w = w_min + (w_max - w_min) q, grid for the two parameters, chosen on the tuning set), `run.py` (tuning, fit, features, eval).
  A test found that a clip of pure digital silence gave NaN features (division by zero); none of the 100,000+ real clips was affected (0 NaN rows in every
  feature file); fixed. First run of the chain 2026-10-08 12:10-16:00: per-clip scores of run 9 on ASV5 and ASVspoof 2019 (reproduce 5.74 / 7.75 / 5.72% and
  5.33 / 8.46 / 5.40%), tuning set scored by both models, features for the six test sets, estimator, evaluation.
- **Estimator quality (held-out tuning clips, never trained on):** echo accuracy 89.3% (AUC 0.938), noisy 86.9% (AUC 0.891); mean q 0.13 on clean clips and 0.69 on
  degraded ones; on v2's known conditions echo 80.7% (AUC 0.846), noisy 81.9% (AUC 0.872). AUC of q for fake vs real: 0.498 on the tuning set, 0.514 / 0.497 /
  0.541 / 0.511 / 0.504 / 0.511 on ASV5 / ASVspoof 2019 / 2021 / v1 / babble / v2: q carries no label information (my intended ceiling was 0.65; the rule text
  of 7.3v lists it as reported, not as a limit).
- **Result: gate chosen on the tuning set = w_min 0, w_max 0.5** (tuning-set mean EER by setting 4.05-4.21%, all within 0.17 points: the choice is not
  sharp). Table and per-condition numbers in `new_plan.md` 7.3w; files `results/gating/{tuning.npz, features_*.npz, estimator.joblib, gate_eval.json}`,
  `results/chain_gate.log`. Against the rule: ASVspoof 2021 9.46% vs limit 9.29% (**missed by 0.17**; +0.32 [+0.09, +0.54] against WavLM alone), fakes caught at the
  10% verify level 90.9% vs 91.0% (**missed by 0.1**), every other limit met (v2 9.41, babble 6.25, echo-only 11.36, echo + noise 17.03, ASVspoof 2019 5.07,
  ASV5 5.68, v1 2.92, fakes caught at the 1% level 65.5%).
- **Verdict by the rule: not adopted, by 0.17 and 0.1 points.** Compared with the fixed 65/35 mix: the phone-line loss falls from +0.72 to +0.32, all call-set gains stay
  (v2 -1.38 [-1.97, -0.91], babble -0.75 [-1.15, -0.27], ASVspoof 2019 -0.26 [-0.41, -0.12]), fakes caught at the strict 1% level +5.5 points (65.5 vs 60.0%) but
  -0.8 at the 10% level. Whisper takes no weight (< 0.1) on 61-93% of the clean-ish clips and on 26-31% of the noisy call sets.
- **Why phone lines still lose a little (hypothesis, not tested):** the estimator never saw a real phone channel (it was fitted on simulated echo / noise),
  so some real phone lines look degraded to it. A gate that treats real phone channels as clean needs real phone-channel examples for the estimator, and the
  only ones available are the ASVspoof 2021 test calls: using them would break the rule that nothing is tuned on a test set. Options for you: keep run 7 alone;
  adopt the gated model as an override (it is better on every noisy call set, slightly worse on real phone lines); or fix a gate v2 whose estimator is also
  fitted on other real phone-channel audio (none is available today).

### Phase 23: second seed of run 7, the run-to-run spread (2026-10-08; measurement, rule in `new_plan.md` 7.3w written before training)

- **Run:** `configs/run7_seed1.yaml` = run 7 with `wavlm.seed` 1 (head / layer-mix initialisation, window sampling, augmentation draws); cached copies reused. Training
  ~13:30 to 19:50, 8 passes of 46-50 min; tuning EER by pass 4.83, 5.68, 3.68, 4.07, 3.48, **2.83** (pass 6 kept), 3.00, 3.09% (seed 0: 5.56, 5.15, 3.64, 4.78, 3.39, 3.28, 2.95,
  2.94 with pass 8 kept). Bundle `artifacts_run7_seed1/`, report `results/run7_seed1/training_report.json`, scores `results/evaluate_*_run7s1*`, comparison `results/run7_seed_spread.json`.
  (My earlier clock times for this run, "16:25 start, 23:00 end", were guesses; the real ones are above.)
- **Results (EER at 10 s, WavLM alone, same clips; 95% paired-bootstrap interval of seed 1 minus seed 0; avg = mean of the two seeds' scores):**

| Test set | seed 0 (run 7) | seed 1 | seed 1 - seed 0 | average of 2 seeds | avg - seed 0 |
|---|---|---|---|---|---|
| ASV5 eval | 5.74% | 5.75% | +0.01 | | |
| ASVspoof2019 eval | 5.33% | 5.48% | +0.15 | | |
| ASVspoof 2021 real phone lines | 9.14% | 9.14% | +0.00 [-0.28, +0.23] | 8.97% | -0.16 [-0.37, +0.04] |
| Call set v1 | 3.06% | 3.07% | +0.01 [-0.18, +0.31] | 2.81% | -0.25 [-0.32, +0.04] |
| Babble | 7.00% | 7.56% | +0.56 [-0.12, +1.08] | 7.12% | +0.12 [-0.25, +0.58] |
| Held-out v2 | 10.78% | 11.14% | +0.36 [-0.16, +0.86] | 10.83% | +0.05 [-0.33, +0.38] |

  v2 by condition (seed 0 / seed 1 / average): no noise, no echo 2.72 / 1.88 / 2.41; echo only 12.58 / 12.74 / 11.81; noise only 5.94 / 6.25 / 5.66; **echo + noise
  19.95 / 21.30 / 20.21**; babble 13.19 / 14.09 / 13.43. (ASV5 and ASVspoof 2019 are from the training logs: the per-clip scores of the first seed were not saved.)
- **Reading:** on clean-speech sets the two seeds agree to 0.00-0.15 points; on noisy sets they differ by 0.36-0.56 on whole sets and up to 1.35 inside a condition (echo + noise).
  Every seed-to-seed interval includes zero (the test clips are the noise that the interval measures; the seed adds to it). **Largest whole-set difference: 0.56 (babble).**
  This is the yardstick for earlier comparisons: run 7 vs run 4 on ASVspoof 2019 (+0.48 / +0.63 against 0.15) and ASVspoof 2021 (+0.33 against 0.00) are larger than the spread on
  those sets, so a real cost of the noise recipe; run 8's babble gain (6.29% vs seeds 7.00 / 7.56) is larger than the spread, its held-out v2 gain (10.47% vs 10.78 / 11.14)
  is within it, and its ASVspoof 2021 loss (+0.28 against a spread of 0.00) is not; the gated Whisper fusion's v2 gain (-1.38) is well beyond it and its phone-line loss
  (+0.32) is too. Differences below ~0.5 points on noisy sets should not be read as real without a second seed.
- **A second WavLM seed as an ensemble:** averaging the two seeds gains -0.16 / -0.25 on ASVspoof 2021 / v1 (intervals touch zero) and nothing on babble and v2: the opposite of
  the Whisper fusion (gains on noisy calls, loss on phone lines). Both cost about twice the single-model compute.

### Phase 24: the model is final (your decision, 2026-10-08)

- **Final model: run 7 seed 0, WavLM-Base+ alone** (`artifacts/`, verify >= 0.0060, block >= 0.90), chosen on the deciding measure you set: fakes missed at the verify
  level (`new_plan.md` 7.3x). It is best or tied on every set except real phone lines, where run 4 misses 3 fewer fakes per 1,000 but misses 75 per 1,000 on noisy v2 calls
  against run 7's 6. Whisper (alone, fixed or gated) and a second WavLM seed stay documented options; none is served.
- **Next (your request):** an explanation agent on a free-tier LLM, fed by SHAP attributions of the served model's decision; then the remaining work (`new_plan.md` 7.3y).

### Phase 25: explanation agent (SHAP + free-tier LLM), built 2026-10-08 for demos

- **Your decisions:** the model is final (run 7 seed 0); explanations by an LLM on a free tier, fed by SHAP; a project, so a free cloud API is fine for demos.
- **Design (`src/audiodf/explain/`):** the explained audio is the first 10 s of speech, prepared as the serving path does. `attribution.py`: 2 s time slices x 4 bands (below
  300 Hz, 300 Hz-1 kHz, 1-3 kHz, 3-8 kHz) = at most 20 regions; a region is removed by attenuating its short-time-spectrum cells by 30 dB; the audio is re-scored exactly as
  the serving path does (2 s windows, 1 s hop, mean, bundle weights); KernelSHAP (160 evaluations) gives each region's contribution; a faithfulness check compares the score
  drop when the 3 top regions are removed with the mean drop for 3 random regions. `explainer.py` builds a facts record (numbers and labels only: verdict, thresholds, top
  regions, window scores, channel estimate, check) and runs the explainer; `llm.py`: Groq / Gemini / OpenRouter through their OpenAI-compatible APIs (key in
  `AUDIODF_LLM_API_KEY`, never stored), every number in the LLM's answer must match a number in the facts, any failure falls back to a fixed template; `plot.py`: a
  spectrogram with each region's contribution and the window scores. CLI `audiodf explain AUDIO [--provider] [--plot] [--out]`, API `POST /explain`. The LLM never sees audio,
  never decides, and runs only when asked (about 10 s per call on this GPU, for flagged calls and demos).
- **Checks on real calls:** on a fake (A32, mu-law, brown noise; action verify) and a genuine call (allow) the explained-audio score equals the served verdict (0.7789 / 0.77889;
  0.0047 / 0.00474) and the faithfulness check passes (fake: removing the top 3 regions drops the score by 0.67 against 0.33 for random regions). Demo files in
  `results/explain_demo/`.
- **Found and fixed while building:** the channel estimate showed "echo 0.80" for a call with no echo. Measured on the call sets: the echo estimate is above 0.5 for 7.5% of
  echo-free calls (up to 15% on G.711 / GSM), and a low value is no proof of absence (11-21% of echo calls score <= 0.05-0.2). The facts now say only "likely" (p >= 0.9:
  wrong on < 1% of echo-free and < 0.3% of noise-free calls) or "not established", never a probability and never "unlikely". Small scores were printed as "0.00" against a
  verify level of 0.0060; now 4 decimals.
- **Tests:** 7 new (a stand-in model whose score is known checks that SHAP finds the right region, that the values add up, the faithfulness check, that the facts hold no
  audio or caller data, that an LLM answer with an invented number or any failure falls back to the template, and the API endpoint); 176 tests green. `shap` and `matplotlib`
  added to `requirements.txt`.

### Phase 26: early decisions and calibration of the final model (2026-10-08; from the saved per-time scores, no new scoring)

- **Early decisions (served levels fixed: verify 0.0060, block 0.90; `results/early_decisions_run7.json`).** Fakes missed at the verify level per 1,000 after 2 / 4 / 6 / 8 / 10 s
  of speech: phone lines 88 / 83 / 83 / 83 / 83 (clips are short: the decision is fixed by 4 s); v1 34 / 15 / 11 / 9 / 9; babble 28 / 12 / 9 / 8 / 8; v2 31 / 11 / 8 / 6 / 6. Genuine
  callers blocked per 1,000: v2 19.1 / 3.1 / 1.2 / 1.2 / 1.2; babble 7.5 / 0.4 / 0 / 0 / 0; v1 2.3 / 0.4 / 0 / 0 / 0; phone lines 19.8 then 19.2. **A verify decision after ~6 s is
  almost as good as at 10 s; a block before ~4-6 s blocks many more genuine callers** (the early score rests on one or two windows).
- **Calibration (`results/calibration_run7_scores.json`).** The raw score is not a probability: 0.0060 already flags 10% of genuine phone-line callers. Platt scaling fitted
  on the tuning set only, calibrated = sigmoid(1.314 logit(score) + 1.594): expected calibration error on phone lines 0.135 -> 0.113, on v2 0.098 -> 0.034 (each set re-based to
  its own share of fakes). A probability also needs the real share of fraud, which is unknown; the likelihood ratio of a flag does not: verify 9.2 / block 37.4 on phone lines,
  verify 2.3 / block 318 on v2 calls.
- **What a flag means when fraud is rare (Bayes, from those ratios).** On phone lines, if 1 call in 1,000 is a deepfake: P(fake | verify flag) = 0.9%, **P(fake | block) = 3.6%**,
  i.e. about 96% of blocked phone-line calls would be genuine customers (the block level blocks 1.92% of genuine phone-line callers). On v2 calls P(fake | block) = 24% at the
  same rate. **Consequence: the block level must not be a hard reject on its own; "block" should mean escalation to the strongest check (e.g. agent callback). A hard reject
  would need a channel-aware block level.** The verify policy stands (a cheap step-up check is exactly what a 1-in-100 precision calls for). This was not examined when
  block 0.90 was chosen in phase 19 (mistake 26).

### Phase 27: out-of-domain test on In-the-Wild (2026-10-08; plan `new_plan.md` 7.3z written before scoring; scored once, nothing changed after)

- **Data:** the authors' upload (`mueller91/In-The-Wild`, CC-BY-SA 4.0), 31,779 clips of 54 public figures; 31,668 with speech (11,806 fake, 19,862 genuine), median 3.2 s.
  `dataset_itw/release_in_the_wild` (zip removed after a verified extraction; it re-downloads in ~2 min). Loader: dataset `itw` (`data/protocol.py`, `AUDIODF_ITW`).
- **Bug found and fixed on the way:** the first scoring run failed on every clip: `SplitIndex.path` rebuilt paths as `<id>.flac`, and In-the-Wild ships `.wav` (index building worked
  because it used the protocol's real paths). The index now keeps each folder's file type; a test indexes and reads a small WAV set; all 81,232 real windows read.
- **Results (EER at 10 s; each model at its own verify level that flags 10% of genuine ASVspoof 2021 callers; `results/itw_report.json`):**

| Model | EER | fakes missed at verify, per 1,000 | genuine asked to verify, per 1,000 | at block 0.90 |
|---|---|---|---|---|
| **Final model (run 7 WavLM)** | **4.47%** | **16** | 89 | 601 fakes / 9.2 genuine blocked per 1,000 |
| Whisper alone | 12.95% | 168 | 101 | |
| Fixed fusion 65/35 | 5.23% | 29 | 85 | |

  - The served thresholds carry over: the verify level flags 8.9% of genuine speakers here (designed 10% on phone lines); per speaker (38 with >= 50 genuine clips) 2.9-31.4%,
    median 8.3%.
  - Clip length: under 4 s of speech (21,092 clips) EER 5.48%, 25 fakes missed per 1,000; 4 s or more (10,576) EER 1.91%, 4 per 1,000.
  - Per speaker (37 with >= 50 fakes): median 0.3% of fakes missed, but **Adam Driver 42% (58 of 137)**, 31% of all 186 missed fakes; next Barack Obama 8% (25 of 331), Donald
    Trump 6% (9 of 155). Without Adam Driver, 11 fakes missed per 1,000. Likely one convincing deepfake source for that voice (not examined).
- **Reading:** on real-world deepfakes made with tools none of the training data contains, the final model holds up (4.47% EER, 16 per 1,000 missed at the served level),
  better than on the lab's real phone lines (9.14%); Whisper and the fusion do worse here too, which agrees with the final choice. Caveats: In-the-Wild deepfakes date from
  around 2020-2022 and are mostly clean studio-like audio; newer cloning tools and phone channels are harder (phone lines: 83 missed per 1,000).

### Phase 28: serving cost of the final model (2026-10-08; `src/audiodf/evaluation/serving_cost.py`, `results/serving_cost.json`)

- **Method:** a live call needs one 2 s window scored per second (1 s hop), so the calls one device can carry is its window throughput. Timed on this laptop (GPU 4.3 GB,
  16 CPU threads) with real call windows; decision check on 300 random calls of set v1 (first 10 s of speech), actions (allow / verify / block) against GPU float32.

| Variant | ms per window, batch 1 | windows / s (best batch) | live calls per device | actions changed / 300 | max score difference |
|---|---|---|---|---|---|
| GPU float32 | 42.7 | 115.8 (32) | ~115 | reference | |
| **GPU float16, autocast (served)** | 53.4 | **236.6 (32)** | **~235** | 1 | 0.0099 |
| CPU float32 | 236.9 | 5.6 (8) | ~5 | 0 | 0.0000 |
| CPU int8, dynamic quantisation of Linear layers | 269.1 | 6.2 (8) | ~6 | **75** | 0.7255 |

- **Reading:** serving needs a GPU: float16 doubles the throughput of float32 for one changed action in 300 (a score near a threshold); one laptop-class GPU carries ~235
  simultaneous calls. A CPU carries ~5 calls; int8 dynamic quantisation gives no speed-up on this CPU and changes a quarter of the actions: **rejected**. Batch-1 latency
  (one call alone) is 43-53 ms on the GPU, well inside the 1 s between windows. Not measured: server GPUs (T4 / L4 / A10), CPU with ONNX Runtime or OpenVINO, layer pruning or
  distillation.

### Phase 29: the high level escalates instead of blocking (your decision, 2026-10-09)

- Action names: `allow` / `verify` / **`escalate`** (was `block`). Escalate = route to the strongest check (an agent callback on the number on file, in-branch verification);
  the system never rejects a call by itself. Reason: phase 26 (at 1 deepfake in 1,000 calls, ~96% of high-level phone-line calls would be genuine). Thresholds unchanged
  (verify 0.0060, escalate 0.90). Changed: `risk.py`, the explanation facts / text / picture, API and README docs, 3 tests. The analysis tools (calibration, call thresholds,
  serving cost) keep calling the high threshold "block" so earlier reports stay comparable.
- Your other decisions the same day: push (done, phase 29 commit); **transcript-based scam intent: keep** (demo-grade module next); deployment only after these.

### Phase 30: transcript-based scam intent, the rules (2026-10-09; plan `new_plan.md` 8.0 written before any rule or test data)

- **Data:** BothBosu/scam-dialogue (Apache-2.0; train 1,280 / test 320 synthetic phone dialogues, half scam; scam types refund, reward, Social Security, tech support;
  normal types delivery, insurance, telemarketing, wrong number) and shakeleoatmeal/phone-scam-detection-synthetic (MIT; test 180, never looked at before the measurement;
  its normal calls are *legitimate* refund / Social Security / support calls). `dataset_intent/`. Only the caller's turns are read.
- **Rules (`src/audiodf/intent/rules.py`):** 8 intent categories (credential request 0.6, remote access 0.6, payment redirection 0.5, bypassing verification 0.5, secrecy 0.35,
  authority impersonation 0.3, lure 0.3, urgency / threat 0.25); score = 1 - prod(1 - weight); caution >= 0.3, high >= 0.6; each match keeps its phrase. Written on the train
  split only: first pass AUC 0.964 (scam 92.7% / normal 3.3% at caution or high); one round of fixes on train ("verify **our** credentials" is the caller offering theirs;
  refund lures "refund you $500"; "account info") -> train AUC 0.981, scam 96.1% / normal 1.2% at caution or high, 86.6% / 0.6% at high. Then frozen.
  The fix itself first carried a literal backspace character where `\b` was meant (the scripted-escape slip of mistake 13 again); found by inspecting the bytes, fixed.
- **Results (measured once, rules unchanged; `results/intent_rules_eval.json`):** BothBosu test: AUC 0.989; caution or high: scam 96.9%, normal 0.6%; high: scam 88.8%, normal
  0.6%. **shakeleoatmeal test (independent): AUC 0.790; caution or high: scam 90.0%, normal 82.2%; high: scam 83.3%, normal 37.8%** (normal refund 10%, Social Security 40%,
  support 63% at high); scams by style: direct 94%, somewhat subtle 76%, very subtle 77% at high.
- **Reading:** the rules find scam *patterns* reliably, but a legitimate call about the same subject uses the same words (a real agency also asks you to verify your identity),
  so on the independent set they flag most legitimate calls. The pre-set demo bar (<= 5% of normal calls at high) is met on BothBosu and **missed on the independent set**.
  Phrase rules alone cannot separate a scam from a legitimate call on the same topic. Consequences: intent never decides alone (as planned); in a bank, money movement
  already gets a step-up check, so the extra friction falls on calls that need one anyway; context-aware classification (the LLM path) is the part that could separate them,
  to be measured once a free API key is set. Both sets are synthetic and scammer-calls-victim; our case is a fraudster calling the bank.

### Phase 31: the intent module around the rules (2026-10-09)

- `intent/asr.py`: local Whisper-small speech-to-text with timestamps (cached weights; the transcript never leaves the machine). On real In-the-Wild clips: first call 25 s
  (model load), then ~1.7 s per clip on this GPU; correct short transcripts.
- `intent/llm_intent.py`: optional context-aware labels from a free-tier LLM (same client as the explanations), with bounded power because the transcript is untrusted: labels
  only from the fixed list, each with a quote of at least 3 words found word for word in the transcript (else dropped); an "unlikely" judgement lowers a rules "high" to
  "caution" and never further; any failure leaves the rules' result. Not yet measured (needs your free API key; `python -m audiodf.intent.evaluate --provider groq --pause 2`).
- `intent/policy.py`: intent raises the check and never lowers the voice action: allow + high intent -> verify; verify + high -> escalate; "caution" is context only (the
  rules flagged most legitimate calls on the independent set).
- `intent/call.py`, CLI `audiodf intent --audio FILE | --text "caller: ..."`; `intent/evaluate.py` reproduces phase 30 exactly. 16 tests (rules, Hinglish phrases, timestamps,
  the policy table, verbatim-quote validation, an injected "ignore your instructions" transcript held at caution, failures, the CLI); 195 tests green.

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
- **(2026-10-07) Verify stays strict for banking** (0.0060): a missed fake can empty an account, a verification costs a step-up check; break-even ~4,000-7,300
  verifications per fraud stopped at an assumed 1-in-1,000 fraud rate.
- **(2026-10-08) Run 8 not adopted; run 7 seed 0 is the final model;** the deciding measure is fakes missed at the verify level, not EER.
- **(2026-10-08) Whisper and the gate are not served** (both rules missed); shadow mode is the way to collect live evidence if it is ever wanted.
- **(2026-10-09) The high level escalates instead of blocking** (phases 26 and 29).
- **(2026-10-09) Transcript intent kept;** it never decides alone and never lowers the voice action; the LLM may not lower a rules "high" below "caution".
- **(2026-10-08) Explanations use a free cloud LLM for demos** (your call; a local model would be the private choice in production); the LLM sees numbers and labels only.

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
22. Phase 17 slips, all caught: (a) I claimed the v2 loss bursts were longer than anything in training while the shortest (3 packets)
    overlapped training's longest (4); my own test caught it and I lengthened them to 5-12; (b) twice the on-screen paired-bootstrap interval
    had the sign of its bracket flipped (the tables in this log are the corrected ones; the denoising table was written with the
    right signs); (c) the vacuous `or True` assertion appeared a third time, in the impairment tests, and was replaced before running;
    (d) the run 6 log message first listed the held-out noise corpora among the sources although training does not use them (they sit
    in the same noise bank), fixed to name only what training uses, with a test that no training draw can name a held-out corpus;
    (e) I could not guard against the dataset folder being renamed during the run (your rename crashed run 6 at 34k copies); the render
    resumed without loss.
21. While adding phase 15 I replaced the heading "## 3. Decisions and the reasoning behind them" with "## 3. Decisions"
    (pushed in `4a69e80`); noticed when a later edit would not match, restored in phase 16. In the call test the vacuous
    `or` assertion from mistake 20 reappeared once in the packet-loss test; replaced with a check that fails (the test
    still passes, now meaningfully).
23. Run 7's decision rule (written 2026-10-06) allowed +0.3 points on ASVspoof2019 (limit 5.15%), while run 6's framework had
    allowed +0.5 (5.35%); I tightened it without saying so. Run 7 scored 5.33%: it fails the rule as written and would have passed
    the old one. Reported both ways in phase 18; the rule is not changed after the fact.
24. Run 8's echo targets were set without looking at the held-out rooms' reverberation times (median ~0.9 s, a third above 1 s,
    far beyond typical call rooms), so they were not reachable by training on more echo; found afterwards and reported in phase 20.
25. I pushed a subset of files that excluded `configs/run8.yaml` while two tests read it; CI failed (`d561fbe`). Fixed by `1db0270`; the pushed commit is now
    tested in a clean checkout before pushing a subset (phase 22).
26. When the block level 0.90 was chosen (phase 19) I compared catch rates and false-block rates but not what a block means at a realistic fraud
    rate: on real phone lines about 96% of blocked calls would be genuine if 1 call in 1,000 were a deepfake. Found in phase 26; block should escalate, not reject.

## 5. Known limitations (current)

- **No real bank calls.** Every call test is simulated (ASV5 clips through VoIP codecs, noise, echo, loss) or from public benchmarks; a small set of real recordings would
  be the best final check.
- **Real phone lines are the weakest condition:** 83 fakes missed per 1,000 at the verify level (ASVspoof 2021, attacks from 2019); newer cloning tools may be harder.
- **The escalate level on real phone lines** reaches 1.92% of genuine callers; at realistic fraud rates most escalations are genuine customers (hence escalate, never reject).
- **Verification load on noisy calls:** the strict verify level asks 37-44% of genuine callers on babble / echoey simulated calls to verify.
- **Very reverberant rooms and echo + noise** remain hard (v2 echo + noise 19.95%); neural codecs too (ASV5 C04 / C07 15-20%).
- **Scam intent** cannot tell a scam from a legitimate call on the same subject (37.8% of legitimate calls flagged at high on the independent set); the datasets are synthetic,
  English and scammer-calls-victim; Hindi / code-mixed speech-to-text is untested.
- **Explanations** show which parts of the audio drove the score, not why the voice is synthetic; the channel estimate is shown only when "likely" (p >= 0.9).
- **A GPU is required** for more than a handful of simultaneous calls.
- **Only 2 of 10 ASV5 eval tars**; the 30k-clip subset is used for every comparison. ASV5 eval and the call sets have been used to compare candidates: never tune on them.
- **The model weights exist only on this disk** until the DVC remote is set up.

## 6. Inventory

- Code: `src/audiodf/` (data, features, models, training, evaluation, inference, serving, streaming, monitoring, explain, intent, gating), `src/run.py`, `src/tests/`
  (195 tests), `src/configs/` (default, run6, run7, run7_seed1, run9; run8 local only), `src/deploy/` (Dockerfile, compose, k8s, Prometheus, Grafana),
  `.github/workflows/ci.yml`, `.dvc/`.
- Docs: `new_plan.md` (section 0 = the resume point), `audit.md`, `src/README.md`.
- Models (git-ignored; DVC-tracked locally except the seed-1 bundle): `artifacts/` = **final (run 7 seed 0)**; `artifacts_run7/` (+ 8 passes), `artifacts_run7_wavlm_only/`,
  `artifacts_run7_seed1/`, `artifacts_run8/` (+ passes), `artifacts_run9/` (WavLM + Whisper, + passes), `artifacts_run6/`, `artifacts_run5_encodec/`,
  `artifacts_run4_wavlm(_only)/`, `artifacts_run3_codecs/`, `artifacts_run2_pooled/`, `artifacts_run1_asv5/`, `artifacts_prev_asv2019/`.
- Results: `results/` (training reports per run, per-clip score CSVs per run and test set, `fakes_missed_at_verify_all_runs.json`, `ensemble_table_seeds_whisper.json`,
  `run7_seed_spread.json`, `gating/`, `explain_demo/`, `itw_report.json`, `early_decisions_run7.json`, `calibration_run7_scores.json`, `serving_cost.json`,
  `intent_rules_eval.json`, `intent_eval_rules.json`, calibration and call-threshold reports, logs).
- Data and cache: see `new_plan.md` 0.4.

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

Added this session:
```
python run.py --config configs/run7.yaml --eval-utts 30000                     # the final model's recipe (seed 0); configs/run7_seed1.yaml = seed 1
python run.py --config configs/run9.yaml --wavlm-checkpoint ../artifacts_run7/wavlm.pt --eval-utts 30000   # Whisper branch next to run 7
python -m audiodf render-calls --speaker-half A --n-genuine 0 --n-per-attack 100 --seed 5 --out ../dataset_calls_cal_v1   # calibration calls
python -m audiodf render-calls-v2 --speaker-half A --n-genuine 0 --n-per-attack 100 --seed 5 --out ../dataset_calls_cal_v2
python -m audiodf call-thresholds --cal <protocol,scores> ... --test v1=<protocol,scores> ... --stored ../artifacts --tag T
python -m audiodf evaluate --dataset itw --eval-utts 0 --bundle ../artifacts_run9 --tag itw_run9 --save-scores   # In-the-Wild
python -m audiodf.gating.run tuning|fit|features|eval ...                      # gated fusion experiment
python -m audiodf.evaluation.serving_cost --clips 300                          # serving cost (AUDIODF_CALLS=../dataset_calls)
python -m audiodf explain call.wav [--provider groq] [--plot out.png]          # explanation (AUDIODF_LLM_API_KEY for an LLM)
python -m audiodf intent --audio call.wav | --text "caller: ... receiver: ..." [--provider groq]
python -m audiodf.intent.evaluate [--provider groq --pause 2]                  # intent evaluation (phase 30)
```

## 8. Open items

| Item | Status |
|---|---|
| Runs 1-9, run 7 seed 1, gated fusion, In-the-Wild, calibration, early decisions, serving cost | done (phases 6-28) |
| Final model | **done: run 7 seed 0** (your decision 2026-10-08) |
| Block -> escalate | done (phase 29) |
| Explanation agent | done (phase 25); the LLM demo needs your API key |
| Transcript intent | rules + module done (phases 30-31); **LLM path unmeasured** (rule fixed in `new_plan.md` 0.2 item 3) |
| Deployment: Docker runtime, Kafka end to end, Grafana, Kubernetes | **waiting for your go** (`new_plan.md` 0.2 item 1) |
| `configs/default.yaml` relative paths | open, minor (part of deployment) |
| DVC remote (DagsHub) + `artifacts_run7_seed1` tracking | **waiting for you** (`new_plan.md` 0.2 item 4) |
| CI results of `3d0c21a` and the final commit | to read (API rate limit) |
| Real bank call recordings | not available |
| Full ASV5 eval (8 more tars, ~68 GB) | optional |
| Final docs (model card), GitHub cleanup | at the very end (your decision) |

## 9. Next steps (updated 2026-10-09, end of session)

The resume point, every open decision with a recommendation, optional work and the rules to keep are in **`new_plan.md` section 0 (RESUME HERE)**. In short:
your go for the deployment stage; a free LLM key for the demos and the LLM intent measurement (its rule is fixed); the DVC remote; then final docs and the GitHub cleanup.

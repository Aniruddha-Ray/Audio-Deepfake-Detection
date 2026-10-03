# Audit log: Real-Time Deepfake Voice Detection

Running record of what was done, what was found, what turned out wrong, and what is open.
Last updated: 2026-10-03. Companion docs: `new_plan.md` (plan and analysis), `src/README.md` (usage).

## 1. Current state at a glance

| Item | State |
|---|---|
| Codebase | `src/audiodf/` (30 modules), 88 passing tests. Last commit `a839261`; **41 files changed since, uncommitted**. |
| Trained model | `artifacts/` = run 3 (pooled data + real-codec augmentation). **Not production-ready.** Runs 1-2 kept alongside. |
| Previous model | `artifacts_prev_asv2019/` (ASV2019 prototype, feature v1, cannot load in current code). Kept, git-ignored. |
| Best honest number | ASV5 eval fused EER **29.4%** (run 3; run 2 31.6%, run 1 33.0%). ASV2019 eval is no longer a cross-dataset test. |
| Deployment | FastAPI tested for real; **Kafka, Docker, Kubernetes, Grafana scaffolds untested**. No real phone-call audio. |
| Disk | C: 87.7 GB free. Data: `dataset/LA` 7.1 GB, `dataset5/` ~75 GB (train, dev, 2 of 10 eval tars). |
| Next | WavLM front end (run 3 missed the 28.6% significance line; phase 10). |

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

## 3. Decisions and the reasoning behind them

- **Train on ASV5 only, drop ASV2019 (run 1).** Chosen because ASV5 dev gives attack-disjoint tuning and ASV2019 has a silence
  shortcut. The cross-dataset result (37.8%) now weakens this: mixed training is coded and tested but not run.
- **Codec augmentation is label-blind** (augmenter takes no label), so it cannot create a "codec means real/fake" shortcut.
- **Fusion weight tie-break:** among weights within 0.05 EER points of the best, pick the one closest to 0.5.
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

## 5. Known limitations (current)

- Weak generalisation to unseen attacks and real codecs (section 2, phase 6). Dev is not a reliable proxy for eval.
- ASV5 clips are crowdsourced speech, not live VoIP audio (packet loss, DTX, echo untested).
- Simulated codecs cover Opus/MP3/Vorbis/G.711/narrowband only; AMR, Speex, EnCodec, AAC, Bluetooth device channels are not simulated.
- SVM recomputed per verdict (~41-49 ms per 10 s buffer); first-verdict compute ~22-27 ms on GPU.
- ASV5 eval has been used to compare candidates (baseline now); never tune on it.
- Cross-dataset (ASV5 -> ASV2019) near chance; reverse direction untested.

## 6. Inventory

- Code: `src/audiodf/` (data, features, models, training, evaluation, inference, serving, streaming, monitoring), `src/run.py`, `src/tests/` (6 test files), `src/deploy/`, `.github/workflows/ci.yml`.
- Docs: `new_plan.md`, `src/README.md`, `audit.md`.
- Results: `results/training_report.json`, `evaluate_asv5_eval.json`, `data_integrity_asv5.json`, `data_integrity_asv19.json`, `segmented_baseline_bilstm.json`, logs (`train_asv5.log`, `asvspoof5_download.log`, ...).
- Cache (outside the repo): `~/.cache/audiodf/prep_vad-45_0.025_0.05_svmv2_melv1/` (indexes + SVM snapshots, 0.38 GB). The obsolete
  ASV2019 v1 feature caches (`seg2s_hop1s*`, 8.60 GB) were **deleted on 2026-10-03** after checking that nothing in `src/` reads them (only
  `experiments/segmented_baseline.py` did; it rebuilds them in ~28 min). Their one model file, the prototype's 5-epoch RCNN, was kept as
  `artifacts_prev_asv2019/experiment_5epoch_cnn_bilstm.pt`. Free disk after cleanup: 87.7 GB.

## 7. Reproduce

```
cd src
python run.py                              # train on ASV5 (audit, index, SVM, RCNN, tune, test)
python run.py --train-on asv5 asv19        # mixed training
python -m audiodf audit --dataset asv5     # integrity audit
python -m audiodf evaluate --dataset asv5 --split eval --eval-utts 30000
python -m audiodf serve --port 8000        # API
python -m pytest                           # 69 tests
```

## 8. Open items

| Item | Status |
|---|---|
| Run 2: pooled data + held-out-attack tuning (options 1+2 combined) | done: ASV5 eval 31.6% (run 1 33.0%) |
| Real-codec augmenter via ffmpeg (encoders confirmed available) | done: run 3, 29.4% (not significant) |
| Pretrained speech front end (WavLM) | **next** |
| Investigate A12 inversion and the cross-dataset collapse | open |
| Download the remaining 8 ASV5 eval tars (~68 GB; needs space) for the final number | optional |
| Delete the obsolete ASV2019 cache | done (8.60 GB freed) |
| Kafka / Docker / Kubernetes / Grafana verification | not started |
| Real call audio (VoIP) evaluation | not started |
| Commit the 41 uncommitted changes | not requested |

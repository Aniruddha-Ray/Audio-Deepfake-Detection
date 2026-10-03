# Audit log: Real-Time Deepfake Voice Detection

Running record of what was done, what was found, what turned out wrong, and what is open.
Last updated: 2026-10-03. Companion docs: `new_plan.md` (plan and analysis), `src/README.md` (usage).

## 1. Current state at a glance

| Item | State |
|---|---|
| Codebase | `src/audiodf/` (29 modules), 69 passing tests. Last commit `a839261`; **41 files changed since, uncommitted**. |
| Trained model | `artifacts/` = run 1 (ASV5 train only, feature version SVM v2 / RCNN v1). **Not production-ready.** |
| Previous model | `artifacts_prev_asv2019/` (ASV2019 prototype, feature v1, cannot load in current code). Kept, git-ignored. |
| Best honest number | ASV5 eval fused EER **33.0%**; ASV2019 eval (cross-dataset) 37.8%; dev 15.9% (not trustworthy as a proxy). |
| Deployment | FastAPI tested for real; **Kafka, Docker, Kubernetes, Grafana scaffolds untested**. No real phone-call audio. |
| Disk | C: 85 GB free. Data: `dataset/LA` 7.1 GB, `dataset5/` ~75 GB (train, dev, 2 of 10 eval tars). |
| Waiting on | Your choice of next experiment (section 8). |

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
- Caches (outside the repo): `~/.cache/audiodf/prep_vad-45_0.025_0.05_svmv2_melv1/` (indexes + SVM snapshots) and **8.8 GB of obsolete ASV2019 v1 features** (safe to delete, awaiting your OK).

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
| Next experiment: (1) held-out-attack tuning + train on ASV5 train+dev, (2) mixed ASV5 + ASV2019, (3) real-codec augmenter via ffmpeg, (4) pretrained speech front end | **awaiting your decision** |
| Investigate A12 inversion and the cross-dataset collapse | open |
| Download the remaining 8 ASV5 eval tars (~68 GB; needs space) for the final number | optional |
| Delete the 8.8 GB obsolete cache | needs your OK |
| Kafka / Docker / Kubernetes / Grafana verification | not started |
| Real call audio (VoIP) evaluation | not started |
| Commit the 41 uncommitted changes | not requested |

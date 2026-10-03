# Replan: Real-Time Deepfake Voice Detection

Base: the existing prototype (`AudioDeepfakeModel.ipynb`: SVM + CNN-LSTM, weighted-average ensemble on ASVspoof2019 LA) and the architecture in `BEing-Techies_PSB_HACKATHON_2026.pptx`.

> **Status update:** Phases 0, 1, 3 and most of 4 are done — see `src/README.md` for the actual
> codebase, confirmed metrics and known limitations. Section 7 below records what we learned
> doing that work and what's still open, including the plan to add ASVspoof5 for generalisation
> (the biggest remaining gap). Sections 1-6 below are the original replan and are kept for context
> on what was decided and why; they are no longer the task list.

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

---

## 7. What we learned building it (post-implementation notes)

This is not a hackathon deliverable anymore — the goal stated explicitly is production-level
performance. That changes several defaults below from "good enough to demo" to "must hold up
on attacks and audio we've never seen."

### 7.1 Confirmed results (ASVspoof2019 LA only, `src/` codebase, RCNN = CNN-BiLSTM)

| | Dev EER (seen attacks A01-A06) | Eval EER (unseen attacks A07-A19) |
|---|---|---|
| SVM (whole-utterance MFCC+LFCC+MGDCC) | 0.24% | 8.58% |
| RCNN, 5 epochs | 2.98% | 15.24% |
| RCNN, 20 epochs (best epoch 6 of 20) | 2.11% | 12.76% |
| **Ensemble 0.7 SVM / 0.3 RCNN** | 0.11-0.14% | **5.64-5.67%** |

Takeaways that should steer everything after this:
- **Eval (unseen attacks) is the only honest number.** Dev reuses training's attack types, so a
  dev EER near 0.1% says nothing about generalisation. Never quote dev EER as "the" accuracy.
- **20 epochs of RCNN training bought ~2.5 points on the RCNN alone but ~0 on the ensemble**,
  because the SVM carries 70% of the fused score and is deterministic. Diminishing returns from
  more epochs; further gains need better/more data, not more training time on the same data.
  Training loss hit 0.0002 by epoch 20 (memorising the 6 known attacks) while dev EER bounced
  between 2.1% and 6.6% epoch to epoch — the best-epoch choice is somewhat noisy on this data.
- **The two branches fail on different attacks** (SVM weak on A10/A12/A17/A18; RCNN weak on
  A10/A13-A15), which is *why* fusion helps (8.6% -> 5.6%) despite the RCNN being the weaker
  model alone. But fusion also actively *hurts* on A10 and A15 where the RCNN is confidently
  wrong and drags a correct SVM score down — the fixed 0.7/0.3 weight is a compromise, not a
  free win everywhere.
- **Risk thresholds (0.80/0.50 from the slides) are not calibrated for unseen attacks.** On eval,
  only ~51% of spoofs hit "block" and ~28% slip through as "allow" (vs. 0.2% of bonafide wrongly
  flagged). The EER operating point sits near P(fake)=0.03-0.12, far from 0.5. Needs calibration
  (see 7.2) before the risk engine's thresholds mean anything in production.
- **Learned fusion (logistic regression on dev) did *worse* than the fixed weight** (12.3% vs.
  5.67% eval EER), because dev has no unseen attacks to learn from — it just learns "trust the
  SVM, dev says it's almost perfect," which is wrong exactly where it matters. This generalises:
  **do not choose/tune/fit anything (epoch, fusion weight, thresholds, a learned fuser) on dev or
  on eval.** Dev shares train's attacks; eval must stay a final, untouched number. Use 7.2 instead.

### 7.2 Must do before trusting any tuned number: attack-disjoint (and speaker-disjoint) validation

Both ASVspoof2019 train and dev use the *same six* attack types (A01-A06), so dev cannot tell you
how a tuning choice will behave on new attacks — it can only tell you how well you've fit the six
you already know. Any number "improved" by looking at dev (fusion weight, epoch choice, risk
thresholds, a learned fuser's weights) is suspect for production use.

Fix: carve a held-out-attack split out of what we already have, no new data required:
1. Pick 1-2 of the 6 training attack types to hold out entirely from training (e.g. train on
   A01-A04, hold out A05-A06).
2. Tune everything (RCNN epoch/checkpoint, fusion weight, risk thresholds, any learned fuser) on
   the held-out A05-A06 bonafide+spoof set, exactly as if it were "eval."
3. Confirm once, at the end, on the real eval set (A07-A19) — never re-tune after seeing it.
4. Optionally rotate which attacks are held out (k-fold over attack types) for a more stable
   estimate, since a single 2-attack holdout is small and noisy.
5. For the final production model, retrain on *all* available attacks (see 7.3) using the
   settings chosen in step 2 — the holdout is for choosing settings, not for shipping a weaker model.

This read on the current ensemble weight stands only provisionally until this is done: the 0.7/0.3
split was carried over from the original notebook and never actually tuned against unseen
attacks; it happens to work, but we don't yet know if 0.6/0.4 or 0.8/0.2 would do better on held-out
attacks. Re-derive it via step 2 before treating it as final.

### 7.3 Train on more data, test on fresh data: adding ASVspoof5

Decision made in conversation: since this is for production (not a competition to be judged on
ASVspoof2019's own eval split), train on more attack variety than ASVspoof2019 LA alone provides,
and reserve a truly external eval set to estimate real generalisation. There is no "ASVspoof 2022";
the options are ASVspoof 2021 (no new train data, only a harder eval set reusing 2019's train/dev
-- doesn't fit "train on it too") and **ASVspoof5 (2024, the current newest)**, which has fully
separate train/dev/eval: ~2000 crowdsourced speakers, 20+ TTS/VC attacks plus 7 adversarial
attacks, telephony-realistic conditions. ASVspoof5 is the right fit for "train on new data, test
on their eval."

**Source:** Zenodo [10.5281/zenodo.14498691](https://zenodo.org/records/14498691) (also mirrored
on Hugging Face as `jungjee/asvspoof5`). Open access; read README.txt/LICENSE.txt on the record
before downloading.

**Confirmed exact sizes** (from the Zenodo file list, not estimated):

| Split | Tars | Size |
|---|---|---|
| train (`flac_T_aa..ae.tar`) | 5 | 37.5 GB |
| dev (`flac_D_aa..ac.tar`) | 3 | 20.0 GB |
| eval (`flac_E_aa..aj.tar`) | 10 | 84.8 GB |
| **Total** | | **142.3 GB** |
| + `ASVspoof5_protocols.tar.gz` | | small (protocol/trial/enrollment TSVs) |

Unlike ASVspoof2019 (where PA was a genuinely separate, unused 16.6 GB replay-attack corpus),
**ASVspoof5 has no equivalent prunable chunk for our use case.** Track 1 (deepfake detection,
our use case) and Track 2 (spoofing-robust ASV) share the *same* audio (`flac_T`/`flac_D`/`flac_E`);
Track 2 only adds a few KB of extra protocol/trial/enrollment `.tsv` files and would additionally
require the separate VoxCeleb2 dataset, which we don't need and won't fetch. So the 142.3 GB of
audio is already the lean set for Track 1 — there's nothing bundled-but-irrelevant left to skip.

**Protocol format differs from 2019** and the data layer needs updating, not just pointed at a new
folder:
- 2019: whitespace-separated `.txt`, columns `speaker utt_id - attack key`.
- ASVspoof5: tab-separated `.tsv` (`ASVspoof5.train.tsv`, `ASVspoof5.dev.track_1.tsv`,
  `ASVspoof5.eval.track_1.tsv`), different column layout, directories `flac_T` / `flac_D` /
  `flac_E_eval` instead of `ASVspoof2019_LA_{train,dev,eval}/flac`.
- `audiodf/data/protocol.py` needs a second parser (or a format-detecting one) and `SPLITS`-style
  mapping for ASVspoof5; `read_protocol`'s label convention (1 = spoof) should stay the API so the
  rest of the pipeline (segmenter, feature extractors, cache) doesn't need to change.

**Disk plan (checked against real free space, not assumed):**
- Measured on this machine: `dataset/LA` 7.1 GB (kept), `dataset/PA` 16.6 GB (deleted — confirmed
  unused, replay-attack data, different problem from ours), `~/.cache/kagglehub` raw archive
  23.7 GB (deleted — redundant once extracted into `dataset/`). Reclaimed 40.3 GB.
- Free space is tight even after cleanup (roughly 140s GB free vs. 142.3 GB needed) — **do not**
  download all ASVspoof5 tars before extracting. Stream it: download one tar, extract it, delete
  the tar, move to the next. This keeps peak usage near the final extracted size instead of 2x.
- **Stage it:** fetch train+dev first (57.5 GB — fits safely even without further cleanup), confirm
  real feature-cache growth (ASVspoof5 train+dev alone is ~322k utterances vs. 2019's ~50k, so the
  on-disk Log-Mel/SVM-feature cache will be several times today's 8.6 GB), then decide on eval
  (84.8 GB) once there's confirmed headroom — more cleanup, an external/network drive, or
  downloading eval in sub-batches of its own 10 tars.
- A Hugging Face mirror (`jungjee/asvspoof5`) exists as a fallback if Zenodo throughput is poor,
  same tar-per-file structure.

**Resulting training plan once ASVspoof5 is in:**
1. Combine ASVspoof2019 LA train + ASVspoof5 train (+ dev, after held-out-attack tuning is settled)
   for the production model — more attack variety directly addresses the generalisation gap in 7.1.
2. Still never train on eval of anything. ASVspoof5's own eval becomes a second, independent
   generalisation check alongside ASVspoof2019 eval — report both, since they stress different
   things (2019 eval = older unseen TTS/VC attacks; ASVspoof5 eval = newer attacks, adversarial
   attacks, telephony-style conditions closer to production).
3. Re-run the 7.2 held-out-attack tuning procedure on the combined, larger attack pool before
   finalising fusion weight / thresholds / epoch for the production model.
4. Even with ASVspoof5, all source audio is still clean-ish crowdsourced/studio recordings, not
   real degraded phone-call audio — augmentation (codec, noise, packet loss) from the original
   plan (7.2-era section 2, Phase 2 "Robustness") is still required on top of this, not instead of it.

### 7.3b ASVspoof5 data audit (done 2026-10-03) and the decisions it forced

Downloaded: ASV5 train (182,357 utts, 400 speakers) + dev (140,950, 785 speakers) into `dataset5/`
via the Hugging Face mirror (Zenodo was ~40x slower). Eval (680,774 utts, 84.8 GB) not yet.
`dataset/PA` (replay attacks, a different problem) and the redundant kagglehub archive were deleted.

**Protocol columns** (official README): `SPEAKER_ID FLAC_FILE_NAME SPEAKER_GENDER CODEC CODEC_Q
CODEC_SEED ATTACK_TAG ATTACK_LABEL KEY TMP`, whitespace-separated despite the `.tsv` name.
`ATTACK_TAG` (AC1/AC2/AC3) is the attacker adaptation condition, NOT a codec. An earlier reading
treated it as a codec and reported a "codec perfectly predicts spoof" shortcut; that was wrong.

**Similarities with ASV2019:** same audio format (16 kHz mono 16-bit FLAC); same ~90/10 spoof/bonafide.
**Differences that matter:**
- Clips ~3.5x longer: ASV5 mean 11.8 s (5.4-24 s) vs 2019 mean 3.35 s (0.9-9.3 s).
- Attack codes are separate namespaces ("A01" means different systems in each dataset).
- ASV5 is attack-disjoint across its own splits (train A01-A08, dev A09-A16, eval A17-A32);
  2019 dev reuses train's attacks.
- **Codec domain gap (the real issue):** train and dev have 0% codec-processed audio; eval has
  ~75% codec-processed audio, *equally* for bonafide (74.7%) and spoof (74.8%), across 11 conditions
  (C01 opus_wb, C02 amr_wb, C03 speex_wb, C04 encodec_wb, C05 mp3_wb, C06 m4a_wb, C07 mp3+encodec,
  C08 opus_nb, C09 amr_nb, C10 speex_nb, C11 Bluetooth/cable device channels). Not a label shortcut,
  but a train/test mismatch that also mirrors production (real calls are codec'd).
  => codec augmentation must be applied to BOTH classes with equal probability. Augmenting only
  bonafide would *create* a "codec => real" shortcut. libsndfile 1.2.2 (bundled with soundfile)
  covers Opus, MP3, Vorbis and G.711 mu/A-law in-process (no ffmpeg); AMR, Speex, EnCodec, M4A and
  Bluetooth are not covered and will show up in the per-codec eval breakdown.

**Decision: train on ASV5 only; drop ASV2019 from training.** Train on ASV5 train, tune everything
(epoch, fusion weight, risk thresholds, calibration) on ASV5 dev (unseen attacks by design), report
once on ASV5 eval; retrain the production model on train+dev with frozen settings. ASV2019 eval is
kept as a free cross-dataset test. Reasons: ASV5 dev is a far stronger tuning set than a hand-carved
2-attack holdout; 2019 is ~8% of the audio hours; 2019 has a documented silence shortcut
(arXiv 2309.11827); mixing studio 2019 with crowdsourced ASV5 invites a dataset-identity shortcut.
Revisit adding 2019 as extra training data only if the 2019-eval cross-test is weak.

**Live-decision design (segments first, then per-model preprocessing):**
- RCNN: fixed 2 s windows (1 s hop), as served. To stop long ASV5 clips dominating, sample a fixed
  number of random windows per utterance per epoch.
- SVM: never isolated 2 s crops (measured: eval EER 8.6% -> 16.9%). Instead "growing-window
  snapshots": per utterance, feature vectors over prefixes of 2, 4, 6, 8, 10 s (capped by clip length),
  matching what `CallSession` already feeds the SVM live (it re-scores the rolling buffer each second).
- Evaluation mimics the call: verdict at 10 s (or end of a shorter clip), plus a time-to-decision
  curve at 2/4/6/8/10 s, per-attack and per-codec EER.

**Scaling decisions (measured, not assumed):**
- No Log-Mel cache for ASV5: ~2M train + ~1.5M dev windows would need ~90 GB at float16 with ~99 GB free.
  RCNN features are computed on the fly from FLAC in DataLoader workers (also lets codec augmentation
  vary every epoch).
- Exact RBF SVM is kept but trained on a stratified subsample of snapshot vectors. On ASV2019, exact
  RBF on a random 50% subsample matched the full fit (8.39% vs 8.58% eval EER), while Nystroem
  kernel approximation (k=3000) lost ~0.55 points (9.13%). `probability=True` was most of the old fit
  time (113 s vs 11 s); probabilities will instead come from calibration on dev.

### 7.3c Data integrity system (built; `python -m audiodf audit|prepare`)

`audit` writes `results/data_integrity_<dataset>.json` and exits non-zero on hard errors
(missing audio, duplicate IDs, speakers shared across splits, bonafide/attack inconsistency, wrong
audio format). Both datasets pass. It also reports shortcut signals, i.e. properties other than the
voice that differ between classes. Findings and what was done about each:

| Finding | Where | Action |
|---|---|---|
| Edge silence: bonafide 0.20/0.26 s lead/trail vs spoof 0.05/0.03 s | ASV5 train (dev is clean) | VAD trim (-45 dBFS, 25 ms frames, 50 ms margin) on every training clip, both classes; `CallSession` skips pre-speech audio with the same rule. Post-trim: 0.025 s vs 0.025 s leading. |
| Same silence shortcut, up to 15x | ASV2019 train/dev/eval | Confirms the documented 2019 issue; another reason 2019 is not used for training. |
| Exact digital silence in 72% of attack A11 clips (7.9% of frames) | ASV5 dev | Exact-zero samples replaced with +-1 LSB noise at every entry point (training loads, uploads, live chunks); real audio stays bit-exact. |
| SVM log features exploded on digital zeros (log(1e-10) = -23) | feature code | Floor raised to the STFT magnitude of a 1-LSB signal (1e-6). SVM features are now version 2; artifacts record feature versions and refuse to load on mismatch (the old ASV2019 bundle must be retrained). |
| Duration: bonafide 14.6 s vs spoof 11.2 s median | ASV5 train | Mostly neutralised by the 10 s decision horizon; kept as a finding. |
| Noise floor: spoof ~7 dB higher than bonafide in quietest frames | ASV5 train | Partly genuine vocoder hiss; recorded, not "fixed". |

Open (measure during training): SVM statistics still move for a few features when exact-zero margins
surround speech (p95 shift ~1.5 between-clip std). Ablation to run: mean/std over voiced frames only.

Preprocessing order (identical for both classes, mirrored in serving):
`load -> fill digital silence -> VAD trim -> [train only: label-blind codec aug, p=0.5] ->`
SVM: prefixes at 2/4/6/8/10 s -> 318-d vectors | RCNN: random 2 s windows read straight from FLAC.
Measured: index ~80+ utt/s, SVM snapshots ~15 utt/s, RCNN windows ~600/s with augmentation (8 workers).

### 7.3d Run 1 results (ASV5 train only; SVM 20k clips, RCNN 10 epochs, codec aug p=0.5)

| | dev tuning subset (unseen attacks A09-A16, codec-aug 75%) | ASV5 eval (30k of 136k downloaded, A17-A32, 11 real codecs) | ASV2019 eval (15k, cross-dataset) |
|---|---|---|---|
| SVM | 21.8% | 33.6% | 41.4% |
| RCNN (best epoch 6) | 17.6% | 37.4% | 33.3% |
| Fused (SVM weight 0.2, tuned on dev) | 15.9% | **33.0%** | 37.8% |

- Not production-ready. Eval per attack: A29 4.7%, A21 7.8%, A24 10.1%, A17 11.0% ... A19 59%, A20 58%,
  A32 47% (chance = 50%). On codec-free eval audio fused EER is 22.8%; real codec conditions 26-41%
  (C08 narrowband Opus 40.9%, C11 device channels 39.0%), so simulated Opus/MP3 augmentation did not
  transfer well to the real codecs.
- The dev-tuned fusion weight (0.2) did not transfer: on eval the SVM is the stronger branch. Dev, even
  codec-augmented, is a weak proxy for eval. ASV5 dev attack A12 was inverted (EER 69%).
- Hypothesis refuted: a classifier on 8 channel statistics (level, noise floor, bandwidth, ...) got 40%
  dev EER with a different per-attack pattern, so the models are not mainly using those cues.
- RCNN stopped improving on dev attacks after epoch ~3 while train loss fell 4x: it fits the 8 training
  attacks. More epochs will not help; more attack variety and real codec coverage should.
- Process notes: eval subset = 2 of 10 tars (136,376 clips, 20%); verified representative (spoof share,
  all 16 attacks, all 12 codec conditions within 0.5 pp). The ASV5 eval is now used for comparing
  candidate models, so choose between candidates only with a pre-agreed rule, never tune on it.

Next experiments, ordered by expected gain per hour (each full run is ~4-5 h):
1. Attack variety: final models train on ASV5 train **+ dev** (16 attacks instead of 8), with settings chosen
   beforehand on held-out attacks (leave-attacks-out) rather than on one dev split. Optionally add ASV2019.
2. Real codec coverage: an ffmpeg-backed augmenter (AMR, Speex, EnCodec, AAC, Bluetooth-like) rendered offline.
3. Stronger front end: a pretrained speech model (WavLM/wav2vec2 family) is the usual route to attack and
   channel robustness; untested here and heavier at inference (latency budget to be measured).

### 7.4 Fusion: don't jump to an ANN yet

Considered a small ANN/learned voting classifier instead of the fixed 0.7/0.3 weight. Verdict:
not yet, and not naively — logistic-regression fusion already failed for exactly the reason an ANN
would fail the same way (see 7.1): trained on dev, it only sees seen-attack behaviour, and a more
flexible model would fit that even more tightly, not less. An ANN becomes worth trying only once
7.2's held-out-attack split exists, and only if it's given more than the two raw probabilities —
e.g. inter-window RCNN score variance, SVM margin, clip length, inter-branch score gap — so it can
learn *when* to trust each branch rather than just re-deriving a fancier fixed weight. Compare
fixed-weight vs. logistic-regression vs. small-ANN fusion on the held-out-attack split; keep the
simplest one that wins, confirm once on eval.

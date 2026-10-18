# Model card: deepfake voice detector for banking calls

## What it is
A detector that listens to a phone call and says how likely the caller's voice is synthetic (a deepfake). It answers with a score and one of three actions:

| Action | Score | Meaning |
|---|---|---|
| allow | below 0.0060 | nothing unusual; no extra step |
| verify | 0.0060 or more | ask the caller for an extra check (call-back, knowledge question) |
| escalate | 0.90 or more | send to a human fraud analyst at once. **Never an automatic reject** |

The score is a ranking value, **not a probability**. The levels were set so that about 1 in 10 genuine callers on real phone lines is asked to verify (cost of a missed fake judged far higher than the cost of a verification).

## Model
- **WavLM-Base+** (self-supervised speech model), top 4 layers fine-tuned, attentive-statistics pooling and a small classifier head. Bundle `artifacts/` (run 7, seed 0).
- Works on 2 s windows with a 1 s hop; the call's score is the mean over windows in the last 10 s. First verdict after 2 s of speech.
- Trained on ASVspoof 5 (train and dev) and ASVspoof 2019 LA (train and dev), with simulated phone impairments (noise, codecs, packet loss, echo). Thresholds tuned on held-out attacks and speakers.
- Around it: an explanation agent (SHAP over 2 s x 4-band regions, faithfulness check, plain-language reasons from a template or an optional cloud LLM) and a scam-intent module (rules over a local Whisper transcript). **Intent only raises the check; it never lowers it and never decides alone.**

## Results (final model; decision measure = fakes missed at the verify level)
Each set has its own verify level set to flag 10% of genuine ASVspoof 2021 callers. "Missed" = deepfakes per 1000 that stay below it.

| Test set | EER | Fakes missed per 1000 |
|---|---|---|
| ASVspoof 2021 LA eval (real phone channels) | 9.14% | 83 |
| Simulated calls v1 | 3.06% | 9 |
| Simulated calls + babble noise | 7.00% | 8 |
| Simulated calls v2 | 10.78% | 6 |
| In-the-Wild (out of domain) | 4.47% | 16 |
| ASVspoof 5 eval | 5.74% | n/a |
| ASVspoof 2019 LA eval | 5.33% | n/a |

Seed-to-seed spread of the missed-fake rate: 0.00 to 0.15 on clean sets, 0.36 to 0.56 on noisy ones. EER is reported but never decides.

## Intended use
Second opinion in a bank's call-security flow, giving the agent or fraud team a reason to add a check. Demonstration and pilot use.

## Not for
- Automatically rejecting or accusing a caller. **At 1 deepfake per 1000 calls, roughly 96% of calls at the highest level would still be genuine people.**
- Use as the only control. It does not replace voice-biometric enrolment, call-back procedures or fraud rules.
- Claims of live-call accuracy. **No live bank-call validation has been done.**

## Known limits
- The weakest set is real phone channels (ASVspoof 2021): 83 of 1000 fakes missed at the verify level. Attacks and codecs unseen in training can do worse; the In-the-Wild check is one out-of-domain sample, not a guarantee.
- Simulated calls are not real calls: packet loss, voice-activity gaps, echo and codecs in production will differ. Collect labelled production audio before relying on the thresholds.
- Languages and accents: trained on mostly English read speech; other languages untested.
- The intent rules flag 37.8% of normal calls on an independent dialogue set at the highest intent level (0.6% on the set they were built on); a cloud LLM did not improve this and is not adopted. Treat intent as context.
- Explanations show where in the audio the score came from. They are not proof that the audio is fake.
- Serving: GPU about 63 ms per 10 s call in the container (about 235 calls per GPU from the earlier cost study); CPU about 5 calls. The Kubernetes manifests are schema-valid but have never run on a cluster.

## Data and licences
ASVspoof 5 / 2019 / 2021, In-the-Wild, DEMAND, ESC-50 and two public scam-dialogue sets were used under their own terms. The datasets are not in this repository and were deleted from the working machine at the end of the project (re-download to reproduce). The model weights are not in git (DVC pointers only; remote not set up yet).

## More
Full history, decisions and mistakes: `audit.md`. How to run a demo: `DEMO_GUIDE.md`. Code and commands: `src/README.md`.

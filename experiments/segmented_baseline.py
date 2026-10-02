"""Segmented re-implementation of AudioDeepfakeModel.ipynb (SVM + CNN-(Bi)LSTM ensemble).

Every utterance is cut into 2 s windows (1 s hop) to mimic the streaming buffer that
Kafka chunks will fill at inference time; both branches are trained and scored per
segment and aggregated per utterance. Labels: 1 = spoof (fake), 0 = bonafide.

Usage:
    python experiments/segmented_baseline.py --limit 300 --epochs 1   # smoke test
    python experiments/segmented_baseline.py --epochs 5               # full run
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as AT
from joblib import Parallel, delayed
from scipy import signal as sp_signal
from scipy.fftpack import dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

SR = 16000
SEG = 2 * SR
SEG_HOP = SR
HOP = 160
N_MELS = 64
N_FRAMES = SEG // HOP
SVM_DIM = 318

SPLITS = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
}


def read_protocol(data_root, split, limit=None):
    proto, audio_dir = SPLITS[split]
    rows = []
    with open(data_root / "ASVspoof2019_LA_cm_protocols" / proto) as f:
        for line in f:
            _, utt, _, attack, key = line.split()
            path = data_root / audio_dir / "flac" / f"{utt}.flac"
            rows.append((str(path), int(key == "spoof"), attack))
    if limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), min(limit, len(rows)), replace=False))]
    return rows


# ----------------------------------------------------------------------------- features

class FeatureExtractor:
    """Feature definitions copied from the notebook (MFCC/LFCC/MGDCC stats + Log-Mel)."""

    def __init__(self):
        self.mfcc = AT.MFCC(sample_rate=SR, n_mfcc=13,
                            melkwargs={"n_fft": 512, "hop_length": HOP, "win_length": 400, "n_mels": 26})
        self.mel = AT.MelSpectrogram(sample_rate=SR, n_fft=512, hop_length=HOP, win_length=400,
                                     n_mels=N_MELS, power=2.0)

    @staticmethod
    def _with_deltas(x):
        d1 = AF.compute_deltas(x)
        return torch.cat([x, d1, AF.compute_deltas(d1)], dim=1)

    @staticmethod
    def _cepstra(waves):
        _, _, stft = sp_signal.stft(waves, fs=SR, window="hann", nperseg=400,
                                    noverlap=400 - HOP, nfft=512, axis=-1)
        mag = np.abs(stft)
        lfcc = dct(np.log(mag + 1e-10), type=2, axis=1, norm="ortho")[:, :20]
        phase = np.angle(stft)
        mgd = np.diff(phase, axis=1, prepend=phase[:, :1]) * mag ** 0.4
        mgdcc = dct(np.log(np.abs(mgd) + 1e-10), type=2, axis=1, norm="ortho")[:, :20]
        return torch.from_numpy(lfcc).float(), torch.from_numpy(mgdcc).float()

    def svm_stats(self, waves):
        """waves: (n, samples) float32 -> (n, 318), same layout as the notebook's CSV."""
        t = torch.from_numpy(waves)
        mfcc = self._with_deltas(self.mfcc(t))
        lfcc, mgdcc = self._cepstra(waves)
        lfcc, mgdcc = self._with_deltas(lfcc), self._with_deltas(mgdcc)
        parts = []
        for feat in (mfcc, lfcc, mgdcc):
            parts += [feat.mean(dim=2), feat.std(dim=2)]
        return torch.cat(parts, dim=1).numpy()

    def logmel(self, waves):
        spec = torch.log(self.mel(torch.from_numpy(waves)) + 1e-2)[:, :, :N_FRAMES]
        mean = spec.mean(dim=(1, 2), keepdim=True)
        std = spec.std(dim=(1, 2), keepdim=True)
        return ((spec - mean) / (std + 1e-6)).numpy()


def segment(wave):
    """Repeat-pad short audio, then cut 2 s windows with 1 s hop; last window is end-aligned."""
    if len(wave) < SEG:
        wave = np.tile(wave, int(np.ceil(SEG / len(wave))))[:SEG]
    starts = list(range(0, len(wave) - SEG + 1, SEG_HOP))
    if (len(wave) - SEG) % SEG_HOP:
        starts.append(len(wave) - SEG)
    return np.stack([wave[s:s + SEG] for s in starts])


class ExtractDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.fx = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.fx is None:
            self.fx = FeatureExtractor()
        wave, sr = sf.read(self.rows[i][0], dtype="float32")
        assert sr == SR, f"unexpected sample rate {sr}"
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        segs = segment(wave)
        return {
            "utt_svm": self.fx.svm_stats(wave[None]).squeeze(0),
            "seg_svm": self.fx.svm_stats(segs),
            "logmel": self.fx.logmel(segs).astype(np.float16),
        }


def _worker_init(_):
    torch.set_num_threads(1)


def extract_split(rows, out_dir, workers):
    if (out_dir / "meta.json").exists():
        print(f"  cached: {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(ExtractDataset(rows), batch_size=None, num_workers=workers,
                        worker_init_fn=_worker_init, prefetch_factor=8 if workers else None)
    utt_svm, seg_svm, seg_utt = [], [], []
    n_seg, t0 = 0, time.time()
    with open(out_dir / "logmel.f16", "wb") as fmel:
        for i, item in enumerate(loader):
            mel = item["logmel"].numpy()
            fmel.write(mel.tobytes())
            utt_svm.append(item["utt_svm"].numpy())
            seg_svm.append(item["seg_svm"].numpy())
            seg_utt.append(np.full(len(mel), i, dtype=np.int32))
            n_seg += len(mel)
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"    {i + 1}/{len(rows)} utts, {n_seg} segs, {rate:.0f} utt/s", flush=True)
    np.save(out_dir / "utt_svm.npy", np.stack(utt_svm))
    np.save(out_dir / "seg_svm.npy", np.concatenate(seg_svm))
    np.save(out_dir / "seg_utt.npy", np.concatenate(seg_utt))
    np.save(out_dir / "utt_label.npy", np.array([r[1] for r in rows], dtype=np.int8))
    np.save(out_dir / "utt_attack.npy", np.array([r[2] for r in rows]))
    (out_dir / "meta.json").write_text(json.dumps({"n_utt": len(rows), "n_seg": n_seg}))
    print(f"  extracted {len(rows)} utts / {n_seg} segs in {time.time() - t0:.0f}s")


class SplitData:
    def __init__(self, d):
        meta = json.loads((d / "meta.json").read_text())
        self.n_seg = meta["n_seg"]
        self.logmel = np.memmap(d / "logmel.f16", dtype=np.float16, mode="r",
                                shape=(self.n_seg, N_MELS, N_FRAMES))
        self.utt_svm = np.load(d / "utt_svm.npy")
        self.seg_svm = np.load(d / "seg_svm.npy")
        self.seg_utt = np.load(d / "seg_utt.npy")
        self.utt_label = np.load(d / "utt_label.npy").astype(np.int64)
        self.utt_attack = np.load(d / "utt_attack.npy")
        self.seg_label = self.utt_label[self.seg_utt]

    def to_utt(self, seg_scores):
        """Mean of segment scores per utterance (the 10 s aggregation window, offline)."""
        sums = np.bincount(self.seg_utt, weights=seg_scores, minlength=len(self.utt_label))
        return sums / np.bincount(self.seg_utt, minlength=len(self.utt_label))


# ----------------------------------------------------------------------------- metrics

def metrics(y, p):
    """y: 1 = spoof, p: P(spoof). Threshold-dependent numbers use the EER threshold."""
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    pred = (p >= thr[i]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "eer_pct": round(float(100 * (fpr[i] + fnr[i]) / 2), 3),
        "auc": round(float(roc_auc_score(y, p)), 5),
        "acc_at_eer_thr": round(float((tp + tn) / len(y)), 5),
        "recall_spoof": round(float(tp / (tp + fn)), 5),
        "fpr_bonafide_flagged": round(float(fp / (fp + tn)), 5),
        "acc_at_0.5": round(float(((p >= 0.5) == y).mean()), 5),
        "eer_threshold": round(float(thr[i]), 5),
    }


def per_attack_eer(y, p, attacks):
    bona = y == 0
    out = {}
    for a in sorted(set(attacks[~bona])):
        m = bona | (attacks == a)
        out[a] = metrics(y[m], p[m])["eer_pct"]
    return out


# ----------------------------------------------------------------------------- SVM branch

def make_svm():
    return Pipeline([("scaler", StandardScaler()),
                     ("svm", SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                                 probability=True, random_state=0))])


def svm_predict(model, X, n_jobs=8, chunk=4000):
    parts = Parallel(n_jobs=n_jobs)(delayed(model.predict_proba)(X[i:i + chunk])
                                    for i in range(0, len(X), chunk))
    return np.concatenate(parts)[:, 1]


# ----------------------------------------------------------------------------- CNN branch

class CNNRNN(nn.Module):
    def __init__(self, bidirectional=True, hidden=128):
        super().__init__()

        def block(i, o, pool):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(pool))

        self.cnn = nn.Sequential(block(1, 32, (2, 2)), block(32, 64, (2, 2)),
                                 block(64, 128, (2, 1)), nn.Dropout(0.2))
        with torch.no_grad():
            _, c, h, _ = self.cnn(torch.zeros(1, 1, N_MELS, N_FRAMES)).shape
        self.rnn = nn.LSTM(c * h, hidden, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        z = self.cnn(x)
        b, c, h, t = z.shape
        out, _ = self.rnn(z.permute(0, 3, 1, 2).reshape(b, t, c * h))
        return self.head(out.mean(dim=1)).squeeze(1)


def batches(data, idx, bs):
    for i in range(0, len(idx), bs):
        b = np.sort(idx[i:i + bs])
        yield torch.from_numpy(data.logmel[b].astype(np.float32)).unsqueeze(1), b


@torch.no_grad()
def cnn_predict(model, data, device, bs=512):
    model.eval()
    out = np.empty(data.n_seg, dtype=np.float32)
    for x, b in batches(data, np.arange(data.n_seg), bs):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out[b] = torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy()
    return out


def train_cnn(train, dev, args, device, ckpt):
    torch.manual_seed(0)
    model = CNNRNN(bidirectional=args.rnn == "bilstm").to(device)
    n_spoof = int(train.seg_label.sum())
    pos_weight = torch.tensor((len(train.seg_label) - n_spoof) / n_spoof, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(train.n_seg / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    labels = torch.from_numpy(train.seg_label.astype(np.float32))
    rng = np.random.default_rng(0)
    best, history = None, []
    for ep in range(args.epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, b in batches(train, rng.permutation(train.n_seg), args.batch):
            x, y = x.to(device, non_blocking=True), labels[b].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
        dev_m = metrics(dev.utt_label, dev.to_utt(cnn_predict(model, dev, device)))
        history.append({"epoch": ep + 1, "loss": round(total / train.n_seg, 5),
                        "dev_utt_eer_pct": dev_m["eer_pct"], "sec": round(time.time() - t0)})
        print(f"  epoch {ep + 1}: {history[-1]}", flush=True)
        if best is None or dev_m["eer_pct"] < best:
            best = dev_m["eer_pct"]
            torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, history


# ----------------------------------------------------------------------------- latency

def latency(model, svm, device):
    fx = FeatureExtractor()
    wave = np.random.default_rng(0).standard_normal(SEG).astype(np.float32) * 0.05
    segs = wave[None]

    def timeit(fn, n=50):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return round(1000 * (time.perf_counter() - t0) / n, 3)

    x_cpu = torch.from_numpy(fx.logmel(segs)).unsqueeze(1)
    svm_x = fx.svm_stats(segs)
    out = {"features_logmel_ms": timeit(lambda: fx.logmel(segs)),
           "features_svm_ms": timeit(lambda: fx.svm_stats(segs)),
           "svm_predict_ms": timeit(lambda: svm.predict_proba(svm_x), n=20)}
    model.eval()
    with torch.no_grad():
        out["cnn_cpu_ms"] = timeit(lambda: model.cpu()(x_cpu))
        if device.type == "cuda":
            model.to(device)
            x = x_cpu.to(device)

            def gpu():
                model(x)
                torch.cuda.synchronize()
            out["cnn_gpu_ms"] = timeit(gpu)
    model.to(device)
    out["note"] = "per single 2 s segment, batch size 1"
    return out


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "dataset" / "LA" / "LA")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "audiodf")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    ap.add_argument("--limit", type=int, default=None, help="utterances per split (smoke test)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rnn", choices=["bilstm", "lstm"], default="bilstm")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    tag = f"seg2s_hop1s{f'_limit{args.limit}' if args.limit else ''}"
    cache = args.cache / tag
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} cache={cache}")

    print("[1/5] feature extraction")
    for split in SPLITS:
        print(f" {split}")
        extract_split(read_protocol(args.data_root, split, args.limit), cache / split, args.workers)
    train, dev, ev = (SplitData(cache / s) for s in SPLITS)
    print(f"  segments: train={train.n_seg} dev={dev.n_seg} eval={ev.n_seg}")

    results = {"config": {k: str(v) for k, v in vars(args).items()},
               "data": {s: {"utts": len(d.utt_label), "segs": d.n_seg, "spoof_utts": int(d.utt_label.sum())}
                        for s, d in zip(SPLITS, (train, dev, ev))}}
    scores = {"dev": {}, "eval": {}}

    print("[2/5] SVM, utterance-level (notebook-faithful)")
    t0 = time.time()
    svm_utt = make_svm().fit(train.utt_svm, train.utt_label)
    for name, d in (("dev", dev), ("eval", ev)):
        scores[name]["svm_utt"] = svm_predict(svm_utt, d.utt_svm)
    print(f"  {time.time() - t0:.0f}s")

    print("[3/5] SVM, segment-level (1 random segment per train utterance)")
    t0 = time.time()
    rng = np.random.default_rng(0)
    first = np.searchsorted(train.seg_utt, np.arange(len(train.utt_label)))
    counts = np.bincount(train.seg_utt, minlength=len(train.utt_label))
    pick = first + (rng.random(len(first)) * counts).astype(int)
    svm_seg = make_svm().fit(train.seg_svm[pick], train.seg_label[pick])
    for name, d in (("dev", dev), ("eval", ev)):
        scores[name]["svm_seg"] = d.to_utt(svm_predict(svm_seg, d.seg_svm))
    print(f"  {time.time() - t0:.0f}s")

    print(f"[4/5] CNN-{args.rnn.upper()} on 2 s segments, {args.epochs} epochs")
    ckpt = cache / f"cnn_{args.rnn}.pt"
    model, history = train_cnn(train, dev, args, device, ckpt)
    results["cnn_history"] = history
    seg_scores = {}
    for name, d in (("dev", dev), ("eval", ev)):
        seg_scores[name] = cnn_predict(model, d, device)
        scores[name]["cnn"] = d.to_utt(seg_scores[name])
    results["cnn_segment_level"] = {n: metrics(d.seg_label, seg_scores[n])
                                    for n, d in (("dev", dev), ("eval", ev))}

    print("[5/5] fusion")
    for name in ("dev", "eval"):
        s = scores[name]
        s["fuse_0.7svm_utt_0.3cnn"] = 0.7 * s["svm_utt"] + 0.3 * s["cnn"]
        s["fuse_0.7svm_seg_0.3cnn"] = 0.7 * s["svm_seg"] + 0.3 * s["cnn"]
    grid = np.round(np.linspace(0, 1, 21), 2)
    dev_eers = [metrics(dev.utt_label, w * scores["dev"]["svm_seg"] + (1 - w) * scores["dev"]["cnn"])["eer_pct"]
                for w in grid]
    w = float(grid[int(np.argmin(dev_eers))])
    results["tuned_svm_weight"] = w
    for name in ("dev", "eval"):
        scores[name]["fuse_tuned"] = w * scores[name]["svm_seg"] + (1 - w) * scores[name]["cnn"]

    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    stack_x = {n: np.column_stack([logit(scores[n]["svm_seg"]), logit(scores[n]["cnn"])]) for n in scores}
    stacker = LogisticRegression(class_weight="balanced").fit(stack_x["dev"], dev.utt_label)
    scores["eval"]["fuse_stacking_lr"] = stacker.predict_proba(stack_x["eval"])[:, 1]

    for name, d in (("dev", dev), ("eval", ev)):
        results[name] = {k: metrics(d.utt_label, v) for k, v in scores[name].items()}
    results["eval_per_attack_eer_pct"] = {k: per_attack_eer(ev.utt_label, scores["eval"][k], ev.utt_attack)
                                          for k in ("svm_utt", "svm_seg", "cnn", "fuse_tuned")}
    results["latency"] = latency(model, svm_seg, device)

    out_file = args.out / f"segmented_baseline_{args.rnn}{'_limit' + str(args.limit) if args.limit else ''}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_file}")
    for split in ("dev", "eval"):
        print(f"\n{split.upper()} (utterance level, spoof = positive)")
        print(f"  {'model':28s} {'EER%':>7s} {'AUC':>8s} {'acc@EER':>8s} {'recall':>8s} {'FPR':>8s}")
        for k, m in results[split].items():
            print(f"  {k:28s} {m['eer_pct']:7.3f} {m['auc']:8.5f} {m['acc_at_eer_thr']:8.5f} "
                  f"{m['recall_spoof']:8.5f} {m['fpr_bonafide_flagged']:8.5f}")
    print("\nlatency:", results["latency"])


if __name__ == "__main__":
    main()

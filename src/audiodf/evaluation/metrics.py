"""Detection metrics. y: 1 = spoof, p: P(spoof). Threshold-dependent numbers use the EER operating point."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def compute_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr[i]).astype(int), labels=[0, 1]).ravel()
    return {
        "eer_pct": round(float(100 * (fpr[i] + fnr[i]) / 2), 3),
        "auc": round(float(roc_auc_score(y, p)), 5),
        "acc_at_eer_thr": round(float((tp + tn) / len(y)), 5),
        "recall_spoof": round(float(tp / (tp + fn)), 5),
        "fpr_bonafide_flagged": round(float(fp / (fp + tn)), 5),
        "acc_at_0.5": round(float(((p >= 0.5) == y).mean()), 5),
        "eer_threshold": round(float(thr[i]), 5),
    }


def per_group_eer(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """EER within each group (e.g. each codec condition), skipping groups missing a class."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    return {str(g): compute_metrics(y[groups == g], p[groups == g])["eer_pct"]
            for g in sorted(set(groups)) if len(set(y[groups == g])) == 2}


def per_attack_eer(y: np.ndarray, p: np.ndarray, attacks: np.ndarray) -> dict[str, float]:
    """EER of each spoofing attack against all bonafide utterances."""
    y, p, attacks = np.asarray(y), np.asarray(p), np.asarray(attacks)
    bona = y == 0
    return {a: compute_metrics(y[bona | (attacks == a)], p[bona | (attacks == a)])["eer_pct"]
            for a in sorted(set(attacks[~bona]))}

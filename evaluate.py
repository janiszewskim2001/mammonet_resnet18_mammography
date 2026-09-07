"""Metryki diagnostyczne, testy istotności i koszt obliczeniowy obu modeli."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2 as chi2_dist
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc

import config
from models import MammoNet, build_resnet18, count_parameters


def wilson_ci(successes, n, z=1.96):
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z / denom * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return p, center - half, center + half


def metrics(labels, preds, probs):
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr, tpr, _ = roc_curve(labels, probs)
    return {
        "cm": cm,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "sensitivity": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp),
        "auc": auc(fpr, tpr),
    }


def mcnemar(labels, preds_a, preds_b):
    """Test McNemara z poprawką Edwardsa na ciągłość."""
    labels = np.asarray(labels)
    a = np.asarray(preds_a) == labels
    b = np.asarray(preds_b) == labels

    both = int((a & b).sum())
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    neither = int((~a & ~b).sum())

    discordant = only_a + only_b
    if discordant == 0:
        return 0.0, 1.0, (both, only_a, only_b, neither)

    chi2 = (abs(only_a - only_b) - 1) ** 2 / discordant
    p = 1 - chi2_dist.cdf(chi2, df=1)
    return chi2, p, (both, only_a, only_b, neither)


def bootstrap_auc_diff(labels, probs_a, probs_b, n_boot=2000, seed=config.SEED):
    """Sparowany bootstrap różnicy AUC."""
    labels = np.asarray(labels)
    probs_a, probs_b = np.asarray(probs_a), np.asarray(probs_b)
    rng = np.random.default_rng(seed)
    n = len(labels)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        diffs.append(roc_auc_score(labels[idx], probs_b[idx])
                     - roc_auc_score(labels[idx], probs_a[idx]))

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(p)


def benchmark(model, shape, runs=50, warmup=5):
    model.eval()
    dummy = torch.randn(1, *shape)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        start = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    return elapsed / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=config.RESULTS_DIR)
    args = ap.parse_args()
    rd = args.results_dir

    pm = json.loads((rd / "mammonet_predictions.json").read_text())
    pr = json.loads((rd / "resnet18_predictions.json").read_text())

    assert pm["labels"] == pr["labels"], "modele oceniano na różnych zbiorach"
    labels = pm["labels"]
    n = len(labels)

    m = metrics(labels, pm["preds"], pm["probs"])
    r = metrics(labels, pr["preds"], pr["probs"])

    table = pd.DataFrame({
        "Metryka": ["Dokładność", "Czułość", "Swoistość", "Precyzja", "AUC"],
        "MammoNet": [m["accuracy"], m["sensitivity"], m["specificity"], m["precision"], m["auc"]],
        "ResNet-18": [r["accuracy"], r["sensitivity"], r["specificity"], r["precision"], r["auc"]],
    })
    print(table.to_string(index=False, float_format="%.3f"))
    table.to_csv(rd / "metryki.csv", index=False, sep=";", encoding="utf-8-sig")

    print("\nDokładność z 95% CI (Wilson)")
    for name, pred in [("MammoNet", pm["preds"]), ("ResNet-18", pr["preds"])]:
        k = int((np.asarray(pred) == np.asarray(labels)).sum())
        p, lo, hi = wilson_ci(k, n)
        print(f"  {name:10s} {p:.4f}  [{lo:.4f}, {hi:.4f}]  ({k}/{n})")

    chi2, p_mc, (both, only_m, only_r, neither) = mcnemar(labels, pm["preds"], pr["preds"])
    print(f"\nMcNemar: oba poprawne {both}, tylko MammoNet {only_m}, "
          f"tylko ResNet-18 {only_r}, oba błędne {neither}")
    print(f"  chi2 = {chi2:.3f}, p = {p_mc:.4g}")

    diff, lo, hi, p_boot = bootstrap_auc_diff(labels, pm["probs"], pr["probs"])
    p_txt = "< 0.001" if p_boot < 0.001 else f"= {p_boot:.4g}"
    print(f"\nRóżnica AUC = {diff:.4f}  [{lo:.4f}, {hi:.4f}]  p {p_txt}")

    mammonet = MammoNet()
    mammonet.load_state_dict(torch.load(rd / "mammonet_best.pt", map_location="cpu"))
    resnet = build_resnet18(pretrained=False)
    resnet.load_state_dict(torch.load(rd / "resnet18_best.pt", map_location="cpu"))

    t_m = benchmark(mammonet, (1, 128, 128))
    t_r = benchmark(resnet, (3, 224, 224))
    n_m, n_r = count_parameters(mammonet), count_parameters(resnet)

    print(f"\nMammoNet:  {n_m:>11,} parametrów, {t_m:6.2f} ms/obraz")
    print(f"ResNet-18: {n_r:>11,} parametrów, {t_r:6.2f} ms/obraz")
    print(f"Stosunek:  {n_r / n_m:.1f}x parametrów, {t_r / t_m:.1f}x czasu")

    pd.DataFrame({
        "Model": ["MammoNet", "ResNet-18"],
        "Parametry": [n_m, n_r],
        "Czas inferencji [ms]": [t_m, t_r],
    }).to_csv(rd / "efektywnosc.csv", index=False, sep=";", encoding="utf-8-sig")

    (rd / "statystyka.json").write_text(json.dumps({
        "mcnemar": {"chi2": chi2, "p": p_mc, "both_correct": both,
                    "only_mammonet": only_m, "only_resnet": only_r, "neither": neither},
        "auc_diff": {"value": diff, "ci_low": lo, "ci_high": hi, "p": p_boot},
    }, indent=2))


if __name__ == "__main__":
    main()

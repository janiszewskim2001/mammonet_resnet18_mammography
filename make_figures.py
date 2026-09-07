"""Ryciny: krzywe uczenia, macierze pomyłek, ROC, mapy Grad-CAM
oraz zależność skuteczności od cech klinicznych.

Zapis w EPS (wektor) i PNG 600 dpi
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

import config
from data import build_datasets, set_seed
from gradcam import GradCAM, target_layer, upsample
from models import MammoNet, build_resnet18

MM = 1 / 25.4
W_SINGLE, W_DOUBLE = 85 * MM, 170 * MM
BLUE, VERM = "#0072B2", "#D55E00"
CMAP = "turbo"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "lines.linewidth": 1.2, "lines.markersize": 3.5,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "ps.fonttype": 42, "pdf.fonttype": 42,
})


def save(fig, out_dir, name):
    fig.savefig(out_dir / f"{name}.eps", format="eps")
    fig.savefig(out_dir / f"{name}.png", dpi=600)
    plt.close(fig)
    print(f"  {name}")


def panel(ax, letter):
    ax.text(-0.02, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="bottom", ha="right")


def learning_curves(history, name, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.38))

    for ax, k_train, k_test, ylabel, loc in [
        (ax1, "train_loss", "test_loss", "Strata", "upper right"),
        (ax2, "train_acc", "test_acc", "Dokładność", "lower right"),
    ]:
        ax.plot(epochs, history[k_train], color=BLUE, marker="o", ls="-", label="Treningowy")
        ax.plot(epochs, history[k_test], color=VERM, marker="s", ls="--", label="Testowy")
        ax.set_xlabel("Epoka")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.legend(loc=loc)

    panel(ax1, "a")
    panel(ax2, "b")
    fig.tight_layout()
    save(fig, out_dir, f"Rys1_Krzywe_uczenia_{name}")


def confusion_matrices(cm_m, cm_r, out_dir):
    vmin = min(cm_m.min(), cm_r.min())
    vmax = max(cm_m.max(), cm_r.max())

    for name, cm, cmap in [("MammoNet", cm_m, "Blues"), ("ResNet-18", cm_r, "Oranges")]:
        fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.88))
        disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASSES)
        disp.plot(cmap=cmap, ax=ax, values_format="d", colorbar=True)
        disp.im_.set_clim(vmin, vmax)   # wspólna skala, inaczej odcienie nie są porównywalne
        ax.set_xlabel("Klasa przewidywana")
        ax.set_ylabel("Klasa rzeczywista")
        ax.set_title(name)
        for t in ax.texts:
            t.set_fontsize(8)
        fig.tight_layout()
        save(fig, out_dir, f"Rys2_Macierz_{name}")


def roc(labels, probs_m, probs_r, out_dir):
    fpr_m, tpr_m, _ = roc_curve(labels, probs_m)
    fpr_r, tpr_r, _ = roc_curve(labels, probs_r)

    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE))
    ax.plot(fpr_m, tpr_m, color=BLUE, ls="-",
            label=f"MammoNet (AUC = {auc(fpr_m, tpr_m):.3f})")
    ax.plot(fpr_r, tpr_r, color=VERM, ls="--",
            label=f"ResNet-18 (AUC = {auc(fpr_r, tpr_r):.3f})")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=0.8, label="Klasyfikator losowy")
    ax.set_xlabel("1 − swoistość")
    ax.set_ylabel("Czułość")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, out_dir, "Rys3_ROC_porownanie")


def overlay(ax, image, cam, alpha_max=0.75, threshold=0.15):
    """Przezroczystość proporcjonalna do wartości - obszary o niskiej atrybucji
    pozostają czystym obrazem, bez tonowania tła."""
    ax.imshow(image, cmap="gray", interpolation="nearest")
    alpha = np.clip((cam - threshold) / (1 - threshold), 0, 1) * alpha_max
    ax.imshow(cam, cmap=CMAP, vmin=0, vmax=1, alpha=alpha, interpolation="bilinear")


def gradcam_figures(models, datasets_, out_dir, n_per_class=3):
    mammonet, resnet = models
    ds_m, ds_r = datasets_

    cam_m = GradCAM(mammonet, target_layer(mammonet, "mammonet"))
    cam_r = GradCAM(resnet, target_layer(resnet, "resnet18"))
    idx_to_class = {v: k for k, v in ds_m.class_to_idx.items()}

    random.seed(config.SEED)
    for cls, cls_name in idx_to_class.items():
        pool = [i for i, (_, lbl) in enumerate(ds_m.samples) if lbl == cls]
        random.shuffle(pool)

        for rank, idx in enumerate(pool[:n_per_class], start=1):
            x_m, _ = ds_m[idx]
            x_r, _ = ds_r[idx]

            raw_m, p_m = cam_m(x_m.unsqueeze(0), cls)
            raw_r, p_r = cam_r(x_r.unsqueeze(0), cls)

            # wspólna skala przestrzenna dla obu modeli
            image = x_r[0].numpy() * 0.229 + 0.485
            heat_m, heat_r = upsample(raw_m, 224), upsample(raw_r, 224)

            fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.36))
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.06)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
            cax = fig.add_subplot(gs[0, 3])

            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect("equal")

            axes[0].imshow(image, cmap="gray", interpolation="nearest")
            axes[0].set_title("Obraz wejściowy", pad=4)
            overlay(axes[1], image, heat_m)
            axes[1].set_title(f"MammoNet ($p$ = {p_m:.2f})", pad=4)
            overlay(axes[2], image, heat_r)
            axes[2].set_title(f"ResNet-18 ($p$ = {p_r:.2f})", pad=4)

            for ax, letter in zip(axes, "abc"):
                ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes,
                        fontsize=8, fontweight="bold", va="top", ha="left", color="white",
                        bbox=dict(boxstyle="square,pad=0.15", fc="black", ec="none", alpha=0.6))

            cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP),
                              cax=cax, ticks=[0, 0.5, 1.0])
            cb.set_label("Znormalizowana atrybucja", fontsize=7, labelpad=4)
            cb.ax.tick_params(labelsize=6)
            cb.outline.set_linewidth(0.6)

            save(fig, out_dir, f"GradCAM_{cls_name}_{rank}")

    cam_m.remove()
    cam_r.remove()


def clinical_bars(csv_path, xcol, xlabel, name, out_dir):
    df = pd.read_csv(csv_path, sep=";")
    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.5, W_SINGLE * 0.85))

    x = np.arange(len(df))
    width = 0.35
    b1 = ax.bar(x - width / 2, df["mammonet_acc"], width, label="MammoNet",
                color=BLUE, edgecolor="black", lw=0.5)
    b2 = ax.bar(x + width / 2, df["resnet_acc"], width, label="ResNet-18",
                color=VERM, edgecolor="black", lw=0.5)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=6)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}\n(n={int(n)})" for v, n in zip(df[xcol], df["n"])])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Dokładność")
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    fig.tight_layout()
    save(fig, out_dir, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--results_dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--out_dir", type=Path, default=config.FIGURES_DIR)
    args = ap.parse_args()

    set_seed()
    rd, out = args.results_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pm = json.loads((rd / "mammonet_predictions.json").read_text())
    pr = json.loads((rd / "resnet18_predictions.json").read_text())
    labels = np.array(pm["labels"])

    print("Krzywe uczenia")
    for arch, name in [("mammonet", "MammoNet"), ("resnet18", "ResNet-18")]:
        history = json.loads((rd / f"{arch}_history.json").read_text())
        learning_curves(history, name, out)

    print("Macierze pomyłek")
    confusion_matrices(confusion_matrix(labels, pm["preds"]),
                       confusion_matrix(labels, pr["preds"]), out)

    print("Krzywe ROC")
    roc(labels, pm["probs"], pr["probs"], out)

    print("Mapy Grad-CAM")
    _, ds_m = build_datasets("mammonet", args.data_dir)
    _, ds_r = build_datasets("resnet18", args.data_dir)
    assert ds_m.samples == ds_r.samples

    mammonet = MammoNet()
    mammonet.load_state_dict(torch.load(rd / "mammonet_best.pt", map_location="cpu"))
    mammonet.eval()
    resnet = build_resnet18(pretrained=False)
    resnet.load_state_dict(torch.load(rd / "resnet18_best.pt", map_location="cpu"))
    resnet.eval()

    gradcam_figures((mammonet, resnet), (ds_m, ds_r), out)

    print("Zależność od cech klinicznych")
    for fname, xcol, xlabel, name in [
        ("dokladnosc_subtlety.csv", "subtlety",
         "Subtelność zmiany (1 = trudna, 5 = łatwa)", "Rys6_Dokladnosc_vs_subtlety"),
        ("dokladnosc_gestosc.csv", "density",
         "Gęstość utkania piersi (BI-RADS)", "Rys7_Dokladnosc_vs_gestosc"),
    ]:
        path = rd / fname
        if path.exists():
            clinical_bars(path, xcol, xlabel, name, out)
        else:
            print(f"  pominięto {fname} (uruchom analyze_clinical.py)")

    print(f"\nRyciny w: {out}")


if __name__ == "__main__":
    main()

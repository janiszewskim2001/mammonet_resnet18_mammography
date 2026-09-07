"""Ilościowa charakterystyka map Grad-CAM.

Ocena zgodności lokalizacyjnej przez porównanie z maskami segmentacji okazała się
niewykonalna. Maski i wycinki klasyfikacyjne w wykorzystanym eksporcie CBIS-DDSM
nie są zapisane w spójnym układzie współrzędnych (maska odpowiada pełnemu
mammogramowi, wycinek jego fragmentowi a współrzędne wycięcia nie są dostępne).
Wyznaczane są miary opisujące sam rozkład mapy niezależne od masek:

  udział energii brzegowej  - ułamek sumy wartości mapy w zewnętrznym pasie kadru
  stereotypowość            - średnia korelacja par map dla różnych obrazów
  rozrzut środka masy       - odchylenie standardowe współrzędnych centroidu
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import config
from data import build_datasets, set_seed
from gradcam import GradCAM, target_layer, upsample
from models import MammoNet, build_resnet18

BORDER_FRACTION = 0.15
COMMON_SIZE = 64


def border_energy(cam, fraction=BORDER_FRACTION):
    h, w = cam.shape
    bh, bw = int(h * fraction), int(w * fraction)
    mask = np.zeros_like(cam, dtype=bool)
    mask[:bh, :] = mask[-bh:, :] = True
    mask[:, :bw] = mask[:, -bw:] = True
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 1e-8 else 0.0


def mean_pairwise_correlation(maps):
    flat = maps.reshape(len(maps), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    corr = (flat / norms) @ (flat / norms).T
    iu = np.triu_indices(len(maps), k=1)
    return float(corr[iu].mean()), float(corr[iu].std())


def centroids(maps):
    h, w = maps.shape[1], maps.shape[2]
    ys, xs = np.indices((h, w))
    out = []
    for cam in maps:
        s = cam.sum()
        if s < 1e-8:
            continue
        out.append(((ys * cam).sum() / s / h, (xs * cam).sum() / s / w))
    return np.array(out)


def collect_maps(model, arch, dataset, indices, image_size):
    cam = GradCAM(model, target_layer(model, arch))
    maps, borders = [], []
    for idx, cls in indices:
        x, _ = dataset[idx]
        raw, _ = cam(x.unsqueeze(0), cls)
        maps.append(upsample(raw, COMMON_SIZE))
        borders.append(border_energy(upsample(raw, image_size)))
    cam.remove()
    return np.array(maps), np.array(borders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--results_dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--n_per_class", type=int, default=60)
    args = ap.parse_args()

    set_seed()
    rd = args.results_dir

    _, test_mammo = build_datasets("mammonet", args.data_dir)
    _, test_resnet = build_datasets("resnet18", args.data_dir)
    assert test_mammo.samples == test_resnet.samples

    mammonet = MammoNet()
    mammonet.load_state_dict(torch.load(rd / "mammonet_best.pt", map_location="cpu"))
    mammonet.eval()

    resnet = build_resnet18(pretrained=False)
    resnet.load_state_dict(torch.load(rd / "resnet18_best.pt", map_location="cpu"))
    resnet.eval()

    idx_to_class = {v: k for k, v in test_mammo.class_to_idx.items()}

    # próbka zrównoważona klasowo
    selection = []
    for cls in idx_to_class:
        pool = [i for i, (_, lbl) in enumerate(test_mammo.samples) if lbl == cls]
        random.shuffle(pool)
        selection += [(i, cls) for i in pool[:args.n_per_class]]
    random.shuffle(selection)
    print(f"Próbka: {len(selection)} obrazów testowych")

    maps_m, border_m = collect_maps(mammonet, "mammonet", test_mammo, selection, 128)
    maps_r, border_r = collect_maps(resnet, "resnet18", test_resnet, selection, 224)

    r_m, sd_m = mean_pairwise_correlation(maps_m)
    r_r, sd_r = mean_pairwise_correlation(maps_r)
    c_m, c_r = centroids(maps_m), centroids(maps_r)

    def summarize(name, corr, corr_sd, border, cent):
        dist = np.sqrt(((cent - 0.5) ** 2).sum(axis=1))
        print(f"\n{name}")
        print(f"  energia brzegowa      {border.mean():.3f} +- {border.std():.3f}")
        print(f"  korelacja par map     {corr:.3f} +- {corr_sd:.3f}")
        print(f"  centroid y            {cent[:, 0].mean():.3f} +- {cent[:, 0].std():.3f}")
        print(f"  centroid x            {cent[:, 1].mean():.3f} +- {cent[:, 1].std():.3f}")
        print(f"  odległość od środka   {dist.mean():.3f} +- {dist.std():.3f}")
        return {
            "border_energy_mean": float(border.mean()),
            "border_energy_sd": float(border.std()),
            "pairwise_corr_mean": corr,
            "pairwise_corr_sd": corr_sd,
            "centroid_y_mean": float(cent[:, 0].mean()),
            "centroid_y_sd": float(cent[:, 0].std()),
            "centroid_x_mean": float(cent[:, 1].mean()),
            "centroid_x_sd": float(cent[:, 1].std()),
            "dist_from_center_mean": float(dist.mean()),
            "dist_from_center_sd": float(dist.std()),
        }

    out = {
        "n_images": len(selection),
        "border_fraction": BORDER_FRACTION,
        "MammoNet": summarize("MammoNet", r_m, sd_m, border_m, c_m),
        "ResNet-18": summarize("ResNet-18", r_r, sd_r, border_r, c_r),
    }
    (rd / "analiza_atrybucji.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

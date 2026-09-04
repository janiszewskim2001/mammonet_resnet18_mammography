"""Skuteczność obu modeli w podgrupach wyznaczonych przez subtelność zmiany
i gęstość utkania piersi.

Obrazy wiązane są z rekordami opisowymi przez indeks zakodowany w nazwie pliku
podczas przygotowania danych.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from torchvision import datasets

import config

META_FILES = {
    "Zwapnienie": "calc_case_description_test_set.csv",
    "Guz": "mass_case_description_test_set.csv",
}


def load_meta(csv_dir):
    out = {}
    for cls, fname in META_FILES.items():
        df = pd.read_csv(csv_dir / fname)
        df.columns = [c.strip().lower() for c in df.columns]
        out[cls] = df
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True,
                    help="katalog z rozpakowanym zbiorem (zawiera csv/)")
    ap.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--results_dir", type=Path, default=config.RESULTS_DIR)
    args = ap.parse_args()

    rd = args.results_dir
    meta = load_meta(args.raw_dir / "csv")

    pm = json.loads((rd / "mammonet_predictions.json").read_text())
    pr = json.loads((rd / "resnet18_predictions.json").read_text())
    assert pm["labels"] == pr["labels"]

    test = datasets.ImageFolder(args.data_dir / "test")
    assert len(test.samples) == len(pm["labels"])
    idx_to_class = {v: k for k, v in test.class_to_idx.items()}

    rows = []
    for (path, label), pred_m, pred_r in zip(test.samples, pm["preds"], pr["preds"]):
        cls = idx_to_class[label]
        row_idx = int(Path(path).stem.split("_")[-1])
        rec = meta[cls].iloc[row_idx]
        rows.append({
            "class": cls,
            "subtlety": rec.get("subtlety", np.nan),
            "density": rec.get("breast density", rec.get("breast_density", np.nan)),
            "mammonet_correct": pred_m == label,
            "resnet_correct": pred_r == label,
        })

    df = pd.DataFrame(rows)
    matched = df["subtlety"].notna().sum()
    print(f"Dopasowano metadane: {matched}/{len(df)}")
    df.to_csv(rd / "metadane_polaczone.csv", index=False, sep=";", encoding="utf-8-sig")

    for col, label, fname in [
        ("subtlety", "subtelność zmiany", "dokladnosc_subtlety.csv"),
        ("density", "gęstość utkania", "dokladnosc_gestosc.csv"),
    ]:
        agg = (df.dropna(subset=[col])
                 .groupby(col)
                 .agg(n=("class", "count"),
                      mammonet_acc=("mammonet_correct", "mean"),
                      resnet_acc=("resnet_correct", "mean"))
                 .reset_index())
        print(f"\nDokładność wg {label}")
        print(agg.to_string(index=False, float_format="%.4f"))
        agg.to_csv(rd / fname, index=False, sep=";", encoding="utf-8-sig")


if __name__ == "__main__":
    main()

"""Porządkuje surowy eksport CBIS-DDSM do struktury ImageFolder."""

import argparse
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_MAP = {"calcification": "Zwapnienie", "mass": "Guz"}

CSV_TASKS = [
    ("calc_case_description_train_set.csv", "calcification", "train"),
    ("calc_case_description_test_set.csv", "calcification", "test"),
    ("mass_case_description_train_set.csv", "mass", "train"),
    ("mass_case_description_test_set.csv", "mass", "test"),
]


def build_lookup(dicom_info_csv):
    """Mapuje SeriesInstanceUID na ścieżkę JPEG i kategorię serii.

    Ścieżki w plikach opisowych wskazują na pliki .dcm, a obrazy leżą w katalogach
    nazwanych identyfikatorami serii. Wycinek zmiany i jego maska często dzielą ten
    sam katalog, więc rozróżnia je wyłącznie pole SeriesDescription.
    """
    df = pd.read_csv(dicom_info_csv)

    def parse(image_path):
        p = str(image_path).strip().replace("\\", "/")
        rel = p.split("jpeg/", 1)[1] if "jpeg/" in p else p
        return rel.split("/")[0], rel

    parsed = df["image_path"].apply(parse)
    df["series_uid"] = parsed.apply(lambda t: t[0])
    df["rel_path"] = parsed.apply(lambda t: t[1])

    def categorize(desc):
        d = str(desc).lower()
        for key in ("crop", "mask", "full"):
            if key in d:
                return key
        return "other"

    df["category"] = df["SeriesDescription"].apply(categorize)
    return df[["series_uid", "category", "rel_path"]]


def series_uid(dcm_path):
    return str(dcm_path).strip().replace("\\", "/").split("/")[-2]


def lookup_jpeg(table, uid, category):
    hit = table[(table.series_uid == uid) & (table.category == category)]
    return hit.iloc[0]["rel_path"] if len(hit) else None


def readable(path):
    try:
        Image.open(path).verify()
        return True
    except Exception:
        return False


def process(csv_path, abnormality, split, raw_dir, out_dir, masks_dir, table):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    cls = CLASS_MAP[abnormality]
    img_dir = out_dir / split / cls
    mask_dir = masks_dir / split / cls
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ok = missing_img = missing_mask = 0

    for i, row in df.iterrows():
        rel_img = lookup_jpeg(table, series_uid(row["cropped image file path"]), "crop")
        rel_mask = lookup_jpeg(table, series_uid(row["roi mask file path"]), "mask")

        if rel_img is None:
            missing_img += 1
            continue

        src = raw_dir / "jpeg" / rel_img
        if not src.exists() or not readable(src):
            missing_img += 1
            continue

        # indeks wiersza w nazwie pliku pozwala później odtworzyć metadane kliniczne
        patient = str(row.get("patient_id", i)).replace("/", "_")
        name = f"{cls}_{split}_{patient}_{i}.jpg"

        shutil.copy(src, img_dir / name)
        ok += 1

        if rel_mask is not None:
            m = raw_dir / "jpeg" / rel_mask
            if m.exists() and readable(m):
                shutil.copy(m, mask_dir / name)
            else:
                missing_mask += 1
        else:
            missing_mask += 1

    print(f"{split}/{cls}: {ok} obrazów, brak obrazu: {missing_img}, brak maski: {missing_mask}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=Path, required=True,
                    help="katalog z rozpakowanym zbiorem (zawiera csv/ i jpeg/)")
    ap.add_argument("--out_dir", type=Path, default=Path("data"))
    ap.add_argument("--masks_dir", type=Path, default=Path("data_masks"))
    args = ap.parse_args()

    csv_dir = args.raw_dir / "csv"
    table = build_lookup(csv_dir / "dicom_info.csv")
    print(f"Wpisów w tabeli DICOM: {len(table)}")

    for fname, abnormality, split in CSV_TASKS:
        path = csv_dir / fname
        if not path.exists():
            print(f"Pominięto brakujący plik: {fname}")
            continue
        process(path, abnormality, split, args.raw_dir, args.out_dir, args.masks_dir, table)


if __name__ == "__main__":
    main()

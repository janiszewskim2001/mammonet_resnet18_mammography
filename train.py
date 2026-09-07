"""Trening MammoNet lub ResNet-18 na wycinkach CBIS-DDSM."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data import build_loaders, get_device, set_seed
from models import MammoNet, build_resnet18, check_attention_gradient, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss = correct = seen = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            seen += y.size(0)

    return total_loss / seen, correct / seen


def predict(model, loader, device):
    model.eval()
    labels, probs, preds = [], [], []
    with torch.no_grad():
        for x, y in loader:
            p = F.softmax(model(x.to(device)), dim=1).cpu().numpy()
            labels += y.numpy().tolist()
            probs += p[:, 1].tolist()
            preds += np.argmax(p, axis=1).tolist()
    return labels, probs, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mammonet", "resnet18"], required=True)
    ap.add_argument("--data_dir", type=Path, default=config.DATA_DIR)
    ap.add_argument("--results_dir", type=Path, default=config.RESULTS_DIR)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    data_dir, results_dir = args.data_dir, args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed()
    device = get_device()
    cfg = config.MAMMONET if args.arch == "mammonet" else config.RESNET
    epochs = args.epochs or cfg["epochs"]

    train_loader, test_loader, train_ds, test_ds = build_loaders(args.arch, data_dir)
    print(f"Urządzenie: {device}")
    print(f"Obrazy: {len(train_ds)} treningowych, {len(test_ds)} testowych")

    if args.arch == "mammonet":
        model = MammoNet().to(device)
        norm = check_attention_gradient(model, device)
        print(f"Norma gradientu warstwy uwagi: {norm:.6f}")
        assert norm > 0, "gradient nie dociera do warstwy uwagi"
    else:
        model = build_resnet18().to(device)

    print(f"Parametry: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    ckpt = results_dir / f"{args.arch}_best.pt"
    history = {k: [] for k in ("train_loss", "test_loss", "train_acc", "test_acc")}
    best = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        te_loss, te_acc = run_epoch(model, test_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["test_loss"].append(te_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        # dokładność testowa waha się między epokami, więc zapisujemy najlepszy stan,
        # a nie ostatni
        flag = ""
        if te_acc > best:
            best = te_acc
            torch.save(model.state_dict(), ckpt)
            flag = "  *"

        print(f"[{epoch:2d}/{epochs}] train {tr_loss:.4f}/{tr_acc:.4f}  "
              f"test {te_loss:.4f}/{te_acc:.4f}{flag}")

    print(f"\nNajlepsza dokładność testowa: {best:.4f}")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    labels, probs, preds = predict(model, test_loader, device)

    (results_dir / f"{args.arch}_history.json").write_text(json.dumps(history, indent=2))
    (results_dir / f"{args.arch}_predictions.json").write_text(json.dumps({
        "labels": labels, "probs": probs, "preds": preds,
        "class_to_idx": test_ds.class_to_idx,
    }, indent=2))


if __name__ == "__main__":
    main()

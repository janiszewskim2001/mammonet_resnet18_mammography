import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(arch):
    cfg = config.MAMMONET if arch == "mammonet" else config.RESNET
    norm = config.NORM_GRAY if arch == "mammonet" else config.NORM_IMAGENET
    return transforms.Compose([
        transforms.Resize((cfg["size"], cfg["size"])),
        transforms.Grayscale(num_output_channels=cfg["channels"]),
        transforms.ToTensor(),
        transforms.Normalize(*norm),
    ])


def build_datasets(arch, data_dir=config.DATA_DIR):
    tf = build_transform(arch)
    train = datasets.ImageFolder(data_dir / "train", transform=tf)
    test = datasets.ImageFolder(data_dir / "test", transform=tf)

    # ImageFolder numeruje klasy alfabetycznie; niezgodność zerwałaby porównanie modeli
    assert train.class_to_idx == config.CLASS_TO_IDX, train.class_to_idx
    assert test.class_to_idx == config.CLASS_TO_IDX, test.class_to_idx

    return train, test


def build_loaders(arch, data_dir=config.DATA_DIR):
    cfg = config.MAMMONET if arch == "mammonet" else config.RESNET
    train, test = build_datasets(arch, data_dir)
    return (
        DataLoader(train, batch_size=cfg["batch_size"], shuffle=True),
        DataLoader(test, batch_size=cfg["batch_size"], shuffle=False),
        train,
        test,
    )

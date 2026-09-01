"""
Train a clean teacher and save it to disk.

The smoke test trained a teacher in memory and then discarded it, which meant every
experiment retrained from scratch. The detector needs a fixed clean teacher to calibrate
against, and it needs the same one on every run, or the calibration drifts between
experiments and the results stop being comparable.

This teacher is clean in every sense: no trigger, no poisoned samples, no hijack. It is
the honest reference the detector learns normal behaviour from, and the victim's starting
point in the threat model.

The defaults are small enough to run on a CPU in a few minutes, which is useful for
checking that the pipeline works. The reported results come from a ResNet-50 trained on
full CIFAR-10 on a GPU. Nothing downstream depends on which is used, because the detector
loads whatever checkpoint it is pointed at.

Usage:
    python train_teacher.py
    python train_teacher.py --n_train 50000 --epochs 30 --model resnet50 --device cuda
"""

import os
import sys
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets, utils  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def get_loaders(n_train, n_test, batch_size, seed=0):
    """Plain [0,1] tensors — no normalisation, so the patch trigger stays a valid image."""
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    if n_train:
        train = Subset(train, range(min(n_train, len(train))))
    if n_test:
        test = Subset(test, range(min(n_test, len(test))))
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g),
            DataLoader(test, batch_size=batch_size, shuffle=False))


def train(model, loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        running, nb = 0.0, 0
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = ce(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        sched.step()
        print(f"  epoch {ep + 1}/{epochs}  loss={running / max(nb, 1):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return model


def main():
    p = argparse.ArgumentParser(description="Train and save a clean teacher")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--n_train", type=int, default=10000, help="0 = full training set")
    p.add_argument("--n_test", type=int, default=2000, help="0 = full test set")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="checkpoint path (default: checkpoints/clean_teacher_<model>.pth)")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds Python, NumPy and torch, including model initialisation")
    args = p.parse_args()

    # Seed everything that affects the trained weights, not only the data order. An earlier
    # version seeded the DataLoader generator alone, which left model initialisation free and
    # meant two runs of the same command did not produce the same teacher. Note that this makes
    # a run repeatable going forward; it cannot make an already-completed run reproducible, and
    # the teachers behind the reported results were trained before this was added.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    os.makedirs(CKP_DIR, exist_ok=True)
    out = args.out or os.path.join(CKP_DIR, f"clean_teacher_{args.model}.pth")

    print("=== Training a CLEAN teacher (no trigger, no poison, no hijack) ===")
    print(f"  model={args.model}  n_train={args.n_train or 'full'}  epochs={args.epochs}  device={device}")

    trainloader, testloader = get_loaders(args.n_train, args.n_test, args.batch_size, seed=args.seed)
    model = nets.get_network(args.dataset, args.model).to(device)
    model = train(model, trainloader, device, args.epochs, args.lr)

    model.eval()
    acc = utils.get_acc_results(model, testloader, device)
    print(f"  clean teacher ACC = {acc:.4f}")

    torch.save(model.state_dict(), out)
    print(f"  saved -> {out}")
    print("\nNext: python calibrate_and_eval.py  (builds the detector's clean baseline)")


if __name__ == "__main__":
    main()

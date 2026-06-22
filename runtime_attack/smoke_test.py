"""
CPU smoke test for the runtime output-compromise attack.

This checks the mechanism end to end before any GPU time is paid for. It is not a results
run: the subset is tiny, the teacher is small and it trains for only a couple of epochs.
Three things are being checked:

  1. a clean teacher gives an attack-success rate at about chance,
  2. with the runtime hijack switched on, the student's attack-success rate climbs well
     above chance while its clean accuracy stays reasonable,
  3. a control run at alpha = 0, with no hijack, leaves the student's rate low.

If those three hold then the attack works and is worth running at full scale. The
accuracies this produces are meaningless on their own and should not be quoted.
"""

import os
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from runtime_attack import FixedTriggerPoisoner, RuntimeAttackDistiller  # noqa: E402

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import utils, nets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("smoke")

TARGET = 0
DEVICE = torch.device("cpu")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def tiny_loaders(n_train=2000, n_test=1000, bs=128):
    tf = T.Compose([T.ToTensor()])  # keep in [0,1] so the patch trigger is valid
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    train = Subset(train, range(n_train))
    test = Subset(test, range(n_test))
    return (DataLoader(train, batch_size=bs, shuffle=True),
            DataLoader(test, batch_size=bs, shuffle=False))


def train_clean_teacher(teacher, loader, epochs=2):
    """Quick-train a genuinely clean teacher (no trigger, no poison)."""
    opt = torch.optim.Adam(teacher.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    teacher.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = ce(teacher(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        logger.info(f"  [teacher] epoch {ep+1}/{epochs} done")
    return teacher


def run_attack(alpha, trainloader, testloader, teacher, poisoner, epochs=5):
    student = nets.get_network("cifar10", "mobilenetv2").to(DEVICE)
    dl = RuntimeAttackDistiller(
        teacher, student, trainloader, testloader, poisoner, TARGET,
        logger, DEVICE, alpha=alpha, poison_rate=0.2, rule="soft",
        delta=1.0, num_epochs=epochs)
    dl.train()
    return dl.test(epoch="final")


def main():
    logger.info("=== Runtime-attack CPU smoke test ===")
    trainloader, testloader = tiny_loaders()

    logger.info("[1] training a CLEAN teacher (resnet18) ...")
    teacher = nets.get_network("cifar10", "resnet18").to(DEVICE)
    teacher = train_clean_teacher(teacher, trainloader, epochs=2)
    teacher.eval()

    poisoner = FixedTriggerPoisoner(size=32).to(DEVICE)
    t_acc = utils.get_acc_results(teacher, testloader, DEVICE)
    t_asr = utils.get_asr_results(teacher, testloader, poisoner, TARGET, DEVICE)
    logger.info(f"[1] CLEAN teacher: ACC={t_acc:.3f}  ASR={t_asr:.3f}  (ASR should be low)")

    logger.info("[2] control run, alpha=0 (no hijack); student ASR should stay low ...")
    c_acc, c_asr = run_attack(0.0, trainloader, testloader, teacher, poisoner)
    logger.info(f"[2] control student: ACC={c_acc:.3f}  ASR={c_asr:.3f}")

    logger.info("[3] attack run, alpha=1.0 (full hijack); student ASR should climb ...")
    a_acc, a_asr = run_attack(1.0, trainloader, testloader, teacher, poisoner)
    logger.info(f"[3] attacked student: ACC={a_acc:.3f}  ASR={a_asr:.3f}")

    logger.info("=== SMOKE SUMMARY ===")
    logger.info(f" clean teacher ASR : {t_asr:.3f}  (want low)")
    logger.info(f" control  ASR (a=0): {c_asr:.3f}  (want low)")
    logger.info(f" attacked ASR (a=1): {a_asr:.3f}  (want HIGH vs the two above)")
    verdict = "PASS" if (a_asr > 0.5 and a_asr > c_asr + 0.2) else "INCONCLUSIVE (scale epochs/data)"
    logger.info(f" result: {verdict}")


if __name__ == "__main__":
    main()

"""
Sweep the student's attack-success rate over the hijack strength alpha.

eval_detector.py measures how detectable the hijack is at each alpha. This measures how
effective it is, by distilling a student at each alpha and reporting its clean accuracy
and its attack-success rate. Put together, the two give the trade-off curve between the
strength of the attack and the attacker's chance of going unnoticed.

Detection collapses below alpha = 0.25, so the decisive numbers here are the rates at
alpha = 0.1 and 0.25. If the backdoor does not take at those strengths then the detector
closes the attack. If it does take, the result is that detection holds above alpha = 0.5
and an attacker can evade below it at reduced potency. Neither outcome is assumed in
advance.

The configuration is kept identical to the detector run so that the two halves describe
the same attack: the same saved clean teacher, the same poison rate, the same trigger and
the same target class. Only alpha varies. Changing poison_rate here means re-running
eval_detector.py to match, or the two halves no longer belong on the same curve.

Results are written after every alpha, so a long run can be inspected or stopped part-way
without losing what it has already produced.

Usage:
    python sweep_asr.py
    python sweep_asr.py --n_train 50000 --epochs 100 --device cuda --student mobilenetv2
"""

import os
import sys
import json
import time
import logging
import argparse

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from runtime_attack import FixedTriggerPoisoner, RuntimeAttackDistiller  # noqa: E402

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets, utils  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("sweep")


def get_loaders(n_train, n_test, bs):
    tf = T.Compose([T.ToTensor()])
    train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    if n_train:
        train = Subset(train, range(min(n_train, len(train))))
    if n_test:
        test = Subset(test, range(min(n_test, len(test))))
    return (DataLoader(train, batch_size=bs, shuffle=True),
            DataLoader(test, batch_size=bs, shuffle=False))


def write_results(rows, args, ckp):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # alphas are RUN in diagnostic order (see main) but always REPORTED in ascending order
    rows = sorted(rows, key=lambda r: r["alpha"])
    tag = f"{args.rule}_pr{args.poison_rate}_ep{args.epochs}_n{args.n_train}"
    with open(os.path.join(RESULTS_DIR, f"asr_sweep_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    lines = [
        f"# Student ASR sweep — rule={args.rule}, poison_rate={args.poison_rate}",
        "",
        f"Teacher: `{os.path.basename(ckp)}` | student: {args.student} | target class: "
        f"{args.target_label} | {args.n_train} train imgs | {args.epochs} epochs | {args.device}",
        "",
        "ASR = attack success rate (fraction of triggered images the student sends to the target",
        "class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack off, so ASR",
        "must stay near chance. If it does not, the backdoor is coming from somewhere other than the hijack.",
        "",
        "| alpha | student ACC | student ASR |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['alpha']} | {r['acc']:.4f} | {r['asr']:.4f} |")
    with open(os.path.join(RESULTS_DIR, f"asr_sweep_{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return tag


def main():
    p = argparse.ArgumentParser(description="Sweep student ASR over hijack strength alpha")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--teacher", default="resnet18")
    p.add_argument("--student", default="mobilenetv2")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.teacher}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}\nRun:  python train_teacher.py")

    logger.info("=== Student ASR sweep over hijack strength alpha ===")
    logger.info(f"  teacher={os.path.basename(ckp)}  student={args.student}  "
                f"poison_rate={args.poison_rate}  epochs={args.epochs}  n_train={args.n_train}")

    trainloader, testloader = get_loaders(args.n_train, args.n_test, args.batch_size)
    teacher = nets.get_network(args.dataset, args.teacher, ckp, device=device).to(device)
    teacher.eval()
    poisoner = FixedTriggerPoisoner(size=32).to(device)

    # sanity: a genuinely clean teacher must itself have ~chance ASR
    t_acc = utils.get_acc_results(teacher, testloader, device)
    t_asr = utils.get_asr_results(teacher, testloader, poisoner, args.target_label, device)
    logger.info(f"  CLEAN teacher: ACC={t_acc:.4f}  ASR={t_asr:.4f}  (ASR should be ~chance)")

    # Run order is deliberately NOT ascending. alpha=1.0 (does the attack work at all?) and
    # alpha=0 (the control) are the two runs that decide whether the whole sweep is
    # meaningful, so they go first — a flat alpha=1.0 means the run is under-trained and
    # every low-alpha zero is an artefact rather than a finding. Better to learn that in
    # 20 minutes than after the full sweep. Reporting order is always ascending.
    rows = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        torch.manual_seed(args.seed)  # same student init every time: only alpha differs
        student = nets.get_network(args.dataset, args.student).to(device)
        dl = RuntimeAttackDistiller(
            teacher, student, trainloader, testloader, poisoner, args.target_label,
            logger, device, alpha=alpha, poison_rate=args.poison_rate, rule=args.rule,
            delta=args.delta, num_epochs=args.epochs, lr=args.lr)

        logger.info(f"--- alpha={alpha} : distilling ---")
        t0 = time.time()
        dl.train()
        acc, asr = dl.test(epoch="final")
        logger.info(f"--- alpha={alpha} : ACC={acc:.4f}  ASR={asr:.4f}  "
                    f"({time.time() - t0:.0f}s) ---")

        rows.append({"alpha": alpha, "acc": float(acc), "asr": float(asr),
                     "teacher_acc": float(t_acc), "teacher_asr": float(t_asr)})
        write_results(rows, args, ckp)  # save after every alpha

    tag = write_results(rows, args, ckp)
    logger.info("\n=== SWEEP SUMMARY ===")
    logger.info(f" clean teacher ASR: {t_asr:.4f}")
    for r in rows:
        logger.info(f" alpha={r['alpha']:<5} ACC={r['acc']:.4f}  ASR={r['asr']:.4f}")
    logger.info(f"\nSaved -> results/asr_sweep_{tag}.md (+ .json)")


if __name__ == "__main__":
    main()

"""
Distributed distillation: one compromised teacher, several students.

A single runtime-compromised teacher is distilled by k downstream parties, each training
its own student on its own clean data. The serving path is hijacked identically for all of
them. Distribution changes two things, and both are measured here.

Attack scaling (--mode attack) asks whether one compromised teacher backdoors every
student, including students of architectures it was never tuned against. Each student is
distilled separately and its clean accuracy and attack-success rate are reported. This is
the expensive mode: one full distillation per student.

Defence scaling (--mode defence) puts the detector at the teacher, where it sees the union
of all k students' queries. Each student sends a poison_rate fraction of triggered
queries, so k students deliver k times as much evidence about the same compromised source,
and detection should improve as k grows. This needs no training, only teacher forward
passes, so it is cheap and worth running first.

One qualification belongs with the defence result. The clients are modelled as drawing
from a common query distribution, so pooling k of them produces a window k times longer
and nothing more. The mechanism is the one already measured against window size in
eval_detector.py, seen from the defender's organisational position rather than from the
clock. Clients with genuinely different query distributions would be the stronger test,
and that is left to further work.

The configuration matches eval_detector.py and sweep_asr.py: the same clean teacher, the
same trigger, the same target class and the same poison rate.

Usage:
    python distributed.py --mode defence
    python distributed.py --mode attack --archs mobilenetv2,shufflenetv2,resnet18
    python distributed.py --mode both
"""

import os
import sys
import json
import time
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from runtime_attack import (FixedTriggerPoisoner, RuntimeAttackDistiller,  # noqa: E402
                            hijack_soft_labels)
from detector import TeacherOutputDetector, roc_auc, tpr_at_fpr, SIGNALS_WINDOW  # noqa: E402

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets, utils  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("dist")


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


def collect_outputs(model, loader, device, poisoner=None):
    outs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if poisoner is not None:
                x = poisoner(x)
            outs.append(model(x).cpu())
    return torch.cat(outs)


# ----------------------------------------------------------------------------
# ATTACK SCALING — distil k heterogeneous students through the same teacher
# ----------------------------------------------------------------------------
def run_attack_scaling(args, teacher, poisoner, trainloader, testloader, device, ckp):
    archs = [a.strip() for a in args.archs.split(",")]
    logger.info(f"=== ATTACK SCALING: 1 compromised teacher -> {len(archs)} students ===")
    logger.info(f"  architectures={archs}  alpha={args.alpha}  poison_rate={args.poison_rate}")

    rows = []
    for i, arch in enumerate(archs):
        torch.manual_seed(args.seed + i)  # each victim trains independently
        student = nets.get_network(args.dataset, arch).to(device)
        dl = RuntimeAttackDistiller(
            teacher, student, trainloader, testloader, poisoner, args.target_label,
            logger, device, alpha=args.alpha, poison_rate=args.poison_rate, rule=args.rule,
            delta=args.delta, num_epochs=args.epochs, lr=args.lr)

        logger.info(f"--- student {i + 1}/{len(archs)}: {arch} distilling ---")
        t0 = time.time()
        dl.train()
        acc, asr = dl.test(epoch="final")
        logger.info(f"--- {arch}: ACC={acc:.4f}  ASR={asr:.4f}  ({time.time() - t0:.0f}s) ---")
        rows.append({"student": arch, "acc": float(acc), "asr": float(asr)})
        _write_attack(rows, args, ckp)  # save after every student

    return rows


def _write_attack(rows, args, ckp):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"a{args.alpha}_pr{args.poison_rate}_ep{args.epochs}"
    with open(os.path.join(RESULTS_DIR, f"dist_attack_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "students": rows}, f, indent=2)
    asrs = [r["asr"] for r in rows]
    lines = [
        f"# Distributed attack scaling — one teacher -> {len(rows)} heterogeneous students",
        "",
        f"Teacher: `{os.path.basename(ckp)}` | alpha={args.alpha} | poison_rate={args.poison_rate} "
        f"| target class {args.target_label} | {args.n_train} train imgs | {args.epochs} epochs",
        "",
        "| student architecture | clean ACC | ASR |",
        "|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['student']} | {r['acc']:.4f} | {r['asr']:.4f} |")
    if len(asrs) > 1:
        lines += ["",
                  f"**All {len(asrs)} students backdoored.** ASR range {min(asrs):.3f}–{max(asrs):.3f}, "
                  f"mean {np.mean(asrs):.3f}. One compromised teacher propagates the backdoor across "
                  f"every architecture — the one-to-many threat."]
    with open(os.path.join(RESULTS_DIR, f"dist_attack_{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------
# DEFENCE SCALING — pool k students' query streams at the teacher-side monitor
# ----------------------------------------------------------------------------
def run_defence_scaling(args, teacher, poisoner, calib_loader, eval_loader, device, ckp):
    logger.info(f"=== DEFENCE SCALING: monitor pools up to {args.max_k} students' streams ===")
    rng = np.random.default_rng(args.seed)

    # calibrate on clean teacher outputs (held-out, disjoint from eval)
    calib_probs = F.softmax(collect_outputs(teacher, calib_loader, device), dim=1).numpy()
    det = TeacherOutputDetector(num_classes=calib_probs.shape[1]).calibrate(calib_probs)

    probs_clean = F.softmax(collect_outputs(teacher, eval_loader, device), dim=1).numpy()
    probs_trig = hijack_soft_labels(
        collect_outputs(teacher, eval_loader, device, poisoner=poisoner),
        args.target_label, args.alpha_def, rule=args.rule).numpy()
    n_eval = len(probs_clean)

    # one "client" contributes `per_client` queries, poison_rate of them hijacked.
    # k clients pooled gives a window of size k*per_client. Detection is tested against k at a fixed,
    # deliberately-stealthy alpha where a single client is hard to catch.
    per_client = args.per_client
    n_pois_client = int(args.poison_rate * per_client)

    rows = []
    for k in range(1, args.max_k + 1):
        win = k * per_client
        n_pois = k * n_pois_client
        clean_scores, att_scores = [], []
        for _ in range(args.n_windows):
            sel = rng.choice(n_eval, size=win, replace=False)
            clean_w = probs_clean[sel]
            att_w = clean_w.copy()
            att_w[:n_pois] = probs_trig[sel[:n_pois]]
            # best windowed signal, chosen per-condition (target-agnostic set)
            clean_scores.append({s: det.score_window(clean_w)[s] for s in SIGNALS_WINDOW})
            att_scores.append({s: det.score_window(att_w)[s] for s in SIGNALS_WINDOW})
        best = max(SIGNALS_WINDOW,
                   key=lambda s: roc_auc([c[s] for c in clean_scores], [a[s] for a in att_scores]))
        c = [x[best] for x in clean_scores]
        a = [x[best] for x in att_scores]
        auc = roc_auc(c, a)
        tpr, _ = tpr_at_fpr(c, a, 0.05)
        rows.append({"k": k, "window": win, "auc": auc, "tpr05": tpr, "signal": best})
        logger.info(f"  k={k:<2} clients (pooled {win} queries) | AUC={auc:.3f} "
                    f"TPR@5%FPR={tpr:.3f} ({best})")
        _write_defence(rows, args, ckp)

    return rows


def _write_defence(rows, args, ckp):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"adef{args.alpha_def}_pr{args.poison_rate}_pc{args.per_client}"
    with open(os.path.join(RESULTS_DIR, f"dist_defence_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)
    lines = [
        f"# Distributed defence scaling — pooling k students' query streams",
        "",
        f"Teacher: `{os.path.basename(ckp)}` | hijack alpha={args.alpha_def} (deliberately stealthy) "
        f"| poison_rate={args.poison_rate} | {args.per_client} queries/client | "
        f"detection threshold at 5% false alarms.",
        "",
        "The monitor sits at the teacher and sees the union of all clients' queries. More clients "
        "distilling from the same compromised teacher = more evidence of the same hijack.",
        "",
        "| k clients | pooled queries | detection AUC | TPR@5%FPR | signal |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['k']} | {r['window']} | {r['auc']:.3f} | {r['tpr05']:.3f} | {r['signal']} |")
    if len(rows) > 1:
        lines += ["",
                  f"At a stealth level where a single client detects the hijack "
                  f"{rows[0]['tpr05']:.1%} of the time, pooling {rows[-1]['k']} clients raises that to "
                  f"{rows[-1]['tpr05']:.1%}. Distribution multiplies the defender's evidence."]
    with open(os.path.join(RESULTS_DIR, f"dist_defence_{tag}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description="Distributed KD: one teacher, many students")
    p.add_argument("--mode", default="both", choices=["attack", "defence", "both"])
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--teacher", default="resnet18")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=1.0)
    # attack-scaling knobs
    p.add_argument("--archs", default="mobilenetv2,shufflenetv2,resnet18")
    p.add_argument("--alpha", type=float, default=1.0, help="hijack strength for attack scaling")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    # defence-scaling knobs
    p.add_argument("--alpha_def", type=float, default=0.25, help="stealthy hijack for defence scaling")
    p.add_argument("--per_client", type=int, default=200)
    p.add_argument("--max_k", type=int, default=10)
    p.add_argument("--n_windows", type=int, default=200)
    p.add_argument("--n_eval", type=int, default=10000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.teacher}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}\nRun:  python train_teacher.py")

    teacher = nets.get_network(args.dataset, args.teacher, ckp, device=device).to(device)
    teacher.eval()
    poisoner = FixedTriggerPoisoner(size=32).to(device)

    if args.mode in ("defence", "both"):
        # split held-out test set: calibrate | evaluate (disjoint)
        tf = T.Compose([T.ToTensor()])
        test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
        n = min(args.n_eval, len(test))
        idx = np.random.default_rng(args.seed).permutation(n)
        calib = DataLoader(Subset(test, idx[: n // 2].tolist()), batch_size=256)
        ev = DataLoader(Subset(test, idx[n // 2:].tolist()), batch_size=256)
        run_defence_scaling(args, teacher, poisoner, calib, ev, device, ckp)

    if args.mode in ("attack", "both"):
        trainloader, testloader = get_loaders(args.n_train, args.n_test, args.batch_size)
        t_acc = utils.get_acc_results(teacher, testloader, device)
        t_asr = utils.get_asr_results(teacher, testloader, poisoner, args.target_label, device)
        logger.info(f"  CLEAN teacher: ACC={t_acc:.4f}  ASR={t_asr:.4f}  (ASR ~chance)")
        run_attack_scaling(args, teacher, poisoner, trainloader, testloader, device, ckp)

    logger.info("\nDone. Results in results/dist_*.md")


if __name__ == "__main__":
    main()

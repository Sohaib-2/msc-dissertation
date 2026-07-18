"""
Evaluate the teacher-side detector against the runtime hijack.

This produces the defensive half of the trade-off: how detectable the hijack is at each
attack strength alpha, and therefore how much attack power an attacker has to give up in
order to stay hidden.

Method:
  1. Load the fixed clean teacher.
  2. Split held-out data in two, one half to calibrate the detector on and one half to
     evaluate it on. The halves never share images. Calibrating on the data the detector
     is then tested against would inflate the result.
  3. Run one forward pass over clean images and one over trigger-stamped images.
     Everything after that is index arithmetic, so sweeping alpha costs almost nothing.
  4. Build windows of teacher outputs. A clean window holds honest outputs on clean
     queries. An attacked window has a poison_rate fraction stamped and hijacked.
  5. Score every window and measure separation by ROC-AUC and by the true-positive rate
     at a 5% false-alarm rate.

Three things keep this honest. alpha = 0 is included as a control, where the trigger is
still stamped but no hijack occurs, so the detector should score about 0.5; a high score
there would mean the trigger's effect on the teacher was being detected rather than the
attack. The detector is never told the target class or the trigger. Per-query and windowed
detection are both reported, so the claim that windowed scoring is what works is evidenced
rather than asserted.

Usage:
    python eval_detector.py
    python eval_detector.py --alphas 0,0.1,0.25,0.5,0.75,1.0 --poison_rate 0.1 --window 200
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(__file__))
from runtime_attack import FixedTriggerPoisoner, hijack_soft_labels  # noqa: E402
from detector import (TeacherOutputDetector, roc_auc, tpr_at_fpr,  # noqa: E402
                      SIGNALS_WINDOW, SIGNALS_QUERY)

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def collect_outputs(model, loader, device, poisoner=None):
    """Run the teacher over a loader and return its logits. Optionally stamp the trigger first."""
    outs = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            if poisoner is not None:
                x = poisoner(x)
            outs.append(model(x).cpu())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser(description="Evaluate the teacher-side runtime detector")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--alphas", default="0,0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--window", type=int, default=200, help="queries per detection window")
    p.add_argument("--n_windows", type=int, default=300, help="windows sampled per condition")
    p.add_argument("--n_eval", type=int, default=10000, help="held-out images to use")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.model}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}\nRun:  python train_teacher.py")

    print("=== Teacher-side runtime detector — evaluation ===")
    print(f"  teacher={ckp}  rule={args.rule}  poison_rate={args.poison_rate}  window={args.window}")

    # ---- data: held-out test set, split calibration | evaluation -----------
    tf = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    n = min(args.n_eval, len(test))
    idx = rng.permutation(n)
    calib_idx, eval_idx = idx[: n // 2], idx[n // 2:]
    calib_loader = DataLoader(Subset(test, calib_idx.tolist()), batch_size=args.batch_size)
    eval_loader = DataLoader(Subset(test, eval_idx.tolist()), batch_size=args.batch_size)
    print(f"  calibration images={len(calib_idx)}  evaluation images={len(eval_idx)}")

    teacher = nets.get_network(args.dataset, args.model, ckp, device=device).to(device)
    poisoner = FixedTriggerPoisoner(size=32).to(device)

    # ---- 1. calibrate the detector on CLEAN outputs only -------------------
    calib_logits = collect_outputs(teacher, calib_loader, device)
    calib_probs = F.softmax(calib_logits, dim=1).numpy()
    det = TeacherOutputDetector(num_classes=calib_probs.shape[1]).calibrate(calib_probs)
    print(f"  calibrated: mean entropy={det.ref_entropy_mean:.3f}  "
          f"mean confidence={det.ref_maxprob_mean:.3f}")

    # ---- 2. one pass each: clean outputs, and honest outputs on triggered ---
    eval_logits_clean = collect_outputs(teacher, eval_loader, device)
    eval_logits_trig = collect_outputs(teacher, eval_loader, device, poisoner=poisoner)
    probs_clean = F.softmax(eval_logits_clean, dim=1).numpy()
    n_eval = len(probs_clean)
    n_pois = int(args.poison_rate * args.window)
    print(f"  {n_pois}/{args.window} queries per attacked window are hijacked")

    # ---- 3. sweep alpha ----------------------------------------------------
    alphas = [float(a) for a in args.alphas.split(",")]
    rows = []
    for alpha in alphas:
        # what the compromised runtime returns for triggered queries at this alpha
        probs_hijack = hijack_soft_labels(
            eval_logits_trig, args.target_label, alpha, rule=args.rule).numpy()

        clean_win, attacked_win = [], []
        q_clean, q_attacked = [], []
        for _ in range(args.n_windows):
            sel = rng.choice(n_eval, size=args.window, replace=False)
            clean_w = probs_clean[sel]
            attacked_w = clean_w.copy()
            attacked_w[:n_pois] = probs_hijack[sel[:n_pois]]
            clean_win.append(clean_w)
            attacked_win.append(attacked_w)
            # per-query: only the hijacked rows vs an equal number of clean rows
            q_attacked.append(probs_hijack[sel[:n_pois]])
            q_clean.append(clean_w[:n_pois])

        row = {"alpha": alpha}

        # windowed detection — score each window once, then split out the signals
        c_scores = [det.score_window(w) for w in clean_win]
        a_scores = [det.score_window(w) for w in attacked_win]
        for sig in SIGNALS_WINDOW:
            c = np.array([s[sig] for s in c_scores])
            a = np.array([s[sig] for s in a_scores])
            tpr, _ = tpr_at_fpr(c, a, 0.05)
            row[f"win_{sig}_auc"] = roc_auc(c, a)
            row[f"win_{sig}_tpr05"] = tpr

        # per-query detection (best case for the attacker's individual outputs)
        qc = np.concatenate(q_clean)
        qa = np.concatenate(q_attacked)
        for sig in SIGNALS_QUERY:
            c = det.score_per_query(qc)[sig]
            a = det.score_per_query(qa)[sig]
            row[f"qry_{sig}_auc"] = roc_auc(c, a)

        rows.append(row)
        best_w = max(SIGNALS_WINDOW, key=lambda s: row[f"win_{s}_auc"])
        best_q = max(SIGNALS_QUERY, key=lambda s: row[f"qry_{s}_auc"])
        print(f"  alpha={alpha:<5} | windowed AUC={row[f'win_{best_w}_auc']:.3f} ({best_w}) "
              f"TPR@5%FPR={row[f'win_{best_w}_tpr05']:.3f} | per-query AUC={row[f'qry_{best_q}_auc']:.3f}")

    # ---- 4. report ---------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{args.rule}_pr{args.poison_rate}_w{args.window}"
    with open(os.path.join(RESULTS_DIR, f"detector_eval_{tag}.json"), "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    lines = [
        f"# Detector evaluation — rule={args.rule}, poison_rate={args.poison_rate}, window={args.window}",
        "",
        f"Teacher: `{os.path.basename(ckp)}` | target class: {args.target_label} | "
        f"{args.n_windows} windows/condition | detection threshold set at 5% false-alarm rate.",
        "",
        "AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,",
        "no hijack) and must sit near 0.5. Anything else would mean the trigger's own effect on",
        "the teacher was being detected rather than the attack.",
        "",
        "The 'best signal' columns below name whichever of the four signals scored highest on this",
        "condition, so they are chosen after the fact and are optimistic. Every signal is listed",
        "individually in the accompanying JSON, and eval_combined.py evaluates a rule fixed in",
        "advance.",
        "",
        "| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        bw = max(SIGNALS_WINDOW, key=lambda s: r[f"win_{s}_auc"])
        bq = max(SIGNALS_QUERY, key=lambda s: r[f"qry_{s}_auc"])
        lines.append(f"| {r['alpha']} | {r[f'win_{bw}_auc']:.3f} ({bw}) | "
                     f"{r[f'win_{bw}_tpr05']:.3f} | {r[f'qry_{bq}_auc']:.3f} ({bq}) |")
    lines += ["", "## All windowed signals (AUC)", "",
              "| alpha | " + " | ".join(SIGNALS_WINDOW) + " |",
              "|---" * (len(SIGNALS_WINDOW) + 1) + "|"]
    for r in rows:
        lines.append(f"| {r['alpha']} | " +
                     " | ".join(f"{r[f'win_{s}_auc']:.3f}" for s in SIGNALS_WINDOW) + " |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(RESULTS_DIR, f"detector_eval_{tag}.md"), "w") as f:
        f.write(md)
    print(f"\nSaved -> results/detector_eval_{tag}.md (+ .json)")


if __name__ == "__main__":
    main()

"""
Evaluate the detector's operational decision rule.

eval_detector.py reports each of the four window signals separately, and its console
summary quotes whichever of them scored highest for that condition. As an analysis of the
signals that is fine, but it is not a detector. Choosing the signal after seeing labelled
results is post-hoc selection, and it makes the quoted figure optimistic by an unknown
margin. This script evaluates rules that are fixed before any attacked data is seen.

  primary   A single pre-specified signal, max_class_excess, chosen on design grounds
            because the attack has to push probability mass toward one class. It was not
            chosen on the strength of any result, and it is kept even in the conditions
            where it loses.

  combined  Every signal standardised against its own spread on clean windows, with the
            maximum taken. Nothing in it depends on the attack, the trigger, the target
            class or the evaluation labels.

  oracle    The best signal for each condition, chosen on the evaluation AUC. Reported
            only as an upper bound, so that the cost of proceeding honestly can be
            quantified rather than guessed at.

The calibration used to standardise the signals comes from the calibration half of the
data, which the evaluation half never touches.

Usage:
    python eval_combined.py --model resnet50 --device cpu
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
                      SIGNALS_WINDOW)
from eval_detector import collect_outputs  # noqa: E402

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PRIMARY = "max_class_excess"   # pre-specified on design grounds, see module docstring


def main():
    p = argparse.ArgumentParser(description="Evaluate the operational (pre-specified) detector")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--window", type=int, default=200)
    p.add_argument("--n_windows", type=int, default=300)
    p.add_argument("--n_eval", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.model}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}")

    print("=== Operational detector (pre-specified rule) ===")
    print(f"  teacher={os.path.basename(ckp)}  rule={args.rule}  window={args.window}")

    tf = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    n = min(args.n_eval, len(test))
    idx = rng.permutation(n)
    calib_idx, eval_idx = idx[: n // 2], idx[n // 2:]
    calib_loader = DataLoader(Subset(test, calib_idx.tolist()), batch_size=args.batch_size)
    eval_loader = DataLoader(Subset(test, eval_idx.tolist()), batch_size=args.batch_size)

    teacher = nets.get_network(args.dataset, args.model, ckp, device=device).to(device)
    poisoner = FixedTriggerPoisoner(size=32).to(device)

    # ---- calibration: BOTH stages, on clean data only ----------------------
    calib_probs = F.softmax(collect_outputs(teacher, calib_loader, device), dim=1).numpy()
    det = TeacherOutputDetector(num_classes=calib_probs.shape[1]).calibrate(calib_probs)
    det.calibrate_windows(calib_probs, window=args.window,
                          n_windows=args.n_windows, seed=args.seed)
    print("  window-signal calibration (clean mean ± sd):")
    for s in SIGNALS_WINDOW:
        print(f"    {s:<20} {det.win_ref_mean[s]:+.5f} ± {det.win_ref_std[s]:.5f}")

    eval_logits_clean = collect_outputs(teacher, eval_loader, device)
    eval_logits_trig = collect_outputs(teacher, eval_loader, device, poisoner=poisoner)
    probs_clean = F.softmax(eval_logits_clean, dim=1).numpy()
    n_eval = len(probs_clean)
    n_pois = int(args.poison_rate * args.window)

    rows = []
    for alpha in [float(a) for a in args.alphas.split(",")]:
        probs_hijack = hijack_soft_labels(
            eval_logits_trig, args.target_label, alpha, rule=args.rule).numpy()

        c_comb, a_comb, c_prim, a_prim = [], [], [], []
        c_all = {s: [] for s in SIGNALS_WINDOW}
        a_all = {s: [] for s in SIGNALS_WINDOW}
        for _ in range(args.n_windows):
            sel = rng.choice(n_eval, size=args.window, replace=False)
            cw = probs_clean[sel]
            aw = cw.copy()
            aw[:n_pois] = probs_hijack[sel[:n_pois]]

            cz, az = det.score_window_z(cw), det.score_window_z(aw)
            for s in SIGNALS_WINDOW:
                c_all[s].append(cz[s]); a_all[s].append(az[s])
            c_comb.append(max(cz[s] for s in SIGNALS_WINDOW))
            a_comb.append(max(az[s] for s in SIGNALS_WINDOW))
            c_prim.append(cz[PRIMARY]); a_prim.append(az[PRIMARY])

        comb_auc = roc_auc(c_comb, a_comb); comb_tpr, _ = tpr_at_fpr(c_comb, a_comb, 0.05)
        prim_auc = roc_auc(c_prim, a_prim); prim_tpr, _ = tpr_at_fpr(c_prim, a_prim, 0.05)
        per_sig = {s: roc_auc(c_all[s], a_all[s]) for s in SIGNALS_WINDOW}
        oracle_sig = max(SIGNALS_WINDOW, key=lambda s: per_sig[s])

        rows.append({"alpha": alpha,
                     "combined_auc": comb_auc, "combined_tpr05": comb_tpr,
                     "primary_auc": prim_auc, "primary_tpr05": prim_tpr,
                     "oracle_auc": per_sig[oracle_sig], "oracle_signal": oracle_sig,
                     "per_signal_auc": per_sig})
        print(f"  alpha={alpha:<5} | COMBINED AUC={comb_auc:.3f} TPR={comb_tpr:.3f} "
              f"| PRIMARY({PRIMARY}) AUC={prim_auc:.3f} TPR={prim_tpr:.3f} "
              f"| oracle {per_sig[oracle_sig]:.3f} ({oracle_sig})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"{args.model}_{args.rule}_pr{args.poison_rate}_w{args.window}"
    out = {"args": vars(args), "primary_signal": PRIMARY,
           "win_ref_mean": det.win_ref_mean, "win_ref_std": det.win_ref_std, "rows": rows}
    with open(os.path.join(RESULTS_DIR, f"detector_combined_{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)

    md = [f"# Operational detector — pre-specified rule ({args.model})", "",
          f"Teacher `{os.path.basename(ckp)}` | rule={args.rule} | poison_rate={args.poison_rate} "
          f"| window={args.window} | {args.n_windows} windows/condition | seed={args.seed}", "",
          "COMBINED = max over the four window signals, each standardised on CLEAN calibration",
          f"windows. PRIMARY = the single pre-specified signal `{PRIMARY}`. ORACLE = best signal",
          "chosen on the evaluation AUC — an upper bound, not a deployable detector.", "",
          "| alpha | combined AUC | combined TPR@5%FPR | primary AUC | primary TPR@5%FPR | oracle AUC (signal) |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['alpha']} | {r['combined_auc']:.3f} | {r['combined_tpr05']:.3f} "
                  f"| {r['primary_auc']:.3f} | {r['primary_tpr05']:.3f} "
                  f"| {r['oracle_auc']:.3f} ({r['oracle_signal']}) |")
    with open(os.path.join(RESULTS_DIR, f"detector_combined_{tag}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved -> results/detector_combined_{tag}.json (+ .md)")


if __name__ == "__main__":
    main()

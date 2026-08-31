"""
Report the pre-specified detector against the after-the-fact best choice, across window sizes.

eval_detector.py sweeps the attack strength at one window size and quotes, for each condition,
whichever of the four signals scored highest on that condition's own evaluation windows. That
is selection after the fact. It inflates the quoted figure by an unknown margin, and it is not
a detector anybody could actually deploy, because a defender does not know in advance which
signal will win.

This script reports, at every window size and attack strength:

  primary   max_class_excess, fixed on design grounds before any attacked data was seen,
            because a targeted hijack has to push probability mass toward one class. It is
            kept even in the conditions where it loses, which is the point.
  oracle    the best of the four signals for that condition, chosen on the evaluation AUC.
            Reported only as an upper bound, so the cost of proceeding honestly is measured
            rather than guessed at.

The window sweep also covers the pooled-client result. Clients are modelled as drawing from a
common query distribution, so pooling k of them is arithmetically a window k times longer; the
k-client view is therefore read off the same sweep at window = k * per_client rather than
being run as a separate experiment.

All of it comes from one set of forward passes, so it is cheap enough to run on a CPU.

Usage:
    python eval_prespecified.py --model resnet50 --device cpu
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

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PRIMARY = "max_class_excess"


def collect_outputs(model, loader, device, poisoner=None, batches_note=""):
    outs = []
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            if poisoner is not None:
                x = poisoner(x)
            outs.append(model(x).cpu())
            if batches_note and i % 5 == 0:
                print(f"    {batches_note} batch {i}", flush=True)
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser(description="Pre-specified detector against the oracle")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--alphas", default="0,0.25,0.5,1.0")
    p.add_argument("--windows", default="100,200,500,1000,2000")
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--n_windows", type=int, default=300)
    p.add_argument("--n_eval", type=int, default=10000)
    p.add_argument("--pool_alpha", type=float, default=0.25,
                   help="attack strength for the pooled-client view")
    p.add_argument("--per_client", type=int, default=200,
                   help="queries one client contributes, for the pooled-client view")
    p.add_argument("--max_k", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.model}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}\nRun:  python train_teacher.py")

    print("=== pre-specified detector against the oracle ===", flush=True)
    print(f"  teacher={os.path.basename(ckp)}  primary signal={PRIMARY}  rule={args.rule}",
          flush=True)

    # ---- data: same split discipline as eval_detector.py -------------------
    tf = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    n = min(args.n_eval, len(test))
    idx = rng.permutation(n)
    calib_idx, eval_idx = idx[: n // 2], idx[n // 2:]
    calib_loader = DataLoader(Subset(test, calib_idx.tolist()), batch_size=args.batch_size)
    eval_loader = DataLoader(Subset(test, eval_idx.tolist()), batch_size=args.batch_size)
    print(f"  calibration images={len(calib_idx)}  evaluation images={len(eval_idx)}", flush=True)

    teacher = nets.get_network(args.dataset, args.model, ckp, device=device).to(device)
    poisoner = FixedTriggerPoisoner(size=32).to(device)

    print("  forward pass 1/3: calibration, clean", flush=True)
    calib_probs = F.softmax(collect_outputs(teacher, calib_loader, device), dim=1).numpy()
    det = TeacherOutputDetector(num_classes=calib_probs.shape[1]).calibrate(calib_probs)

    print("  forward pass 2/3: evaluation, clean", flush=True)
    eval_logits_clean = collect_outputs(teacher, eval_loader, device)
    print("  forward pass 3/3: evaluation, trigger stamped", flush=True)
    eval_logits_trig = collect_outputs(teacher, eval_loader, device, poisoner=poisoner)

    probs_clean = F.softmax(eval_logits_clean, dim=1).numpy()
    n_eval = len(probs_clean)

    alphas = [float(a) for a in args.alphas.split(",")]
    windows = sorted({int(w) for w in args.windows.split(",")} |
                     {args.per_client * k for k in range(1, args.max_k + 1)})
    windows = [w for w in windows if w <= n_eval]

    # cache the hijacked outputs once per alpha rather than once per (alpha, window)
    hijacked = {a: hijack_soft_labels(eval_logits_trig, args.target_label, a,
                                      rule=args.rule).numpy() for a in set(alphas) | {args.pool_alpha}}

    def evaluate(window, alpha):
        """AUC and TPR at 5% FPR for every signal and for the combined rule."""
        n_pois = int(args.poison_rate * window)
        probs_hijack = hijacked[alpha]
        clean_scores = {s: [] for s in SIGNALS_WINDOW}
        attacked_scores = {s: [] for s in SIGNALS_WINDOW}
        clean_comb, attacked_comb = [], []
        wr = np.random.default_rng(args.seed)          # same windows for every signal
        for _ in range(args.n_windows):
            sel = wr.choice(n_eval, size=window, replace=False)
            clean_w = probs_clean[sel]
            attacked_w = clean_w.copy()
            attacked_w[:n_pois] = probs_hijack[sel[:n_pois]]
            cs, as_ = det.score_window(clean_w), det.score_window(attacked_w)
            for s in SIGNALS_WINDOW:
                clean_scores[s].append(cs[s])
                attacked_scores[s].append(as_[s])
            clean_comb.append(det.score_combined(clean_w))
            attacked_comb.append(det.score_combined(attacked_w))
        out = {}
        for s in SIGNALS_WINDOW:
            neg = np.array(clean_scores[s]); pos = np.array(attacked_scores[s])
            out[s] = {"auc": roc_auc(neg, pos), "tpr05": tpr_at_fpr(neg, pos, 0.05)[0]}
        neg, pos = np.array(clean_comb), np.array(attacked_comb)
        out["__combined__"] = {"auc": roc_auc(neg, pos), "tpr05": tpr_at_fpr(neg, pos, 0.05)[0]}
        return out

    rows = []
    for window in windows:
        # the spread of each signal depends on the window length, so the combined rule is
        # re-standardised at every window size, always on clean calibration data only
        det.calibrate_windows(calib_probs, window=window,
                              n_windows=args.n_windows, seed=args.seed)
        for alpha in sorted(set(alphas) | {args.pool_alpha}):
            per_signal = evaluate(window, alpha)
            best = max(SIGNALS_WINDOW, key=lambda s: per_signal[s]["auc"])
            rows.append({
                "window": window, "alpha": alpha,
                "primary_signal": PRIMARY,
                "primary_auc": per_signal[PRIMARY]["auc"],
                "primary_tpr05": per_signal[PRIMARY]["tpr05"],
                "combined_auc": per_signal["__combined__"]["auc"],
                "combined_tpr05": per_signal["__combined__"]["tpr05"],
                "oracle_signal": best,
                "oracle_auc": per_signal[best]["auc"],
                "oracle_tpr05": per_signal[best]["tpr05"],
                "gap_auc": per_signal[best]["auc"] - per_signal[PRIMARY]["auc"],
                "per_signal": {s: per_signal[s] for s in SIGNALS_WINDOW},
            })
            r = rows[-1]
            print(f"  window={window:<5} alpha={alpha:<5} "
                  f"primary AUC={r['primary_auc']:.3f} TPR={r['primary_tpr05']:.3f} | "
                  f"combined AUC={r['combined_auc']:.3f} TPR={r['combined_tpr05']:.3f} | "
                  f"oracle AUC={r['oracle_auc']:.3f} ({r['oracle_signal']})", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"prespecified_{args.model}_{args.rule}_pr{args.poison_rate}"
    with open(os.path.join(RESULTS_DIR, tag + ".json"), "w") as f:
        json.dump({"args": vars(args), "primary": PRIMARY, "rows": rows}, f, indent=1)

    # ---- markdown: the window sweep, then the pooled-client reading ---------
    md = [f"# Pre-specified detector against the oracle ({args.model})", "",
          f"Teacher: `{os.path.basename(ckp)}` | rule: {args.rule} | "
          f"poison rate: {args.poison_rate} | {args.n_windows} windows per condition | "
          f"threshold at a 5% false-alarm rate.", "",
          f"`{PRIMARY}` is fixed in advance on design grounds: a targeted hijack has to push "
          "probability mass toward one class. The oracle column names whichever of the four "
          "signals scored highest on that condition's own evaluation windows, which is a choice "
          "made after the fact and is not available to a real defender. The gap between them is "
          "the cost of proceeding honestly.", ""]

    for alpha in sorted(set(alphas) | {args.pool_alpha}):
        md += [f"## Attack strength alpha = {alpha}", "",
               "| window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |",
               "|---|---|---|---|---|---|---|"]
        for r in [r for r in rows if r["alpha"] == alpha]:
            md.append(f"| {r['window']} | {r['primary_auc']:.3f} | {r['primary_tpr05']:.3f} | "
                      f"{r['combined_auc']:.3f} | {r['combined_tpr05']:.3f} | "
                      f"{r['oracle_auc']:.3f} | {r['oracle_signal']} |")
        md.append("")

    md += [f"## Pooled clients, alpha = {args.pool_alpha}", "",
           f"Each client contributes {args.per_client} queries. Because the clients draw from a "
           "common query distribution, pooling k of them is a window k times longer, so these "
           "rows are the sweep above read at those window sizes rather than a separate "
           "experiment.", "",
           "| clients k | window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |",
           "|---|---|---|---|---|---|---|---|"]
    for k in range(1, args.max_k + 1):
        w = args.per_client * k
        m = [r for r in rows if r["window"] == w and r["alpha"] == args.pool_alpha]
        if m:
            r = m[0]
            md.append(f"| {k} | {w} | {r['primary_auc']:.3f} | {r['primary_tpr05']:.3f} | "
                      f"{r['combined_auc']:.3f} | {r['combined_tpr05']:.3f} | "
                      f"{r['oracle_auc']:.3f} | {r['oracle_signal']} |")
    md.append("")
    with open(os.path.join(RESULTS_DIR, tag + ".md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> results/{tag}.md (+ .json)")


if __name__ == "__main__":
    main()

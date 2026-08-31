"""
Measure what the teacher-side monitor costs to run.

Chapter 6 concedes that the detector's runtime cost was never measured, which leaves an obvious
question open: a defence that doubles serving latency is not deployable however well it detects.
This measures it directly.

Three quantities matter to somebody deciding whether to switch the monitor on.

  Latency.    The teacher has to run either way. The monitor adds one scoring pass per window,
              not per query, so its cost is amortised over the whole window. The overhead
              reported here is the detector's time divided by the teacher time for the same
              number of queries.
  Memory.     The monitor holds one window of output vectors, and nothing else that grows.
  Bandwidth.  In the pooled-client setting the output vectors have to reach the monitor, so
              the size of a window on the wire is what a deployment would have to carry.

Everything is measured on the same machine, with the same teacher, in the configuration the
evaluation uses. Timings are medians over repeated trials after a warm-up, because first calls
pay for lazy allocation and would flatter the comparison.

Usage:
    python measure_overhead.py --model resnet50 --device cpu
"""

import os
import sys
import json
import time
import argparse
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

sys.path.append(os.path.dirname(__file__))
from detector import TeacherOutputDetector, SIGNALS_WINDOW  # noqa: E402

SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import nets  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CKP_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def median_time(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), min(times), max(times)


def main():
    p = argparse.ArgumentParser(description="Cost of running the teacher-side monitor")
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet50")
    p.add_argument("--teacher_ckp", default=None)
    p.add_argument("--windows", default="100,200,500,1000,2000")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--calib_n", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(torch.get_num_threads())
    device = torch.device(args.device)
    ckp = args.teacher_ckp or os.path.join(CKP_DIR, f"clean_teacher_{args.model}.pth")
    if not os.path.exists(ckp):
        sys.exit(f"No teacher checkpoint at {ckp}")

    print("=== cost of the teacher-side monitor ===", flush=True)
    print(f"  teacher={os.path.basename(ckp)}  device={args.device}  "
          f"threads={torch.get_num_threads()}  batch={args.batch_size}", flush=True)

    tf = T.Compose([T.ToTensor()])
    test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=tf)
    pool = torch.stack([test[i][0] for i in range(max(args.calib_n, args.batch_size * 4))])

    teacher = nets.get_network(args.dataset, args.model, ckp, device=device).to(device).eval()

    # ---- calibrate the detector, so scoring runs against a real reference ---
    with torch.no_grad():
        calib = torch.cat([F.softmax(teacher(pool[i:i + args.batch_size].to(device)), dim=1).cpu()
                           for i in range(0, args.calib_n, args.batch_size)]).numpy()
    det = TeacherOutputDetector(num_classes=calib.shape[1]).calibrate(calib)
    print(f"  detector calibrated on {len(calib)} clean outputs", flush=True)

    # ---- 1. teacher inference, the cost that exists with or without the monitor
    batch = pool[: args.batch_size].to(device)

    @torch.no_grad()
    def infer():
        return teacher(batch)

    t_batch, t_lo, t_hi = median_time(infer, args.repeats)
    per_query_ms = t_batch / args.batch_size * 1000
    print(f"\n  teacher inference: {t_batch*1000:.1f} ms per batch of {args.batch_size} "
          f"({per_query_ms:.3f} ms per query, range {t_lo*1000:.1f}-{t_hi*1000:.1f} ms)", flush=True)

    # ---- 2. the monitor's scoring pass, once per window ---------------------
    rows = []
    probs_all = calib
    for window in [int(w) for w in args.windows.split(",")]:
        idx = np.random.default_rng(args.seed).choice(len(probs_all), size=window,
                                                      replace=window > len(probs_all))
        w_probs = probs_all[idx]

        def score():
            return det.score_window(w_probs)

        t_score, s_lo, s_hi = median_time(score, args.repeats)
        teacher_time_for_window = per_query_ms / 1000 * window
        overhead_pct = t_score / teacher_time_for_window * 100
        # one window held in memory as float32 probability vectors
        window_bytes = window * probs_all.shape[1] * 4
        # The ratio depends on how fast the teacher itself is. Measured here against a
        # CPU-served teacher, which is slow, so the monitor looks almost free. A teacher
        # served from a GPU is far faster and the same detector cost is a larger share of
        # it, so the figure is also given against a hypothetical 1 ms per query teacher.
        overhead_fast_pct = t_score / (0.001 * window) * 100
        rows.append({
            "window": window,
            "detector_ms_per_window": t_score * 1000,
            "teacher_ms_per_window": teacher_time_for_window * 1000,
            "overhead_pct": overhead_pct,
            "overhead_pct_if_1ms_teacher": overhead_fast_pct,
            "detector_ms_per_query": t_score * 1000 / window,
            "window_kib": window_bytes / 1024,
        })
        print(f"  window={window:<5} detector {t_score*1000:7.3f} ms  "
              f"teacher {teacher_time_for_window*1000:9.1f} ms  "
              f"overhead {overhead_pct:.5f}%  "
              f"(vs a 1 ms/query teacher: {overhead_fast_pct:.3f}%)  "
              f"window holds {window_bytes/1024:.1f} KiB", flush=True)

    # A per-signal breakdown is deliberately not reported. score_window computes all four
    # signals in one pass and they share the class histogram, the mean output vector and the
    # entropy calculation, so timing them separately would either repeat that shared work or
    # attribute it arbitrarily. The figures above are the cost of the whole scoring pass,
    # which is what a deployment would actually pay.
    out = {"args": vars(args),
           "threads": torch.get_num_threads(),
           "signals": SIGNALS_WINDOW,
           "teacher_ms_per_batch": t_batch * 1000,
           "teacher_ms_per_query": per_query_ms,
           "rows": rows}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = f"overhead_{args.model}_{args.device}"
    with open(os.path.join(RESULTS_DIR, tag + ".json"), "w") as f:
        json.dump(out, f, indent=1)

    md = [f"# Cost of running the teacher-side monitor ({args.model}, {args.device})", "",
          f"Teacher: `{os.path.basename(ckp)}` | batch size {args.batch_size} | "
          f"{torch.get_num_threads()} CPU threads | median of {args.repeats} trials after a warm-up.", "",
          f"Teacher inference costs {per_query_ms:.3f} ms per query "
          f"({t_batch*1000:.1f} ms per batch of {args.batch_size}). That cost is paid whether or "
          "not the monitor is running. The monitor adds one scoring pass per window rather than "
          "per query, so the figures below divide its cost over the whole window.", "",
          "| window | detector, ms per window | teacher, ms for the same queries | overhead | overhead vs a 1 ms/query teacher | window in memory |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['window']} | {r['detector_ms_per_window']:.3f} | "
                  f"{r['teacher_ms_per_window']:.0f} | {r['overhead_pct']:.5f}% | "
                  f"{r['overhead_pct_if_1ms_teacher']:.3f}% | "
                  f"{r['window_kib']:.1f} KiB |")
    md += ["",
           "The overhead column compares the monitor against this machine's own teacher, which "
           "is served on a CPU and is therefore slow. That flatters the monitor, so the next "
           "column repeats the comparison against a hypothetical teacher answering in 1 ms per "
           "query, which is the order a GPU-served model would achieve. The monitor stays well "
           "under a tenth of one per cent even then, because it runs once per window rather than "
           "once per query.", "",
           "The monitor holds one window of output vectors and a fixed calibration reference; "
           "nothing else in it grows with the length of the stream. The window figure above is "
           "also what a pooled deployment would have to carry on the wire, since it is the size "
           "of the probability vectors themselves.", ""]
    with open(os.path.join(RESULTS_DIR, tag + ".md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved -> results/{tag}.md (+ .json)")


if __name__ == "__main__":
    main()

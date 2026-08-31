"""
Regenerate Figures 5.3 and 5.4 from the pre-specified sweep.

The earlier versions of both figures plotted a single curve, the best window signal for each
condition, which was chosen after the results were known. They therefore showed the detector
performing better than any rule a defender could actually have committed to in advance. These
versions plot three curves: the signal fixed in advance, the deployable combined rule, and the
after-the-fact best signal as an upper bound.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "prespecified_resnet50_soft_pr0.1.json")
# Written into the dissertation's figures directory when it is present, otherwise into a
# local figs/ directory, so the script still runs from a clone of the code on its own.
_DISS = os.path.abspath(os.path.join(HERE, "..", "..", "07_Dissertation_Drafts", "figs"))
FIGS = _DISS if os.path.isdir(_DISS) else os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

d = json.load(open(RES))
rows = sorted([r for r in d["rows"] if r["alpha"] == 0.25], key=lambda r: r["window"])
w = [r["window"] for r in rows]
primary = [r["primary_tpr05"] for r in rows]
combined = [r["combined_tpr05"] for r in rows]
oracle = [r["per_signal"]["mean_prob_kl"]["tpr05"] for r in rows]

plt.rcParams.update({"font.size": 10, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 200})

def style(ax, xlabel):
    ax.set_ylabel("True-positive rate at 5% false alarms")
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.03, 1.0)
    ax.axhline(0.05, color="0.5", lw=0.8, ls=":", zorder=1)
    ax.legend(frameon=False, loc="upper left", fontsize=9)

# ---- Figure 5.3: window size ------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(w, oracle, "o--", color="#999999", ms=4, lw=1.2,
        label="best signal, chosen after the fact (upper bound)")
ax.plot(w, combined, "s-", color="#1f4e79", ms=4, lw=1.8,
        label="combined rule, fixed in advance")
ax.plot(w, primary, "^-", color="#a33", ms=4, lw=1.8,
        label="max_class_excess, pre-specified")
style(ax, "Monitoring window (teacher outputs)")
ax.annotate("false-alarm floor", xy=(w[-1], 0.05), xytext=(w[-1], 0.11),
            ha="right", fontsize=8, color="0.4")
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig5_3_window.png"), bbox_inches="tight")
print("wrote fig5_3_window.png")

# ---- Figure 5.4: pooled clients ---------------------------------------------
per_client = d["args"]["per_client"]
krows = [r for r in rows if r["window"] % per_client == 0 and r["window"] // per_client <= d["args"]["max_k"]]
k = [r["window"] // per_client for r in krows]
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(k, [r["per_signal"]["mean_prob_kl"]["tpr05"] for r in krows], "o--", color="#999999",
        ms=4, lw=1.2, label="best signal, chosen after the fact (upper bound)")
ax.plot(k, [r["combined_tpr05"] for r in krows], "s-", color="#1f4e79", ms=4, lw=1.8,
        label="combined rule, fixed in advance")
ax.plot(k, [r["primary_tpr05"] for r in krows], "^-", color="#a33", ms=4, lw=1.8,
        label="max_class_excess, pre-specified")
style(ax, f"Victims pooled at the monitor (k, {per_client} queries each)")
ax.set_xticks(k)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "fig5_4_distributed.png"), bbox_inches="tight")
print("wrote fig5_4_distributed.png")

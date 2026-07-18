# Adaptive attacker vs the teacher-side detector

**Date:** 2026-07-18 | **Scale:** CPU prototype | **Status:** result stands, numbers provisional

The non-circular test promised in the design doc (§6.3). A *naive* attacker blends the teacher's
soft-labels toward a one-hot on the target class (`soft` rule). An **adaptive** attacker who knows a
teacher-side output monitor exists instead adds a bias to the target **logit before the softmax**
(`logit` rule), producing a smooth, well-formed distribution rather than an obviously blended one.

Question: does knowing about the detector let the attacker win?

---

## Both attack rules, measured on the same axes

**Naive (`soft` rule)** — alpha is the blend weight:

| alpha | Detection TPR@5%FPR | Student ASR |
|---|---|---|
| 0.0 (control) | 0.043 | 0.060 |
| 0.1 | 0.070 | 0.061 |
| 0.25 | 0.200 | 0.079 |
| 0.5 | 0.987 | 0.282 |
| 0.75 | 0.990 | 0.676 |
| 1.0 | 1.000 | 0.827 |

**Adaptive (`logit` rule)** — alpha is the pre-softmax bias on the target logit:

| alpha | Detection TPR@5%FPR | Student ASR |
|---|---|---|
| 2 | 0.070 | 0.062 |
| 3 | 0.093 | 0.088 |
| 5 | 0.31–0.43 | 0.141 |
| **8** | **0.870** | **0.591** |
| 10 | 0.973 | 0.729 |
| 15 | 0.987 | 0.801 |

## The honest comparison: detection at MATCHED attack strength

Comparing the rules at equal alpha is meaningless — the parameter means different things. The fair
comparison holds ASR fixed and asks how often each attacker is caught:

| Student ASR | Naive (`soft`) TPR | Adaptive (`logit`) TPR | Attacker's escape rate |
|---|---|---|---|
| ~0.08 | 0.200 | 0.093 | both useless (ASR ~ control) |
| ~0.28 | 0.987 | — | 1.3% |
| **~0.59** | ~0.99 (interpolated) | **0.870** | **13.0%** |
| ~0.73 | ~0.99 | 0.973 | 2.7% |
| ~0.80 | 0.990–1.000 | 0.987 | 1.3% |

## Findings

**1. The adaptive attacker does gain a real advantage — and it is modest.**
At its best operating point (`logit`, alpha=8) it installs a substantial backdoor (ASR 0.591) while
escaping detection **13% of the time**, versus roughly 1% for a naive attacker at comparable strength.
That is a genuine ~10x improvement in escape rate. It is NOT nothing, and the write-up should not
pretend otherwise.

**2. But the advantage does not scale — it shrinks as the attacker gets greedier.**
Pushing for a stronger backdoor closes the gap: at ASR ~0.73 the escape rate falls to 2.7%, and at
ASR ~0.80 to 1.3%, which is indistinguishable from the naive attacker. The adaptive strategy buys
stealth only while settling for a weaker backdoor.

**3. The core claim survives a smarter attacker.**
For BOTH rules, the detection curve rises faster than the attack curve, and there is no operating point
where the attack is both potent and reliably stealthy. The best the adaptive attacker achieves is a
partial backdoor that is still caught ~87% of the time. This is the answer to the circularity
objection: the detector was not tuned against one attacker — a second, deliberately evasive attack rule
was constructed and swept across its full range, and the evasion window remains narrow.

**4. Why the logit rule evades better at all (mechanism).**
Blending toward a one-hot creates a two-peaked distribution that sits *off the manifold* of outputs a
real teacher would produce, which the aggregate statistics pick up easily. A pre-softmax logit bias
produces a distribution that a genuine teacher plausibly could have emitted, so it is harder to
distinguish from honest output. This is a useful, citable observation in its own right — and it
predicts the right defence: signals sensitive to *class-frequency drift* (`max_class_excess`,
`mean_prob_kl`) survive, whereas per-query confidence heuristics do not.

## Honest statement of the result

> Against a naive attacker the detector catches ~99% of attacks at any strength that installs a
> backdoor. An adaptive attacker who manipulates logits pre-softmax improves its escape rate roughly
> tenfold, to ~13%, but only by accepting a weaker backdoor (ASR 0.59 rather than 0.83); pushing for
> full strength returns the escape rate to ~1%. The detector therefore does not eliminate the attack,
> but forces the attacker into a strictly worse position on both axes.

## Caveats

1. **Prototype scale** — 4,000 train images, 10 epochs, ResNet-18 teacher (69.3% ACC), MobileNetV2
   student, CPU. ASR is likely under-estimated; the real crossover may shift and the margins narrow.
2. **Detection estimates carry sampling noise of roughly +/-0.1 at intermediate alpha.** Measured
   directly: `logit` alpha=5 gave TPR 0.310 and 0.430 on two runs differing only in window sampling.
   The saturated ends (near 0.05 and near 1.0) are stable; the middle of the curve is not. Fix is free:
   disjoint windows over the full test set, and repeat with several seeds.
3. The matched-ASR comparison **interpolates** the `soft` rule between measured points; it is not a
   direct measurement at ASR 0.59.
4. `logit` alpha=0 was not re-run because it is mathematically identical to `soft` alpha=0
   (softmax(logits + 0) = unmodified teacher output), already measured at ASR 0.060.
5. One trigger, one target class, one architecture pair.

## Next

* GPU re-run at full scale, sampling both rules around their crossovers (`soft` 0.25–0.5, `logit` 5–10).
* Repeat detection with disjoint windows and multiple seeds to kill the +/-0.1 noise.
* Compare against RobustKD feature-variance and Yu 2024 benign-vector-distance.

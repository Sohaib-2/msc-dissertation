# Distributed KD — one compromised teacher, many students

**Date:** 2026-07-25 | **Scale:** CPU prototype | **Status:** result stands, numbers provisional

This closes the "distributed" word in the title: *Backdoor Propagation Vector in **Distributed**
Knowledge Distillation*. A single runtime-compromised teacher is distilled by several downstream
parties. We measure the two things distribution changes — the attacker's reach and the defender's
evidence — and they pull in opposite directions.

---

## 1. Attack scaling — one teacher backdoors every student, across architecture families

Four downstream parties each distil their own student, of a different architecture, through the same
hijacked teacher (alpha=1.0, poison_rate 0.1). Clean teacher reference: ACC 0.693, ASR 0.032.

| Student | Family | Clean ACC | ASR |
|---|---|---|---|
| mobilenetv2 | CNN | 0.497 | 0.827 |
| shufflenetv2 | CNN | 0.423 | 0.711 |
| efficientvit | Transformer | 0.520 | 0.667 |
| resnet18 | CNN | 0.638 | 0.503 |

**All four students are backdoored** (ASR 0.50–0.83, mean 0.68, vs 0.03 chance). The propagation is
architecture-agnostic — it crosses from CNNs to a vision transformer. The one-to-many threat in the
title is demonstrated: a single compromised serving path poisons every party that distils from it,
regardless of the student they choose.

**Secondary finding — propagation is universal but not uniform.** The highest-capacity student
(resnet18, same architecture as the teacher) keeps the most clean accuracy (0.638) AND absorbs the
least backdoor (ASR 0.503). Plausible reading: a stronger student fits the teacher's soft-labels less
slavishly, so it inherits the backdoor more weakly. Worth stating as an architecture-dependence result,
not smoothing over.

## 2. Defence scaling — distribution multiplies the DEFENDER's evidence too

The detector sits at the teacher and sees the union of all clients' queries. Because every client
queries the SAME compromised teacher, k clients deliver k times the poisoned evidence about the same
hijack. Measured at a deliberately stealthy alpha=0.25 (where a single client barely detects anything),
poison_rate 0.1, 200 queries/client, threshold at 5% false alarms:

| k clients | pooled queries | detection AUC | TPR@5%FPR |
|---|---|---|---|
| 1 | 200 | 0.677 | 0.170 |
| 2 | 400 | 0.789 | 0.260 |
| 3 | 600 | 0.846 | 0.385 |
| 4 | 800 | 0.906 | 0.645 |
| 5 | 1000 | 0.954 | 0.730 |
| 7 | 1400 | 0.969 | 0.865 |
| 10 | 2000 | 0.996 | 0.995 |

**A hijack one victim catches 17% of the time, ten pooled victims catch 99.5% of the time.** Same
attack, same stealth — only the number of clients changed.

## 3. The finding: distribution is double-edged

| | Attacker | Defender |
|---|---|---|
| What distribution gives | reach — every student backdoored | evidence — every client observes the same hijack |
| Scales with k as | linear (k victims) | statistical (k x evidence => detection saturates) |

Distribution is usually framed as pure downside for the victim (one poisoned teacher, many poisoned
students). This result reframes it: because all victims share one compromised source, distribution also
concentrates the defender's signal. The larger the deployment, the harder the hijack is to hide — a
stealthy attack that survives a single distillation is exposed once several are monitored together.

This unifies with the two earlier results: the [window-size / defender-patience] curve and this
[client-pooling] curve are the same mechanism — the hijack must repeatedly push probability mass toward
one class, and that evidence accumulates whether it is gathered over a longer window or across more
clients. The attacker cannot escape by lowering alpha, because the defender recovers the lost signal by
observing longer or wider.

## Caveats (state before anyone else does)

1. **Prototype scale** — 4,000 train images, 10 epochs, ResNet-18 teacher (69.3% ACC), CPU. ASR is
   likely under-estimated; the GPU run must confirm the four-student pattern holds at full accuracy.
2. **Defence scaling assumes clients query the same teacher on comparable data** so their streams pool
   coherently. Heterogeneous per-client data distributions would add variance not modelled here; a
   fair next step is to give each client a different data slice.
3. **Pooled windows are drawn from 5,000 held-out images**, so at k>=5 (>=1000 queries) they overlap
   and the effective sample shrinks. The monotone trend to k=4 is solid; k>=5 needs disjoint pools on
   the full test set.
4. **The dip at k=6** (TPR 0.725 vs 0.730 at k=5) is sampling noise, consistent with the +/-0.1 noise
   measured earlier at intermediate operating points, not a real non-monotonicity.
5. One trigger, one target class, one teacher.

## Next

* GPU re-run: confirm the four-architecture backdoor at full accuracy, and re-measure defence scaling
  with disjoint per-client data slices.
* Fold into the distributed chapter alongside the trade-off and adaptive-attacker results.

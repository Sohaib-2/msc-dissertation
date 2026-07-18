# Detector evaluation — rule=soft, poison_rate=0.1, window=1000

Teacher: `clean_teacher_resnet18.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 0.0 | 0.497 (mean_prob_kl) | 0.050 | 0.497 (entropy_drop) |
| 0.1 | 0.702 (mean_prob_kl) | 0.227 | 0.399 (confidence_spike) |
| 0.25 | 0.926 (mean_prob_kl) | 0.723 | 0.354 (kl_from_clean_mean) |
| 0.5 | 1.000 (hist_kl) | 1.000 | 0.391 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 0.0 | 0.496 | 0.493 | 0.489 | 0.497 |
| 0.1 | 0.521 | 0.521 | 0.162 | 0.702 |
| 0.25 | 0.631 | 0.654 | 0.076 | 0.926 |
| 0.5 | 1.000 | 1.000 | 0.145 | 1.000 |

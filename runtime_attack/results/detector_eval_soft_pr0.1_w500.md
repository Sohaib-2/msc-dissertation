# Detector evaluation — rule=soft, poison_rate=0.1, window=500

Teacher: `clean_teacher_resnet18.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 0.0 | 0.499 (mean_prob_kl) | 0.050 | 0.498 (entropy_drop) |
| 0.1 | 0.610 (mean_prob_kl) | 0.113 | 0.400 (confidence_spike) |
| 0.25 | 0.825 (mean_prob_kl) | 0.330 | 0.353 (kl_from_clean_mean) |
| 0.5 | 1.000 (hist_kl) | 1.000 | 0.389 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 0.0 | 0.499 | 0.498 | 0.494 | 0.499 |
| 0.1 | 0.510 | 0.510 | 0.252 | 0.610 |
| 0.25 | 0.581 | 0.590 | 0.188 | 0.825 |
| 0.5 | 1.000 | 1.000 | 0.230 | 0.993 |

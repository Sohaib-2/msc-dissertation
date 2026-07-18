# Detector evaluation — rule=soft, poison_rate=0.1, window=100

Teacher: `clean_teacher_resnet18.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 0.0 | 0.502 (hist_kl) | 0.003 | 0.497 (entropy_drop) |
| 0.1 | 0.520 (mean_prob_kl) | 0.060 | 0.395 (confidence_spike) |
| 0.25 | 0.588 (mean_prob_kl) | 0.073 | 0.353 (kl_from_clean_mean) |
| 0.5 | 0.930 (max_class_excess) | 0.573 | 0.388 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 0.0 | 0.502 | 0.500 | 0.497 | 0.499 |
| 0.1 | 0.503 | 0.506 | 0.386 | 0.520 |
| 0.25 | 0.518 | 0.527 | 0.334 | 0.588 |
| 0.5 | 0.894 | 0.930 | 0.366 | 0.740 |

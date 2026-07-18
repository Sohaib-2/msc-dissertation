# Detector evaluation — rule=soft, poison_rate=0.1, window=2000

Teacher: `clean_teacher_resnet18.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 0.0 | 0.497 (mean_prob_kl) | 0.050 | 0.497 (entropy_drop) |
| 0.1 | 0.808 (mean_prob_kl) | 0.313 | 0.400 (confidence_spike) |
| 0.25 | 0.997 (mean_prob_kl) | 0.997 | 0.353 (kl_from_clean_mean) |
| 0.5 | 1.000 (hist_kl) | 1.000 | 0.390 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 0.0 | 0.492 | 0.482 | 0.485 | 0.497 |
| 0.1 | 0.533 | 0.546 | 0.056 | 0.808 |
| 0.25 | 0.737 | 0.802 | 0.010 | 0.997 |
| 0.5 | 1.000 | 1.000 | 0.042 | 1.000 |

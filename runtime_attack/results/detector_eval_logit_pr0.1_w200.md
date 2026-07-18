# Detector evaluation — rule=logit, poison_rate=0.1, window=200

Teacher: `clean_teacher_resnet50.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 3.0 | 0.505 (mean_prob_kl) | 0.060 | 0.502 (kl_from_clean_mean) |
| 5.0 | 0.512 (hist_kl) | 0.050 | 0.509 (kl_from_clean_mean) |
| 8.0 | 0.517 (max_class_excess) | 0.067 | 0.503 (kl_from_clean_mean) |
| 10.0 | 0.522 (mean_prob_kl) | 0.063 | 0.505 (kl_from_clean_mean) |
| 15.0 | 0.554 (max_class_excess) | 0.063 | 0.507 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 3.0 | 0.503 | 0.502 | 0.495 | 0.505 |
| 5.0 | 0.512 | 0.509 | 0.500 | 0.511 |
| 8.0 | 0.517 | 0.517 | 0.495 | 0.515 |
| 10.0 | 0.521 | 0.521 | 0.493 | 0.522 |
| 15.0 | 0.551 | 0.554 | 0.472 | 0.548 |

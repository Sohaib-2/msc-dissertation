# Detector evaluation — rule=soft, poison_rate=0.1, window=200

Teacher: `clean_teacher_resnet50.pth` | target class: 0 | 300 windows/condition | detection threshold set at 5% false-alarm rate.

AUC 1.0 = perfect detection, 0.5 = coin flip. alpha=0 is the control (trigger stamped,
no hijack) and MUST sit near 0.5 — anything else means we are detecting the trigger
rather than the attack.

| alpha | windowed AUC (best signal) | TPR@5%FPR | per-query AUC (best signal) |
|---|---|---|---|
| 0.0 | 0.502 (max_class_excess) | 0.040 | 0.500 (kl_from_clean_mean) |
| 0.25 | 0.603 (mean_prob_kl) | 0.113 | 0.135 (kl_from_clean_mean) |
| 0.5 | 0.994 (max_class_excess) | 0.980 | 0.118 (kl_from_clean_mean) |
| 0.75 | 0.995 (max_class_excess) | 0.987 | 0.143 (kl_from_clean_mean) |
| 1.0 | 0.994 (max_class_excess) | 0.980 | 0.995 (kl_from_clean_mean) |

## All windowed signals (AUC)

| alpha | hist_kl | max_class_excess | mean_entropy_drop | mean_prob_kl |
|---|---|---|---|---|
| 0.0 | 0.499 | 0.502 | 0.500 | 0.500 |
| 0.25 | 0.500 | 0.501 | 0.006 | 0.603 |
| 0.5 | 0.973 | 0.994 | 0.001 | 0.763 |
| 0.75 | 0.974 | 0.995 | 0.010 | 0.917 |
| 1.0 | 0.975 | 0.994 | 0.649 | 0.981 |

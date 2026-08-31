# Operational detector — pre-specified rule (resnet50)

Teacher `clean_teacher_resnet50.pth` | rule=soft | poison_rate=0.1 | window=200 | 300 windows/condition | seed=0

COMBINED = max over the four window signals, each standardised on CLEAN calibration
windows. PRIMARY = the single pre-specified signal `max_class_excess`. ORACLE = best signal
chosen on the evaluation AUC — an upper bound, not a deployable detector.

| alpha | combined AUC | combined TPR@5%FPR | primary AUC | primary TPR@5%FPR | oracle AUC (signal) |
|---|---|---|---|---|---|
| 0.0 | 0.499 | 0.050 | 0.502 | 0.040 | 0.502 (max_class_excess) |
| 0.25 | 0.518 | 0.080 | 0.501 | 0.043 | 0.603 (mean_prob_kl) |
| 0.5 | 0.989 | 0.957 | 0.994 | 0.980 | 0.994 (max_class_excess) |
| 0.75 | 0.987 | 0.930 | 0.995 | 0.987 | 0.995 (max_class_excess) |
| 1.0 | 0.988 | 0.957 | 0.994 | 0.980 | 0.994 (max_class_excess) |

# Operational detector — pre-specified rule (resnet18)

Teacher `clean_teacher_resnet18.pth` | rule=soft | poison_rate=0.1 | window=200 | 300 windows/condition | seed=0

COMBINED = max over the four window signals, each standardised on CLEAN calibration
windows. PRIMARY = the single pre-specified signal `max_class_excess`. ORACLE = best signal
chosen on the evaluation AUC — an upper bound, not a deployable detector.

| alpha | combined AUC | combined TPR@5%FPR | primary AUC | primary TPR@5%FPR | oracle AUC (signal) |
|---|---|---|---|---|---|
| 0.0 | 0.498 | 0.050 | 0.497 | 0.047 | 0.500 (hist_kl) |
| 0.25 | 0.545 | 0.090 | 0.544 | 0.060 | 0.673 (mean_prob_kl) |
| 0.5 | 0.993 | 0.970 | 0.998 | 0.997 | 0.998 (max_class_excess) |
| 0.75 | 0.986 | 0.937 | 0.997 | 0.987 | 0.997 (max_class_excess) |
| 1.0 | 0.997 | 0.990 | 0.997 | 0.990 | 0.998 (mean_prob_kl) |

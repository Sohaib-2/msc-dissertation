# Pre-specified detector against the oracle (resnet50)

Teacher: `clean_teacher_resnet50.pth` | rule: soft | poison rate: 0.1 | 300 windows per condition | threshold at a 5% false-alarm rate.

`max_class_excess` is fixed in advance on design grounds: a targeted hijack has to push probability mass toward one class. The oracle column names whichever of the four signals scored highest on that condition's own evaluation windows, which is a choice made after the fact and is not available to a real defender. The gap between them is the cost of proceeding honestly.

## Attack strength alpha = 0.0

| window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |
|---|---|---|---|---|---|---|
| 100 | 0.500 | 0.047 | 0.498 | 0.050 | 0.500 | max_class_excess |
| 200 | 0.498 | 0.040 | 0.500 | 0.047 | 0.501 | hist_kl |
| 400 | 0.499 | 0.043 | 0.500 | 0.053 | 0.502 | hist_kl |
| 500 | 0.500 | 0.050 | 0.500 | 0.050 | 0.502 | hist_kl |
| 600 | 0.501 | 0.053 | 0.503 | 0.053 | 0.502 | hist_kl |
| 800 | 0.501 | 0.050 | 0.502 | 0.050 | 0.504 | mean_prob_kl |
| 1000 | 0.502 | 0.047 | 0.502 | 0.057 | 0.504 | mean_prob_kl |
| 1200 | 0.500 | 0.047 | 0.504 | 0.057 | 0.505 | mean_prob_kl |
| 1400 | 0.502 | 0.050 | 0.503 | 0.053 | 0.504 | mean_prob_kl |
| 1600 | 0.504 | 0.047 | 0.505 | 0.050 | 0.507 | hist_kl |
| 1800 | 0.501 | 0.047 | 0.506 | 0.053 | 0.506 | mean_prob_kl |
| 2000 | 0.504 | 0.050 | 0.506 | 0.063 | 0.507 | mean_prob_kl |

## Attack strength alpha = 0.25

| window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |
|---|---|---|---|---|---|---|
| 100 | 0.500 | 0.047 | 0.436 | 0.070 | 0.547 | mean_prob_kl |
| 200 | 0.499 | 0.040 | 0.517 | 0.087 | 0.613 | mean_prob_kl |
| 400 | 0.501 | 0.043 | 0.623 | 0.167 | 0.686 | mean_prob_kl |
| 500 | 0.502 | 0.050 | 0.680 | 0.210 | 0.723 | mean_prob_kl |
| 600 | 0.503 | 0.050 | 0.718 | 0.233 | 0.747 | mean_prob_kl |
| 800 | 0.504 | 0.050 | 0.804 | 0.327 | 0.824 | mean_prob_kl |
| 1000 | 0.505 | 0.047 | 0.859 | 0.460 | 0.873 | mean_prob_kl |
| 1200 | 0.505 | 0.047 | 0.894 | 0.580 | 0.902 | mean_prob_kl |
| 1400 | 0.507 | 0.053 | 0.916 | 0.577 | 0.923 | mean_prob_kl |
| 1600 | 0.509 | 0.047 | 0.935 | 0.647 | 0.944 | mean_prob_kl |
| 1800 | 0.506 | 0.053 | 0.957 | 0.847 | 0.961 | mean_prob_kl |
| 2000 | 0.511 | 0.050 | 0.980 | 0.923 | 0.982 | mean_prob_kl |

## Attack strength alpha = 0.5

| window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |
|---|---|---|---|---|---|---|
| 100 | 0.924 | 0.617 | 0.900 | 0.567 | 0.924 | max_class_excess |
| 200 | 0.993 | 0.950 | 0.989 | 0.930 | 0.993 | max_class_excess |
| 400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 2000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |

## Attack strength alpha = 1.0

| window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |
|---|---|---|---|---|---|---|
| 100 | 0.924 | 0.617 | 0.912 | 0.573 | 0.924 | max_class_excess |
| 200 | 0.993 | 0.950 | 0.990 | 0.940 | 0.993 | max_class_excess |
| 400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 500 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | max_class_excess |
| 800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1200 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 1800 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |
| 2000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | hist_kl |

## Pooled clients, alpha = 0.25

Each client contributes 200 queries. Because the clients draw from a common query distribution, pooling k of them is a window k times longer, so these rows are the sweep above read at those window sizes rather than a separate experiment.

| clients k | window | pre-specified AUC | pre-specified TPR | combined AUC | combined TPR | oracle AUC | oracle signal |
|---|---|---|---|---|---|---|---|
| 1 | 200 | 0.499 | 0.040 | 0.517 | 0.087 | 0.613 | mean_prob_kl |
| 2 | 400 | 0.501 | 0.043 | 0.623 | 0.167 | 0.686 | mean_prob_kl |
| 3 | 600 | 0.503 | 0.050 | 0.718 | 0.233 | 0.747 | mean_prob_kl |
| 4 | 800 | 0.504 | 0.050 | 0.804 | 0.327 | 0.824 | mean_prob_kl |
| 5 | 1000 | 0.505 | 0.047 | 0.859 | 0.460 | 0.873 | mean_prob_kl |
| 6 | 1200 | 0.505 | 0.047 | 0.894 | 0.580 | 0.902 | mean_prob_kl |
| 7 | 1400 | 0.507 | 0.053 | 0.916 | 0.577 | 0.923 | mean_prob_kl |
| 8 | 1600 | 0.509 | 0.047 | 0.935 | 0.647 | 0.944 | mean_prob_kl |
| 9 | 1800 | 0.506 | 0.053 | 0.957 | 0.847 | 0.961 | mean_prob_kl |
| 10 | 2000 | 0.511 | 0.050 | 0.980 | 0.923 | 0.982 | mean_prob_kl |

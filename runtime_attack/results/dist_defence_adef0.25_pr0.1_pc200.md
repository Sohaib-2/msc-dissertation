# Distributed defence scaling — pooling k students' query streams

Teacher: `clean_teacher_resnet50.pth` | hijack alpha=0.25 (deliberately stealthy) | poison_rate=0.1 | 200 queries/client | detection threshold at 5% false alarms.

The monitor sits at the teacher and sees the union of all clients' queries. More clients distilling from the same compromised teacher = more evidence of the same hijack.

| k clients | pooled queries | detection AUC | TPR@5%FPR | signal |
|---|---|---|---|---|
| 1 | 200 | 0.626 | 0.120 | mean_prob_kl |
| 2 | 400 | 0.711 | 0.210 | mean_prob_kl |
| 3 | 600 | 0.780 | 0.360 | mean_prob_kl |
| 4 | 800 | 0.835 | 0.385 | mean_prob_kl |
| 5 | 1000 | 0.887 | 0.590 | mean_prob_kl |
| 6 | 1200 | 0.909 | 0.640 | mean_prob_kl |
| 7 | 1400 | 0.936 | 0.625 | mean_prob_kl |
| 8 | 1600 | 0.952 | 0.775 | mean_prob_kl |
| 9 | 1800 | 0.971 | 0.805 | mean_prob_kl |
| 10 | 2000 | 0.980 | 0.935 | mean_prob_kl |

At a stealth level where a single client detects the hijack 12.0% of the time, pooling 10 clients raises that to 93.5%. Distribution multiplies the defender's evidence.

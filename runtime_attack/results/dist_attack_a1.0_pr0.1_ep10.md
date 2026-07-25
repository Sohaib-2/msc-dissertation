# Distributed attack scaling — one teacher -> 4 heterogeneous students

Teacher: `clean_teacher_resnet18.pth` | alpha=1.0 | poison_rate=0.1 | target class 0 | 4000 train imgs | 10 epochs

| student architecture | clean ACC | ASR |
|---|---|---|
| mobilenetv2 | 0.4965 | 0.8271 |
| shufflenetv2 | 0.4225 | 0.7106 |
| resnet18 | 0.6375 | 0.5028 |
| efficientvit | 0.5200 | 0.6657 |

**All 4 students backdoored.** ASR range 0.503–0.827, mean 0.677. One compromised teacher propagates the backdoor across every architecture — the one-to-many threat.

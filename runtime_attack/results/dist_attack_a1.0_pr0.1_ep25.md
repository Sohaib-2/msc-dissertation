# Distributed attack scaling — one teacher -> 4 heterogeneous students

Teacher: `clean_teacher_resnet50.pth` | alpha=1.0 | poison_rate=0.1 | target class 0 | 0 train imgs | 25 epochs

| student architecture | clean ACC | ASR |
|---|---|---|
| mobilenetv2 | 0.8362 | 0.6132 |
| shufflenetv2 | 0.7804 | 0.6656 |
| resnet18 | 0.8752 | 0.6351 |
| efficientvit | 0.7361 | 0.6146 |

**All 4 students backdoored.** ASR range 0.613–0.666, mean 0.632. One compromised teacher propagates the backdoor across every architecture — the one-to-many threat.

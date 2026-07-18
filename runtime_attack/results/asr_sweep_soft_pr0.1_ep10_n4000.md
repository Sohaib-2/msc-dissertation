# Student ASR sweep — rule=soft, poison_rate=0.1

Teacher: `clean_teacher_resnet18.pth` | student: mobilenetv2 | target class: 0 | 4000 train imgs | 10 epochs | cpu

ASR = attack success rate (fraction of triggered images the student sends to the target
class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack OFF, so ASR
must stay near chance — otherwise the backdoor is coming from somewhere other than our attack.

| alpha | student ACC | student ASR |
|---|---|---|
| 0.0 | 0.5055 | 0.0604 |
| 0.1 | 0.4905 | 0.0610 |
| 0.25 | 0.4965 | 0.0793 |
| 0.5 | 0.4880 | 0.2816 |
| 0.75 | 0.4900 | 0.6757 |
| 1.0 | 0.4965 | 0.8271 |

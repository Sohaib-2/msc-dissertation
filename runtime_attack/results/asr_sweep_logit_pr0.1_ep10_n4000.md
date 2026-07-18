# Student ASR sweep — rule=logit, poison_rate=0.1

Teacher: `clean_teacher_resnet18.pth` | student: mobilenetv2 | target class: 0 | 4000 train imgs | 10 epochs | cpu

ASR = attack success rate (fraction of triggered images the student sends to the target
class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack OFF, so ASR
must stay near chance — otherwise the backdoor is coming from somewhere other than our attack.

| alpha | student ACC | student ASR |
|---|---|---|
| 8.0 | 0.4965 | 0.5909 |
| 10.0 | 0.4975 | 0.7289 |
| 15.0 | 0.4955 | 0.8010 |

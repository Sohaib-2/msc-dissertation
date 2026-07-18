# Student ASR sweep — rule=logit, poison_rate=0.1

Teacher: `clean_teacher_resnet18.pth` | student: mobilenetv2 | target class: 0 | 4000 train imgs | 10 epochs | cpu

ASR = attack success rate (fraction of triggered images the student sends to the target
class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack OFF, so ASR
must stay near chance — otherwise the backdoor is coming from somewhere other than our attack.

| alpha | student ACC | student ASR |
|---|---|---|
| 2.0 | 0.4970 | 0.0621 |
| 3.0 | 0.4935 | 0.0881 |
| 5.0 | 0.4995 | 0.1414 |

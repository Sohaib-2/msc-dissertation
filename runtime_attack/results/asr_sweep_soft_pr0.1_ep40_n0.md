# Student ASR sweep — rule=soft, poison_rate=0.1

Teacher: `clean_teacher_resnet50.pth` | student: mobilenetv2 | target class: 0 | 0 train imgs | 40 epochs | cuda

ASR = attack success rate (fraction of triggered images the student sends to the target
class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack OFF, so ASR
must stay near chance — otherwise the backdoor is coming from somewhere other than our attack.

| alpha | student ACC | student ASR |
|---|---|---|
| 0.0 | 0.8415 | 0.0166 |
| 0.25 | 0.8445 | 0.0828 |
| 0.5 | 0.8386 | 0.1581 |
| 0.75 | 0.8469 | 0.2603 |
| 1.0 | 0.8431 | 0.6190 |

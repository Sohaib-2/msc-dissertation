# Student ASR sweep — rule=logit, poison_rate=0.1

Teacher: `clean_teacher_resnet50.pth` | student: mobilenetv2 | target class: 0 | 0 train imgs | 40 epochs | cuda

ASR = attack success rate (fraction of triggered images the student sends to the target
class). ACC = clean accuracy. alpha=0 is the control: trigger stamped, hijack OFF, so ASR
must stay near chance — otherwise the backdoor is coming from somewhere other than our attack.

| alpha | student ACC | student ASR |
|---|---|---|
| 3.0 | 0.8408 | 0.0181 |
| 5.0 | 0.8393 | 0.0179 |
| 8.0 | 0.8344 | 0.0256 |
| 10.0 | 0.8410 | 0.0367 |
| 15.0 | 0.8472 | 0.0791 |

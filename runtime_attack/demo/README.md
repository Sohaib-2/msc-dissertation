# Live demo: the attack and the detector, running

A single-page dashboard that runs the real pipeline on a local machine and shows the
teacher-side monitor catching the hijack. Nothing on the page is simulated: it imports and calls
the project's own `FixedTriggerPoisoner`, `hijack_soft_labels` and `TeacherOutputDetector`, so
what appears on screen is the mechanism the dissertation describes, computed live on each
request.

The page is written for someone who has not met knowledge distillation before. It opens with a
plain explanation and then a four-step walkthrough that drives the controls.

## Running it

```bash
cd runtime_attack/demo
bash run_demo.sh
```

Then open http://127.0.0.1:8000. The first load takes two to three minutes, because a ResNet-50
on a CPU has to load, measure its clean accuracy and calibrate the detector. It prints
`demo ready` when it is finished.

There is nothing to install beyond what the project already needs: Python's standard-library
`http.server`, plus torch, torchvision, numpy and PIL. It runs on a CPU.

## The walkthrough

Each step sets the controls, resets the monitor and explains what to watch for. The four steps
correspond to the conditions in Chapter 5.

| Step | Setting | Alarm rate over 400 windows |
|---|---|---|
| 1, normal service | attack off | 6%, which is the 5% false-alarm rate the thresholds were set for |
| 2, a loud attack | alpha = 1.0, rho = 0.12 | 100% on both signals |
| 3, a sneaky attack | alpha = 0.25, rho = 0.12 | 22% on drift, 6% on class over-representation |
| 4, too quiet to see | alpha = 0.10, rho = 0.08 | 9%, against a 6% clean floor |

Those are measured rates rather than impressions. `../measure_demo_rates.py` samples 400 windows
per condition at the demo's own window size and thresholds. Step 3 is the one worth dwelling on:
drift is roughly four times more sensitive to a gentle nudge than class over-representation is,
which is the argument for carrying two signals, but 22% still means a single window usually looks
clean. That gap is why Section 5.6 answers a quiet attack with a longer window or pooled victims
rather than with a cleverer statistic. The page says so, and warns that one window is a sample
and not a verdict.

## What the two panels show

**A single question.** The teacher's honest answer beside the answer actually handed to the
student. With the attack on, a marked image's served answer is pushed toward the target class
while the teacher's own computation is untouched. The panel reports how much probability moved
onto the target rather than only which class won, because at low alpha the attack shifts the
soft-label without changing the winning class at all, which is the point of attacking a
soft-label. With the attack live the demo also prefers an image the teacher does not already
call the target class, so that the change is visible. That is presentation only, it is disclosed
on the page, and the attack itself treats every marked query alike.

**The monitor.** Streams questions past the teacher and scores a rolling window with two of the
signals:

- `max_class_excess`, the pre-specified primary signal, fixed on design grounds before any
  attacked data was seen, because the attack has to push mass toward one class.
- `mean_prob_kl`, the drift of the window's mean output vector from the calibrated benign mean.
  It shares its underlying principle, distance from an estimated benign reference, with the
  nearest existing defence (Yu et al., 2024), but it is not that method: theirs estimates
  class-conditioned references and reweights clients in federated distillation.

## Calibration

Thresholds are the 95th percentile of clean windows, which is the 5% false-alarm operating point
used throughout Chapter 5. About one clean window in twenty will therefore trip an alarm. That
is the calibration working rather than a fault, and the page says as much: a single alarm is not
evidence, a sustained one is.

## What this does and does not show

- The mark has to be present in the images the student trains on. No student can learn a mark it
  is never shown, and that is true of every backdoor of this kind. What is different here is that
  the labels stay honest, so nothing is mislabelled and the whole malicious signal travels in the
  teacher's answers, which is why watching those answers works.
- This is the teacher's side only. It does not train a student. The effect on students, an attack
  success rate rising from 1.7% to 62% while clean accuracy stays flat, is measured in Chapter 5.
- The teacher is the preserved ResNet-50 checkpoint in `../checkpoints/`, trained on full
  CIFAR-10 in the same configuration as the reported results. It is an independent retrain rather
  than the original file, which was held on a rented instance and not kept. It reaches 86.49%
  clean accuracy on the full test set against the original's 86.32%, and the detector results
  reproduce on it to within about 0.01 of AUC.
- The monitor never sees the trigger, the target class or the model's weights. It sees only the
  outgoing answers.

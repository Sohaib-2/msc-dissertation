# Attribution

## SCAR

The datasets, model definitions, distillation loss and the ACC/ASR measurement code come from
the reference implementation of SCAR, and are used unmodified so that the results here are
directly comparable with that baseline.

| | |
|---|---|
| Paper | Chen et al., *Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack*, NeurIPS 2025, arXiv:2509.23871 |
| Repository | https://github.com/WhitolfChen/SCAR |
| Commit used | `7a7b5b89969d2eb5edc4b71f14983e8d233c66e0`, dated 2025-11-19 |

The clone is not redistributed in this repository. Fetch it at that commit, into `scar_baseline/`,
as described in the README. It is never edited; the modifications made for the reproduction are
kept separately in `scar_repro/` and documented there.

## Work compared against

- Wu et al., arXiv:2504.21323, which poisons the distillation data with adversarial examples so
  that an honest teacher's own outputs teach the backdoor. This is the closest prior attack, and
  the point of departure: it places the malicious signal in the data, where the attack here
  places it in the served outputs.
- Yu et al., arXiv:2401.17746, *Logit Poisoning Attack in Distillation-based Federated Learning
  and its Countermeasures*. Their countermeasure identifies malicious clients at the aggregation
  point of federated distillation using distance from an estimated benign vector. It is the
  nearest existing defence, and the detector's `mean_prob_kl` signal is compared against it
  directly on identical output streams.

## Original work

The runtime output-compromise attack, the teacher-side detector, the evaluation and sweep
scripts, the distributed harness and the demo are the author's own.

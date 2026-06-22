"""
Runtime output-compromise attack on a distillation teacher.

Threat model:
  - The teacher's weights are trained normally and are never modified.
  - The ground-truth labels of the distillation set are never altered.
  - The attacker controls the teacher's serving path, so it can read each query
    and rewrite the soft-labels returned for it.
  - Whenever a trigger appears in the input, the served soft-labels are pushed
    toward a target class, and the student learns the backdoor from a teacher
    that is honest in every other respect.

One condition is worth stating openly, because the threat is easy to overstate.
Trigger-carrying inputs have to reach the distillation set: no student can learn a
mark it is never shown, and that is true of every backdoor of this kind. What is
specific here is that those inputs keep their true labels, so nothing in the
victim's data is mislabelled and the whole of the malicious signal travels in the
teacher's soft-labels.

How this relates to the attacks it is compared against, all run in the same harness:

  SCAR (Chen et al., 2025)  poisons the teacher's weights at training time.
  Wu et al. (2025)          poisons the distillation data with adversarial examples.
  FDLA (Yu et al., 2024)    reverses logits in federated distillation, which is
                            disruption rather than a targeted backdoor.
  This work                 leaves the weights and the labels alone and rewrites
                            only the served outputs, conditioned on a trigger.

Implementation note: this changes exactly one step of SCAR's response-based distiller.
After the clean teacher has produced its logits, the soft-labels of the triggered
samples are overwritten. The dataset, models, KD loss and the ACC/ASR metrics are all
reused unchanged, so the numbers stay directly comparable with the baseline.
"""

import os
import sys
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the SCAR harness (datasets / nets / utils) so numbers are comparable.
SCAR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scar_baseline"))
sys.path.append(SCAR_ROOT)
from core import utils, nets, datasets  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Trigger.  A fixed (non-learned) patch trigger, self-contained so it does not
#    depend on SCAR's pod-side learned trigger.  Same call signature as SCAR's
#    Poisoner (forward(images) -> triggered images) so utils.get_asr_results
#    works on it unchanged.  Swap-in note: for a head-to-head potency comparison
#    with SCAR, load SCAR's learned additive Poisoner here instead.
# ----------------------------------------------------------------------------
class FixedTriggerPoisoner(nn.Module):
    def __init__(self, size=32, patch=3, value=1.0):
        super().__init__()
        # A small bright square stamped in the bottom-right corner (BadNets-style).
        mask = torch.zeros(3, size, size)
        mask[:, size - patch:, size - patch:] = 1.0
        pattern = torch.ones(3, size, size) * value
        # register as buffers (not parameters) -> fixed, not learned, moves with .to(device)
        self.register_buffer("mask", mask)
        self.register_buffer("pattern", pattern)

    def forward(self, inputs):
        return inputs * (1 - self.mask) + self.pattern * self.mask


# ----------------------------------------------------------------------------
# 2. The hijack.  Soft-nudge the teacher's soft-labels of the triggered samples
#    toward the target class:   p' = (1 - alpha) * p_teacher + alpha * onehot(c*)
#    alpha = 0  -> no attack (clean teacher output)
#    alpha = 1  -> hard override (100% target class)
#    (Primary rule from the design doc; 'hard' and 'logit' variants provided too.)
# ----------------------------------------------------------------------------
def hijack_soft_labels(teacher_logits, target_label, alpha, rule="soft"):
    num_classes = teacher_logits.size(1)
    teacher_soft = F.softmax(teacher_logits, dim=1)
    onehot = F.one_hot(
        torch.full((teacher_logits.size(0),), target_label,
                   device=teacher_logits.device, dtype=torch.long),
        num_classes=num_classes,
    ).float()

    if rule == "hard":
        return onehot
    if rule == "soft":
        return (1 - alpha) * teacher_soft + alpha * onehot
    if rule == "logit":
        # add a bias to the target logit before softmax (sneakiest / adaptive)
        biased = teacher_logits.clone()
        biased[:, target_label] += alpha
        return F.softmax(biased, dim=1)
    raise ValueError(f"unknown rule: {rule}")


# ----------------------------------------------------------------------------
# 3. The distiller.  Mirrors SCAR's ResponseBased.train() exactly, except a
#    fraction `poison_rate` of every batch is stamped with the trigger and has
#    its teacher soft-labels hijacked.  The student's own CE loss still uses the
#    true labels (the attacker controls only the teacher's OUTPUT, not the
#    distiller's labels) — a deliberately honest, harder setting.
# ----------------------------------------------------------------------------
class RuntimeAttackDistiller:
    def __init__(self, teacher_model, student_model, trainloader, testloader,
                 poisoner, target_label, logger, device,
                 alpha=1.0, poison_rate=0.1, rule="soft",
                 delta=1.0, num_epochs=150, lr=1e-3):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.poisoner = poisoner
        self.target_label = target_label
        self.logger = logger
        self.device = device
        self.alpha = alpha
        self.poison_rate = poison_rate
        self.rule = rule
        self.delta = delta
        self.num_epochs = num_epochs
        self.optimizer = torch.optim.Adam(self.student_model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs)

    def train(self):
        criterion_kl = nn.KLDivLoss(reduction="batchmean")
        criterion_ce = nn.CrossEntropyLoss()
        self.teacher_model.eval()

        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch: {epoch + 1} starts...")
            self.student_model.train()

            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # --- attacker stamps the trigger on a fraction of the batch ---
                n_pois = int(self.poison_rate * images.size(0))
                if n_pois > 0:
                    images = images.clone()
                    images[:n_pois] = self.poisoner(images[:n_pois])

                # --- clean teacher produces its (honest) logits ---
                with torch.no_grad():
                    teacher_logits = self.teacher_model(images).detach()
                teacher_soft = F.softmax(teacher_logits, dim=1)

                # --- HIJACK: overwrite soft-labels of the triggered samples ---
                if n_pois > 0 and self.alpha > 0:
                    teacher_soft[:n_pois] = hijack_soft_labels(
                        teacher_logits[:n_pois], self.target_label,
                        self.alpha, self.rule)

                # --- student learns from the (partly hijacked) soft-labels ---
                student_logits = self.student_model(images)
                ce_loss = criterion_ce(student_logits, labels)
                kl_loss = criterion_kl(
                    F.log_softmax(student_logits, dim=1), teacher_soft)
                loss = ce_loss + self.delta * kl_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.test(epoch + 1)

    def test(self, epoch=None):
        acc = utils.get_acc_results(self.student_model, self.testloader, self.device)
        asr = utils.get_asr_results(self.student_model, self.testloader,
                                    self.poisoner, self.target_label, self.device)
        tag = f"[epoch {epoch}] " if epoch else ""
        self.logger.info(f"{tag}student_model ACC: {acc:.4f}  ASR: {asr:.4f}")
        return acc, asr


# ----------------------------------------------------------------------------
# CLI for the real (GPU) run.  Needs a CLEAN teacher checkpoint.
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Runtime teacher-compromise attack")
    p.add_argument("--dataset", "-d", default="cifar10")
    p.add_argument("--teacher_model", "-t", default="resnet50")
    p.add_argument("--student_model", "-s", default="mobilenetv2")
    p.add_argument("--teacher_ckp", required=True, help="path to a CLEAN teacher .pth")
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--poison_rate", type=float, default=0.1)
    p.add_argument("--rule", default="soft", choices=["soft", "hard", "logit"])
    p.add_argument("--delta", type=float, default=1.0)
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--gpu_id", "-g", default="0")
    args = p.parse_args()

    save_path = f"runtime_attack_out/{args.dataset}/{args.teacher_model}/{args.student_model}/"
    os.makedirs(save_path, exist_ok=True)
    logger = logging.getLogger(__name__)
    utils.config_logging(save_path)
    for k, v in args.__dict__.items():
        logger.info(f"{k}:{v}")

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = datasets.get_dataloader(args.dataset, args.batch_size)

    teacher = nets.get_network(args.dataset, args.teacher_model, args.teacher_ckp,
                               device=device).to(device)
    teacher.eval()
    poisoner = FixedTriggerPoisoner(size=32).to(device)
    student = nets.get_network(args.dataset, args.student_model).to(device)

    # sanity: a clean teacher should have ~0 ASR (nothing in the weights)
    t_acc = utils.get_acc_results(teacher, testloader, device)
    t_asr = utils.get_asr_results(teacher, testloader, poisoner, args.target_label, device)
    logger.info(f"CLEAN teacher ACC: {t_acc:.4f}  ASR: {t_asr:.4f}  (ASR should be low)")

    dl = RuntimeAttackDistiller(
        teacher, student, trainloader, testloader, poisoner, args.target_label,
        logger, device, alpha=args.alpha, poison_rate=args.poison_rate,
        rule=args.rule, delta=args.delta, num_epochs=args.num_epochs)
    dl.train()


if __name__ == "__main__":
    main()

"""CPU smoke test: exercises the real SCAR modules + the bilevel autograd chain
on a tiny fake batch. Catches import/shape/grad bugs WITHOUT a GPU or CIFAR download.
Run from scar_repro/ with the smoke venv active."""
import os, sys
sys.path.insert(0, os.path.abspath("../scar_baseline"))

import torch, torch.nn as nn
from core import nets

dev = "cpu"
print("[1] importing + building models (resnet18 teacher, mobilenetv2 student, poisoner)...")
teacher = nets.get_network("cifar10", "resnet18").to(dev)
student = nets.get_network("cifar10", "mobilenetv2").to(dev)
poisoner = nets.get_network("cifar10", "poisoner").to(dev)

x = torch.rand(4, 3, 32, 32)          # tiny fake CIFAR batch
y = torch.randint(0, 10, (4,))
print("    ok. teacher out:", tuple(teacher(x).shape), "student out:", tuple(student(x).shape))

print("[2] poisoner forward + project...")
xp = poisoner(x)
poisoner.project(0.1)
assert xp.shape == x.shape

print("[3] KD loss (response) forward/backward...")
ce, kl = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction="batchmean")
s_log, t_log = student(x), teacher(x).detach()
loss = ce(s_log, y) + kl(torch.log_softmax(s_log, 1), torch.softmax(t_log, 1))
loss.backward()
print("    ok. KD loss =", round(loss.item(), 4))

print("[4] THE RISKY PART: bilevel hypergradient (Neumann series) like SCAR.py...")
teacher.train(); student.eval()
lmd, omg = list(teacher.parameters()), list(student.parameters())
t_clean, s_clean = teacher(x), student(x)
outer = ce(t_clean, y) + ce(s_clean, y)
g_lmd = torch.autograd.grad(outer, lmd, retain_graph=True)
g_omg = torch.autograd.grad(outer, omg, retain_graph=True)
# NOTE: teacher is intentionally NOT detached here — SCAR keeps it in the inner
# graph so the hypergradient can flow back to the teacher (SCAR.py:143-144).
inner = ce(s_clean, y) + kl(torch.log_softmax(s_clean,1), torch.softmax(t_clean,1))
inner_grad = torch.autograd.grad(inner, omg, create_graph=True)
F = [w - 1e-5*g for w, g in zip(omg, inner_grad)]
vs = [torch.zeros_like(w) for w in omg]
for _ in range(3):  # 3 Neumann iters is enough to prove it runs
    vs = torch.autograd.grad(F, omg, grad_outputs=vs, retain_graph=True)
    vs = [v + go for v, go in zip(vs, g_omg)]
JFlmd = torch.autograd.grad(F, lmd, grad_outputs=vs)
print("    ok. hypergradient chain ran,", len(JFlmd), "teacher-param grads computed.")

print("\n*** SMOKE TEST PASSED — SCAR code runs on this stack. ***")

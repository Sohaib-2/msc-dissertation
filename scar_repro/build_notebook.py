"""Generates Colab_SCAR_short.ipynb (valid JSON via nbformat).
Run:  python build_notebook.py   (needs `pip install nbformat`)
The notebook is self-contained: it embeds patch_epochs.py so no private-repo auth is
needed on Colab. Designed for the FREE T4 + short config, resilient to disconnects
(each stage syncs to Google Drive; reruns resume because pretrain.py skips existing ckpts)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

PIN = "7a7b5b89969d2eb5edc4b71f14983e8d233c66e0"  # SCAR upstream commit we pinned
nb = new_notebook()
c = []

c.append(new_markdown_cell(
"""# SCAR — short "proof-it-works" run (free Colab T4)

**Goal:** run the whole SCAR pipeline end-to-end and get a *nonzero student ASR* — proof the
backdoor mechanism works. This is the deliberately-shrunk config (~2–3 h). The paper-faithful
run comes later on RunPod/MMU.

### Before you run
1. **Runtime → Change runtime type → T4 GPU → Save.**
2. **Runtime → Run all.** Then watch the three numbers explained at the bottom.

Each stage saves to your Google Drive, so if Colab disconnects you can just re-run — it resumes."""))

c.append(new_code_cell(
"""# [1] GPU check — must show a Tesla T4 (or better). If it says CPU, fix the runtime type.
import torch
print("CUDA available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU! Runtime > Change runtime type > T4 GPU."
print("GPU:", torch.cuda.get_device_name(0))
print("torch:", torch.__version__)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""))

c.append(new_code_cell(
"""# [2] Mount Google Drive (persist checkpoints/logs across disconnects)
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE_DIR = '/content/drive/MyDrive/msc_dissertation/scar_short'  # change if you like
os.makedirs(DRIVE_DIR, exist_ok=True)
print("Outputs will be backed up to:", DRIVE_DIR)"""))

c.append(new_code_cell(
f"""# [3] Clone the EXACT SCAR version we pinned, then install the 2 missing deps.
# (torch/torchvision are already on Colab — do NOT reinstall them.)
import os
if not os.path.isdir('/content/SCAR'):
    !git clone https://github.com/WhitolfChen/SCAR /content/SCAR
!cd /content/SCAR && git checkout -q {PIN} && echo "checked out pinned commit"
!pip install -q timm einops
print("deps ready")"""))

c.append(new_code_cell(
'''# [4] Write our epoch-patcher into the Colab session (keeps SCAR's clone otherwise pristine).
%%writefile /content/patch_epochs.py
import sys, os, re
def patch(path, replacements, ensure_import_os=False):
    with open(path) as f: src = f.read()
    orig = src
    if ensure_import_os and re.search(r'^\\s*import os\\b', src, re.M) is None:
        src = re.sub(r'(\\nimport torch[^\\n]*\\n)', r'\\1import os\\n', src, count=1)
    for old, new in replacements:
        if new in src:
            print("  [skip]", os.path.basename(path), "::", old); continue
        if old not in src:
            raise SystemExit("  [FAIL] not found in "+path+": "+old)
        src = src.replace(old, new); print("  [ok]  ", os.path.basename(path), "::", old)
    if src != orig:
        with open(path, "w") as f: f.write(src)
def main(repo):
    patch(os.path.join(repo,"SCAR","pretrain.py"), [
        ("num_epochs = 200", "num_epochs = int(os.environ.get('TEACHER_EPOCHS','200'))"),
        ("num_epochs = 50",  "num_epochs = int(os.environ.get('STAGE1_EPOCHS','50'))")])
    patch(os.path.join(repo,"core","distillation.py"), [
        ("self.num_epochs = 150","self.num_epochs = int(os.environ.get('DISTILL_EPOCHS','150'))")],
        ensure_import_os=True)
if __name__=="__main__": main(sys.argv[1])'''))

c.append(new_code_cell(
"""# [5] Apply the patch + restore any previous progress from Drive (resume after disconnect)
!python /content/patch_epochs.py /content/SCAR
import shutil, os
for sub in ['pretrain','attack','distillation']:
    src=os.path.join(DRIVE_DIR,sub); dst=f'/content/SCAR/SCAR/{sub}'
    if os.path.isdir(src):
        shutil.copytree(src,dst,dirs_exist_ok=True); print("restored",sub,"from Drive")"""))

c.append(new_code_cell(
"""# [6] STAGE 1/3 — pretrain the trigger (teacher resnet50 + surrogate resnet18 + poisoner)
%cd /content/SCAR/SCAR
!TEACHER_EPOCHS=30 STAGE1_EPOCHS=15 python pretrain.py -d cifar10 -t resnet50 -s resnet18 -g 0
!cp -r /content/SCAR/SCAR/pretrain {DRIVE_DIR}/ 2>/dev/null
print("Stage 1 done; synced pretrain -> Drive")"""))

c.append(new_code_cell(
"""# [7] STAGE 2/3 — SCAR attack: bake the DORMANT backdoor into the teacher (the long one, ~1h)
%cd /content/SCAR/SCAR
!python SCAR.py -d cifar10 -t resnet50 -s resnet18 -g 0 --num_epochs 20 --inner_steps 8 --K 40
!cp -r /content/SCAR/SCAR/attack {DRIVE_DIR}/ 2>/dev/null
print("Stage 2 done; synced attack -> Drive")"""))

c.append(new_code_cell(
"""# [8] STAGE 3/3 — distill the VICTIM student (mobilenetv2) & measure ASR
%cd /content/SCAR/SCAR
!DISTILL_EPOCHS=40 python test_distillation.py -d cifar10 -t resnet50 -r resnet18 -s mobilenetv2 -m response -g 0
!cp -r /content/SCAR/SCAR/distillation {DRIVE_DIR}/ 2>/dev/null
print("Stage 3 done; synced distillation -> Drive")"""))

c.append(new_markdown_cell(
"""## How to read the result

Look at the **last lines** of Stage 3's output (and Stage 2's per-epoch logs):

| Number | What you want | Meaning |
|---|---|---|
| **teacher ACC** | high (~85%+) | teacher still classifies normally |
| **teacher ASR** | **LOW** (<~10%) | teacher looks *clean* on triggers — the stealth |
| **student ASR** | **above chance / high** | backdoor **woke up** after distillation = **attack worked** |

If `student ASR` is clearly above 10% (random would be ~10% for 10 classes) while `teacher ASR`
stays low → **the distillation-conditional backdoor reproduced.** Numbers will be weaker than the
paper (we cut epochs ~5–10×); that's expected for this proof run.

Record the three numbers before deciding the next step: scaling up on RunPod, or moving on to the detector.

> Note: this run uses the patched-in epoch *dials*; with no env vars set the code defaults to the
> exact paper values, so we haven't changed the baseline — only added an off-switch."""))

nb['cells'] = c
nb['metadata'] = {"accelerator":"GPU","colab":{"provenance":[]},
                  "kernelspec":{"name":"python3","display_name":"Python 3"}}
with open("Colab_SCAR_short.ipynb","w") as f:
    nbf.write(nb, f)
print("wrote Colab_SCAR_short.ipynb with", len(c), "cells")

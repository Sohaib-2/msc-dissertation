"""Make SCAR's hardcoded epoch counts adjustable via environment variables.

SCAR bakes epoch counts into the source (pretrain.py: 200/50/50; distillation.py: 150),
so we can't run a fast "proof" config without editing. This applies the MINIMAL change:
each hardcoded count becomes `int(os.environ.get('<VAR>', <original_default>))`.

Defaults are UNCHANGED — with no env vars set, behaviour is identical to upstream paper
settings. Set the env vars (see run_short.sh) to shrink for a quick run.

Usage:  python patch_epochs.py /path/to/SCAR        (the cloned repo root)
Idempotent: safe to run twice (skips already-patched lines).
"""
import sys, os, re

def patch(path, replacements, ensure_import_os=False):
    with open(path) as f:
        src = f.read()
    orig = src
    if ensure_import_os and re.search(r'^\s*import os\b', src, re.M) is None:
        # add `import os` after the first import line
        src = re.sub(r'(\nimport torch[^\n]*\n)', r'\1import os\n', src, count=1)
    for old, new in replacements:
        if new in src:
            print(f"  [skip] already patched: {os.path.basename(path)} :: {old!r}")
            continue
        if old not in src:
            raise SystemExit(f"  [FAIL] pattern not found in {path}: {old!r}")
        src = src.replace(old, new)
        print(f"  [ok]   {os.path.basename(path)} :: {old!r} -> env dial")
    if src != orig:
        with open(path, "w") as f:
            f.write(src)

def main(repo):
    pretrain = os.path.join(repo, "SCAR", "pretrain.py")
    distill  = os.path.join(repo, "core", "distillation.py")

    print("Patching", pretrain)
    patch(pretrain, [
        ("num_epochs = 200", "num_epochs = int(os.environ.get('TEACHER_EPOCHS', '200'))"),
        ("num_epochs = 50",  "num_epochs = int(os.environ.get('STAGE1_EPOCHS', '50'))"),
    ])
    print("Patching", distill)
    patch(distill, [
        ("self.num_epochs = 150", "self.num_epochs = int(os.environ.get('DISTILL_EPOCHS', '150'))"),
    ], ensure_import_os=True)
    print("Done. (SCAR.py epochs are already CLI flags: --num_epochs --inner_steps --K)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python patch_epochs.py /path/to/SCAR")
    main(sys.argv[1])

"""
Measure the demo's actual alarm rate for each step of its walkthrough, at the window size
and thresholds the demo itself uses.

The rates quoted in the demo's README and on the page come from this script. Running the
demo by hand a few times gives an impression of how often it alarms, and that impression
turned out to be wrong, so the figures are measured over several hundred windows instead.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F, torchvision
import torchvision.transforms as T
RA=os.path.abspath("."); sys.path.insert(0,RA)
sys.path.insert(0,os.path.abspath(os.path.join(RA,"..","scar_baseline")))
from core import nets
from runtime_attack import FixedTriggerPoisoner, hijack_soft_labels
from detector import TeacherOutputDetector

WINDOW=400; CALIB_N=4000; CALIB_WINDOWS=600; FPR=95; POOL=4000
np.random.seed(0); torch.manual_seed(0)
tf=T.Compose([T.ToTensor()])
ts=torchvision.datasets.CIFAR10(os.path.join(RA,"data"),train=False,download=False,transform=tf)
imgs=torch.stack([ts[i][0] for i in range(POOL)])
t=nets.get_network("cifar10","resnet50").eval()
t.load_state_dict(torch.load(os.path.join(RA,"checkpoints","clean_teacher_resnet50.pth"),map_location="cpu"))
pois=FixedTriggerPoisoner()

@torch.no_grad()
def logits(x,bs=256):
    return torch.cat([t(x[i:i+bs]) for i in range(0,len(x),bs)])

lg_clean=logits(imgs)
lg_trig =logits(pois(imgs))
p_clean=F.softmax(lg_clean,1).numpy()

det=TeacherOutputDetector(10).calibrate(p_clean[:CALIB_N])
SIG=["max_class_excess","mean_prob_kl"]
rng=np.random.default_rng(0)
cal=[det.score_window(p_clean[rng.choice(len(p_clean),WINDOW,replace=False)]) for _ in range(CALIB_WINDOWS)]
TH={s:float(np.percentile([c[s] for c in cal],FPR)) for s in SIG}
print("demo thresholds:",{k:round(v,5) for k,v in TH.items()},"\n")

def rate(alpha,poison,attack,n=400):
    ph=hijack_soft_labels(lg_trig,0,alpha,rule="soft").numpy() if attack else None
    hits={s:0 for s in SIG}; both=0
    for _ in range(n):
        sel=rng.choice(len(p_clean),WINDOW,replace=False)
        w=p_clean[sel].copy()
        if attack:
            k=int(poison*WINDOW); w[:k]=ph[sel[:k]]
        sc=det.score_window(w); a={s:sc[s]>TH[s] for s in SIG}
        for s in SIG: hits[s]+=a[s]
        both+= (a[SIG[0]] or a[SIG[1]])
    return {s:hits[s]/n for s in SIG}, both/n

print(f"{'step':<26}{'class-over-rep':>16}{'drift':>10}{'any alarm':>12}")
for lbl,atk,a,p in [("1 control (off)",0,0.5,0.12),("2 loud   a=1.0",1,1.0,0.12),
                    ("3 sneaky a=0.25",1,0.25,0.12),("4 quiet  a=0.10",1,0.10,0.08)]:
    r,any_=rate(a,p,atk)
    print(f"  {lbl:<24}{r['max_class_excess']*100:>13.0f}% {r['mean_prob_kl']*100:>8.0f}% {any_*100:>10.0f}%")

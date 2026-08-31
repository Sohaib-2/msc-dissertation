"""
Live demo: the runtime output-compromise attack and the teacher-side detector.

A single-page dashboard that runs the real pipeline on the local machine. A clean
ResNet-50 teacher serves CIFAR-10 soft-labels. With the attack switched on, the
soft-labels of trigger-stamped queries are pushed toward a target class. A separate
teacher-side monitor watches the outgoing stream and raises an alarm when the class
balance or the average output vector drifts away from a clean baseline.

Thresholds sit at the 95th percentile of clean windows, which is the 5% false-alarm
operating point used throughout the evaluation, so roughly one clean window in twenty
trips an alarm by design. The page says as much: a single alarm is not evidence, a
sustained one is.

The trigger is stamped into the images the teacher is asked about, because no student
can learn a mark it is never shown. What stays untouched are the teacher's weights and
the ground-truth labels. The whole of the malicious signal travels in the served
soft-labels, which is why watching those soft-labels works.

The demo imports the project's own code rather than reimplementing it, so what appears
on screen is the same mechanism the dissertation describes:
    FixedTriggerPoisoner, hijack_soft_labels    from runtime_attack.py
    TeacherOutputDetector.score_window          from detector.py

The teacher is the preserved ResNet-50 checkpoint, trained on full CIFAR-10 in the same
configuration as the reported results. It is an independent retrain rather than the
original file, which was held on a rented instance and not kept; the detector results
reproduce on it to within about 0.01 of AUC.

Run:
    python app.py            (with the project environment active)
then open http://127.0.0.1:8000
"""
import os, sys, json, io, base64, random
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RA = os.path.dirname(HERE)                      # runtime_attack/
SCAR = os.path.abspath(os.path.join(RA, "..", "scar_baseline"))
sys.path.insert(0, RA)
sys.path.insert(0, SCAR)

from core import nets                                            # noqa: E402
from runtime_attack import FixedTriggerPoisoner, hijack_soft_labels  # noqa: E402
from detector import TeacherOutputDetector                       # noqa: E402

# ---------------------------------------------------------------- config
MODEL = "resnet50"
CKPT = os.path.join(RA, "checkpoints", "clean_teacher_resnet50.pth")
DATA = os.path.join(RA, "data")
WINDOW = 400            # detector window (outputs) — larger = lower clean-window variance
CALIB_N = 4000          # clean outputs used to calibrate the detector
CALIB_WINDOWS = 600     # clean windows used to set the low-false-alarm thresholds
FPR_PCT = 95            # 95th percentile of clean windows = the 5% false-alarm operating
                        # point used throughout Chapter 5. About 1 clean window in 20 will
                        # therefore trip the alarm; that is the calibration working, not a bug.
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
DEVICE = torch.device("cpu")
random.seed(0); np.random.seed(0); torch.manual_seed(0)

print(">>> loading teacher + CIFAR-10 ...", flush=True)
_tf = T.Compose([T.ToTensor()])
_testset = torchvision.datasets.CIFAR10(DATA, train=False, download=True, transform=_tf)
# a fixed pool of images to serve from
_POOL_N = 4000
_pool_idx = list(range(min(_POOL_N, len(_testset))))
_images = torch.stack([_testset[i][0] for i in _pool_idx])       # (N,3,32,32) in [0,1]
_labels = np.array([_testset[i][1] for i in _pool_idx])

teacher = nets.get_network("cifar10", MODEL).to(DEVICE)
teacher.load_state_dict(torch.load(CKPT, map_location=DEVICE))
teacher.eval()
poisoner = FixedTriggerPoisoner().to(DEVICE)


@torch.no_grad()
def teacher_logits(x):
    return teacher(x.to(DEVICE))


# ---- clean accuracy (for the header) ----
@torch.no_grad()
def _clean_acc():
    correct = 0
    for i in range(0, len(_images), 256):
        lg = teacher_logits(_images[i:i + 256])
        correct += (lg.argmax(1).cpu().numpy() == _labels[i:i + 256]).sum()
    return correct / len(_images)


TEACHER_ACC = float(_clean_acc())
print(f">>> teacher clean accuracy = {TEACHER_ACC:.3f}", flush=True)

# ---- calibrate the detector on clean outputs ----
print(">>> calibrating detector ...", flush=True)
@torch.no_grad()
def _probs_batched(imgs, bs=256):
    """Forward in batches. A single 4000-image pass fits ResNet-18 but exhausts memory
    on ResNet-50, whose intermediate activations are far larger."""
    out = []
    for i in range(0, len(imgs), bs):
        out.append(F.softmax(teacher_logits(imgs[i:i + bs]), dim=1).cpu())
    return torch.cat(out).numpy()


_clean_probs = _probs_batched(_images[:CALIB_N])
detector = TeacherOutputDetector(num_classes=10)
detector.calibrate(_clean_probs)

# ---- set per-signal thresholds at ~5% FPR from clean windows ----
SIGNALS = ["max_class_excess", "mean_prob_kl"]
# PRIMARY is fixed on design grounds before any attacked data is seen: the attack has to
# push probability mass toward one class, so class over-representation is the statistic the
# threat model implies. The second signal is the primitive the closest prior defence uses.
PRIMARY = "max_class_excess"
LABELS = {"max_class_excess": "Is one class showing up too often?",
          "mean_prob_kl": "Has the average answer drifted?"}
TECH = {"max_class_excess": "max_class_excess  (pre-specified primary signal)",
        "mean_prob_kl": "mean_prob_kl  (related to the nearest prior defence)"}
_clean_scores = {s: [] for s in SIGNALS}
for _ in range(CALIB_WINDOWS):
    idx = np.random.choice(len(_clean_probs), WINDOW, replace=False)
    sc = detector.score_window(_clean_probs[idx])
    for s in SIGNALS:
        _clean_scores[s].append(sc[s])
THRESH = {s: float(np.percentile(_clean_scores[s], FPR_PCT)) for s in SIGNALS}
print(f">>> thresholds ({FPR_PCT}th pct clean = ~{100-FPR_PCT}% false alarms): {THRESH}", flush=True)

# ---- live monitor state ----
_window = deque(maxlen=WINDOW)
_counts = {"total": 0, "triggered": 0}


def _img_b64(tensor):
    arr = (tensor.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    im = Image.fromarray(arr, "RGB")
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _serve_query(triggered, attack, alpha, target, informative=True):
    """
    Run one image through the teacher; return clean + served probs.

    When the attack is live, prefer an image the teacher does not already call the
    target class. An image the teacher already calls "airplane", attacked toward
    "airplane", shows an identical before/after and teaches nothing about the
    mechanism. This is presentation only -- the attack itself treats every marked
    query alike, and the page states that the example is chosen this way.
    """
    tries = 30 if (informative and attack and triggered) else 1
    for _ in range(tries):
        i = random.randrange(len(_images))
        x = _images[i:i + 1].clone()
        if triggered:
            x = poisoner(x)
        lg = teacher_logits(x)
        clean = F.softmax(lg, dim=1)
        if int(clean[0].argmax()) != target:
            break
    if attack and triggered:
        served = hijack_soft_labels(lg, target, alpha, rule="soft")
    else:
        served = clean
    return x[0], clean[0].cpu().numpy(), served[0].cpu().numpy(), int(_labels[i])


# ---------------------------------------------------------------- HTTP
def _json(handler, obj, code=200):
    body = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # quiet

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/api/config":
            return _json(self, {"classes": CLASSES, "window": WINDOW,
                                "thresholds": THRESH, "signals": SIGNALS,
                                "labels": LABELS, "tech": TECH, "primary": PRIMARY,
                                "teacher_acc": TEACHER_ACC, "model": MODEL,
                                "fpr": 100 - FPR_PCT})
        if u.path == "/api/query":
            trig = q.get("trigger", "0") == "1"
            attack = q.get("attack", "0") == "1"
            alpha = float(q.get("alpha", "1"))
            target = int(q.get("target", "0"))
            img, clean, served, label = _serve_query(trig, attack, alpha, target)
            return _json(self, {
                "image": _img_b64(img), "true_label": CLASSES[label],
                "clean": clean.tolist(), "served": served.tolist(),
                "pred_clean": CLASSES[int(np.argmax(clean))],
                "pred_served": CLASSES[int(np.argmax(served))],
                "triggered": trig, "hijacked": bool(attack and trig),
                "target": CLASSES[target],
                # the soft-label story: how much probability moved onto the target,
                # whether or not that was enough to change the winning class
                "target_p_clean": float(clean[target]),
                "target_p_served": float(served[target]),
                "flipped": bool(int(np.argmax(clean)) != int(np.argmax(served)))})
        if u.path == "/api/stream_step":
            attack = q.get("attack", "0") == "1"
            alpha = float(q.get("alpha", "0.5"))
            poison = float(q.get("poison", "0.1"))
            target = int(q.get("target", "0"))
            n = int(q.get("n", "25"))
            # build a batch of n queries
            idx = np.random.randint(0, len(_images), n)
            x = _images[idx].clone()
            trig_mask = np.random.random(n) < poison
            if trig_mask.any():
                x[trig_mask] = poisoner(x[trig_mask])
            lg = teacher_logits(x)
            probs = F.softmax(lg, dim=1)
            if attack and trig_mask.any():
                tmask = torch.from_numpy(trig_mask)
                probs[tmask] = hijack_soft_labels(lg[tmask], target, alpha, rule="soft")
            probs = probs.cpu().numpy()
            for row in probs:
                _window.append(row)
            _counts["total"] += n
            _counts["triggered"] += int(trig_mask.sum())
            out = {"fill": len(_window), "window": WINDOW,
                   "total": _counts["total"], "triggered": _counts["triggered"],
                   "scores": {}, "thresholds": THRESH, "alarms": {}, "any_alarm": False}
            if len(_window) >= WINDOW:
                sc = detector.score_window(np.array(_window))
                for s in SIGNALS:
                    out["scores"][s] = float(sc[s])
                    a = sc[s] > THRESH[s]
                    out["alarms"][s] = bool(a)
                    out["any_alarm"] = out["any_alarm"] or a
            return _json(self, out)
        if u.path == "/api/reset":
            _window.clear()
            _counts["total"] = 0; _counts["triggered"] = 0
            return _json(self, {"ok": True})
        self.send_response(404); self.end_headers()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>How a compromised AI teacher poisons its students &mdash; live demo</title>
<style>
:root{--bg:#0e1320;--card:#171f30;--card2:#1d2739;--ink:#e8eef7;--mut:#93a6c4;--dim:#6f81a0;
--line:#2a3550;--red:#ef5350;--green:#3ecf8e;--amber:#f0b429;--blue:#5b9cff;--tgt:#ef5350;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.62 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--blue)}
.page{max-width:1180px;margin:0 auto;padding:0 22px 60px}
header{border-bottom:1px solid var(--line);margin-bottom:22px}
header .page{padding-top:26px;padding-bottom:20px}
h1{font-size:24px;margin:0 0 6px;letter-spacing:-.01em}
.sub{color:var(--mut);font-size:14px}
.badge{display:inline-block;background:#233047;border:1px solid var(--line);border-radius:20px;
padding:3px 11px;font-size:12px;color:var(--mut);margin-right:6px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:18px}
.card h2{font-size:13px;margin:0 0 14px;color:var(--dim);text-transform:uppercase;letter-spacing:.09em;font-weight:600}
.card h3{font-size:15px;margin:0 0 8px}
.lede p{margin:0 0 11px;max-width:78ch}
.lede p:last-child{margin-bottom:0}
.lede b{color:#fff;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.steps{display:flex;gap:10px;flex-wrap:wrap}
.step{flex:1;min-width:170px;text-align:left;background:var(--card2);border:1px solid var(--line);
border-radius:10px;padding:12px 14px;cursor:pointer;color:var(--ink);font:inherit;transition:.15s}
.step:hover{border-color:var(--blue);background:#1b2740}
.step.active{border-color:var(--blue);background:#1e2c47}
.step .n{color:var(--dim);font-size:11px;letter-spacing:.08em;text-transform:uppercase}
.step .t{font-weight:600;margin:3px 0 2px}
.step .d{color:var(--mut);font-size:12.5px;line-height:1.45}
.note{background:#1a2436;border-left:3px solid var(--blue);border-radius:0 8px 8px 0;
padding:12px 15px;margin-top:14px;font-size:14px;color:var(--mut)}
.note b{color:var(--ink)}
.row{display:flex;align-items:center;gap:12px;margin:11px 0;flex-wrap:wrap}
button{background:#26324a;color:var(--ink);border:1px solid var(--line);border-radius:8px;
padding:9px 15px;cursor:pointer;font-size:14px;font-family:inherit}
button:hover{background:#2f3e5c}
button.pri{background:var(--blue);border-color:var(--blue);color:#08111f;font-weight:600}
button.on{background:var(--red);border-color:var(--red);color:#fff}
img#pic{width:132px;height:132px;image-rendering:pixelated;border-radius:8px;border:1px solid var(--line);display:block}
.qcol{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.outs{flex:1;min-width:240px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:620px){.outs{grid-template-columns:1fr}}
.outs .h{font-size:12px;color:var(--dim);margin-bottom:5px;text-transform:uppercase;letter-spacing:.06em}
.bar{display:flex;align-items:center;gap:7px;margin:2px 0;font-size:11.5px}
.bar .lab{width:66px;color:var(--mut);text-align:right;white-space:nowrap}
.bar .track{flex:1;background:#0b1120;border-radius:3px;height:12px;overflow:hidden}
.bar .fill{height:100%;background:var(--blue)}
.bar.tgt .lab{color:var(--tgt)}.bar.tgt .fill{background:var(--tgt)}
.bar .val{width:26px;color:var(--dim);text-align:right}
.verdict{margin-top:12px;font-size:13.5px;padding:10px 13px;border-radius:8px;background:#0f1728;color:var(--mut)}
.verdict b{color:var(--ink)}
input[type=range]{width:150px;accent-color:var(--blue)}
select{background:#26324a;color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:7px 9px;font:inherit}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font-size:12.5px;background:#26324a;min-width:44px;text-align:center}
.meter{margin:16px 0}
.meter .top{display:flex;justify-content:space-between;align-items:baseline;gap:10px;margin-bottom:5px}
.meter .nm{font-size:14px}
.meter .tech{font-size:11px;color:var(--dim);font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meter .v{font-size:13px;color:var(--mut);white-space:nowrap}
.meter .track{position:relative;height:22px;background:#0b1120;border-radius:6px;overflow:hidden}
.meter .fill{height:100%;background:var(--green);transition:width .25s}
.meter.alarm .fill{background:var(--red)}
.meter .thr{position:absolute;top:0;bottom:0;width:2px;background:var(--amber)}
.meter .thrlab{position:absolute;top:-1px;font-size:10px;color:var(--amber);transform:translateX(4px)}
.banner{padding:15px;border-radius:9px;text-align:center;font-weight:700;font-size:15px;margin-top:14px;letter-spacing:.01em}
.banner.idle{background:#1a2436;color:var(--mut);border:1px solid var(--line)}
.banner.clean{background:rgba(62,207,142,.13);color:var(--green);border:1px solid rgba(62,207,142,.45)}
.banner.alarm{background:rgba(239,83,80,.14);color:var(--red);border:1px solid rgba(239,83,80,.5)}
.mut{color:var(--mut);font-size:12.5px}
.legend{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.legend{grid-template-columns:1fr}}
.legend div b{display:block;margin-bottom:4px}
.legend p{margin:0;color:var(--mut);font-size:13.5px}
.honest li{color:var(--mut);margin-bottom:9px;max-width:82ch}
.honest b{color:var(--ink)}
.honest ul{padding-left:20px;margin:0}
</style></head><body>
<header><div class="page">
<h1>How a compromised AI teacher poisons the models it trains</h1>
<div class="sub">This page runs real teacher inference and the dissertation&rsquo;s own detector code.
Student training is not rerun live &mdash; what you see is the stream of answers a student would receive.</div>
<div style="margin-top:10px">
  <span class="badge">teacher: <span id="mname"></span></span>
  <span class="badge">clean accuracy <span id="macc"></span></span>
  <span class="badge">monitoring window <span id="mwin"></span> queries</span>
  <span class="badge"><span id="mfpr"></span>% false-alarm setting</span>
</div>
</div></header>

<div class="page">

<div class="card lede">
  <h3>What am I looking at?</h3>
  <p>Large AI models are expensive to run, so one big <b>teacher</b> model is trained once and its
  knowledge is copied into many small <b>student</b> models that fit on phones and cameras. The student
  learns by sending the teacher example images and copying the teacher&rsquo;s answers.</p>
  <p>This page shows what an attacker can do by compromising only the <b>software that serves the
  teacher&rsquo;s answers</b> &mdash; not the model itself. The teacher&rsquo;s weights are never modified
  and the true labels are never changed. The attacker simply edits the answer the teacher gives back
  whenever an image carries a small hidden mark, so every student trained from it inherits a hidden
  rule: <i>see the mark &rarr; say the attacker&rsquo;s chosen class</i>.</p>
  <p>The second half of the page is the defence: a monitor that sits beside the teacher and watches
  only the stream of answers going out. It is never told what the mark looks like or which class the
  attacker wants.</p>
</div>

<div class="card">
  <h2>Guided walkthrough</h2>
  <div class="steps" id="steps"></div>
  <div class="note" id="stepnote"><b>Start anywhere.</b> Each step sets the controls below and starts the
  monitor for you. Give it a few seconds &mdash; the monitor needs a full window of answers before it can judge
  anything. One window is a single sample, not a verdict: the percentages quoted in each step were measured
  over 400 windows, so on a gentle attack you may need to Reset and run again to see an alarm.</div>
</div>

<div class="grid">
  <div class="card">
    <h2>1 &middot; A single question to the teacher</h2>
    <div class="row">
      <button onclick="q(false)">Ask about a normal image</button>
      <button onclick="q(true)" class="pri">Ask about a marked image</button>
    </div>
    <div class="qcol">
      <div>
        <img id="pic" alt="the image sent to the teacher">
        <div class="mut" style="text-align:center;margin-top:5px">actually a <b id="truth"></b></div>
      </div>
      <div class="outs">
        <div><div class="h">What the teacher really thinks</div><div id="cbars"></div></div>
        <div><div class="h">What the student is told</div><div id="sbars"></div></div>
      </div>
    </div>
    <div class="verdict" id="qinfo"></div>
  </div>

  <div class="card">
    <h2>2 &middot; The monitor beside the teacher</h2>
    <div class="row">
      <button id="atk" onclick="toggleAtk()">Attack: OFF</button>
      <span class="mut">attacker wants everything called</span>
      <select id="target"></select>
    </div>
    <div class="row"><span class="mut" style="width:118px">how hard it pushes</span>
      <input type="range" id="alpha" min="0" max="1" step="0.05" value="0.5">
      <span class="pill" id="alphav">0.50</span></div>
    <div class="row"><span class="mut" style="width:118px">share of marked images</span>
      <input type="range" id="poison" min="0.02" max="0.3" step="0.02" value="0.12">
      <span class="pill" id="poisonv">0.12</span></div>
    <div class="row">
      <button id="run" class="pri" onclick="toggleRun()">Start the monitor</button>
      <button onclick="reset()">Reset</button>
    </div>
    <div class="meter" id="m0"><div class="top"><span><span class="nm"></span><br><span class="tech"></span></span><span class="v"></span></div>
      <div class="track"><div class="fill"></div><div class="thr"></div><span class="thrlab">alarm line</span></div></div>
    <div class="meter" id="m1"><div class="top"><span><span class="nm"></span><br><span class="tech"></span></span><span class="v"></span></div>
      <div class="track"><div class="fill"></div><div class="thr"></div><span class="thrlab">alarm line</span></div></div>
    <div class="mut" id="counts">not started</div>
    <div class="banner idle" id="banner">MONITOR IDLE</div>
  </div>
</div>

<div class="card">
  <h2>What the two meters mean</h2>
  <div class="legend">
    <div><b>Is one class showing up too often?</b>
    <p>The attack has to push answers toward one chosen class, over and over. This meter tracks how far
    the most over-represented class has risen above its normal share. It is the signal the dissertation
    commits to <i>in advance</i>, because it is the one the threat model implies &mdash; not the one that
    happened to score best afterwards.</p></div>
    <div><b>Has the average answer drifted?</b>
    <p>Compares the average answer across the whole window against a healthy teacher&rsquo;s. It notices
    nudges too small to change any single prediction, which is exactly how a cautious attacker operates.
    It shares its underlying principle &mdash; distance from an estimated benign reference &mdash; with the
    nearest existing defence (Yu et al., 2024), but it is not that method: theirs estimates
    class-conditioned references and reweights clients in federated distillation, whereas this scores one
    teacher&rsquo;s window against a single unconditional mean.</p></div>
  </div>
  <div class="note"><b>Why an occasional alarm on a clean stream is normal.</b> The alarm lines are set
  from clean traffic so that roughly <span id="fprtxt"></span> of clean windows trip them &mdash; the same
  operating point used throughout the evaluation. A single alarm is not proof of an attack; a
  <i>sustained</i> one is what matters.</div>
</div>

<div class="card honest">
  <h2>Being straight about what this does and does not show</h2>
  <ul>
    <li><b>The mark has to be in the images the student trains on.</b> No student can learn a mark it is
    never shown &mdash; that is true of every backdoor of this kind. What is different here is that the
    <i>labels stay honest</i>: nothing in the data is mislabelled, and the entire malicious signal travels
    in the teacher&rsquo;s answers. That is precisely why watching those answers works as a defence.</li>
    <li><b>This page shows the teacher&rsquo;s side only.</b> It does not train a student, because that
    takes GPU minutes per run. The effect on real students &mdash; attack success climbing from 2% to 62%
    while their ordinary accuracy never moves &mdash; is measured in Chapter&nbsp;5.</li>
    <li><b>This runs on the same architecture and configuration as Chapter&nbsp;5.</b> The teacher is a
    <span id="mname2"></span> trained on full CIFAR-10. It is not literally the same file: the teacher
    behind the 25&nbsp;August results was held on a rented GPU instance and was not preserved, so this
    is an independent retrain of the identical configuration. Measured across the whole CIFAR-10 test
    set it reaches 86.49% clean accuracy against the original&rsquo;s 86.32%. The badge at the top of
    the page reads <span id="macc2"></span> instead, because that one is measured on the smaller pool
    of images this page samples from. The detector results reproduce on the retrained teacher to within
    about 0.01 of AUC, which is reported in Section&nbsp;5.3, so what you see here should land close to
    the dissertation&rsquo;s numbers rather than merely resemble them.</li>
    <li><b>The monitor is deliberately kept ignorant.</b> It never sees the mark, never learns the
    attacker&rsquo;s target class, and never inspects the model&rsquo;s weights. It only ever sees the
    outgoing answers.</li>
  </ul>
</div>

</div>
<script>
let CFG,running=false,atk=false,timer=null,active=null;
const $=id=>document.getElementById(id);

const STEPS=[
 {n:"Step 1",t:"Normal service",d:"Attack off. See what healthy traffic looks like.",
  atk:false,alpha:0.5,poison:0.12,
  note:"The teacher is answering honestly, so both meters should sit well below their alarm lines. Measured over 400 clean windows on this teacher, an alarm fires on 6% of them, which is the 5% false-alarm rate the thresholds were set for. An occasional alarm here is the calibration working rather than a fault."},
 {n:"Step 2",t:"A loud attack",d:"Push hard toward one class. Watch both meters climb.",
  atk:true,alpha:1.0,poison:0.12,
  note:"At full strength the marked images are answered as the target class outright, so that class floods the stream. Measured over 400 windows, both meters fire on all of them. This is the easy case for the defender, and it is also the strongest version of the attack."},
 {n:"Step 3",t:"A sneaky attack",d:"Push gently instead. One meter goes quiet.",
  atk:true,alpha:0.25,poison:0.12,
  note:"This is the case worth waiting on. Over 400 windows the drift meter fires on 22% of them and the class meter on only 6%, so drift is roughly four times more sensitive to a gentle nudge, which is the argument for carrying two signals. But 22% is not the same as caught: a single window of 400 queries usually still looks clean, so press Reset and run it again a few times before it trips. That gap is why Section 5.6 answers a quiet attack with a longer window or pooled victims rather than with a cleverer statistic."},
 {n:"Step 4",t:"Too quiet to see",d:"Push barely at all — and see the cost of hiding.",
  atk:true,alpha:0.10,poison:0.08,
  note:"At this window the attack is effectively invisible: 9% of windows alarm against the 6% produced by clean traffic, which is barely distinguishable from the false-alarm floor. It is also barely an attack. This is the trade-off the dissertation is really about, and Chapter 5 puts the cost of hiding at roughly 87% of the attacker's power. What is left is answered by watching for longer, not by a better statistic."}
];

function drawSteps(){
  const c=$('steps');c.innerHTML='';
  STEPS.forEach((s,i)=>{
    const b=document.createElement('button');b.className='step'+(active===i?' active':'');
    b.innerHTML=`<div class="n">${s.n}</div><div class="t">${s.t}</div><div class="d">${s.d}</div>`;
    b.onclick=()=>runStep(i);c.appendChild(b);
  });
}
async function runStep(i){
  const s=STEPS[i];active=i;
  atk=s.atk;$('atk').textContent='Attack: '+(atk?'ON':'OFF');$('atk').className=atk?'on':'';
  $('alpha').value=s.alpha;$('alphav').textContent=(+s.alpha).toFixed(2);
  $('poison').value=s.poison;$('poisonv').textContent=(+s.poison).toFixed(2);
  $('stepnote').innerHTML=s.note;
  drawSteps();
  await fetch('/api/reset');
  if(!running){running=true;$('run').textContent='Stop the monitor';timer=setInterval(step,420);}
  q(true);
}

function bars(el,probs,tgt){el.innerHTML='';
  probs.forEach((p,i)=>{const d=document.createElement('div');d.className='bar'+(i===tgt?' tgt':'');
  d.innerHTML=`<span class="lab">${CFG.classes[i]}</span><span class="track"><span class="fill" style="width:${(p*100).toFixed(1)}%"></span></span><span class="val">${(p*100).toFixed(0)}</span>`;
  el.appendChild(d);});}

async function q(trig){
  const t=$('target').value,a=$('alpha').value;
  const d=await(await fetch(`/api/query?trigger=${trig?1:0}&attack=${atk?1:0}&alpha=${a}&target=${t}`)).json();
  $('pic').src=d.image;$('truth').textContent=d.true_label;
  bars($('cbars'),d.clean,d.hijacked?+t:-1);bars($('sbars'),d.served,d.hijacked?+t:-1);
  const v=$('qinfo');
  if(d.hijacked){
    const pc=(d.target_p_clean*100).toFixed(0), ps=(d.target_p_served*100).toFixed(0);
    const tgt=CFG.classes[+t];
    v.innerHTML=`This image carries the mark &mdash; the small bright square in the bottom-right corner.
      The teacher itself still thinks <b>${d.pred_clean}</b>, and its weights were never touched.
      The edit moves the chance the student is told for <b>${tgt}</b> from <b>${pc}%</b> to <b>${ps}%</b>.
      ` + (d.flipped
        ? `That is enough to change the answer outright: the student is told <b>${d.pred_served}</b>.`
        : `That is not yet enough to change the winning class &mdash; the student is still told
           <b>${d.pred_served}</b> &mdash; and that is exactly what makes a gentle attack hard to spot
           one query at a time. It still shifts the average, which is what the monitor watches.`)
      + ` <span class="mut">(With the attack on, the demo picks an image the teacher does not already
        call &ldquo;${tgt}&rdquo;, so the change is visible. The attack itself treats every marked query alike.)</span>`;
  }else if(d.triggered){
    v.innerHTML=`Marked image, but the attack is switched off, so the answer leaves untouched
      (<b>${d.pred_served}</b>). Turn the attack on to see the same image answered differently.`;
  }else{
    v.innerHTML=`An ordinary image with no mark. The attacker leaves these completely alone &mdash;
      which is why the teacher keeps looking healthy on everyday traffic.`;
  }
}
function toggleAtk(){atk=!atk;$('atk').textContent='Attack: '+(atk?'ON':'OFF');
  $('atk').className=atk?'on':'';active=null;drawSteps();}

function setMeter(id,sig,score,thr){
  const m=$(id),full=Math.max(thr*2.2,1e-9),pct=Math.min(100,100*score/full),tp=Math.min(100,100*thr/full);
  const al=score>thr;m.className='meter'+(al?' alarm':'');
  m.querySelector('.v').textContent=al?'over the line':'below the line';
  m.querySelector('.fill').style.width=pct+'%';
  m.querySelector('.thr').style.left=tp+'%';
  m.querySelector('.thrlab').style.left=tp+'%';
}
async function step(){
  const a=$('alpha').value,p=$('poison').value,t=$('target').value;
  const d=await(await fetch(`/api/stream_step?attack=${atk?1:0}&alpha=${a}&poison=${p}&target=${t}&n=40`)).json();
  $('counts').textContent=`${d.total} questions asked · ${d.triggered} of them marked · window ${d.fill}/${d.window}`;
  const b=$('banner');
  if(d.scores[CFG.signals[0]]!==undefined){
    CFG.signals.forEach((s,k)=>setMeter('m'+k,s,d.scores[s],d.thresholds[s]));
    if(d.any_alarm){b.className='banner alarm';b.textContent='⚠  SOMETHING IS WRONG WITH THIS TEACHER’S ANSWERS';}
    else{b.className='banner clean';b.textContent='✓  THESE ANSWERS LOOK HEALTHY';}
  }else{b.className='banner idle';
    b.textContent=`collecting answers… ${d.fill} of ${d.window} needed before the monitor can judge`;}
}
function toggleRun(){running=!running;$('run').textContent=running?'Stop the monitor':'Start the monitor';
  if(running){timer=setInterval(step,420);}else{clearInterval(timer);}}
async function reset(){await fetch('/api/reset');active=null;drawSteps();
  $('banner').className='banner idle';$('banner').textContent='MONITOR IDLE';
  ['m0','m1'].forEach(i=>{$(i).className='meter';$(i).querySelector('.fill').style.width='0%';
    $(i).querySelector('.v').textContent='';});$('counts').textContent='not started';}
$('alpha').oninput=e=>{$('alphav').textContent=(+e.target.value).toFixed(2);active=null;drawSteps();};
$('poison').oninput=e=>{$('poisonv').textContent=(+e.target.value).toFixed(2);active=null;drawSteps();};

(async()=>{CFG=await(await fetch('/api/config')).json();
  const acc=(CFG.teacher_acc*100).toFixed(1)+'%';
  $('mname').textContent=CFG.model;$('macc').textContent=acc;
  $('mname2').textContent=CFG.model;$('macc2').textContent=acc;
  $('mwin').textContent=CFG.window;$('mfpr').textContent=CFG.fpr;
  $('fprtxt').textContent=CFG.fpr+'%';
  const sel=$('target');CFG.classes.forEach((c,i)=>{const o=document.createElement('option');
    o.value=i;o.textContent=c;sel.appendChild(o);});
  CFG.signals.forEach((s,k)=>{$('m'+k).querySelector('.nm').textContent=CFG.labels[s];
    $('m'+k).querySelector('.tech').textContent=CFG.tech[s];});
  drawSteps();q(true);})();
</script></body></html>"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f">>> demo ready → http://127.0.0.1:{port}  (Ctrl-C to stop)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

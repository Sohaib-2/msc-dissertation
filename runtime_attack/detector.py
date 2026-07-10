"""
Teacher-side runtime detector.

The detector watches the stream of soft-labels a teacher serves during distillation and
flags when that stream stops looking like a healthy teacher's. It never sees the weights,
never trains a student and never inspects the inputs; it sees only the output vectors.
That is the surface the runtime attack has to touch in order to work, which is why it is
the place to watch.

The underlying idea is not new in itself. Monitoring a model's output distribution for
drift is ordinary practice in deployed machine-learning systems. What is applied here is
that idea, at the teacher, against a compromise that is conditional on a trigger.

Where the detector sits. The attacker owns the serving wrapper, so the detector cannot
live inside it or it could simply be switched off. It is modelled as a separate monitoring
process that samples the outgoing stream. A victim auditing the soft-labels it receives
before distilling is an equally valid reading of the same code.

Two constraints are imposed on the detector deliberately:

  Target-agnostic. It is not told the attacker's target class. It looks for any class
  becoming over-represented. Telling it the target would make the evaluation circular.

  Trigger-agnostic. It does not know what the trigger looks like, or which queries carry
  it. It sees an undifferentiated stream of outputs.

Detection is measured at two levels. Per-query scoring treats each output vector on its
own and is expected to be weak, because a clean teacher is often legitimately confident
and a confident hijacked output resembles a confident honest one. Windowed scoring treats
a batch of outputs together, which is where the signal should live: the hijack has to push
probability mass toward one class repeatedly, and that accumulates even when no individual
output looks unusual. Both are measured rather than assumed.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-12


# ----------------------------------------------------------------------------
# metrics and small numeric helpers
# ----------------------------------------------------------------------------
def entropy(probs):
    """Shannon entropy (nats) of each row of a (N, C) probability matrix."""
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def kl_divergence(p, q):
    """KL(p || q) for 1-D distributions."""
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), EPS, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def class_histogram(probs, num_classes):
    """Normalised frequency of the argmax (hard prediction) per class."""
    preds = np.argmax(probs, axis=1)
    counts = np.bincount(preds, minlength=num_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def _labelled(neg_scores, pos_scores):
    """Pack clean(=0) and attacked(=1) scores into the (y_true, y_score) form sklearn wants."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    y_true = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y_score = np.concatenate([neg, pos])
    return y_true, y_score


def roc_auc(neg_scores, pos_scores):
    """
    ROC-AUC separating CLEAN windows (negatives) from ATTACKED windows (positives).
    1.0 = perfect detection, 0.5 = the detector is guessing.

    Uses sklearn's reference implementation rather than a bespoke one: these numbers go in
    the dissertation, and a standard, widely-audited metric is not something an examiner
    has to take on trust.
    """
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return float("nan")
    y_true, y_score = _labelled(neg_scores, pos_scores)
    return float(roc_auc_score(y_true, y_score))


def tpr_at_fpr(neg_scores, pos_scores, target_fpr=0.05):
    """
    Detection rate achievable while holding false alarms down to `target_fpr`.
    This is the number that actually matters operationally: a detector that screams
    constantly is useless no matter how good its AUC looks.

    Returns (tpr, threshold) at the largest operating point whose FPR <= target_fpr.
    """
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return float("nan"), float("nan")
    y_true, y_score = _labelled(neg_scores, pos_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ok = fpr <= target_fpr
    if not ok.any():
        return 0.0, float("inf")
    i = np.max(np.flatnonzero(ok))
    return float(tpr[i]), float(thresholds[i])


# ----------------------------------------------------------------------------
# The detector
# ----------------------------------------------------------------------------
class TeacherOutputDetector:
    """
    Calibrate on a clean teacher's outputs, then score unseen output streams by how far
    they deviate from that baseline. Higher score = more suspicious.
    """

    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.calibrated = False

    # -- 1. learn what "normal" looks like -----------------------------------
    def calibrate(self, clean_probs):
        """
        clean_probs: (N, C) soft-labels from the honest teacher on clean, held-out data.
        Everything the detector believes about "normal" comes from here and nowhere else.
        """
        clean_probs = np.asarray(clean_probs, dtype=np.float64)
        ent = entropy(clean_probs)
        maxp = clean_probs.max(axis=1)

        self.ref_entropy_mean = float(ent.mean())
        self.ref_entropy_std = float(ent.std() + EPS)
        self.ref_maxprob_mean = float(maxp.mean())
        self.ref_maxprob_std = float(maxp.std() + EPS)
        # the average output vector, and the hard-prediction class balance
        self.ref_mean_prob = clean_probs.mean(axis=0)
        self.ref_hist = class_histogram(clean_probs, self.num_classes)
        self.calibrated = True
        return self

    def _check(self):
        if not self.calibrated:
            raise RuntimeError("detector must be calibrated on clean teacher outputs first")

    # -- 2. per-query signals (expected to be weak — measured, not assumed) ---
    def score_per_query(self, probs):
        """
        Returns a dict of (N,) score arrays, one per signal. Higher = more suspicious.
        A hijack toward one class tends to lower entropy and raise confidence, so the
        flip the sign on those to keep "higher = worse" consistent across signals.
        """
        self._check()
        probs = np.asarray(probs, dtype=np.float64)
        ent = entropy(probs)
        maxp = probs.max(axis=1)
        # deviation of each individual output from the average clean output
        dev = np.array([kl_divergence(row, self.ref_mean_prob) for row in probs])
        return {
            "entropy_drop": (self.ref_entropy_mean - ent) / self.ref_entropy_std,
            "confidence_spike": (maxp - self.ref_maxprob_mean) / self.ref_maxprob_std,
            "kl_from_clean_mean": dev,
        }

    # -- 3. windowed / stream signals (where the signal should live) ---------
    def score_window(self, probs):
        """
        Score a whole window of outputs together. Returns a dict of scalars.
        Higher is more suspicious. Every signal is target-agnostic: it looks for any class
        becoming over-represented, never a specific one.
        """
        self._check()
        probs = np.asarray(probs, dtype=np.float64)

        hist = class_histogram(probs, self.num_classes)
        mean_prob = probs.mean(axis=0)
        ent = entropy(probs)

        return {
            # how far the window's class balance has drifted from the clean balance
            "hist_kl": kl_divergence(hist, self.ref_hist),
            # the single most over-represented class. Target-agnostic: take the max,
            # since the detector is not told which class the attacker wants.
            "max_class_excess": float(np.max(hist - self.ref_hist)),
            # average sharpening of the outputs
            "mean_entropy_drop": float(self.ref_entropy_mean - ent.mean()),
            # drift of the average soft-label vector itself (uses the full distribution,
            # not just the argmax — catches nudges too small to flip a prediction)
            "mean_prob_kl": kl_divergence(mean_prob, self.ref_mean_prob),
        }


    # -- 4. the operational decision rule -----------------------------------
    #
    # score_window returns four raw statistics on four different scales: a KL
    # divergence, a probability difference, a nats difference, another KL. A raw
    # maximum over them would simply track whichever happens to have the largest
    # magnitude, which is not a detector. To combine them the detector first
    # learns, from CLEAN windows only, what each signal's ordinary spread looks
    # like, and then scores an unseen window by how many standard deviations
    # each signal has moved -- putting all four on a common, dimensionless
    # scale -- before taking the maximum.
    #
    # Everything here is fixed before any attacked data is seen. The calibration
    # windows are clean, the signal set is the full SIGNALS_WINDOW list, and the
    # combination is a plain maximum, so no choice in this rule depends on
    # knowing the attack, the trigger, the target class, or the evaluation
    # labels. That is what makes it a deployable rule rather than a post-hoc
    # comparison of signals.

    def calibrate_windows(self, clean_probs, window=200, n_windows=300, seed=0):
        """
        Second calibration stage: learn the mean and spread of each window signal
        on CLEAN windows, so the signals can be standardised and combined.

        clean_probs: (N, C) honest-teacher outputs, disjoint from anything scored later.
        """
        self._check()
        clean_probs = np.asarray(clean_probs, dtype=np.float64)
        if len(clean_probs) < window:
            raise ValueError(f"need at least {window} clean outputs, got {len(clean_probs)}")
        rng = np.random.default_rng(seed)

        collected = {s: [] for s in SIGNALS_WINDOW}
        for _ in range(n_windows):
            sel = rng.choice(len(clean_probs), size=window, replace=False)
            sc = self.score_window(clean_probs[sel])
            for s in SIGNALS_WINDOW:
                collected[s].append(sc[s])

        self.win_ref_mean = {s: float(np.mean(v)) for s, v in collected.items()}
        self.win_ref_std = {s: float(np.std(v) + EPS) for s, v in collected.items()}
        self.win_calibrated = True
        return self

    def _check_windows(self):
        if not getattr(self, "win_calibrated", False):
            raise RuntimeError("call calibrate_windows() before using the combined rule")

    def score_window_z(self, probs):
        """Each window signal expressed in clean standard deviations. Higher = more suspicious."""
        self._check_windows()
        raw = self.score_window(probs)
        return {s: (raw[s] - self.win_ref_mean[s]) / self.win_ref_std[s]
                for s in SIGNALS_WINDOW}

    def score_combined(self, probs, signals=None):
        """
        THE detector's decision statistic: the largest standardised deviation across
        the signal set. One scalar per window, comparable across conditions, fixed
        in advance of any evaluation.
        """
        z = self.score_window_z(probs)
        use = SIGNALS_WINDOW if signals is None else signals
        return float(max(z[s] for s in use))

    def window_scores(self, probs, window=200):
        """
        Chop a stream into consecutive windows and score each one.
        Returns {signal_name: np.array of per-window scores}.
        """
        self._check()
        probs = np.asarray(probs, dtype=np.float64)
        n_win = len(probs) // window
        if n_win == 0:
            raise ValueError(f"stream of {len(probs)} too short for window={window}")
        out = {}
        for i in range(n_win):
            s = self.score_window(probs[i * window:(i + 1) * window])
            for k, v in s.items():
                out.setdefault(k, []).append(v)
        return {k: np.array(v) for k, v in out.items()}


SIGNALS_WINDOW = ["hist_kl", "max_class_excess", "mean_entropy_drop", "mean_prob_kl"]
SIGNALS_QUERY = ["entropy_drop", "confidence_spike", "kl_from_clean_mean"]

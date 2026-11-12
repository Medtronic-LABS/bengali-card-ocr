"""Read a two-line EPI date as a real date, not just a digit string.

The split date's top line holds day + separator + month, the bottom line the
year. Greedy CTC reads each line character by character and can return a
month of 00, a day of 35 or a dropped separator. Here every valid value is
scored instead (day 1-31 for its month, month 1-12, separator / . -, retaining
the year actually read by the recognizer) with the full CTC
probability of the model's own output, and the most likely one is kept. The
network is unchanged; this only restricts which strings it may produce.

The day/month/year split happens in the model output (one score per part),
not by cutting the image, so no second segmentation step can fail.
"""

import datetime

import numpy as np

BN_DIGIT = "০১২৩৪৫৬৭৮৯"
EN_DIGIT = "0123456789"
SEPARATORS = "/.-"
DAYS_IN_MONTH = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
NEG = -1e30


def _num(n, digits, pad):
    s = "".join(digits[int(c)] for c in str(n))
    return s.rjust(2, digits[0]) if pad else s


def top_candidates(digits):
    """Every 'day sep month' string, with and without zero padding."""
    out = []
    for m in range(1, 13):
        months = {_num(m, digits, False), _num(m, digits, True)}
        for d in range(1, DAYS_IN_MONTH[m - 1] + 1):
            for day in {_num(d, digits, False), _num(d, digits, True)}:
                out += [day + sep + month for month in months for sep in SEPARATORS]
    return out


def is_valid_date(text, year=None):
    """Check a complete calendar date without rejecting it based on today's year."""
    import re

    t = (text or "").strip().translate(str.maketrans(BN_DIGIT, EN_DIGIT))
    match = re.fullmatch(r"(\d{1,2})[./-](\d{1,2})[./\n-](\d{2}|\d{4})", t)
    if not match:
        return False
    d, m, y = (int(g) for g in match.groups())
    if len(match.group(3)) == 2:
        y += 2000
    try:
        datetime.date(y, m, d)
    except ValueError:
        return False
    return True


def ctc_log_likelihoods(logp, labels):
    """log P(label | frames) for many labels at once (batched CTC forward).

    logp: (T, K) log-probabilities, column 0 the CTC blank. labels: lists of
    column indices >= 1.
    """
    n, longest = len(labels), max(len(lab) for lab in labels)
    ext = np.zeros((n, 2 * longest + 1), dtype=np.int64)
    lengths = np.array([len(lab) for lab in labels])
    for i, lab in enumerate(labels):
        ext[i, 1 : 2 * len(lab) : 2] = lab
    skip = np.zeros_like(ext, dtype=bool)
    skip[:, 2:] = (ext[:, 2:] != 0) & (ext[:, 2:] != ext[:, :-2])
    alpha = np.full(ext.shape, NEG)
    alpha[:, 0] = logp[0, 0]
    alpha[:, 1] = logp[0, ext[:, 1]]
    pad1, pad2 = np.full((n, 1), NEG), np.full((n, 2), NEG)
    for t in range(1, logp.shape[0]):
        prev1 = np.hstack([pad1, alpha[:, :-1]])
        prev2 = np.where(skip, np.hstack([pad2, alpha[:, :-2]]), NEG)
        alpha = np.logaddexp(np.logaddexp(alpha, prev1), prev2) + logp[t, ext]
    rows = np.arange(n)
    ends = 2 * lengths
    return np.logaddexp(alpha[rows, ends], alpha[rows, ends - 1])


def best_of(logp_by_model, candidates, column):
    """Most likely candidate, summing log-likelihoods over models, and its share of
    the probability mass among all candidates (an uncalibrated confidence)."""
    labels = [[column[c] for c in cand] for cand in candidates]
    score = sum(ctc_log_likelihoods(lp, labels) for lp in logp_by_model)
    i = int(np.argmax(score))
    share = float(1.0 / np.exp(score - score[i]).sum())
    return candidates[i], float(score[i]), share


def read_two_line_date(top_logits, year_logits, chars, year=None):
    """(top, year, confidence) for the best valid date in one numeral script.

    top_logits / year_logits: one (T, C) logit array per cell model, C = 1 +
    len(chars) with column 0 the CTC blank. Returns None if no model output.
    """
    best = None
    for digits in (BN_DIGIT, EN_DIGIT):
        keep = [0] + [i + 1 for i, c in enumerate(chars) if c in digits + SEPARATORS]
        column = {chars[k - 1]: j for j, k in enumerate(keep) if k}
        if any(c not in column for c in digits + SEPARATORS):
            continue

        def restricted(logits):
            sub = logits[:, keep]
            sub = sub - sub.max(-1, keepdims=True)
            return sub - np.log(np.exp(sub).sum(-1, keepdims=True))

        top, top_score, top_share = best_of([restricted(lg) for lg in top_logits], top_candidates(digits), column)
        # Preserve the recognizer's actual year; do not silently replace it with
        # a plausible-looking year from an arbitrary, moving time window.
        lps = [restricted(lg) for lg in year_logits]
        ids = lps[0].argmax(-1)
        keep_ids = (ids != 0) & np.concatenate([[True], ids[1:] != ids[:-1]])
        reverse = {v: k for k, v in column.items()}
        yr = ''.join(reverse[int(i)] for i in ids[keep_ids])
        if len(yr) not in (2, 4) or any(c not in digits for c in yr):
            continue
        if not is_valid_date(top + "\n" + yr):
            continue
        yr, yr_score, yr_share = best_of(lps, [yr], column)
        if best is None or top_score + yr_score > best[0]:
            best = (top_score + yr_score, top, yr, min(top_share, yr_share))
    return None if best is None else best[1:]

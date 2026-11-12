"""Field syntax checks, and strict date/BP formatting of model text.

syntax_ok never infers anything. format_structured may place separators the
model dropped or misread, but only when exactly one valid value fits the
observed digits; the result says whether anything was inferred.
"""

import datetime
import itertools
import re

BN2EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
# Observed separator -> separator used in the formatted value.
SEPARATORS = {
    "/": "/",
    "|": "/",
    "│": "/",
    "।": "/",
    "\n": "/",
    ".": ".",
    ",": ".",
    "-": "-",
}


def syntax_ok(raw, value_type):
    if not raw:
        return False
    t = raw.translate(BN2EN).strip()
    if value_type == "numeric":
        return bool(re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?°?", t))
    if value_type == "blood_pressure":
        return bool(re.fullmatch(r"\d{2,3}\s*[/|│]\s*\d{2,3}", t))
    if value_type == "date":
        return bool(re.fullmatch(r"\d{1,2}[./-]\d{1,2}(?:[./\n-]\d{2,4})?", t))
    if value_type == "mark":
        return bool(re.fullmatch(r'[()\s✓√✔xX×╳+−–—\-〃"±]+', t))
    return bool(t)


def _tokens(text):
    """Digit runs and the separator after each, or None for other characters."""
    groups, seps = [""], []
    for ch in text.strip():
        if ch.translate(BN2EN).isdigit():
            groups[-1] += ch
        elif ch in SEPARATORS:
            if groups[-1]:
                seps.append(SEPARATORS[ch])
                groups.append("")
        elif ch in " \t":
            continue
        else:
            return None
    if not groups[-1]:
        groups.pop()
        seps = seps[: len(groups) - 1]
    return (groups, seps) if groups else None


def _splits(digits, sizes):
    """Every way to cut digits into len(sizes) parts with allowed part lengths."""
    if len(sizes) == 1:
        return [[digits]] if len(digits) in sizes[0] else []
    out = []
    for n in sizes[0]:
        if n < len(digits):
            out += [[digits[:n]] + rest for rest in _splits(digits[n:], sizes[1:])]
    return out


def _valid(parts, value_type, year):
    n = [int(p.translate(BN2EN)) for p in parts]
    if value_type == "date":
        d, m, y = n
        if len(parts[2]) == 2:
            y += 2000
        try:
            datetime.date(y, m, d)
        except ValueError:
            return False
        return True
    systolic, diastolic = n
    return 70 <= systolic <= 250 and 40 <= diastolic <= 150 and systolic > diastolic


def _candidates(groups, seps, value_type, year):
    sizes = [(1, 2), (1, 2), (2, 4)] if value_type == "date" else [(2, 3), (2, 3)]
    if len(groups) > len(sizes):
        return set()
    default = max(set(seps), key=seps.count) if seps else "/"
    found = set()
    # Observed separators stay where they are; missing ones are cut inside a
    # digit run. counts[i] = how many value parts digit run i holds.
    for counts in itertools.product(range(1, len(sizes) + 1), repeat=len(groups)):
        if sum(counts) != len(sizes):
            continue
        k, per_group = 0, []
        for group, c in zip(groups, counts):
            per_group.append(_splits(group, sizes[k : k + c]))
            k += c
        for choice in itertools.product(*per_group):
            parts = [p for chunk in choice for p in chunk]
            if not _valid(parts, value_type, year):
                continue
            out_seps = []
            for i, chunk in enumerate(choice):
                out_seps += [default] * (len(chunk) - 1)
                if i < len(choice) - 1:
                    out_seps.append(seps[i])
            found.add(parts[0] + "".join(s + p for s, p in zip(out_seps, parts[1:])))
    return found


def format_structured(raw, value_type, year=None):
    """Return (value, inferred) when exactly one valid date/BP fits raw, else None.

    inferred is False only if every separator was observed as a canonical
    separator. A digit 1 that stands where a written bar/slash should be is
    reinterpreted as a separator only if no reading without that works.
    """
    if value_type not in ("date", "blood_pressure") or not raw:
        return None
    parsed = _tokens(raw)
    if not parsed:
        return None
    groups, seps = parsed
    found = _candidates(groups, seps, value_type, year)
    if not found:
        ones = [i for i, ch in enumerate(raw) if ch in "1১"]
        for n in (1, 2):
            for chosen in itertools.combinations(ones, n):
                text = "".join("/" if i in chosen else ch for i, ch in enumerate(raw))
                again = _tokens(text)
                if again:
                    found |= _candidates(*again, value_type, year)
            if found:
                break
    if len(found) != 1:
        return None
    value = found.pop()
    observed = re.sub(r"\s", "", raw)
    return value, value != observed

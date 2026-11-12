"""Field validation — flag suspicious values; never invent clinical numbers."""

from __future__ import annotations

import re
from typing import List, Optional

BN_DIGIT = "০১২৩৪৫৬৭৮৯"
EN_DIGIT = "0123456789"
BN2EN = str.maketrans(BN_DIGIT, EN_DIGIT)


def _west(s: str) -> str:
    return s.translate(BN2EN)


def validate_field(raw: Optional[str], value_type: str, status: str) -> List[str]:
    warnings: List[str] = []
    if status in ("blank", "unreadable") or raw is None or raw == "":
        return warnings

    t = _west(raw.replace("\n", ".").replace("|", "/"))
    from .structured import syntax_ok

    if not syntax_ok(raw, value_type):
        warnings.append("Field syntax requires human review")

    if value_type == "blood_pressure":
        m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", t)
        if not m:
            warnings.append("BP format not recognized (expected systolic/diastolic)")
        else:
            sys_v, dia = int(m.group(1)), int(m.group(2))
            if not (70 <= sys_v <= 250 and 40 <= dia <= 150):
                warnings.append(f"BP out of typical range: {sys_v}/{dia}")
            if sys_v < dia:
                warnings.append("Systolic < diastolic")

    elif value_type == "date":
        digits = re.findall(r"\d+", t)
        if not digits:
            warnings.append("No digits found in date field")
        else:
            # Soft check only
            nums = [int(x) for x in digits if x]
            if any(n > 31 and n < 100 and i == 0 for i, n in enumerate(nums[:1])):
                pass
            if nums and nums[0] > 31:
                warnings.append("Day component > 31")

    elif value_type == "numeric":
        m = re.search(r"\d+(\.\d+)?", t)
        if not m:
            warnings.append("No numeric value found")
        elif re.fullmatch(r"(\d)\1{2,}", t.strip()):
            # Printed rules in near-empty cells read as ০০০০; no card value looks like this.
            warnings.append("One digit repeated; likely printed lines, not a value")

    return warnings

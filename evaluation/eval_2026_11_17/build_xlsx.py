"""Build EPI_ANC_NCD_eval_matrix.xlsx from the run outputs in this folder.

Usage: python3 build_xlsx.py <eval_dir> <bengali-card-ocr dir>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

EVAL, REPO = Path(sys.argv[1]), Path(sys.argv[2])
sys.argv = [sys.argv[0], str(EVAL), str(REPO)]
sys.path.insert(0, str(EVAL))
import analyze as A  # noqa: E402

new = A.load(sorted(EVAL.glob("data/work/eval_pages_cv*.json")))
old = A.load(sorted(REPO.glob("data/work/eval_pages_cv*.json")))
M = json.loads((EVAL / "eval_matrix.json").read_text())
H = json.loads((EVAL / "human_labels_result.json").read_text())
CARDS = ["ANC", "NCD", "EPI"]
sel = lambda rs, c: rs if c == "ALL" else [r for r in rs if r["card"] == c]  # noqa: E731

F = "Arial"
HEAD = Font(name=F, bold=True, color="FFFFFF")
HFILL = PatternFill("solid", start_color="1F3A5F")
BOLD = Font(name=F, bold=True)
BASE = Font(name=F)
NOTE = Font(name=F, italic=True, color="555555", size=9)
TITLE = Font(name=F, bold=True, size=14)
ALLFILL = PatternFill("solid", start_color="E8EEF5")
KEYFILL = PatternFill("solid", start_color="FFF4CC")
thin = Side(style="thin", color="BBBBBB")
BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
PCT = "0.0%"

wb = Workbook()


def header(ws, row, cols, widths=None):
    for i, c in enumerate(cols, 1):
        cell = ws.cell(row=row, column=i, value=c)
        cell.font, cell.fill, cell.border = HEAD, HFILL, BOX
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[row].height = 58
    if widths:
        for i, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w


def put(ws, row, values, fmts=None, font=BASE, fill=None):
    for i, v in enumerate(values, 1):
        cell = ws.cell(row=row, column=i, value=v)
        cell.font, cell.border = font, BOX
        if fill:
            cell.fill = fill
        if fmts and fmts.get(i):
            cell.number_format = fmts[i]
        elif isinstance(v, int) or (isinstance(v, str) and v.startswith("=SUM")):
            cell.number_format = "#,##0"
        if i > 1:
            cell.alignment = Alignment(horizontal="center")


def note(ws, row, text, ncols=10):
    ws.cell(row=row, column=1, value=text).font = NOTE
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    ws.cell(row=row, column=1).alignment = Alignment(wrap_text=True, vertical="top")
    width = sum((ws.column_dimensions[get_column_letter(k)].width or 9) for k in range(1, ncols + 1))
    lines = -(-len(text) // max(20, int(width * 1.15)))
    ws.row_dimensions[row].height = 13 * lines + 4


# ---------------- Sheet 1: Eval matrix ----------------
ws = wb.active
ws.title = "Eval matrix"
ws["A1"] = "Bengali card OCR (PP-Cards) — eval on EPI_ANC_NCD_cards.zip"
ws["A1"].font = TITLE

ws["A4"] = "1. Field accuracy"
ws["A4"].font = BOLD
cols = ["Card type", "Cards", "Filled fields (AI-labelled)", "Correct (AI labels)", "Accuracy vs AI labels",
        "95% CI low", "95% CI high", "Filled fields (human-labelled)", "Correct (human labels)",
        "Accuracy vs human labels", "Human 95% CI low", "Human 95% CI high"]
header(ws, 5, cols, [14, 8, 13, 12, 13, 10, 10, 13, 12, 13, 11, 11, 12, 12, 12, 12, 12, 12])
r0 = 6
for i, c in enumerate(CARDS):
    rs = sel(new, c)
    f = A.filled(rs)
    lo, hi = A.page_boot(rs, A.acc)
    h = H["filled"][c]
    hl, hh = h["value_ci"]
    rr = r0 + i
    put(ws, rr, [c, len({r["page"] for r in rs}), len(f), sum(r["ok"] for r in f), f"=D{rr}/C{rr}", lo, hi,
                 h["n"], h["value"], f"=I{rr}/H{rr}", hl, hh],
        {5: PCT, 6: PCT, 7: PCT, 10: PCT, 11: PCT, 12: PCT})
ra = r0 + 3
lo, hi = A.page_boot(new, A.acc)
h = H["filled"]["ALL"]
put(ws, ra, ["All", f"=SUM(B{r0}:B{ra-1})", f"=SUM(C{r0}:C{ra-1})", f"=SUM(D{r0}:D{ra-1})", f"=D{ra}/C{ra}", lo, hi,
             f"=SUM(H{r0}:H{ra-1})", f"=SUM(I{r0}:I{ra-1})", f"=I{ra}/H{ra}", h["value_ci"][0], h["value_ci"][1]],
    {5: PCT, 6: PCT, 7: PCT, 10: PCT, 11: PCT, 12: PCT}, font=BOLD, fill=ALLFILL)
for col in ("E", "J"):
    ws[f"{col}{ra}"].fill = KEYFILL
ws["F5"].comment = Comment("Bootstrap over whole cards (2,000 resamples), since cells on one card are correlated.", "eval")
ws["K5"].comment = Comment("Wilson interval over fields. Optimistic: cells on one card are correlated.", "eval")
ws["H5"].comment = Comment("Hand labels made by the project team on 9 test cards (3 per type), without seeing the "
                           "model output. Read by the shipped model, which never trained on these 9 cards.", "eval")
note(ws, ra + 1, "AI labels = cells where two independent AI transcriptions (Codex, Claude) agree; cells they disagree on "
     "are not scored. Correct = same value (Bengali/Western digits equal, whitespace ignored). Human labels are the "
     "stricter, more honest number but cover only 9 cards.", 12)

# Section 2: review / safety
s = ra + 3
ws.cell(row=s, column=1, value="2. Review queue and safety (AI labels, all 142 cards)").font = BOLD
cols2 = ["Card type", "Filled fields", "Marked routine", "Routine share", "Routine & correct", "Routine accuracy",
         "Silent errors (wrong + routine)", "Scored cells (filled + blank)", "Silent error rate",
         "Blank cells", "Value invented in blank cell", "Invented rate", "Filled cell read as blank"]
header(ws, s + 1, cols2)
for i, c in enumerate(CARDS):
    rs = sel(new, c)
    f = A.filled(rs)
    rt = [r for r in f if A.routine(r)]
    rr = s + 2 + i
    put(ws, rr, [c, len(f), len(rt), f"=C{rr}/B{rr}", sum(r["ok"] for r in rt), f'=IF(C{rr}=0,"n/a",E{rr}/C{rr})',
                 len(A.silent(rs)), len(f) + len(A.blank(rs)), f"=G{rr}/H{rr}", len(A.blank(rs)),
                 sum(A.has(r) for r in A.blank(rs)), f"=K{rr}/J{rr}", sum(not A.has(r) for r in f)],
        {4: PCT, 6: PCT, 9: "0.00%", 12: "0.00%"})
ra2 = s + 5
put(ws, ra2, ["All"] + [f"=SUM({get_column_letter(k)}{s+2}:{get_column_letter(k)}{ra2-1})" if k in (2, 3, 5, 7, 8, 10, 11, 13)
                        else None for k in range(2, 14)],
    {4: PCT, 6: PCT, 9: "0.00%", 12: "0.00%"}, font=BOLD, fill=ALLFILL)
ws[f"D{ra2}"] = f"=C{ra2}/B{ra2}"
ws[f"F{ra2}"] = f"=E{ra2}/C{ra2}"
ws[f"I{ra2}"] = f"=G{ra2}/H{ra2}"
ws[f"L{ra2}"] = f"=K{ra2}/J{ra2}"
for col in ("F", "I"):
    ws[f"{col}{ra2}"].fill = KEYFILL
note(ws, ra2 + 1, "Routine = the app lets the reviewer bulk-accept it; everything else is flagged 'check'. A silent error is a "
     "wrong value (or a value invented in an empty cell) that is marked routine — the error a busy reviewer would miss. "
     "The routine cutoffs were fit on these same cross-validation results, so these safety numbers are slightly optimistic. "
     "EPI dates never pass the cutoff, so every written EPI field goes to review.", 13)

# Section 3: card level + speed
s3 = ra2 + 3
ws.cell(row=s3, column=1, value="3. Whole cards and speed").font = BOLD
cols3 = ["Card type", "Cards", "Every field right", "Share", "No silent error", "Share",
         "Median sec / card", "Slowest sec / card"]
header(ws, s3 + 1, cols3)
CL = M["card_level"]
sp = H["speed_sec"]
mx = defaultdict(float)
for i, c in enumerate(CARDS):
    rr = s3 + 2 + i
    put(ws, rr, [c, CL[c]["cards"], CL[c]["all_correct"], f"=C{rr}/B{rr}", CL[c]["no_silent_error"], f"=E{rr}/B{rr}",
                 sp["by_type_median"][c], sp["by_type_max"][c]], {4: PCT, 6: PCT, 7: "0.00", 8: "0.00"})
ra3 = s3 + 5
put(ws, ra3, ["All", f"=SUM(B{s3+2}:B{ra3-1})", f"=SUM(C{s3+2}:C{ra3-1})", f"=C{ra3}/B{ra3}",
              f"=SUM(E{s3+2}:E{ra3-1})", f"=E{ra3}/B{ra3}", sp["median"], f"=MAX(H{s3+2}:H{ra3-1})"],
    {2: "#,##0", 3: "#,##0", 4: PCT, 5: "#,##0", 6: PCT, 7: "0.00", 8: "0.00"}, font=BOLD, fill=ALLFILL)
note(ws, ra3 + 1, "Speed: full pipeline on one card, CPU only, offline, Apple M-series laptop, model already loaded. "
     "Target is under 5 s. ANC cards have about 55 filled fields each, so 'every field right' is a hard bar for ANC.", 8)
note(ws, 2, "Run 2026-11-17. The 142 cards in the zip are the same cards as our existing data set (re-exported scans), "
     "so every card here was read by a cross-validation model that never trained on that card.", 13)
ws.freeze_panes = "B6"

# ---------------- Sheet 2: By value type ----------------
w2 = wb.create_sheet("By value type")
header(w2, 1, ["Value type", "Filled fields (AI)", "Correct (AI)", "Accuracy vs AI labels", "Routine share",
               "Silent errors", "Filled fields (human)", "Correct (human)", "Accuracy vs human labels"],
       [26, 13, 12, 13, 12, 11, 13, 12, 13])
names = {"numeric": "Numbers", "date": "Dates", "blood_pressure": "Blood pressure", "mark": "Marks (✓ × −)",
         "text": "Text (eye exam, referral)"}
for i, vt in enumerate(["numeric", "date", "blood_pressure", "mark", "text"]):
    rs = [r for r in new if r["value_type"] == vt]
    f = A.filled(rs)
    hv = H["filled"].get("type:" + vt, {"n": None, "value": None})
    rr = 2 + i
    put(w2, rr, [names[vt], len(f), sum(r["ok"] for r in f), f"=C{rr}/B{rr}",
                 sum(A.routine(r) for r in f) / len(f), len(A.silent(rs)), hv["n"], hv["value"],
                 f'=IF(G{rr}="","n/a",H{rr}/G{rr})' if hv["n"] else "n/a"], {4: PCT, 5: PCT, 9: PCT})
put(w2, 7, ["All", "=SUM(B2:B6)", "=SUM(C2:C6)", "=C7/B7", None, "=SUM(F2:F6)", "=SUM(G2:G6)", "=SUM(H2:H6)", "=H7/G7"],
    {4: PCT, 9: PCT}, font=BOLD, fill=ALLFILL)
w2["E7"].value = None
note(w2, 9, "Dates are the weak spot: 94.6% vs AI labels but only 69% vs human labels. Most date errors are one digit "
     "misread (Bengali 6 read as 9 or 5, 2 read as 1). That is why every written date is sent to review.", 9)

w2.cell(row=11, column=1, value="Error kinds (AI labels, filled fields read wrong)").font = BOLD
header(w2, 12, ["Kind", "Count", "Share"])
ek = M["error_kinds"]
for i, (k, v) in enumerate(ek.items()):
    put(w2, 13 + i, [k, v, f"=B{13+i}/B${13+len(ek)}"], {3: PCT})
put(w2, 13 + len(ek), ["Total", f"=SUM(B13:B{12+len(ek)})", None], font=BOLD, fill=ALLFILL)

# ---------------- Sheet 3: Per field ----------------
w3 = wb.create_sheet("Per field")
header(w3, 1, ["Card", "Field", "Filled fields", "Correct", "Accuracy vs AI labels"], [10, 22, 12, 10, 14])
for i, d in enumerate(M["per_field"]):
    card, field = d["field"].split(":", 1)
    rr = 2 + i
    put(w3, rr, [card, field, d["n"], d["ok"], f"=D{rr}/C{rr}"], {5: PCT})
w3.auto_filter.ref = f"A1:E{1+len(M['per_field'])}"
w3.freeze_panes = "A2"

# ---------------- Sheet 4: Detection + marks confusion ----------------
w4 = wb.create_sheet("Confusion")
w4["A1"] = "Blank vs filled detection (cells both AI transcriptions agree on)"
w4["A1"].font = BOLD
header(w4, 2, ["Truth \\ Model", "Model read a value", "Model read blank", "Total"], [22, 18, 18, 12])
D = M["detection_confusion"]
put(w4, 3, ["Filled cell", D.get("filled->value", 0), D.get("filled->blank", 0), "=B3+C3"], {4: "#,##0"})
put(w4, 4, ["Empty cell", D.get("blank->value", 0), D.get("blank->blank", 0), "=B4+C4"], {4: "#,##0"})
w4["A6"] = "Marks (✓ / × / −) on mark cells"
w4["A6"].font = BOLD
marks = defaultdict(lambda: defaultdict(int))
for r in new:
    if r["value_type"] != "mark" or r["field_key"] in ("eye_exam", "referral"):
        continue
    if r["truth"] is None and not r["both_blank"]:
        continue
    t = A.mark_norm(r["truth"]) if r["truth"] is not None else "blank"
    p = A.mark_norm(r["pred"])
    p = p if p in ("✓", "×", "−", "blank") else "other"
    if t in ("✓", "×", "−", "blank"):
        marks[t][p] += 1
    elif t == "〃":
        marks["〃 (ditto)"][p] += 1
cats = ["✓", "×", "−", "blank", "other"]
header(w4, 7, ["Truth \\ Model"] + cats + ["Total", "Accuracy"])
for i, t in enumerate(["✓", "×", "−", "blank"]):
    rr = 8 + i
    diag = {"✓": "B", "×": "C", "−": "D", "blank": "E"}.get(t)
    put(w4, rr, [t] + [marks[t][c] for c in cats] + [f"=SUM(B{rr}:F{rr})", f"={diag}{rr}/G{rr}" if diag else "n/a"],
        {7: "#,##0", 8: PCT})
note(w4, 13, "Rows = what the card says (both AI transcriptions agree), columns = what the model read. "
     "'other' = a digit or stray text read in a mark cell.", 8)
for col in "BCDEFGH":
    w4.column_dimensions[col].width = 13

# ---------------- Sheet 5: Scan format ----------------
w5 = wb.create_sheet("New vs old scans")
w5["A1"] = "Same 5 cross-validation models, same cards: zip scans (rotated portrait, higher res) vs our original scans"
w5["A1"].font = BOLD
header(w5, 2, ["Card type", "Filled fields", "Correct, original scans", "Accuracy, original scans",
               "Correct, zip scans", "Accuracy, zip scans", "Change (points)"], [32, 12, 14, 14, 14, 14, 12])
for i, c in enumerate(CARDS + ["ALL"]):
    fo, fn = A.filled(sel(old, c)), A.filled(sel(new, c))
    rr = 3 + i
    put(w5, rr, [c if c != "ALL" else "All", len(fn), sum(r["ok"] for r in fo), f"=C{rr}/B{rr}",
                 sum(r["ok"] for r in fn), f"=E{rr}/B{rr}", f"=(F{rr}-D{rr})*100"],
        {4: PCT, 6: PCT, 7: "+0.0;-0.0;0.0"}, font=BOLD if c == "ALL" else BASE, fill=ALLFILL if c == "ALL" else None)
fr = M["format_robustness"]
w5["A9"] = "Same cell, filled fields"
w5["A9"].font = BOLD
header(w5, 10, ["Outcome", "Fields"])
for i, (k, lab) in enumerate([("filled_old_ok_new_ok", "Right on both"), ("fixed", "Wrong before, right on zip scan"),
                              ("broken", "Right before, wrong on zip scan"), ("both_wrong", "Wrong on both")]):
    put(w5, 11 + i, [lab, fr[k]])
note(w5, 16, f"{fr['prediction_changed']} of {fr['cells_compared']} cell readings changed at all. The pipeline handles "
     "the rotation and higher resolution; the difference is within noise.", 7)

# ---------------- Sheet 6: Notes ----------------
w6 = wb.create_sheet("Method and caveats")
w6.column_dimensions["A"].width = 120
lines = [
    ("What was tested", True),
    ("Model: PP-Cards (PP-OCRv6_small_rec fine-tuned on our cards, Apache-2.0) inside the full offline pipeline "
     "(card detection, alignment, cell crops, reading, checks, review flags).", False),
    ("Data: EPI_ANC_NCD_cards.zip — 142 cards (EPI 44, ANC 50, NCD 48).", False),
    ("", False),
    ("Important: this is not new data", True),
    ("The 142 cards in the zip are the same cards the model was developed on, re-exported as rotated portrait scans "
     "(p-000 = our page-001, and so on). The shipped model trained on 100 of them.", False),
    ("To get a fair number anyway, the cards were dealt into 5 groups; each group was read by a model trained only on "
     "the other 4 (5-fold cross-validation). So every card was read by a model that never saw it.", False),
    ("Hand-label scores use the 9 test cards the shipped model never trained on.", False),
    ("", False),
    ("Two kinds of ground truth", True),
    ("AI labels: two independent AI transcriptions; only cells where they agree are scored. Covers all 3,811 filled "
     "fields, but it is easier than reality (cells that are hard to read tend to be the ones the AIs disagree on, "
     "so they drop out).", False),
    ("Human labels: 349 fields on 9 cards labelled by hand by the project team. Fewer fields, but the more honest "
     "number. Labels from the clinical reviewer are still pending.", False),
    ("", False),
    ("What would finalise the model", True),
    ("A new batch of cards the model has never seen (ideally including phone photos, which is how the app will "
     "capture them), with hand labels from a clinical reviewer. Pass bar: 95% of filled fields correct against those "
     "labels, plus a silent error rate under ~1%.", False),
]
for i, (t, b) in enumerate(lines, 1):
    c = w6.cell(row=i, column=1, value=t)
    c.font = BOLD if b else BASE
    c.alignment = Alignment(wrap_text=True, vertical="top")

for sh in wb.worksheets:
    sh.page_setup.orientation = "landscape"
    sh.page_setup.fitToWidth = 1
    sh.page_setup.fitToHeight = 0
    sh.sheet_properties.pageSetUpPr.fitToPage = True
    sh.sheet_view.showGridLines = False
wb.calculation.fullCalcOnLoad = True
out = EVAL / "EPI_ANC_NCD_eval_matrix.xlsx"
wb.save(out)
print(out)

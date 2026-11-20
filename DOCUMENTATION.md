# Bengali Card OCR — Technical Documentation

**Leap Well Project — Medtronic LABS** · October–November 2026

This document covers the purpose, requirements, data, method, experiments and results of the offline OCR for handwritten Bengali health cards. Setup and usage are in [README.md](README.md). Detailed result tables are in [docs/RESULTS.md](docs/RESULTS.md), and the training procedure is in [docs/TRAINING.md](docs/TRAINING.md).

## Contents

1. [Objective and requirements](#1-objective-and-requirements)
2. [Data](#2-data)
3. [Labels and evaluation method](#3-labels-and-evaluation-method)
4. [Pipeline design](#4-pipeline-design)
5. [Recognizer development](#5-recognizer-development)
6. [Error analysis and targeted fixes](#6-error-analysis-and-targeted-fixes)
7. [Experiments that were not kept](#7-experiments-that-were-not-kept)
8. [Making it trustworthy: cross-validation, calibration and the ship gate](#8-making-it-trustworthy-cross-validation-calibration-and-the-ship-gate)
9. [Results](#9-results)
10. [Evaluation on the re-scanned card set](#10-evaluation-on-the-re-scanned-card-set)
11. [Phone-photo check](#11-phone-photo-check)
12. [Deployment path](#12-deployment-path)
13. [Limitations](#13-limitations)
14. [Next steps](#14-next-steps)
15. [Timeline](#15-timeline)

---

## 1. Objective and requirements

In the Leap Well programme, health workers record EPI (vaccination), ANC (antenatal) and NCD (chronic disease) visits on paper cards, then re-enter them into the app. The goal is to **read the handwritten table values from a card photo or scan and pre-fill the form**, with a person verifying every value before it reaches Leap Well.

| Requirement | Target | Status |
|---|---|---|
| Runs offline | Priority | ✅ Local CPU inference; verified with the network blocked |
| Speed | < 5 s per card | ✅ ≈ 0.4 s per card, up to ≈ 1 s for the first card |
| Accuracy | Preferably > 95% | Cross-validation 97.1% exact against AI labels; **81.7% exact / 87.9% same value against human labels**. Routine fields 98.3% accurate on human labels |
| Scope | Table blocks first, over full-page extraction | ✅ Table fields and visit grids |
| Verification | Manual verification before Leap Well | ✅ Nothing is saved without review; unreviewed values are `null` |
| Licensing | Usable commercially | ✅ Apache-2.0 base model; no non-commercial components |

## 2. Data

### 2.1 Card set

The dataset is **142 scanned cards from one clinic**, split once by whole page with a fixed seed:

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| EPI | 24 | 7 | 13 | 44 |
| ANC | 27 | 8 | 15 | 50 |
| NCD | 27 | 7 | 14 | 48 |
| **All** | **78** | **22** | **42** | **142** |

EPI is the smallest set (44 cards), which is a main reason EPI dates remain the weakest field type.

### 2.2 Card content

| Card | Main written content | Characteristic difficulty |
|---|---|---|
| EPI | Vaccine dates per dose | Dates written across **two lines** (day/month above, year below) |
| ANC | Visit grid: weight, BP, Hb, glucose, fetal movement, fundal height, marks (✓ × ±) | Numbers touching grid lines or spilling into the next row |
| NCD | BP, glucose, height, weight, eye exam, referral | The slash inside blood pressure; dashes and repeat marks |

Printed field labels were translated by a native Bengali reviewer. The translations are used in the app and the notebook, for example for glucose units (mmol/L) and temperature (°F).

### 2.3 Data handling

Card scans, labels, alignment templates and weights are patient-derived. They are **never committed**, and `tests/test_no_patient_data.py` enforces this. All processing is local.

## 3. Labels and evaluation method

### 3.1 AI reference labels

No verified transcriptions existed at the start. Two independent vision-language transcriptions of every page were produced, one by OpenAI Codex and one by Anthropic Claude. A cell is labelled only **where both agree**. These labels:
- cover most cells, but skip the hardest ones, where the two disagree;
- are used for training and cross-validation;
- measure **agreement, not accuracy**, and run higher than human-label scores.

### 3.2 Human labels

**Project-team labels:** 349 fields on 9 test cards (3 per type) were labelled by hand without seeing the model output. 289 of them contain writing.

**Clinical reviewer labels:** a blind labelling page was built for the native Bengali clinical reviewer, with Bengali instructions and no model guesses shown. It covers 368 fields from 9 held-out cards and saves progress as it goes. These labels are pending and will be the official accuracy figure.

### 3.3 Metrics

| Metric | Definition |
|---|---|
| **Exact** | Identical text. The only allowance: a line break between the two lines of a date counts as `/` |
| **Same value** | What the form would receive. Bengali and Western digits are equal, and so are mark spellings and the date separators `/ . -` |
| **Routine / check** | Routine fields can be accepted in one click; check fields are reviewed individually, with a reason |
| **Silent error** | A wrong value marked routine. **The primary safety metric** |
| **Values in blank cells** | A value invented in an empty cell |
| Intervals | 95% Wilson intervals (optimistic, since cells on one card are correlated) |

### 3.4 Evaluation discipline

- Design decisions are made on **validation** pages.
- The fixed test split was scored once per final change. It was eventually reused several times, so cross-validation over all 142 cards became the primary held-out measure ([§8](#8-making-it-trustworthy-cross-validation-calibration-and-the-ship-gate)).
- Validation decisions use a **train-only** model. The shipped model is trained on train + validation, so it cannot be scored on validation.

## 4. Pipeline design

```
image/PDF → align to card template → locate fields → read cell → check & flag → human review → JSON export
```

1. **Alignment.**
   - SIFT keypoints with RANSAC homography match the image against one template per card type. The best match also determines the card type.
   - If no template matches reliably, the card is rejected. The pipeline never guesses field positions.
   - EPI phone photos have an additional grid-based path (`pipeline/epi_photo.py`).
2. **Field layout.**
   - Field boxes come from `config/layout_templates.json` and are snapped to the printed grid lines of the templates.
   - On the lower ANC rows (row 19 and below), boxes use the per-column grid lines. This fixes a one-row drift ([§6](#6-error-analysis-and-targeted-fixes)).
   - Empty cells are detected from pixels and never sent to the model.
3. **Recognition.**
   - A fine-tuned PP-OCRv6_small_rec reads each cell (ONNX, 12 MB, 211-character card set).
   - Decoding is restricted to the characters allowed for the field type, and each value to one numeral script. The script choice uses the read's fit to the full model output, not mean confidence, which had favoured reads that dropped digits.
   - Two-line EPI dates and blood pressure are read as separate parts.
   - A small CNN reads ✓, ×, (−) and − marks. A "minus gate" stops minus signs being read as numbers.
4. **Checks.**
   - Dates and blood pressure must parse.
   - An impossible two-line EPI date is replaced by the most likely valid date, and the original reading is kept.
   - NCD eye-exam and referral fields snap to the usual values from the training pages, and are always flagged.
   - A repeated-digit flag catches stuck reads.
5. **Review policy.** A calibrated probability and the per-field ship gate decide routine or check, together with the low-ink rule ([§8](#8-making-it-trustworthy-cross-validation-calibration-and-the-ship-gate)).
6. **Review and export.**
   - The reviewer accepts or corrects each field, or accepts all routine fields in one logged click.
   - The JSON export keeps the model reading, the reviewer decision and the source image hash separately, grouped by Leap Well form sections.

## 5. Recognizer development

### 5.1 Baselines

All measured on the same 42 test cards (AI labels):

| Model | Agreement | Notes |
|---|---|---|
| Leap Well CRNN (existing team model) | 33% | Built for a different crop style; Western digits only |
| Orus-OCR (original, Bengali handwriting) | 64% | Strong base, but licensed "MIT (Non-Commercial)" |

### 5.2 Orus-Cards (Orus fine-tuned on card cells)

Orus was pretrained on synthetic handwritten digits from NumtaDB, then fine-tuned on real card crops:
- **Test:** 90.9% (ANC 94.0%, NCD 84.4%, EPI 80.7%) once the EPI date rule was added, which took EPI from 75% to 81%.
- **Rejected follow-ups:** a date-heavy variant and a hard-example retrain gave no gain.

### 5.3 PP-Cards (PP-OCRv6 fine-tuned): the shipped model

**Why the switch:** Orus is non-commercial, so it could not go into business use without legal approval. PP-OCRv6_small_rec is **Apache-2.0**.

**Recipe** ([docs/TRAINING.md](docs/TRAINING.md)):
- The 18,710-class output head was replaced by the 211-character card set.
- **Stage 1:** pretraining on 12,000 synthetic NumtaDB cells.
- **Stage 2:** fine-tuning on 1,640 real card crops.
- **Final model:** retrained on train + validation crops (2,063) on a fixed 12-epoch schedule.
- Training ran in PyTorch through `transformers`, with no Paddle installation, then the model was exported to ONNX.

**Results:**
- **EPI dates first:** on validation, PP-OCRv6 read EPI dates much better than Orus-Cards (89.8–93.2% vs 79.7%). Shipping PP for EPI dates alone took test EPI from 81.6% to 91.2%.
- **Then every cell:** once the script-choice fix and the NCD rules were in, PP-only matched the Orus/PP mix on validation (541 vs 540 of 559 fields). It was then made the only engine, so no Orus code is loaded, which was verified in a clean process.
- **Test after the switch, scored once:** **97.3%** same value (ANC 98.5%, NCD 95.0%, EPI 93.0%), up from 95.5%.

## 6. Error analysis and targeted fixes

A structured error analysis covered all 142 cards, with the train, validation and test splits kept separate. Each card type showed a characteristic failure:

| Card | Dominant error | Fix |
|---|---|---|
| EPI | Two-line dates: single-digit confusions (৬/৯/৫, ২/১), dropped digits | Date grammar and repair of impossible dates; PP-OCRv6 for date parts; all EPI dates flagged check |
| ANC | Values in empty last-column cells; boxes on rows 19–21 sitting about one row high | **Box snapping** to per-column grid lines from row 19 down: fake values in blank cells fell **20 → 12** with no real value lost |
| NCD | Slash inside BP; glucose dashes; eye-exam and referral text | Dash check for glucose; eye-exam `হ্যাঁ` completion; referral clinic-name snapping. NCD test 88.4% → 95.5% |
| All | Minus signs read as numbers | Minus gate in the mark reader: validation 93.2% → 94.5% (14 fixes) |

A box and crop audit of misread fields found three causes: crops cutting off digits, crops catching writing from the row above, and a few clear crops that were still misread. The first two drove the layout fixes.

## 7. Experiments that were not kept

Negative results are recorded so they are not repeated.

| Experiment | Result |
|---|---|
| Image cleanup filters (edge marks, long lines, printed text) | Removed real handwriting too; rejected |
| Gray-cell detection for invented values | Explained only 2 of 20 fake values; small win, not pursued |
| `Sarjinkhan2003/bengali-ocr-recognition` (MIT) | 33.8% overall, 1.3% on numbers vs 93.0% for Orus-Cards; trained on printed words |
| `medocr-pharmacy/bengali-ocr-bn` (int8 ONNX) | No published alphabet; numbers 17.8%, dates 0%, BP 0%; confuses scripts; 6× slower |
| GraDeT-HTR | Weights not public; academic use only |
| Vision-language models (2B+ parameters) | Too slow for a phone, and wrong on single cells |
| Four EPI ideas: multi-view reads, column voting, printed-line erase, split day/month decoding | Each either did not help, or helped on validation but hurt on test |
| Hard-example retrain (Orus) | No gain |
| Copying a date from neighbouring vaccines | Fixes some reads but breaks correct ones; kept only as a flagged *same-day suggestion*, never applied automatically |
| **Digit expert:** a small CNN that re-reads confusable single digits | Holdout digit accuracy 82%; field accuracy unchanged (97.0 → 97.0% holdout, 94.0 → 94.0% validation). Shipped **disabled** (`pipeline/digits.py`, no-op when disabled) |

## 8. Making it trustworthy: cross-validation, calibration and the ship gate

**The problem.** A single score cutoff per value type let **19 of 349 human-labelled fields (5.4%)** through as routine while wrong. Most of them were EPI dates. For a review tool, these silent errors matter more than overall accuracy.

### 8.1 Cross-validation

All 142 cards are dealt into 5 folds by page, balanced by card type. For each fold, a recognizer is trained on the other four with the shipped recipe. The last epoch is kept, so nothing is selected on the held-out fold. The full pipeline is then scored on the held-out cards, so every card is read by a model that never saw it.

### 8.2 Calibration

For numbers, dates and BP, Platt scaling maps the model's raw score to the probability that the reading is right. The held-out expected calibration error fell from **1.1 to 0.3 points**.

### 8.3 Per-field ship gate

For each of the 45 card fields, the gate finds the lowest probability cutoff at which the **lower bound of the 95% Wilson interval** of held-out accuracy reaches **95%**, over at least 20 cells.
- Readings above the cutoff are routine.
- A field with no such cutoff is always check.
- Cutoffs are chosen on four folds and applied to the fifth, so the reported results are held out too.

**Fields that pass:**
- 10 value fields: ANC fetal movement, fundal height, gestation month, glucose, hemoglobin, pulse and weight; NCD random glucose, height and weight.
- 7 ANC mark fields.
- The blank calls of all three cards.

**No written EPI date passes yet.**

### 8.4 Low-ink rule

A value read from fewer than 30 ink pixels, with no stroke of 30 pixels or more, is always check. On the cross-validation it flags 39 of 91 values read in cells both transcriptions call blank, and only 3 of 3,811 real values.

### 8.5 Effect

Readings are unchanged on all 6,245 validation and test fields; only the flags differ.

| | One score cutoff | Per-field gate |
|---|---|---|
| Silent errors, human labels | 19 / 349 (5.4%) | **4 (1.1%)** |
| Accuracy of routine fields, human labels | 94.1% | **98.3%** |
| Silent errors, cross-validation, filled fields | 1.02% | **0.31%** |
| Accuracy of routine cells, cross-validation | 99.5% | **99.8%** |
| Filled fields marked routine, cross-validation | 91.2% | 66.1% |

The cost is review load: fewer fields are routine. Routine share by card, counting blanks: ANC 94%, NCD 79%, EPI 53% (EPI's routine cells are its blank calls).

## 9. Results

### 9.1 Cross-validation (AI labels, 142 cards, 3,811 filled fields)

| | Exact | Same value |
|---|---|---|
| **All** | **97.1%** (96.6–97.6) | **97.9%** (97.4–98.3) |
| ANC (2,754) | 97.8% | 98.6% |
| NCD (693) | 96.0% | 97.0% |
| EPI (364) | 94.2% | 94.2% |

### 9.2 Human labels (9 test cards, 289 filled fields)

| | Exact | Same value |
|---|---|---|
| **All** | **81.7%** (77–86) | **87.9%** (84–91) |
| ANC | 85.5% | 94.2% |
| NCD | 79.5% | 79.5% |
| EPI | 63.2% | 63.2% |
| Numbers | 95.1% | 95.1% |
| Marks | 80.7% | 92.0% |
| Blood pressure | 85.0% | 85.0% |
| Dates | 53.4% | 67.2% |

- **Blank calls:** all 54 cells the model called blank were blank.
- **Date errors:** mostly single-digit confusions.
- **Silent errors:** the 4 that remain are three numbers (২৪ → 28, ৫৮ → ৫০, ৫.৬ → ৫.৯) and one mark.

### 9.3 Gap between AI-label and human-label scores

The AI-agreement labels exclude cells where the two transcriptions disagree, and those are mostly the hard cells. The human-label set is small (9 cards) but unfiltered. The honest planning figure is therefore **≈ 88% same value on scans**, with **≈ 98% accuracy on fields marked routine**. The clinical reviewer's labels will settle the official number.

### 9.4 Speed

About 0.4 s per card on an Apple M5 laptop CPU once loaded, and up to about 1 s for the first card (median 0.44 s, p95 0.61 s across 142 cards). Phone speed has not been measured.

## 10. Evaluation on the re-scanned card set

A second export of the card scans (`EPI_ANC_NCD_cards.zip`) was evaluated for robustness. The full write-up is in [evaluation/eval_2026_11_17/](evaluation/eval_2026_11_17/README.md).

**What the set is:** the same 142 cards, re-exported as rotated, higher-resolution portrait A4. So it is not a new test set. Every card was read by a cross-validation fold model that never saw it.

**AI labels:** 97.6% same value (ANC 98.0%, NCD 96.8%, EPI 95.3%), against 97.9% on the original scans. 437 of 13,462 predictions changed: 16 fixed, 28 broken.

**Human labels:** 88.2% same value, against 87.9% on the original scans. 4 silent errors.

**Conclusion:** the scan export format has little effect.

## 11. Phone-photo check

Three phone photos were tested: one EPI card shot twice, and one ANC card. All three aligned, in 0.4–0.8 s each. The reference values were AI-read, not human-verified.

**EPI (13 written dates):**
- Photo A read 6/13, photo B 8/13, and the two photos disagree on 5 cells.
- Failure modes:
  - visit-1 dates misread;
  - tick and signature strokes read as extra year digits;
  - dotted placeholder lines read as digits;
  - a newer card version with a TCV row the template does not know.
- **Every EPI field was flagged check, so there were no silent errors.**

**ANC (35 written cells):**
- About 15/35 (43%) correct. Visit dates, glucose, Hb, temperature, height and (−) marks were read well.
- Marks were often wrong: ✓ read as ×, and (+) read as (−).
- 26 empty cells were read as "−" because of grid lines inside the crops of a curved photo.
- **ANC photos currently get no photo-specific handling.** About 33 wrong values were marked routine on one card, against about 4 in 349 on scans.

**Conclusion:** scans read at ≈ 88%, phone photos at ≈ 45–60%. The ship gate was fitted on scans, so **photo inputs must be reviewed field by field** until photo handling is added (see [§14](#14-next-steps)).

## 12. Deployment path

**Current app path:**
- Cards are scanned in the Leap Well Flutter app, which has scan screens for EPI, ANC and NCD.
- **Online,** the app reads cards with Gemini through `/ai-scribe/card-extraction/extract`.
- **Offline,** it uses ML Kit or Tesseract, which cannot read handwriting.

**Target:** this model should replace the **offline path**, returning the same JSON (field code, value, confidence). The plan is to package the recognizer, mark classifier, layouts and review policy as a **card pack** and export to TFLite or ONNX Runtime Mobile. Model size is not a constraint: the recognizer is about 12 MB.

## 13. Limitations

- **One clinic, 142 flat scans.** Writer-disjoint testing has not been done, and there is no second clinic.
- **EPI dates** are the weakest field type (94% held out against AI labels, 63% exact against human labels). Every written EPI date is flagged check.
- **Labels:** training and cross-validation labels come from AI agreement. Human labels cover 9 cards, and the clinical reviewer's labels are pending.
- **Reused test split.** The fixed test split has been scored several times. A new batch of cards is needed for a clean final number.
- **Phone photos** are not yet handled safely for ANC ([§11](#11-phone-photo-check)).
- **Card versions:** newer EPI cards with a TCV row are not in the templates.
- **Field-label conflicts:** two newborn labels and the last three ANC columns have translation conflicts that need confirmation.

## 14. Next steps

1. **Clinical reviewer labels** on the 368-field blind labelling page, giving the official human-verified accuracy. Corrections on new cards then feed retraining (`training/human_labels_to_crops.py`).
2. **Photo safety:** force check for any photo input, on all card types, and fix ANC photo row alignment.
3. **More phone photos**, labelled by the reviewer, to measure and fit a photo-specific gate.
4. **EPI:** more labelled cards, then retrain on all 44 EPI cards once a fresh test set exists.
5. **A new scan batch from another clinic**, for writer-disjoint testing.
6. **Templates** for the newer EPI card version (TCV row).
7. **Mobile export:** a card pack plus TFLite for the Leap Well app's offline path.

## 15. Timeline

| Date (2026) | Milestone |
|---|---|
| Oct | Baselines (Leap Well CRNN 33%, Orus 64%); pipeline built (alignment, layout, recognition, review app) |
| 29 Oct | Orus-Cards (synthetic NumtaDB pretraining + card fine-tune) scored on 42 test cards |
| 30 Oct | EPI date rule (test 90.9%); blind labelling page for the clinical reviewer |
| 2 Nov | Walkthrough notebook aligned with the app and reported numbers |
| 5 Nov | First phone photos; box and crop audit; 142-card error analysis; label translations integrated; ANC row-drift diagnosis |
| 8 Nov | External model search: three candidates rejected; NCD dash check |
| 9 Nov | PP-OCRv6 fine-tune; PP-Cards becomes the only engine (Apache-2.0); NCD rules; minus gate; ANC box snap made permanent; test 97.3% |
| 11 Nov | Clean repository: PP-Cards only, outputs verified identical on 6,245 fields, clean-install tested |
| 12 Nov | 5-fold cross-validation, calibration, per-field ship gate, low-ink rule |
| 13 Nov | Phone-photo check |
| 17 Nov | Evaluation on the re-scanned card set; digit-expert experiment (no gain, shipped disabled) |

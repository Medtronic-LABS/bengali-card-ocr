# Results

All numbers are for the shipped model (`models/ppocr-cards/ppocrv6_cards.onnx`) and the pipeline in this repository.

## Data and split

142 scanned cards, split once by whole page with a fixed seed:

| | Train | Validation | Test | Total |
|---|---|---|---|---|
| EPI | 24 | 7 | 13 | 44 |
| ANC | 27 | 8 | 15 | 50 |
| NCD | 27 | 7 | 14 | 48 |
| **All** | **78** | **22** | **42** | **142** |

Training labels come from two independent AI transcriptions (Codex and Claude). A cell is labeled only where both agree. Design choices were picked on validation pages. The shipped model was then retrained on train + validation with the same fixed schedule.

## Cross-validation: every card, held out

All 142 cards are dealt into 5 folds, balanced by card type (`training/cross_validate.py`). For each fold a recognizer is trained on the other four with the shipped recipe (the last epoch is kept, so nothing is chosen on the held-out fold) and the full pipeline is scored on the held-out cards. Every card is therefore read by a model that never saw it. Labels are the AI-agreed values, so these numbers run higher than the human-label ones below.

| Filled fields | Exact | Same value |
|---|---|---|
| **All** (3,811) | **97.1%** (96.6–97.6%) | **97.9%** (97.4–98.3%) |
| ANC (2,754) | 97.8% (97.2–98.3%) | 98.6% (98.1–99.0%) |
| NCD (693) | 96.0% (94.2–97.2%) | 97.0% (95.4–98.0%) |
| EPI (364) | 94.2% (91.3–96.2%) | 94.2% (91.3–96.2%) |

The shared mark classifier was trained on the train pages, so mark results on those pages are not fully held out. The shipped model is not one of the five; it was trained on train + validation pages.

## Ship gate, calibration and silent errors

`evaluation/ship_gate.py` turns the cross-validation into the review policy in `config/review_policy.json`:

1. **Calibration.** For numbers, dates and BP, a logistic fit (Platt scaling) maps the model's raw score to the probability that the reading is right. Expected calibration error, held out: **1.1 → 0.3 points**.
2. **Ship gate per field.** For each card field (45 in all), the lowest probability cutoff at which the lower end of the 95% Wilson interval of held-out accuracy reaches **95%**, over at least 20 cells. Readings above the cutoff are routine; a field with no such cutoff is always check. Marks and blank calls pass or fail on the same test. 10 value fields (ANC fetal movement, fundal height, gestation month, glucose, hemoglobin, pulse, weight; NCD random glucose, height, weight), 7 ANC mark fields and the blank calls of all three cards pass.
3. **Very little ink.** A value read from fewer than 30 ink pixels with no stroke of 30 pixels or more is always check. On the cross-validation it flags 39 of the 91 values read in cells both transcriptions call blank, and 3 of 3,811 real values.

The cutoffs reported below were chosen on four folds and applied to the fifth, so they are held out too. A **silent error** is a wrong value that is marked routine.

| Cross-validation, 11,417 scored cells (3,811 filled) | Before (one score cutoff per value type) | Now |
|---|---|---|
| Cells marked routine | 95.7% | 87.2% |
| Accuracy of routine cells | 99.5% | **99.8%** (99.7–99.9%) |
| Silent errors, share of all cells | 0.50% (57) | **0.17% (19)** |
| Filled fields marked routine | 91.2% | 66.1% |
| Silent errors, share of filled fields | 1.02% | **0.31%** |
| Values in blank cells marked routine | 18 | 7 |

| Routine share by card (all cells) | ANC | NCD | EPI |
|---|---|---|---|
| Now | 94% | 79% | 53% |

EPI routine cells are its blank calls; no written EPI date passes the gate yet.

## Against human labels

On 9 test cards (3 per type), 349 fields were labeled by hand without seeing the model's output. 289 of them have writing. Two scores, with 95% Wilson intervals (optimistic, since cells on one card are correlated):

- **Exact:** the text is identical. The only allowance is that a line break between the two written lines of a date counts as `/`, because the app keeps a two-line date as two lines.
- **Same value:** what the form would receive. Bengali and Western digits are equal, and so are mark spellings (✓ / y) and the date separators `/ . -`.

| Filled fields | Exact | Same value |
|---|---|---|
| **All** | **81.7%** (236/289; 77–86%) | **87.9%** (254/289; 84–91%) |
| ANC | 85.5% (177/207; 80–90%) | 94.2% (195/207; 90–97%) |
| NCD | 79.5% (35/44; 66–89%) | 79.5% (35/44; 66–89%) |
| EPI | 63.2% (24/38; 47–77%) | 63.2% (24/38; 47–77%) |
| Numbers | 95.1% (117/123; 90–98%) | 95.1% (117/123; 90–98%) |
| Marks (✓ × −) | 80.7% (71/88; 71–88%) | 92.0% (81/88; 84–96%) |
| Blood pressure | 85.0% (17/20; 64–95%) | 85.0% (17/20; 64–95%) |
| Dates | 53.4% (31/58; 41–66%) | 67.2% (39/58; 54–78%) |

| Review flags (all 349 labeled fields) | Before (one score cutoff) | Now (per-field gate) |
|---|---|---|
| Marked routine | 323 (93%) | 238 (68%) |
| Accuracy of routine fields | 94.1% | **98.3%** |
| Silent errors (wrong and routine) | 19 (5.4%) | **4 (1.1%)** |

All 54 cells the model called blank were blank. The date errors are mostly single-digit confusions, such as ৬ read as ৯ or ৫, or ২ read as ১. Before the gate, 12 wrong dates were marked routine; now every written EPI date is flagged check. The 4 remaining silent errors are three numbers (২৪ read as 28, ৫৮ as ৫০, ৫.৬ as ৫.৯) and one mark. Writing a date across two lines instead of with `/` is not counted as an error.

These labels were made by the project team. Labels from the clinical reviewer are still pending, and they will be the official number. Score a label file with:

```bash
.venv/bin/python evaluation/verified_accuracy.py --gold <labels.json>
```

## Against AI labels, fixed test split

The original fixed split, scored with the shipped model. Superseded by the cross-validation above, which covers every card.

| | Overall | ANC | NCD | EPI |
|---|---|---|---|---|
| Test, exact (42 cards, 1,126 filled fields) | 96.3% | 97.4% | 93.5% | 93.0% |
| Test, same value | 97.3% | 98.5% | 95.0% | 93.0% |
| Validation, same value (22 cards, 559 filled fields, train-only model) | 96.8% | | | |

The test cards have been scored after several development changes, so they are no longer a fully untouched held-out set. A new batch of cards is needed for a clean final number.

## Speed

For the full pipeline on an Apple M5 laptop CPU: about 0.4 s per card once the model is loaded, and up to about 1 s for the first card. The time on a phone has not been measured.

## Models tried and not kept

| Model | Why not |
|---|---|
| Orus-OCR (original and fine-tuned on the cards) | Licensed for non-commercial use only. Accuracy was similar to the current model. |
| Leapwell CRNN | Built for a different crop style; 33% when run on this pipeline's crops, and it outputs only Western digits. |
| Sarjinkhan2003/bengali-ocr-recognition | Trained on printed words: 1% on validation numbers. |
| medocr-pharmacy/bengali-ocr-bn | No published character list; 18% on validation numbers, 0% on dates. |
| Vision-language models (2B+ parameters) | Too slow for the phone, and wrong on single cells. |

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

Training labels come from two independent AI transcriptions (Codex and Claude). A cell is labeled only where both agree. Design choices and review thresholds were picked on validation pages. The shipped model was then retrained on train + validation with the same fixed schedule.

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

| Review flag (all labeled fields) | Correct |
|---|---|
| Routine | 94.1% (304/323) |
| Check | 15.4% (4/26) |

All 54 cells the model called blank were blank.

The date errors are mostly single-digit confusions, such as ৬ read as ৯ or ৫, or ২ read as ১. **13 of the 19 wrong dates were marked routine**, so for dates the routine flag is much less reliable than the overall numbers suggest. Writing a date across two lines instead of with `/` is not counted as an error.

These labels were made by the project team. Labels from the clinical reviewer are still pending, and they will be the official number. Score a label file with:

```bash
.venv/bin/python evaluation/verified_accuracy.py --gold <labels.json>
```

## Against AI labels

A larger but weaker check: agreement with fields where both AI transcriptions agree. Fields the AIs disagreed on are left out, and those tend to be the hard ones, so these numbers run higher than the human-label ones.

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

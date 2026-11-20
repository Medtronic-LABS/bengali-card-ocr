# Eval matrix: EPI_ANC_NCD_cards.zip (2026-11-17)

Requested by Aditya: "test your model and send an eval matrix" on `EPI_ANC_NCD_cards.zip`.

## What the zip turned out to be

The zip holds our existing 142 cards, re-exported as portrait A4 scans (~1655×2338, rotated). `p-N.jpg` is `page-(N+1)`. There are no labels in it. So it is **not a new test set**: the shipped model trained on 100 of these cards.

To keep the run fair, every card was read by one of the 5 cross-validation fold models from Nov 12, so no card was read by a model that had seen it.

## Results

**Against AI reference labels** (cells where two separate AI transcriptions agree; filled fields, same value):

| | Cards | Filled fields | Same value | 95% CI |
|---|---|---|---|---|
| All | 142 | 3,811 | **97.6%** | 96.9–98.1% |
| ANC | 50 | 2,754 | 98.0% | 97.4–98.6% |
| NCD | 48 | 693 | 96.8% | 94.9–98.4% |
| EPI | 44 | 364 | 95.3% | 92.4–97.7% |

On the original scans the same models scored 97.9%. On the new scans, 437 of 13,462 predictions changed: 16 fixed and 28 broken. The new scan format barely matters.

**Against human hand labels** (349 fields on 9 test cards, shipped model): **88.2%** same value (ANC 94.2%, NCD 79.5%, EPI 65.8%; dates 40/58). On the old scans this was 87.9%. 4 silent errors.

**Other numbers:**
- Fields marked "routine" are right 99.1% of the time: 30 silent errors in 11,417 cells. The cutoffs were fit on these same CV rows, so this is slightly optimistic.
- No EPI field is ever routine.
- 92 values were read in 7,606 blank cells.
- 60 of 142 cards are fully correct (ANC 7/50).
- Speed: median 0.44 s per card, p95 0.61 s, max 0.88 s.

## Files

| File | What |
|---|---|
| `analyze.py` | Builds the eval matrix from the CV rows (new scans vs old scans, plus the shipped model by split) |
| `human_labels.py` | Scores the shipped model on the new scans against the hand labels, and times every card |
| `build_xlsx.py` | Writes the Excel eval matrix that was sent to Aditya |
| `eval_matrix_summary.json` | Output of `analyze.py`, aggregates only |
| `human_labels_summary.json` | Output of `human_labels.py`, aggregates only |

Per-field values, card images and labels are left out on purpose (patient data; see the repo `.gitignore`). The summaries drop the per-field item list and the example cells.

```bash
python evaluation/eval_2026_11_17/analyze.py <eval_dir> .
python evaluation/eval_2026_11_17/build_xlsx.py <eval_dir> .
CARD_DATA_DIR=<eval_dir>/data .venv/bin/python evaluation/eval_2026_11_17/human_labels.py <gold.json> <out.json>
```

# Evaluation: re-scanned card set (2026-11-17)

An evaluation of the pipeline on `EPI_ANC_NCD_cards.zip`, a second export of the project's card scans, producing the full evaluation matrix.

## What the set contains

The zip holds the same 142 cards as the main dataset, re-exported as portrait A4 scans (~1655×2338, rotated). `p-N.jpg` corresponds to `page-(N+1)`. Identity was confirmed by thumbnail matching and visual spot checks of the handwriting. The zip contains no labels.

**It is therefore not a new test set:** the shipped model was trained on 100 of these cards. To keep the evaluation fair, every card was read by one of the five cross-validation fold models, so no card was read by a model that had seen it. The run measures **robustness to a different scan export** of the same cards.

## Results

**Against AI reference labels** (filled fields, same value; cells where two independent AI transcriptions agree):

| | Cards | Filled fields | Same value | 95% CI |
|---|---|---|---|---|
| All | 142 | 3,811 | **97.6%** | 96.9–98.1% |
| ANC | 50 | 2,754 | 98.0% | 97.4–98.6% |
| NCD | 48 | 693 | 96.8% | 94.9–98.4% |
| EPI | 44 | 364 | 95.3% | 92.4–97.7% |

**Original vs re-exported scans:**
- The same models scored 97.9% on the original scans and 97.6% on the re-exported ones.
- 437 of 13,462 predictions changed: 16 fixed, 28 broken.
- The scan format has little effect.

**Against human hand labels** (349 fields on 9 test cards, shipped model):
- **88.2%** same value overall: ANC 94.2%, NCD 79.5%, EPI 65.8%. Dates were 40/58.
- On the original scans the same score was 87.9%.
- 4 silent errors.

**Review policy and errors:**
- Fields marked "routine" were right 99.1% of the time: 30 silent errors in 11,417 cells. The gate cutoffs were fit on these same CV rows, so this is slightly optimistic.
- No EPI field is marked routine.
- 92 values were read in 7,606 blank cells.

**Card level and speed:**
- 60 of 142 cards are fully correct (ANC 7/50).
- Median 0.44 s per card, p95 0.61 s, max 0.88 s.

## Files

| File | Purpose |
|---|---|
| `analyze.py` | Builds the evaluation matrix from the cross-validation rows: re-exported vs original scans, plus the shipped model by split |
| `human_labels.py` | Scores the shipped model on the re-exported scans against the hand labels, and times every card |
| `build_xlsx.py` | Writes the evaluation matrix as an Excel workbook |
| `eval_matrix_summary.json` | Output of `analyze.py` (aggregates only) |
| `human_labels_summary.json` | Output of `human_labels.py` (aggregates only) |

Per-field values, card images and labels are excluded by design. They are patient data; see the repository `.gitignore`. The summaries omit the per-field item list and the example cells.

```bash
python evaluation/eval_2026_11_17/analyze.py <eval_dir> .
python evaluation/eval_2026_11_17/build_xlsx.py <eval_dir> .
CARD_DATA_DIR=<eval_dir>/data .venv/bin/python evaluation/eval_2026_11_17/human_labels.py <gold.json> <out.json>
```

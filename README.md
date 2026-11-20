# Bengali Card OCR (Offline)

**Leap Well Project — Medtronic LABS**

Reads handwritten values from paper health cards and turns them into structured fields that a health worker reviews before anything is saved. It supports three card types: **EPI** (vaccination), **ANC** (antenatal) and **NCD** (chronic disease). Everything runs locally on the CPU, with no network calls and no cloud APIs.

> **Status:** pilot-ready as a review tool. A person confirms every card. Fields that have proven accurate on cards the model never saw are pre-marked **routine** and can be accepted in one click. Everything else is flagged **check**, with a reason. The full method, experiments and results are in [DOCUMENTATION.md](DOCUMENTATION.md).

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Results at a Glance](#results-at-a-glance)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running](#running)
- [Export Format](#export-format)
- [Tests](#tests)
- [Project Layout](#project-layout)
- [Data](#data)
- [Models and Licenses](#models-and-licenses)
- [Troubleshooting](#troubleshooting)
- [Security and Safety](#security-and-safety)
- [Roadmap](#roadmap)
- [License](#license)

## Features

- **Fully offline.** The recognizer is a 12 MB ONNX model on the CPU. Inference was verified with the network blocked.
- **Fast.** About 0.4 s per card on a laptop CPU, up to about 1 s for the first card.
- **Automatic card-type detection and alignment.** SIFT + RANSAC match each card to its template. A card that cannot be matched safely is rejected; the app never guesses field positions.
- **Field-aware reading.**
  - Decoding is restricted per field type: a number field can only produce digits.
  - Each value uses one numeral script, Bengali or Western.
  - Two-line EPI dates and blood pressure halves are read separately.
- **Mark classifier.** A small CNN reads ✓, ×, (−) and − marks. A possible "+" is never filled in automatically.
- **Calibrated confidence.** The model's score is mapped to the probability that the reading is right (Platt scaling; calibration error 1.1 → 0.3 points).
- **Per-field ship gate.** A field may be marked routine only if, on held-out cards, the lower end of the 95% interval of its accuracy reaches 95%.
- **Every flag explains itself.** The reasons are: not yet proven for this field, too little ink, an unparseable date or BP, an alternative reading, or a same-day suggestion from other vaccines.
- **Review app.** A Streamlit app to accept or correct each field, or to accept all routine fields in one logged click.
- **Leap Well-ready export.** JSON schema 2.0 keeps the model reading, the reviewer decision and the source image hash, grouped by Leap Well form sections.
- **Patient data never committed.** A test fails if any image, label, model file or notebook output would be committed.

## Architecture

```
┌─────────────┐   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ Image / PDF │──►│ Align to template    │──►│ Locate fields        │──►│ Read each cell       │
│ (scan or    │   │ SIFT + RANSAC,       │   │ layout config,       │   │ PP-OCRv6 (ONNX,      │
│  photo)     │   │ detects card type,   │   │ blank detection from │   │ 12 MB), per-field    │
└─────────────┘   │ rejects unknown cards│   │ pixels (blanks never │   │ charset; mark CNN    │
                  └──────────────────────┘   │ sent to the model)   │   │ for ✓ × (−) −        │
                                             └──────────────────────┘   └──────────┬───────────┘
                                                                                   ▼
┌──────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────────┐
│ JSON export (2.0)    │◄──│ Human review (Streamlit)     │◄──│ Check and flag                   │
│ model reading +      │   │ accept / correct each field, │   │ date & BP syntax, EPI date       │
│ reviewer decision +  │   │ or accept all routine (one   │   │ repair, calibrated probability,  │
│ image hash; Leap Well│   │ logged click); unreviewed    │   │ per-field ship gate, low-ink     │
│ form grouping        │   │ values stay null             │   │ rule → routine / check + reason  │
└──────────────────────┘   └──────────────────────────────┘   └──────────────────────────────────┘
```

| Stage | Code | Configuration |
|---|---|---|
| Alignment and card detection | `pipeline/alignment.py`, `pipeline/epi_photo.py` | `assets/registration/*.png` (templates, not committed) |
| Field layout and blank detection | `pipeline/layout.py` | `config/layout_templates.json` |
| Recognition | `pipeline/recognize.py`, `pipeline/marks.py` | `models/ppocr-cards/`, `models/marks/` |
| Checks and review policy | `pipeline/validate.py`, `date_grammar.py`, `review.py` | `config/review_policy.json`, `config/low_ink_rule.json` |
| Export and Leap Well mapping | `pipeline/export.py` | `config/form_schemas.json`, `config/leapwell/` |

## Results at a Glance

All numbers are held out: measured on cards the model did not train on. Full tables with 95% intervals: [docs/RESULTS.md](docs/RESULTS.md).

**Safety: wrong values that would be auto-accepted.** This is the number that matters for a review tool.

| | Before (one score cutoff) | Now (per-field gate) |
|---|---|---|
| Silent errors, human labels (9 cards) | 19 of 349 cells (5.4%) | **4 (1.1%)** |
| Accuracy of routine fields, human labels | 94.1% | **98.3%** |
| Silent errors, cross-validation (142 cards, filled fields) | 1.0% | **0.3%** |
| Filled fields auto-accepted, cross-validation | 91% | 66% |

**Accuracy per card**, filled fields, exact match:

| | Overall | ANC | NCD | EPI |
|---|---|---|---|---|
| Cross-validation, AI labels (142 cards, 3,811 fields) | **97.1%** | 97.8% | 96.0% | 94.2% |
| Human labels (9 cards, 289 fields) | **81.7%** | 85.5% | 79.5% | 63.2% |
| Human labels, *same value* | **87.9%** | 94.2% | 79.5% | 63.2% |

- **Cross-validation:**
  - 5 models, each trained without one fifth of the cards and scored on that fifth.
  - The labels are fields where two independent AI transcriptions agree. They skip the hardest fields, so these numbers run higher than the human-label ones.
- **Human labels:** the stricter test, but they cover only 9 cards.
- **What ships:**
  - 10 value fields (7 ANC, 3 NCD), 7 ANC mark fields, and blank calls on all three cards pass the gate.
  - EPI dates do not yet, so every written EPI date is flagged check.
- **Re-scanned card set** ([evaluation/eval_2026_11_17/](evaluation/eval_2026_11_17/)): 97.6% against AI labels and 88.2% against human labels. A different scan export has little effect.

## Prerequisites

- Python 3.12 or newer (tested on 3.12 and 3.14, macOS on Apple silicon).
- The model files and alignment templates, obtained from the project team ([Setup](#setup)).
- For training only: the private card dataset and `requirements-train.txt`.

## Setup

```bash
git clone https://github.com/Medtronic-LABS/bengali-card-ocr.git
cd bengali-card-ocr

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Model files and alignment templates (needed once).** Both are derived from real patient cards, so they are **not in this repository**. Get them from the project team and unpack them into the repository folder:

```
models/ppocr-cards/ppocrv6_cards.onnx         recognizer (12 MB)
models/ppocr-cards/ppocrv6_cards.chars.json   its character list
models/marks/mark_cnn.onnx, mark_cnn.json     mark classifier
assets/registration/EPI.png, ANC.png, NCD.png one alignment template per card type
```

The templates can also be rebuilt from your own scans with `training/build_registration_assets.py`.

Verify the installation:

```bash
.venv/bin/python scripts/preflight.py     # should print "Preflight: PASS"
```

## Configuration

| Setting | Where | Purpose |
|---|---|---|
| `CARD_DATA_DIR` | Environment variable | Folder that holds card scans, labels and work outputs (default `data/`) |
| Field layout | `config/layout_templates.json` | Field boxes per card type, snapped to the printed grid |
| Review policy | `config/review_policy.json` | Calibration and per-field ship-gate cutoffs, generated by `evaluation/ship_gate.py` |
| Low-ink rule | `config/low_ink_rule.json` | When a value read from very little ink is forced to check |
| Fallback thresholds | `config/review_thresholds.json` | Per-value-type cutoffs, used only when no review policy exists |
| Form schemas | `config/form_schemas.json`, `config/leapwell/` | Field definitions and the Leap Well form mapping |

Regenerate `review_policy.json` after any retraining, so the gate always describes the model that ships ([docs/TRAINING.md](docs/TRAINING.md)).

## Running

**Review app:**

```bash
./run.sh                                  # opens on http://127.0.0.1:8501
```

Upload a JPG, PNG or PDF of an EPI, ANC or NCD card and click **Extract Card**. Cards placed in `data/samples/` are also listed in the sidebar.

**From Python:**

```python
from pipeline.pipeline import extract_card
from pipeline.export import build_export_payload

result = extract_card("card.jpg")            # card type is detected automatically
for f in result["fields"]:
    print(f["field_id"], f["value_bn_raw"], f["status"], f["review_priority"])

payload = build_export_payload(result, reviews={})   # every final_value is null until reviewed
```

**Walkthrough notebook.** [POC_WALKTHROUGH.ipynb](POC_WALKTHROUGH.ipynb) follows one card through every stage of the pipeline.

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --prefix .venv --name bengali-card-ocr --display-name "Bengali Card OCR (.venv)"
```

Open it, select the **Bengali Card OCR (.venv)** kernel and Run All. It reads sample cards from `data/pages/`.

**Evaluation:**

```bash
.venv/bin/python evaluation/verified_accuracy.py --gold <labels.json>   # score against human labels
.venv/bin/python evaluation/ship_gate.py                                # calibration + per-field gate
```

## Export Format

The export JSON (`schema_version: "2.0"`, built by `pipeline/export.py`) stores the source file's SHA-256, the card type, timings and a review summary. For every field it stores:
- `model_prediction`: the raw reading, the original reading before any repair, the normalized value (Bengali and Western), the status and review priority, `probability_correct`, alternative candidates and warnings;
- `human_review`: the reviewer's action, any edit, notes and a UTC timestamp;
- `final_value`: the reviewed value, which stays `null` until the field is approved;
- the field geometry (polygon, crop box) and its printed label in Bengali and English.

It also groups the fields by Leap Well form section. The mapping is in [docs/LEAPWELL_FORM_MAPPING.md](docs/LEAPWELL_FORM_MAPPING.md).

## Tests

```bash
.venv/bin/python -m unittest discover tests
```

59 tests. Tests that need the alignment templates or real card scans are skipped without them; set `CARD_DATA_DIR` to run the scan tests. `tests/test_no_patient_data.py` fails if any image, label, model file or notebook output would be committed. GitHub Actions runs the suite on every push.

## Project Layout

```
app.py                  Streamlit review app
pipeline/               alignment, field layout, recognition, marks, review policy, export
config/                 card layouts, form schemas, review policy (ship gate + calibration), Leap Well mapping
models/                 recognizer and mark classifier (not committed; see Setup)
assets/registration/    alignment templates (not committed)
training/               scripts that built the models (need the private card data)
evaluation/             accuracy scoring, ship gate and calibration, labeling page, evaluation runs
scripts/                preflight check, notebook runner
tests/                  unit and regression tests
docs/                   results, training procedure, Leap Well mapping
POC_WALKTHROUGH.ipynb   step-by-step pipeline walkthrough
```

## Data

Card scans, labels, alignment templates and trained weights all come from patient records and are **never committed**. `data/`, `assets/registration/` and `models/` are gitignored, and a test enforces it. Training, evaluation and the notebook read them from `data/`, or from the folder in `CARD_DATA_DIR`:

```
data/
  pages/page-001.jpg ...     scanned cards
  split.json                 train / val / test page ids
  ai-labels/codex/, claude/  two independent AI transcriptions per page (reference labels)
  samples/                   optional: cards listed in the app's sample menu
  work/                      outputs of training/ and evaluation/
```

How the models were trained: [docs/TRAINING.md](docs/TRAINING.md).

## Models and Licenses

| Component | Source | License |
|---|---|---|
| Recognizer | PP-OCRv6_small_rec (PaddlePaddle), fine-tuned on card cells | Apache-2.0 |
| Synthetic pretraining data | NumtaDB handwritten Bengali digits | CC BY-SA 4.0 |
| Mark classifier | Trained from scratch on card cells | This project (MIT) |

No non-commercial model or code is included. Orus-OCR, a strong earlier candidate, was dropped because its license is non-commercial.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Preflight fails on missing model or template | Model files or `assets/registration/` not unpacked | Get them from the project team ([Setup](#setup)) |
| "Card layout could not be registered safely" | Not an EPI, ANC or NCD card; heavy blur; the card is cut off or rotated | Rescan flat and fully in frame, or use the rotation / card-type override. Unknown layouts are refused by design |
| Every EPI date is "check" | Expected: EPI dates have not passed the ship gate | Review them manually; more labelled EPI cards are needed |
| Many empty cells read as "−" on a phone photo | Grid lines inside the crops of a curved photo | Use flat scans; phone-photo handling is on the roadmap |
| App does not open | Nothing listening on port 8501 | Start it with `./run.sh`, then open http://127.0.0.1:8501 |
| Scan tests are skipped | No card data available | Set `CARD_DATA_DIR` to the private data folder |

## Security and Safety

- **Nothing is saved without a person.** Unreviewed values are `null` in the export, and "accept all routine" is one explicit, logged click.
- **Routine must be earned per field**, on held-out cards, with enough cells for a 95% interval. The gate is refitted from evidence (`evaluation/ship_gate.py`), not set by hand.
- **Every check says why.**
- **Unknown cards are refused.** A scan that matches none of the three layouts is rejected, not guessed.
- **Patient data never leaves the machine.** There are no network calls, and a test blocks patient data from being committed.
- **Phone photos:** the gate was fitted on flat scans. Until phone photos are measured and handled, review photo inputs field by field ([DOCUMENTATION.md §11](DOCUMENTATION.md#11-phone-photo-check)).

## Roadmap

**Completed ✅**
- ✅ Offline pipeline for EPI, ANC and NCD cards (alignment, layout, recognition, marks, export)
- ✅ PP-OCRv6 fine-tune (Apache-2.0) replacing the non-commercial Orus model
- ✅ 5-fold cross-validation, calibration, per-field ship gate and low-ink rule
- ✅ Review app, walkthrough notebook, Leap Well form mapping
- ✅ Evaluation on the re-scanned card set

**In progress 🚧**
- 🚧 Clinical reviewer labels: 368 fields on 9 held-out cards, blind labeling page ready
- 🚧 Phone-photo handling: force check for photo inputs, fix ANC photo row alignment

**Planned 📋**
- More labelled EPI cards; retrain on all 44 EPI cards once a fresh test set exists
- A new scan batch from another clinic, for writer-disjoint testing
- A card pack and TFLite export for the Leap Well phone app (replacing its ML Kit / Tesseract offline path)

## License

Code is released under the MIT License; see [LICENSE](LICENSE). Model weights and card data are not distributed with this repository. The base model (Apache-2.0) and NumtaDB (CC BY-SA 4.0) keep their own licenses.

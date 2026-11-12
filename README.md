# Bengali card OCR (offline)

Reads handwritten values from BRAC paper health cards (**EPI** vaccination, **ANC** antenatal, **NCD** chronic disease) and turns them into structured fields that a health worker reviews before they are saved. Everything runs locally on the CPU: no network calls, no cloud APIs.

**Status: proof of concept.** It is not accurate enough for unattended use, and every field must be reviewed by a person. See [Results](#results).

## How it works

```
image / PDF ─► align to card template ─► cut out each field ─► read the cell ─► check & flag ─► human review ─► JSON
               (SIFT + RANSAC,            (layout config,        (PP-OCRv6,       (date/BP syntax,  (Accept / Correct,
                detects card type)         blank detection)       ONNX, 12 MB)     routine / check)  nothing auto-saved)
```

1. **Align.** The photo or scan is matched to a template of each card type and warped onto it. A card that can't be matched safely is rejected; the app never guesses field positions.
2. **Locate fields.** Field boxes come from `config/layout_templates.json`. Empty cells are detected from the pixels and never sent to the model.
3. **Read.** A [PP-OCRv6_small_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors) recognizer, fine-tuned on card cells, reads each cell. Decoding is restricted per field type (a number field can only produce digits) and to one numeral script per value. Tick/cross/(±) marks are read by a small separate classifier.
4. **Check.** Dates and blood pressure must be well-formed. Impossible two-line EPI dates are replaced by the most likely valid date. Each field is marked **routine** (confident and valid) or **check** (needs a closer look).
5. **Review.** The reviewer accepts or corrects each field, or accepts all routine fields with one logged click. Unreviewed values stay `null` in the export.

The export JSON (schema 2.0) keeps the model's reading, the reviewer's decision and the source image hash separately. It also includes a grouping by the LEAPWELL app's form sections ([docs/LEAPWELL_FORM_MAPPING.md](docs/LEAPWELL_FORM_MAPPING.md)).

## Results

Measured on held-out cards that were not used for training or tuning, with 95% intervals. Full details are in [docs/RESULTS.md](docs/RESULTS.md).

**Against human labels** (9 cards, 289 filled fields):

| | Overall | ANC | NCD | EPI |
|---|---|---|---|---|
| **Exact match** | **81.7%** (77–86%) | 85.5% (80–90%) | 79.5% (66–89%) | 63.2% (47–77%) |
| Same value | 87.9% (84–91%) | 94.2% (90–97%) | 79.5% (66–89%) | 63.2% (47–77%) |

*Exact* means the text is identical; only a line break between the two lines of a written date counts as `/`. *Same value* also treats Bengali and Western digits, mark spellings (✓ / y) and date separators (`/ . -`) as equal.

Against AI labels on 42 cards (1,126 filled fields) the model scores 96.3% exact and 97.3% same value. Those labels skip the fields the two AIs disagreed on, which are the hard ones, so they run high.

- The human-label number is the one that counts. The **95% target is not met yet**, mainly because handwritten dates are read at 67%.
- Flags mostly point to the errors: fields marked routine were right 94% of the time, fields marked check 15%. **Dates are the exception:** most wrong dates were still marked routine.
- All 54 cells called blank really were blank.
- Speed: about 0.4 s per card on a laptop CPU (Apple M5), up to 1 s for the first card.

## Quick start

Requires Python 3.12 or newer (tested on 3.12 and 3.14, macOS on Apple Silicon).

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Model files and alignment templates (needed once).** Both come from real patient cards, so they are **not in this repository**; get them from the project team and unpack them into the repo folder:

```
models/ppocr-cards/ppocrv6_cards.onnx         recognizer (12 MB)
models/ppocr-cards/ppocrv6_cards.chars.json   its character list
models/marks/mark_cnn.onnx, mark_cnn.json     mark classifier
assets/registration/EPI.png, ANC.png, NCD.png one alignment template per card type
```

The templates can also be rebuilt from your own scans with `training/build_registration_assets.py`.

Check the setup, then start the app:

```bash
.venv/bin/python scripts/preflight.py     # should print "Preflight: PASS"
./run.sh                                  # opens on http://127.0.0.1:8501
```

Upload a JPG, PNG or PDF of an EPI, ANC or NCD card and click **Extract Card**. Cards placed in `data/samples/` are also listed in the sidebar.

### From Python

```python
from pipeline.pipeline import extract_card
from pipeline.export import build_export_payload

result = extract_card("card.jpg")            # card type is detected automatically
for f in result["fields"]:
    print(f["field_id"], f["value_bn_raw"], f["status"], f["review_priority"])

payload = build_export_payload(result, reviews={})   # every final_value is null until reviewed
```

### Walkthrough notebook

[POC_WALKTHROUGH.ipynb](POC_WALKTHROUGH.ipynb) follows one card through every step of the pipeline.

```bash
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --prefix .venv --name bengali-card-ocr --display-name "Bengali Card OCR (.venv)"
```

Open it, pick the **Bengali Card OCR (.venv)** kernel and Run All. It reads sample cards from `data/pages/`.

## Tests

```bash
.venv/bin/python -m unittest discover tests
```

Tests that need the alignment templates or real card scans are skipped without them; set `CARD_DATA_DIR` (below) to run the scan tests. A test also fails if any image, label, model file or notebook output would be committed. GitHub Actions runs the suite on every push.

## Project layout

```
app.py                  Streamlit review app
pipeline/               alignment, field layout, recognition, review, export
config/                 card layouts, form schemas, review thresholds, LEAPWELL form mapping
models/                 recognizer and mark classifier (not committed; see Quick start)
training/               scripts that built the models (need the private card data)
evaluation/             human labeling page builder and accuracy scoring
scripts/                preflight check, notebook runner
tests/                  unit and regression tests
docs/                   results, training notes, LEAPWELL mapping
```

## Data

Card scans, labels, alignment templates and trained weights all come from patient records and are **never committed** (`data/`, `assets/registration/` and `models/` are gitignored, and a test enforces it). Training, evaluation and the notebook read them from `data/`, or from the folder in `CARD_DATA_DIR`:

```
data/
  pages/page-001.jpg ...     scanned cards
  split.json                 train / val / test page ids
  ai-labels/codex/, claude/  two independent AI transcriptions per page
  samples/                   optional: cards listed in the app's sample menu
  work/                      outputs of training/ and evaluation/
```

How the models were trained: [docs/TRAINING.md](docs/TRAINING.md).

## Models and licenses

| Component | Source | License |
|---|---|---|
| Recognizer | PP-OCRv6_small_rec (PaddlePaddle), fine-tuned on card cells | Apache-2.0 |
| Synthetic pretraining data | NumtaDB handwritten Bengali digits | CC BY-SA 4.0 |
| Mark classifier | Trained from scratch on card cells | (this project) |

No non-commercial model or code is included. A license for this repository's own code has not been chosen yet.

## Known limitations

- **Accuracy is below target**, especially handwritten dates, and wrong dates are often still marked routine. Every value needs human review, and confidence scores are not calibrated.
- **"Readable" means well-formed, not correct.** A wrong number can still be a valid number.
- **Phone photos are not measured.** Accuracy figures are for flat scans. EPI phone photos are always flagged for review.
- **Small, narrow data.** 142 scanned cards from one clinic's writers, with only 44 EPI cards. The human-label test covers 9 cards.
- **Only three card layouts.** Other forms, or heavily redesigned versions of these, are rejected at alignment.
- **Values in blank cells.** The model sometimes reads print or noise in empty ANC last-column cells as a value. Most are flagged check, but not all.
- **No persistence or sign-in.** Reviews live in the browser session until exported. There is no user identity or audit signature.
- **Not integrated yet.** This is a desktop review app. Running on the phone inside the LEAPWELL app is future work.

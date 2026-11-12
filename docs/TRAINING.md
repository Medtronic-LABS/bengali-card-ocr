# Training

How the two shipped models were built. Every step needs the private card data laid out as described in the README, plus `requirements-train.txt`:

```bash
.venv/bin/python -m pip install -r requirements-train.txt
```

Labels are fields where two independent AI transcriptions agree. They are not human-verified, so validation scores below are agreement, not accuracy. Test pages are never opened by any training script.

## Recognizer (`models/ppocr-cards/`)

Base model: [PaddlePaddle/PP-OCRv6_small_rec_safetensors](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_rec_safetensors) (Apache-2.0). Download it to `models/base/PP-OCRv6_small_rec/`. Its 18,710-class output layer is replaced by the 211-character card set in `ppocrv6_cards.chars.json` (Bengali and Western digits, Bengali letters, ASCII, a few symbols). Rows for characters the base model already knows are copied over; Bengali rows start fresh.

1. **Cut training crops** from train and validation pages, using the app's own alignment and cell preparation, so the model sees exactly what the app feeds it:

   ```bash
   .venv/bin/python training/build_card_crops.py --out crops.pkl
   ```

   Two-line EPI dates and blood pressure are split into their parts, as the app reads them. Mark fields are skipped (they have their own model).

2. **Make synthetic cells** from [NumtaDB](https://www.kaggle.com/datasets/BengaliAI/numta) handwritten Bengali digits (CC BY-SA 4.0): numbers, decimals, temperatures, dates and BP halves with drawn separators and jitter. No card page is used.

   ```bash
   .venv/bin/python training/make_synthetic_cells.py <numtadb_dir> data/work --n 12000 --val-from crops.pkl --out synthetic.pkl
   ```

3. **Stage 1: pretrain on synthetic cells** (12,000 crops, 4 epochs; best validation epoch kept):

   ```bash
   .venv/bin/python training/finetune_ppocrv6.py --crops synthetic.pkl --epochs 4 --out stage1.pt --report report_stage1.json
   ```

4. **Stage 2: fine-tune on real card crops** (1,640 train crops, 30-epoch cosine schedule; best validation epoch was 11, at 87.9% exact crop agreement). This is the **train-only model**, used for every validation decision:

   ```bash
   .venv/bin/python training/finetune_ppocrv6.py --crops crops.pkl --init data/work/stage1.pt --out ppocrv6_cards_trainonly.pt --report report_stage2.json
   ```

5. **Final model: train + validation crops** (2,063 crops) with the same recipe, on a 12-epoch schedule that ends at the last epoch. No validation-based selection is possible here, so validation no longer measures this model:

   ```bash
   .venv/bin/python training/finetune_ppocrv6.py --crops crops.pkl --init data/work/stage1.pt --epochs 12 --train-on-val 12 --out ppocrv6_cards.pt --report report_final.json
   ```

6. **Export to ONNX** (also writes the `.chars.json` and checks ONNX against PyTorch):

   ```bash
   .venv/bin/python training/export_ppocrv6_onnx.py data/work/ppocrv6_cards.pt models/ppocr-cards/ppocrv6_cards.onnx
   ```

Human-checked values can be added as extra training crops with `training/human_labels_to_crops.py`. It refuses test pages, so human labels on test pages stay usable for scoring.

## Mark classifier (`models/marks/`)

A small CNN that reads ✓, ×, (−) and − in fields whose printed label offers "(✓/x)" or "(+/-)". It is trained on train pages (726 cells). Validation pages (200 cells) choose the epoch and the confidence cutoff: below the cutoff, stroke-shape rules decide or the field is flagged. A possible "+" is never filled in automatically.

```bash
.venv/bin/python training/train_mark_classifier.py
```

## Review thresholds (`config/review_thresholds.json`)

The routine/check cutoff per value type is the lowest model score at which at least 95% of validation fields marked routine agree with the AI labels. It is chosen with the train-only model:

```bash
.venv/bin/python evaluation/ai_label_agreement.py val --model models/ppocr-cards/ppocrv6_cards_trainonly.onnx --tag trainonly
.venv/bin/python evaluation/choose_thresholds.py trainonly
```

## Card templates

- `training/build_registration_assets.py` builds the alignment templates in `assets/registration/` from 8 scans per card type.
- `training/snap_layout_to_template.py` snaps the field boxes in `config/layout_templates.json` to the printed grid lines of those templates.

## Next steps

- Retrain on all 44 EPI cards once a fresh test set exists (a new scan batch, or human labels on new cards).
- Add human-checked crops, especially two-line EPI dates.

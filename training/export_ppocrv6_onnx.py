"""Export a PP-OCRv6 card checkpoint (training/finetune_ppocrv6.py) to ONNX for the app.

Input "x": float32 [1, 3, 48, W] (W >= 128), output "logits": [1, W/8, 1 + len(chars)].
Also writes <name>.chars.json (the character list; class 0 is the CTC blank)
and checks the ONNX logits against PyTorch on a few validation crops from data/work/crops.pkl.

Usage: .venv/bin/python training/export_ppocrv6_onnx.py data/work/ppocrv6_cards.pt models/ppocr-cards/ppocrv6_cards.onnx
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))
import finetune_ppocrv6 as ft  # noqa: E402
from pipeline.data_paths import WORK  # noqa: E402


class Logits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return ft.logits_of(self.model, x)


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    model = Logits(ft.build(str(src))).eval()
    torch.onnx.export(model, (torch.ones(1, 3, 48, 256),), str(dest), input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {3: "width"}, "logits": {1: "steps"}}, opset_version=17, dynamo=False)
    dest.with_suffix(".chars.json").write_text(json.dumps(ft.CHARS, ensure_ascii=False))
    import onnxruntime as ort

    session = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
    val = [s for s in pickle.load(open(WORK / "crops.pkl", "rb"))
           if s["subset"] == "val"][:40]
    worst = 0.0
    with torch.no_grad():
        for s in val:
            x = ft.batch([s["image"]])
            worst = max(worst, float(np.abs(session.run(None, {"x": x.numpy()})[0] - model(x).numpy()).max()))
    print(f"wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB); max |onnx - torch| over {len(val)} crops = {worst:.2e}")


if __name__ == "__main__":
    main()

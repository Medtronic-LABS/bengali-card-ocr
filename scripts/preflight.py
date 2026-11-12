"""Fast, local-only startup check. Never downloads dependencies or assets."""

from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.recognize import engine_status  # noqa: E402


def main():
    missing = []
    for name in ("streamlit", "PIL", "numpy", "cv2", "onnxruntime", "pypdfium2", "rapidfuzz"):
        if importlib.util.find_spec(name) is None:
            missing.append("Python module: " + name)
    for name in ("EPI", "ANC", "NCD"):
        if not (ROOT / "assets/registration" / f"{name}.png").is_file():
            missing.append(f"alignment template: assets/registration/{name}.png (see README)")
    for name in ("layout_templates.json", "form_schemas.json", "review_thresholds.json"):
        if not (ROOT / "config" / name).is_file():
            missing.append("config: " + name)
    if not (ROOT / "models/marks/mark_cnn.onnx").is_file():
        missing.append("mark model: models/marks/mark_cnn.onnx")
    status = engine_status()
    if not status["available"]:
        missing.append("recognizer: " + str(status["error"]))
    print("Recognizer (PP-OCRv6-cards):", "ready" if status["available"] else "unavailable")
    for item in missing:
        print("MISSING:", item)
    print("Preflight:", "PASS" if not missing else "FAIL")
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())

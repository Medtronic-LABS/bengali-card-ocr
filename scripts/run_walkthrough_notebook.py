"""Execute the walkthrough using the project's kernel; save sensitive outputs locally."""

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.data_paths import WORK  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-controls",
        action="store_true",
        help="Exercise widget callbacks then reset; never export test approvals",
    )
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    import nbformat
    from nbclient import NotebookClient

    source = ROOT / "POC_WALKTHROUGH.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    assert all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    ), "Source notebook must have cleared outputs"
    if args.test_controls:
        notebook.cells.append(
            nbformat.v4.new_code_cell("""
# AUTOMATED CALLBACK TEST ONLY: transient operator simulations are NOT gold labels.
selected = next(f for f in result['fields'] if f['value_bn_raw'] is not None)
field_picker.value = selected['field_id']
reviewed_text.value = 'NOTEBOOK_CALLBACK_TEST_ONLY'
correct_button.click()
assert reviews[selected['field_id']]['edited_value'] == 'NOTEBOOK_CALLBACK_TEST_ONLY'
assert reviews[selected['field_id']]['action'] == 'correct'
reset_button.click()
assert selected['field_id'] not in reviews
accept_button.click()
assert reviews[selected['field_id']]['edited_value'] == selected['value_bn_raw']
assert reviews[selected['field_id']]['action'] == 'accept'
reset_button.click()
assert not reviews
bulk_button.click()
assert reviews and all(field_map[fid]['review_priority'] == 'routine' for fid in reviews)
assert all(r['verification_origin'] == 'explicit_operator_bulk_accept_routine' for r in reviews.values())
reviews.clear()
show_review_field()
assert build_export_payload(result, reviews)['review_summary']['reviewed'] == 0
assert len(exported_paths) == 1, 'Test approvals must never be exported'
print('Notebook callback regression checks: PASS; all simulated reviews reset; no test approvals exported.')
""")
        )
    kernel_data = ROOT / ".venv/share/jupyter"
    os.environ["JUPYTER_PATH"] = str(kernel_data) + (
        os.pathsep + os.environ["JUPYTER_PATH"]
        if os.environ.get("JUPYTER_PATH")
        else ""
    )
    # Restrict transient Jupyter connection files to the local ignored workspace.
    runtime = WORK / "notebook/jupyter-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    output = WORK / "notebook/POC_WALKTHROUGH.executed.ipynb"
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="bengali-card-ocr",
        resources={"metadata": {"path": str(ROOT)}},
        store_widget_state=True,
    )
    def progress(cell, cell_index, **kwargs):
        print(f"Executing cell {cell_index + 1}/{len(notebook.cells)} ({cell.cell_type})", flush=True)
    client.on_cell_start = progress
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output)
    errors = [
        out
        for cell in notebook.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, f"Notebook contains {len(errors)} failed cells"
    reports = []
    for cell in notebook.cells:
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                for line in out.get("text", "").splitlines():
                    if line.startswith("WALKTHROUGH_RUN_REPORT="):
                        reports.append(json.loads(line.split("=", 1)[1]))
    assert reports, "Missing notebook execution report"
    report = reports[-1]
    assert report["reviewed_fields"] == 0, "Run All must not approve any clinical value"
    assert report["pending_fields"] == report["fields"]
    assert set(report["three_card_types_run"]) == {"EPI", "ANC", "NCD"}
    report.update(
        notebook=str(source.relative_to(ROOT)),
        executed_copy=str(output),
        executed_code_cells=sum(c.cell_type == "code" for c in notebook.cells),
        callback_tests_passed=args.test_controls,
        errors=0,
    )
    (output.parent / "execution-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

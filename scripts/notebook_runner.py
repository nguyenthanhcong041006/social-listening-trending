import json
import os
import sys
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr

def run_notebook(nb_path):
    print(f"\n{'='*70}\nRunning Notebook: {nb_path}\n{'='*70}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Isolated global scope for notebook execution
    nb_globals = {
        "__name__": "__main__",
        "__file__": os.path.abspath(nb_path),
    }

    # Set working directory to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    cells = nb.get("cells", [])
    errors = []
    executed_count = 0

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        executed_count += 1
        source_lines = cell.get("source", [])
        if isinstance(source_lines, list):
            code = "".join(source_lines)
        else:
            code = str(source_lines)

        # Filter out ipython magic commands like %matplotlib inline, !pip, etc.
        filtered_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("%") or stripped.startswith("!"):
                filtered_lines.append(f"# {line}")
            else:
                filtered_lines.append(line)
        executable_code = "\n".join(filtered_lines)

        if not executable_code.strip():
            continue

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(executable_code, nb_globals)
            out_str = stdout_buf.getvalue()
            first_out = out_str.strip().split("\n")[0] if out_str.strip() else ""
            print(f"  [Cell {idx+1}] OK {f'-> {first_out[:60]}' if first_out else ''}")
        except Exception as e:
            err_msg = traceback.format_exc()
            print(f"  [Cell {idx+1}] FAILED: {type(e).__name__}: {e}")
            errors.append({
                "cell_index": idx,
                "cell_number": idx + 1,
                "code": code,
                "error": str(e),
                "traceback": err_msg
            })

    print(f"\nSummary for {nb_path}: {executed_count} code cells, {len(errors)} errors.")
    return nb, errors

if __name__ == "__main__":
    notebooks = [
        "notebooks/01_data_exploration_and_preprocessing.ipynb",
        "notebooks/02_topic_modeling_bertopic.ipynb",
        "notebooks/03_trend_prediction_evaluation.ipynb",
        "notebooks/04_model_evaluation_and_visualization.ipynb"
    ]
    all_results = {}
    for nb in notebooks:
        if os.path.exists(nb):
            try:
                _, errs = run_notebook(nb)
                all_results[nb] = errs
            except Exception as e:
                print(f"Fatal error reading {nb}: {e}")
                all_results[nb] = [{"error": str(e)}]
        else:
            print(f"Not found: {nb}")

    print("\n" + "="*70)
    print("TOTAL ERRORS OVERVIEW")
    print("="*70)
    for nb, errs in all_results.items():
        print(f"{nb}: {len(errs)} error(s)")
        for err in errs:
            print(f"  - Cell {err.get('cell_number', '?')}: {err.get('error', '')}")

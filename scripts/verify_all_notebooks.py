import os
import sys
import json
import traceback

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

def run_notebook(nb_path):
    print(f"\n{'='*60}")
    print(f"RUNNING: {nb_path}")
    print(f"{'='*60}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *args, **kwargs: None

    def dummy_display(*args, **kwargs):
        pass

    nb_globals = {
        "__name__": "__main__",
        "__file__": os.path.abspath(nb_path),
        "display": dummy_display,
    }

    cell_results = []
    code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]

    for idx, cell in enumerate(code_cells):
        lines = cell.get("source", [])
        code = "".join(lines) if isinstance(lines, list) else str(lines)

        exec_lines = []
        for line in code.splitlines():
            s = line.strip()
            if s.startswith("%") or s.startswith("!"):
                exec_lines.append(f"# {line}")
            else:
                exec_lines.append(line)
        exec_code = "\n".join(exec_lines)
        if not exec_code.strip():
            continue

        try:
            exec(exec_code, nb_globals)
            cell_results.append((idx + 1, "PASS", None))
            print(f"  [Cell {idx+1}/{len(code_cells)}] PASS")
        except Exception as e:
            tb = traceback.format_exc()
            cell_results.append((idx + 1, "FAIL", f"{type(e).__name__}: {e}"))
            print(f"  [Cell {idx+1}/{len(code_cells)}] FAIL: {type(e).__name__}: {e}")

    failed = [r for r in cell_results if r[1] == "FAIL"]
    print(f"\n--> {nb_path}: {len(code_cells)} code cells, {len(code_cells) - len(failed)} passed, {len(failed)} failed.")
    return len(failed) == 0, cell_results

if __name__ == "__main__":
    notebooks = [
        "notebooks/01_data_exploration_and_preprocessing.ipynb",
        "notebooks/02_topic_modeling_bertopic.ipynb",
        "notebooks/03_trend_prediction_evaluation.ipynb",
        "notebooks/04_model_evaluation_and_visualization.ipynb",
    ]

    summary = {}
    all_ok = True
    for nb in notebooks:
        ok, res = run_notebook(nb)
        summary[nb] = (ok, res)
        if not ok:
            all_ok = False

    print(f"\n{'='*60}")
    print("FINAL SUMMARY REPORT")
    print(f"{'='*60}")
    for nb, (ok, res) in summary.items():
        total = len(res)
        passed = sum(1 for _, status, _ in res if status == "PASS")
        status_str = "SUCCESS (100% PASS)" if ok else "FAILED"
        print(f"• {nb:55s} : {status_str} ({passed}/{total} cells)")

    if all_ok:
        print("\nALL NOTEBOOKS EXECUTED WITH ZERO ERRORS!")
        sys.exit(0)
    else:
        print("\nSOME NOTEBOOKS HAVE FAILING CELLS!")
        sys.exit(1)

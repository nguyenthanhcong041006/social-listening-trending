import json
import os
import sys
import traceback

def test_notebook(nb_path):
    print(f"\n==================================================")
    print(f"TESTING: {nb_path}")
    print(f"==================================================")
    
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

    errors = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        
        lines = cell.get("source", [])
        code = "".join(lines) if isinstance(lines, list) else str(lines)
        
        # Strip magics
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
            print(f"  [Cell {idx+1}] PASS")
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  [Cell {idx+1}] FAIL: {type(e).__name__}: {e}")
            print(tb)
            errors.append((idx + 1, exec_code, type(e).__name__, str(e), tb))

    print(f"Finished {nb_path}: {len(errors)} error(s)")
    return errors

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "notebooks/01_data_exploration_and_preprocessing.ipynb"
    test_notebook(target)

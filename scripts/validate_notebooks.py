import json
import glob
import sys
import traceback

print("=" * 70)
print("VALIDATING SYNTAX OF ALL CODE CELLS IN NOTEBOOKS")
print("=" * 70)

has_error = False
for nb_path in sorted(glob.glob("notebooks/*.ipynb")):
    print(f"\nChecking: {nb_path}")
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        print(f"  [ERROR] Invalid JSON in {nb_path}: {e}")
        has_error = True
        continue
    
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"  Total cells: {len(nb['cells'])}, Code cells: {len(code_cells)}")
    
    for idx, c in enumerate(code_cells):
        src = "".join(c.get("source", []))
        try:
            compile(src, f"{nb_path} [Cell {idx}]", "exec")
        except SyntaxError as e:
            print(f"  [SYNTAX ERROR in Code Cell {idx}]: {e}")
            print("  --- Code preview ---")
            print("\n".join(src.split("\n")[:10]))
            has_error = True

if has_error:
    print("\nFAILED: Syntax errors detected.")
    sys.exit(1)
else:
    print("\nSUCCESS: All notebook code cells compiled with 100% valid Python syntax!")

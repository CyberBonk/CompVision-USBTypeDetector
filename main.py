
import os
import sys
import subprocess

def run(cmd, *, stdin_data=None):

    res = subprocess.run(
        [sys.executable, cmd],
        input=stdin_data,
        text=True,      # let us pass / receive str instead of bytes
    )
    if res.returncode:
        raise RuntimeError(f"{cmd} exited with status {res.returncode}")

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    ms1_path = os.path.join(base_dir, "milestone1.py")
    ms2_path = os.path.join(base_dir, "milestone2.py")
    out_dir = os.path.join(base_dir, "output_images")

    # 1) run milestone-1
    print("============  MILESTONE-1  ============")
    run(ms1_path)                       # executes milestone1.py main()

    # 2)  run milestone-2   
    # milestone2.py expects a single line of user input (folder path).
    print("\n============  MILESTONE-2  ============")
    print(f"(Sending '{out_dir}' to milestone2.py stdin)\n")
    run(ms2_path, stdin_data=out_dir + "\n")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import subprocess
import sys

def main():
    # 20 → 100 in steps of 5, and ensure 101 runs as well.
    energies = list(range(20, 101, 5))


    # Forward any extra args to process_energy.py (e.g. --skip-existing, --jobs 32)
    extra_args = sys.argv[1:]

    for e in energies:
        cmd = ["python", "/home/dwong/DELight_mtr/trigger_study/HTCondor/process_energy.py", "--energy", str(e)] + extra_args
        print("\n=== Running:", " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"[WARN] process_energy.py failed for energy={e} (exit {ret}). Continuing...")

if __name__ == "__main__":
    main()

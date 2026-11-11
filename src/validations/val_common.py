"""
val_common.py

Helpers shared by the validation scripts for both cartpole and mspacman.
"""

import sys
import copy

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

RESULTS = []

def check(name, ok, detail=""):
    """Record and print one PASS or FAIL line."""
    RESULTS.append((name, bool(ok)))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f"\t{detail}" if detail else ""))

def section(title):
    print("\n" + "#" * 10 + f"\n{title}\n" + "#" * 10)

def finish():
    failed = [n for n, ok in RESULTS if not ok]
    section("SUMMARY")
    print(f"{len(RESULTS) - len(failed)} of {len(RESULTS)} checks passed")
    for n in failed:
        print("  FAILED:", n)
    sys.exit(1 if failed else 0)


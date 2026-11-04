#!/usr/bin/env python3
"""Check the LLVM maintenance split without configuring or building LLVM.

This structural gate checks source ownership, not generated code or hardware.
Use --baseline with the archived pre-split commit to verify that all unrelated
files, including Tensix and upstream vector/GPU sources, remain byte-identical.
"""

import argparse
from pathlib import Path
import re
import subprocess
import sys


SHARED = {
    "llvm/include/llvm/CodeGen/ValueTypes.td",
    "llvm/include/llvm/IR/Intrinsics.h",
    "llvm/include/llvm/IR/Intrinsics.td",
    "llvm/include/llvm/IR/IntrinsicsRISCV.td",
    "llvm/lib/CodeGen/ValueTypes.cpp",
    "llvm/lib/IR/Intrinsics.cpp",
    "llvm/lib/IR/Type.cpp",
    "llvm/unittests/Target/RISCV/CMakeLists.txt",
}
SHARED.update(
    "llvm/lib/Target/RISCV/" + name
    for name in (
        "CMakeLists.txt", "RISCVAsmPrinter.cpp", "RISCVCodeGenPassBuilder.cpp",
        "RISCVFeatures.td", "RISCVISelLowering.cpp", "RISCVInsertVSETVLI.cpp",
        "RISCVInstrInfo.cpp", "RISCVInstrInfo.td", "RISCVRegisterContents.cpp",
        "RISCVRegisterInfo.cpp", "RISCVRegisterInfo.td", "RISCVTargetMachine.cpp",
        "RISCVVLOptimizer.cpp", "RISCVVMV0Elimination.cpp",
        "RISCVVectorPeephole.cpp",
    )
)
SELF = "llvm/utils/trex/check-backend-partition.py"
SYMBOL = re.compile(r"BOSC|bosc|XBOSC")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def owned(path):
    return "bosc" in path.lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bosc", choices=("absent", "present"), required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    tracked = git(root, "ls-files").splitlines()
    errors = []
    active = [p for p in tracked if (root / p).is_file()]
    dedicated = [p for p in active if owned(p)]
    if args.bosc == "absent":
        errors += [f"BOSC-owned path remains: {p}" for p in dedicated]
        for path in sorted(SHARED):
            for number, line in enumerate((root / path).read_text().splitlines(), 1):
                if SYMBOL.search(line):
                    errors.append(f"BOSC shared seam remains: {path}:{number}")
    else:
        for path in (
            "llvm/include/llvm/IR/IntrinsicsRISCVBOSC.td",
            "llvm/lib/Target/RISCV/RISCVBOSCAME.cpp",
            "llvm/lib/Target/RISCV/RISCVBOSCBound.cpp",
            "llvm/test/CodeGen/RISCV/bosc/contents-ame.ll",
        ):
            if path not in dedicated:
                errors.append(f"BOSC restoration missing: {path}")
    protected = (
        "llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp",
        "mlir/lib/Dialect/GPU/IR/GPUDialect.cpp",
        "mlir/lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
        "llvm/lib/Target/RISCV/RISCVInstrInfoV.td",
        "llvm/lib/Target/RISCV/RISCVTensixLowering.cpp",
        "llvm/include/llvm/IR/IntrinsicsRISCVTensix.td",
    )
    for path in protected:
        if not (root / path).is_file():
            errors.append(f"Preserved implementation missing: {path}")
    if args.baseline:
        baseline = git(root, "ls-tree", "-r", "--name-only", args.baseline).splitlines()
        changed = git(root, "diff", "--name-only", args.baseline, "--").splitlines()
        for path in changed:
            if path != SELF and path not in SHARED and not owned(path):
                errors.append(f"Unrelated baseline change: {path}")
        for path in baseline:
            if path == SELF or path in SHARED or owned(path):
                continue
            if not (root / path).is_file() and not (root / path).is_symlink():
                errors.append(f"Unrelated baseline file missing: {path}")
        if args.bosc == "present":
            errors += [f"Restored tree differs from baseline: {p}" for p in changed if p != SELF]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"FAIL: {len(errors)} partition violations", file=sys.stderr)
        return 1
    print(f"PASS: BOSC {args.bosc}; preserved Tensix, RVV, GPU and NVPTX sources")
    if args.baseline:
        print(f"PASS: unrelated files unchanged from {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

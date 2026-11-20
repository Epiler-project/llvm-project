#!/usr/bin/env python3
"""Check the LLVM maintenance split and Tensix stages without building LLVM.

This structural gate checks source ownership, middle-end freeze boundaries, and
physical intrinsic reception invariants without configuring or compiling LLVM:
  - --bosc {absent,present}: verifies BOSC accelerator partition invariants.
  - --tensix-stage l0-frozen: freezes legacy Tensix middle-end files
    (ContentsGroup, RegisterContents, TensixMOPSequence, auto TensixReplay*,
    Ordinary/SFPU shims, and mlir SymbolTable fork) at --freeze-ref.
    New middle-end files and code are forbidden.
  - --tensix-stage l-bind: enforces Slice 4 (W6) one-step cutover to
    llvm/include/llvm/IR/IntrinsicsTT.td (@llvm.tt.{wh,bh}.*) with zero legacy
    @llvm.riscv.tt.bound.* symbols remaining outside the sole rejection test.
  - --tensix-stage l1-slimmed: enforces W3-L1 post-W4 physical removal
    of all transitional middle-end files, clean mlir SymbolTable.{h,cpp} vs
    --merge-base, and zero legacy @llvm.riscv.tt.* or virtual v32i32
    SFPU carriers.
"""

import argparse
import hashlib
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
        "CMakeLists.txt",
        "RISCVAsmPrinter.cpp",
        "RISCVCodeGenPassBuilder.cpp",
        "RISCVFeatures.td",
        "RISCVISelLowering.cpp",
        "RISCVInsertVSETVLI.cpp",
        "RISCVInstrInfo.cpp",
        "RISCVInstrInfo.td",
        "RISCVRegisterContents.cpp",
        "RISCVRegisterInfo.cpp",
        "RISCVRegisterInfo.td",
        "RISCVTargetMachine.cpp",
        "RISCVVLOptimizer.cpp",
        "RISCVVMV0Elimination.cpp",
        "RISCVVectorPeephole.cpp",
    )
)
SELF = "llvm/utils/trex/check-backend-partition.py"
SYMBOL = re.compile(r"BOSC|bosc|XBOSC")

DEFAULT_FREEZE_REF = "37df17873d3396c9d9c46796f404f29c4056aff4"
DEFAULT_MERGE_BASE = "6b5630dc25728baad8a4daf8697db2da15e7ed58"

FROZEN_MIDDLE_END_LLVM_FILES = (
    "llvm/include/llvm/IR/IntrinsicsRISCVTensixContentsGroup.td",
    "llvm/lib/Target/RISCV/RISCVInstrInfoTensixContentsGroup.td",
    "llvm/lib/Target/RISCV/RISCVTensixContentsGroup.h",
    "llvm/lib/Target/RISCV/RISCVTensixContentsGroup.cpp",
    "llvm/lib/Target/RISCV/RISCVRegisterContents.h",
    "llvm/lib/Target/RISCV/RISCVRegisterContents.cpp",
    "llvm/include/llvm/Support/TensixMOPSequence.h",
    "llvm/lib/Support/TensixMOPSequence.cpp",
    "llvm/lib/Target/RISCV/RISCVTensixReplay.h",
    "llvm/lib/Target/RISCV/RISCVTensixReplay.cpp",
    "llvm/lib/Target/RISCV/RISCVTensixReplaySelection.h",
    "llvm/lib/Target/RISCV/RISCVTensixReplaySelection.cpp",
    "llvm/lib/Target/RISCV/RISCVTensixReplayTemplate.h",
    "llvm/lib/Target/RISCV/RISCVTensixReplayTemplate.cpp",
    "llvm/include/llvm/IR/IntrinsicsRISCVTensixOrdinary.td",
    "llvm/include/llvm/IR/IntrinsicsRISCVTensixSFPUExtra.td",
    "llvm/lib/Target/RISCV/RISCVInstrInfoTensixOrdinary.td",
    "llvm/lib/Target/RISCV/RISCVTensixLowering.h",
    "llvm/lib/Target/RISCV/RISCVTensixLowering.cpp",
    "llvm/lib/Target/RISCV/RISCVTensixCopyCCCleanup.cpp",
    "llvm/lib/Target/RISCV/RISCVTensixSFPUPhysicalIngress.cpp",
)

L1_DELETED_EXTRA_LLVM_FILES = (
    "llvm/include/llvm/IR/IntrinsicsRISCVTensix.td",
    "llvm/include/llvm/IR/IntrinsicsRISCVTensixBound.td",
)

FROZEN_MLIR_SYMBOL_TABLE_FILES = (
    "mlir/include/mlir/IR/SymbolTable.h",
    "mlir/lib/IR/SymbolTable.cpp",
)

ALLOWED_EXPLICIT_TENSIX_FILES = {
    "llvm/lib/Target/RISCV/RISCVTensixExplicitReplay.cpp",
}

# W6 moves the existing physical word receiver out of the frozen automatic
# owner. Permit only its forwarding edge and its existing selected-immediate
# rejection dependency. No automatic algorithm gains an exemption; L1 removes
# the transitional dependency along with ContentsGroup.
L_BIND_EXTRACTION_LINES = dict(
    [
        (
            "llvm/lib/Target/RISCV/RISCVTensixReplay.h",
            '#include "RISCVTensixExplicitReplay.h"',
        ),
        (
            "llvm/lib/Target/RISCV/RISCVTensixExplicitReplay.cpp",
            '#include "RISCVTensixContentsGroup.h"',
        ),
    ]
)

# The physical bank/scanner extraction leaves only these reviewed adapter
# additions in the frozen automatic owner. Pin their complete ordered sequence
# against the original freeze ref, never a filename or an individual matching
# line. Any added, changed, duplicated or reordered line loses the allowance.
# Native BUNDLE adapters also refer to the frozen placement owner. These
# reviewed added sequences receive and preserve already authored windows;
# they do not allocate SFPRs, repair CC or extract automatic Replay/MOP.
# Pin ordered additions only; this is not an exemption for arbitrary removals.
L_BIND_EXTRACTION_PATCHES = {
    # Exact forwarding to native descriptor selection and actual-ABI semantic
    # projection. The frozen owner gains no transfer table, allocation, CC
    # repair or Dst interpretation; all reprojection stays in BoundContract.
    "llvm/lib/Target/RISCV/RISCVTensixLowering.cpp": (
        9,
        "33103d335916b8dc6fa24374de9d806e0af5fac2369df5215dceed55afe548bd",
    ),
    "llvm/lib/Target/RISCV/RISCVTargetMachine.cpp": (
        9,
        "a1327049553da10da684d14169233e10aa8a2d1afc78c9f9ac033d6ae3be76fb",
    ),
    "llvm/lib/Target/RISCV/RISCVTensixExplicitReplay.cpp": (
        1698,
        "c34d9b990ad2ed9364fb94f9e07b24dac412077f4d96a0156b188b9d438f9dac",
    ),
    "llvm/lib/Target/RISCV/RISCVTensixExplicitReplay.h": (
        268,
        "3c42a03165508ddca9e45392cded79192dcb9f5a9bf20c02bcdbec3e14e7e493",
    ),
    "llvm/lib/Target/RISCV/RISCVTensixHazards.cpp": (
        31,
        "8b8dd840232b41157a7e4f8984b7fad84a9573ebfe98700083ad3e65e783bd4a",
    ),
    # Decode the seven-bit Wormhole fields at native reception. The existing
    # sequence graph remains frozen until its transitional consumers leave.
    "llvm/include/llvm/Target/RISCV/RISCVTensixMOP.h": (
        41,
        "7ec4ade6f7c7fd9d05ac427727c189342e92b6a86d2484c43cb72f84ec96862d",
    ),
    "llvm/lib/Target/RISCV/RISCVTensixReplay.cpp": (
        181,
        "11ae53605bc92a575d81907355df989c0ecb9768674679bb6c9329f87fd2d246",
    ),
    "llvm/lib/Target/RISCV/RISCVTensixReplay.h": (
        2,
        "22442137897417e86078576ac8e21aa45a2163201fdc1245cea6a83d31e5f65b",
    ),
}

MIDDLE_END_OWNER_PATTERN = (
    r"(?:SFPU|Tensix|LReg)\w*(?:RegisterAlloc|RegAlloc|Allocator|"
    r"CCRepair|CCCleanup|AutoReplay|MOPSequence)"
)
FORBIDDEN_MIDDLE_END_PATH_RE = re.compile(
    r"(ContentsGroup|RegisterContents|TensixMOPSequence|TensixReplay|"
    r"TensixOrdinary|SFPUExtra|CopyCCCleanup|SFPUPhysicalIngress|"
    r"llvm/(?:include/llvm|lib)/Support/.*Tensix|"
    + MIDDLE_END_OWNER_PATTERN
    + r")"
)

LEGACY_BOUND_SYMBOL_RE = re.compile(
    r"(?:(?:int_)?riscv_tt_bound_|llvm\.riscv\.tt\.bound\.)"
)
LEGACY_ANY_TT_SYMBOL_RE = re.compile(
    r"(?:(?:int_)?riscv_tt_|llvm\.riscv\.tt\.)"
)
VIRTUAL_SFPU_CARRIER_RE = re.compile(r"(?:\bv32i32\b|<32\s*x\s*i32>)")
MIDDLE_END_CODE_RE = re.compile(
    r"ContentsGroup|RISCVRegisterContents|TensixMOPSequence|"
    r"RISCVTensixReplay(?:Selection|Template)?|"
    r"RISCVTensixCopyCCCleanup|RISCVTensixSFPUPhysicalIngress|"
    + MIDDLE_END_OWNER_PATTERN
)
LLVM_CODE_ROOTS = ("llvm/include/", "llvm/lib/")
MIDDLE_END_CODE_ROOTS = LLVM_CODE_ROOTS + ("mlir/include/", "mlir/lib/")
LLVM_INTRINSIC_ROOTS = LLVM_CODE_ROOTS + ("llvm/test/", "llvm/unittests/")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True)


def git_commit_exists(root, rev):
    proc = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{rev}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def owned(path):
    return "bosc" in path.lower()


def is_test_path(path):
    return (
        path.startswith("llvm/test/")
        or path.startswith("mlir/test/")
        or "/unittests/" in f"/{path}"
    )


def candidate_source_files(root):
    listed = git(
        root, "ls-files", "--cached", "--others", "--exclude-standard"
    ).splitlines()
    seen = set()
    result = []
    for path in listed:
        if path not in seen and (root / path).is_file():
            seen.add(path)
            result.append(path)
    return result


def files_at(root, ref):
    return set(git(root, "ls-tree", "-r", "--name-only", "-z", ref).split("\0"))


def added_source_lines(root, ref, all_files, baseline_files):
    diff = git(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--unified=0",
        ref,
        "--",
        *MIDDLE_END_CODE_ROOTS,
    )
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and line.startswith("+") and not line.startswith("+++"):
            yield path, line[1:]
    # Git diff omits untracked files; a familiar filename grants no exemption.
    for path in all_files:
        if path in baseline_files or not path.startswith(MIDDLE_END_CODE_ROOTS):
            continue
        for line in (
            (root / path)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        ):
            yield path, line


def is_extraction_line(root, path, line, allowed_lines):
    return (
        allowed_lines.get(path) == line
        and (root / path).read_text(encoding="utf-8").splitlines().count(line)
        == 1
    )


def reviewed_extraction_patches(root, ref, approved):
    if not approved:
        return set()
    diff = git(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--unified=0",
        ref,
        "--",
        *approved,
    )
    additions = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and line.startswith("+") and not line.startswith("+++"):
            additions.setdefault(path, []).append(line[1:])
    # A newly introduced native header has no tracked diff yet. Its complete
    # source is the added sequence; the same count/digest and owner checks apply.
    for path in git(
        root, "ls-files", "--others", "--exclude-standard", "--", *approved
    ).splitlines():
        additions[path] = (root / path).read_text(encoding="utf-8").splitlines()
    accepted = set()
    for path, lines in additions.items():
        digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
        if approved.get(path) == (len(lines), digest):
            accepted.add(path)
    return accepted


def check_new_middle_end_code(
    root, ref, all_files, baseline_files, allowed_lines, reviewed_patches=()
):
    violations = set()
    for path, line in added_source_lines(root, ref, all_files, baseline_files):
        if path in reviewed_patches:
            continue
        if is_extraction_line(root, path, line, allowed_lines):
            continue
        if MIDDLE_END_CODE_RE.search(line) or (
            VIRTUAL_SFPU_CARRIER_RE.search(line)
            and ("Tensix" in path or re.search(r"SFPU|Tensix", line))
        ):
            violations.add(path)
    return [
        f"Forbidden new LLVM middle-end code vs {ref[:12]}: {path}"
        for path in sorted(violations)
    ]


def diff_has_added_lines(root, ref, paths, allowed_lines, reviewed_patches=()):
    existing = [p for p in paths]
    if not existing:
        return []
    out = git(
        root,
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--unified=0",
        ref,
        "--",
        *existing,
    )
    violating = []
    current = None
    for line in out.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            if current in reviewed_patches:
                continue
            if current and is_extraction_line(
                root, current, line[1:], allowed_lines
            ):
                continue
            if current and current not in violating:
                violating.append(current)
    return violating


def is_retired_bound_rejection_test(path, source):
    """Keep one executable ABI rejection witness, never a legacy caller."""
    if path != "llvm/test/CodeGen/RISCV/tt-sfpu-retired.ll":
        return False
    symbol = "llvm.riscv.tt.bound.sfpnop"
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    prefix = "; RUN: not llc -mtriple=riscv32"
    suffix = " %s -o /dev/null 2>&1 | FileCheck %s"
    architecture_runs = {
        prefix + f" -mattr=+xtttensix{arch}" + suffix for arch in ("wh", "bh")
    }
    allowed_runs = architecture_runs | {
        prefix + suffix,
        prefix + " -enable-new-pm" + suffix,
    }
    # lit finds directives anywhere in a line. No skip/XFAIL/alternate RUN may
    # disable the witness; exact commands also prevent overriding -mattr.
    directive = re.compile(
        r"\b(?:RUN:|XFAIL:|REQUIRES:|UNSUPPORTED:|ALLOW_RETRIES:|"
        r"END\.|DEFINE:|REDEFINE:)"
    )
    if any(
        directive.search(line) and line not in allowed_runs for line in lines
    ):
        return False
    runs = [line for line in lines if line.startswith("; RUN:")]
    if not architecture_runs.issubset(runs):
        return False
    check = f"; CHECK: unsupported Tensix intrinsic ABI: {symbol}"
    if [line for line in lines if line.startswith("; CHECK")] != [check]:
        return False
    if [line for line in lines if not line.startswith(";")] != [
        f"declare void @{symbol}()",
        'define void @retired() "tensix-executor"="trisc1" {',
        f"call void @{symbol}()",
        "ret void",
        "}",
    ]:
        return False
    allowed = {check, f"declare void @{symbol}()", f"call void @{symbol}()"}
    return all(
        line in allowed
        for line in lines
        if LEGACY_ANY_TT_SYMBOL_RE.search(line)
    )


def scan_files_for_regex(
    root,
    paths,
    pattern,
    label,
    *,
    include_tests=False,
    allow_retired_bound_rejection=False,
):
    hits = []
    for path in sorted(paths):
        if path == SELF or (not include_tests and is_test_path(path)):
            continue
        full = root / path
        if not full.is_file():
            continue
        source = full.read_text(encoding="utf-8", errors="replace")
        if allow_retired_bound_rejection and is_retired_bound_rejection_test(
            path, source
        ):
            continue
        for lineno, line in enumerate(source.splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{label} in {path}:{lineno}")
                break
    return hits


def source_text(root, path):
    full = root / path
    if not full.is_file():
        return ""
    text = full.read_text(encoding="utf-8", errors="replace")
    # Preserve quoted strings while discarding C++/TableGen comments.
    text = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
        lambda match: match[0] if match[0].startswith('"') else " ",
        text,
        flags=re.DOTALL,
    )
    if path.endswith("CMakeLists.txt"):
        text = re.sub(r"#[^\n]*", "", text)
    return text


def check_intrinsics_tt_contract(root):
    errors = []
    td_path = root / "llvm/include/llvm/IR/IntrinsicsTT.td"
    if not td_path.is_file():
        errors.append(
            "Missing physical TT intrinsic table: "
            "llvm/include/llvm/IR/IntrinsicsTT.td"
        )
        return errors
    text = source_text(root, "llvm/include/llvm/IR/IntrinsicsTT.td")
    if not re.search(r'TargetPrefix\s*=\s*"tt"', text):
        errors.append('IntrinsicsTT.td missing TargetPrefix = "tt" declaration')
    if not re.search(r"\bdef(?:m)?\s+int_tt_wh_\w+", text):
        errors.append(
            "IntrinsicsTT.td missing Wormhole B0 "
            "int_tt_wh_* intrinsic definitions"
        )
    if not re.search(r"\bdef(?:m)?\s+int_tt_bh_\w+", text):
        errors.append(
            "IntrinsicsTT.td missing Blackhole "
            "int_tt_bh_* intrinsic definitions"
        )
    if not re.search(
        r'\binclude\s+"llvm/IR/IntrinsicsTT\.td"',
        source_text(root, "llvm/include/llvm/IR/Intrinsics.td"),
    ):
        errors.append("Missing active IntrinsicsTT.td include in Intrinsics.td")
    if not re.search(
        r"\btablegen\s*\(\s*LLVM\s+IntrinsicsTT\.h\s+"
        r"-gen-intrinsic-enums\s+-intrinsic-prefix=tt\s*\)",
        source_text(root, "llvm/include/llvm/IR/CMakeLists.txt"),
    ):
        errors.append("Missing IntrinsicsTT.h TableGen enum producer")
    if not re.search(
        r'#\s*include\s+"llvm/IR/IntrinsicsTT\.h"',
        source_text(root, "llvm/lib/IR/Intrinsics.cpp"),
    ):
        errors.append(
            "Missing generated IntrinsicsTT.h include in Intrinsics.cpp"
        )
    return errors


def check_tensix_stage(root, stage, freeze_ref, merge_base):
    errors = []
    all_files = candidate_source_files(root)

    if stage in ("l0-frozen", "l-bind"):
        eff_freeze = freeze_ref or DEFAULT_FREEZE_REF
        if not git_commit_exists(root, eff_freeze):
            return [f"Freeze reference commit not found: {eff_freeze}"]
        baseline_files = files_at(root, eff_freeze)
        frozen_targets = list(FROZEN_MIDDLE_END_LLVM_FILES) + list(
            FROZEN_MLIR_SYMBOL_TABLE_FILES
        )
        if stage == "l0-frozen":
            frozen_targets.append(
                "llvm/include/llvm/IR/IntrinsicsRISCVTensixBound.td"
            )
        frozen_allowlist = set(frozen_targets) & baseline_files
        for path in all_files:
            if path == SELF or is_test_path(path):
                continue
            if (
                path in frozen_allowlist
                or path in ALLOWED_EXPLICIT_TENSIX_FILES
            ):
                continue
            if FORBIDDEN_MIDDLE_END_PATH_RE.search(path):
                errors.append(
                    "Forbidden new LLVM middle-end Tensix file "
                    f"({stage}): {path}"
                )
        for path in sorted(frozen_allowlist):
            if not (root / path).is_file():
                errors.append(
                    f"Frozen transitional file missing before L1: {path}"
                )
        extraction_lines = L_BIND_EXTRACTION_LINES if stage == "l-bind" else {}
        extraction_patches = reviewed_extraction_patches(
            root,
            eff_freeze,
            L_BIND_EXTRACTION_PATCHES if stage == "l-bind" else {},
        )
        grown_paths = diff_has_added_lines(
            root,
            eff_freeze,
            frozen_targets,
            extraction_lines,
            extraction_patches,
        )
        for path in grown_paths:
            errors.append(
                "Frozen transitional file gained new lines "
                f"vs {eff_freeze[:12]}: {path}"
            )
        if stage == "l0-frozen":
            changed_paths = git(
                root, "diff", "--name-only", eff_freeze, "--", *frozen_targets
            ).splitlines()
            for path in changed_paths:
                if path not in grown_paths and (root / path).is_file():
                    errors.append(
                        "Frozen transitional file changed "
                        f"vs {eff_freeze[:12]}: {path}"
                    )
        errors.extend(
            check_new_middle_end_code(
                root,
                eff_freeze,
                all_files,
                baseline_files,
                extraction_lines,
                extraction_patches,
            )
        )

    if stage in ("l-bind", "l1-slimmed"):
        errors.extend(check_intrinsics_tt_contract(root))
        llvm_code_files = [
            p for p in all_files if p.startswith(LLVM_INTRINSIC_ROOTS)
        ]
        errors.extend(
            scan_files_for_regex(
                root,
                llvm_code_files,
                LEGACY_BOUND_SYMBOL_RE,
                "Legacy @llvm.riscv.tt.bound.* symbol remains",
                include_tests=True,
                allow_retired_bound_rejection=True,
            )
        )

    if stage == "l1-slimmed":
        eff_mb = merge_base or DEFAULT_MERGE_BASE
        if not git_commit_exists(root, eff_mb):
            return errors + [f"Merge-base reference commit not found: {eff_mb}"]
        must_be_absent = list(FROZEN_MIDDLE_END_LLVM_FILES) + list(
            L1_DELETED_EXTRA_LLVM_FILES
        )
        for path in must_be_absent:
            if (root / path).exists():
                errors.append(
                    f"Transitional LLVM file must be deleted in L1: {path}"
                )
        for path in all_files:
            if path == SELF or is_test_path(path):
                continue
            if path in ALLOWED_EXPLICIT_TENSIX_FILES:
                continue
            if FORBIDDEN_MIDDLE_END_PATH_RE.search(path):
                errors.append(
                    "Forbidden LLVM middle-end Tensix path remains in L1: "
                    f"{path}"
                )
        llvm_code_files = [
            p for p in all_files if p.startswith(LLVM_INTRINSIC_ROOTS)
        ]
        errors.extend(
            scan_files_for_regex(
                root,
                llvm_code_files,
                LEGACY_ANY_TT_SYMBOL_RE,
                "Legacy @llvm.riscv.tt.* symbol remains in L1",
                include_tests=True,
                allow_retired_bound_rejection=True,
            )
        )
        tensix_code_files = [
            p
            for p in llvm_code_files
            if "Tensix" in p or p.endswith("IntrinsicsTT.td")
        ]
        errors.extend(
            scan_files_for_regex(
                root,
                tensix_code_files,
                VIRTUAL_SFPU_CARRIER_RE,
                "Virtual v32i32 SFPU carrier remains in Tensix backend",
            )
        )
        errors.extend(
            check_new_middle_end_code(
                root, eff_mb, all_files, files_at(root, eff_mb), {}
            )
        )
        sym_diff = git(
            root,
            "diff",
            "--name-only",
            eff_mb,
            "--",
            *FROZEN_MLIR_SYMBOL_TABLE_FILES,
        ).splitlines()
        for path in sym_diff:
            errors.append(
                "MLIR SymbolTable fork remains "
                f"vs merge-base {eff_mb[:12]}: {path}"
            )

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bosc", choices=("absent", "present"), required=True)
    parser.add_argument("--baseline")
    parser.add_argument(
        "--tensix-stage",
        choices=("l0-frozen", "l-bind", "l1-slimmed"),
        help="Enforce staged Tensix LLVM middle-end freeze/slimming rules.",
    )
    parser.add_argument(
        "--freeze-ref",
        help=(
            "Git commit ref used by l0-frozen/l-bind to freeze "
            "transitional files."
        ),
    )
    parser.add_argument(
        "--merge-base",
        help=(
            "Upstream LLVM merge-base used by l1-slimmed "
            "to verify clean mlir SymbolTable."
        ),
    )
    parser.add_argument(
        "--repo-root",
        help="Override the llvm-project repository root for testing.",
    )
    args = parser.parse_args()
    root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[3]
    )
    tracked = git(root, "ls-files").splitlines()
    errors = []
    active = [p for p in tracked if (root / p).is_file()]
    dedicated = [p for p in active if owned(p)]
    if args.bosc == "absent":
        errors += [f"BOSC-owned path remains: {p}" for p in dedicated]
        for path in sorted(SHARED):
            full = root / path
            if not full.is_file():
                if args.tensix_stage == "l1-slimmed" and path.endswith(
                    "RISCVRegisterContents.cpp"
                ):
                    continue
                errors.append(f"Expected shared file missing: {path}")
                continue
            for number, line in enumerate(
                full.read_text(encoding="utf-8", errors="replace").splitlines(),
                1,
            ):
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

    base_protected = [
        "llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp",
        "mlir/lib/Dialect/GPU/IR/GPUDialect.cpp",
        "mlir/lib/Dialect/LLVMIR/IR/NVVMDialect.cpp",
        "llvm/lib/Target/RISCV/RISCVInstrInfoV.td",
    ]
    if args.tensix_stage == "l1-slimmed":
        base_protected.extend(
            [
                "llvm/include/llvm/IR/IntrinsicsTT.td",
                "llvm/lib/Target/RISCV/RISCVTensixBoundLowering.cpp",
                "llvm/lib/Target/RISCV/RISCVTensixExplicitReplay.cpp",
            ]
        )
    elif args.tensix_stage == "l-bind":
        base_protected.extend(
            [
                "llvm/include/llvm/IR/IntrinsicsTT.td",
                "llvm/lib/Target/RISCV/RISCVTensixBoundLowering.cpp",
            ]
        )
    else:
        base_protected.extend(
            [
                "llvm/lib/Target/RISCV/RISCVTensixLowering.cpp",
                "llvm/include/llvm/IR/IntrinsicsRISCVTensix.td",
            ]
        )
    for path in base_protected:
        if not (root / path).is_file():
            errors.append(f"Preserved implementation missing: {path}")

    if args.tensix_stage:
        errors.extend(
            check_tensix_stage(
                root, args.tensix_stage, args.freeze_ref, args.merge_base
            )
        )

    if args.baseline:
        baseline = git(
            root, "ls-tree", "-r", "--name-only", args.baseline
        ).splitlines()
        changed = git(
            root, "diff", "--name-only", args.baseline, "--"
        ).splitlines()
        for path in changed:
            if path != SELF and path not in SHARED and not owned(path):
                errors.append(f"Unrelated baseline change: {path}")
        for path in baseline:
            if path == SELF or path in SHARED or owned(path):
                continue
            if not (root / path).is_file() and not (root / path).is_symlink():
                errors.append(f"Unrelated baseline file missing: {path}")
        if args.bosc == "present":
            errors += [
                f"Restored tree differs from baseline: {p}"
                for p in changed
                if p != SELF
            ]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"FAIL: {len(errors)} partition violations", file=sys.stderr)
        return 1
    stage_msg = (
        f"; tensix-stage={args.tensix_stage}" if args.tensix_stage else ""
    )
    print(
        f"PASS: BOSC {args.bosc}{stage_msg}; "
        "preserved Tensix, RVV, GPU and NVPTX sources"
    )
    if args.baseline:
        print(f"PASS: unrelated files unchanged from {args.baseline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

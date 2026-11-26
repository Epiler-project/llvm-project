"""Reject corrupted state replay effects through the actual LLVM receiver."""

import pathlib
import re
import subprocess
import sys


source = pathlib.Path(sys.argv[1]).read_text()
output = pathlib.Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
llc = sys.argv[3]
cases = 0


def reject(name, old, replacement, diagnostic):
    global cases
    assert source.count(old) == 1, (name, "ambiguous mutation target")
    path = output / f"{name}.mir"
    path.write_text(source.replace(old, replacement))
    result = subprocess.run(
        [llc, "-mtriple=riscv32", "-mattr=+xtttensixbh",
         "-run-pass=riscv-tensix-bound-verify", str(path), "-o", "/dev/null"],
        capture_output=True, text=True)
    assert result.returncode != 0, (name, "malformed effects accepted")
    diagnostics = (diagnostic,) if isinstance(diagnostic, str) else diagnostic
    assert all(message in result.stderr for message in diagnostics), (name, result.stderr)
    cases += 1


for route in ("structured", "raw"):
    for kind in ("encc", "creg", "lane"):
        name = f"state_{kind}_{route}_mode0"
        block = re.search(r"^name:\s*" + name + r"\s*$.*?^\.\.\.", source, re.M | re.S)
        assert block, name
        lines = [line for line in block[0].splitlines()
                 if "PseudoTTExplicitSFPUReplay 0, 0," in line]
        for index, line in enumerate(lines):
            # Include the function prefix in the unique replacement target:
            # several authored executes intentionally have identical effects.
            line_at = block[0].index(line)
            if index:
                prior = block[0].index(lines[index - 1])
                line_at = block[0].index(line, prior + len(lines[index - 1]))
            prefix = block[0][:line_at]
            old = prefix + line
            operands = re.findall(r", implicit(?:-def)?(?: dead| killed)? \$tt_\w+", line)
            assert operands
            for operand in operands:
                field = operand.split("$")[1]
                role = "def" if "implicit-def" in operand else "use"
                label = f"{name}-{index}-{field}-{role}"
                # Fixed descriptor operands are rejected by the MIR parser;
                # template-dependent state remains the bound verifier's job.
                fixed = field == "tt_issue" or (field == "tt_config" and role == "use")
                missing = ("missing implicit register operand 'implicit" +
                           ("-def" if role == "def" else "") +
                           " $" + field + "'" if fixed else
                           "execution physical effects differ from its reaching words")
                reject(label + "-missing", old, prefix + line.replace(operand, "", 1),
                       missing)
                reject(label + "-duplicate", old, prefix + line.replace(operand, operand * 2, 1),
                       "execution physical effects differ from its reaching words")
            if kind == "lane" and index == 1:
                # Reset must never acquire a fictional staging-register read.
                assert "$tt_l0" not in line
                reject(name + "-reset-l0", old,
                       prefix + line.replace(" :: ", ", implicit $tt_l0 :: "),
                       "execution physical effects differ from its reaching words")

        payload = next(line for line in block[0].splitlines()
                       if "PseudoTTSFPURecordWord " in line)
        old = block[0][:block[0].index(payload)] + payload
        reject(name + "-payload-cc", old, old + ", implicit $tt_cc",
               "record-only word must have only issue ordering effects")

    # A native mode-1 state word cannot silently drop an arbitrary MMO.
    # CONFIG has no memory descriptor, so MIR's mandatory machine verifier
    # rejects both invented flags before the replay verifier is invoked.
    name = f"state_creg_{route}_mode1"
    block = re.search(r"^name:\s*" + name + r"\s*$.*?^\.\.\.", source, re.M | re.S)
    line = next(line for line in block[0].splitlines() if "TTSFPCONFIGC11 " in line)
    old = block[0][:block[0].index(line)] + line
    reject(name + "-body-mmo", old, old + " :: (volatile load store (s32))",
           ("Bad machine code: Missing mayLoad flag",
            "Bad machine code: Missing mayStore flag"))

print(f"State replay receiver rejected {cases} malformed effect cases.")

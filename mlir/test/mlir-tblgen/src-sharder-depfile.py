# RUN: %python %s mlir-src-sharder %t

"""Check source-sharder dependency output and unchanged-output behavior."""

import os
from pathlib import Path
import subprocess
import sys


tool, directory = sys.argv[1:]
root = Path(directory)
root.mkdir(parents=True, exist_ok=True)
source = root / "input.cpp"
output = root / "output.cpp"
depfile = root / "output.d"
source.write_text("// source for shard generation\n")


def run(*args, stdin=None):
    return subprocess.run(
        [tool, "-op-shard-index=3", *map(str, args)],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
    )


run(source, "-o", output, "-d", depfile, "--write-if-changed")
assert output.read_text() == "#define GET_OP_DEFS_3\n" + source.read_text()
expected_dependencies = f"{output}: {source}\n"
assert depfile.read_text() == expected_dependencies, depfile.read_text()

# A depfile with no prerequisites is discarded by CMake's depfile transform.
# Recreate the depfile even when --write-if-changed keeps the old output.
old_time = 1_000_000_000_000_000_000
os.utime(output, ns=(old_time, old_time))
preserved_time = output.stat().st_mtime_ns
depfile.unlink()
run(source, "-o", output, "-d", depfile, "--write-if-changed")
assert depfile.read_text() == expected_dependencies
assert output.stat().st_mtime_ns == preserved_time

# A changed input must still regenerate the actual shard.
source.write_text("// changed source\n")
run(source, "-o", output, "-d", depfile, "--write-if-changed")
assert output.read_text() == "#define GET_OP_DEFS_3\n" + source.read_text()
assert output.stat().st_mtime_ns != preserved_time

# stdin has no filesystem dependency; stdout continues to work without -d.
run("-", "-o", output, "-d", depfile, stdin=source.read_text())
assert depfile.read_text() == f"{output}:\n"
assert run(source).stdout == "#define GET_OP_DEFS_3\n" + source.read_text()
invalid = subprocess.run(
    [tool, str(source), "-d", str(depfile)], capture_output=True, text=True
)
assert invalid.returncode != 0
assert "the option -d must be used together with -o" in invalid.stderr

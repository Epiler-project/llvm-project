"""Corrupt one prepared template contract without depending on opcode numbers."""

import pathlib
import sys


source, destination, mutation = sys.argv[1:]
lines = pathlib.Path(source).read_text().splitlines(keepends=True)
executes = [i for i, line in enumerate(lines)
            if line.lstrip().startswith("PseudoTTReplayTemplateExecute 302,")]
assert len(executes) == 1
execute = executes[0]
line = lines[execute]
if mutation == "missing-read":
    assert line.count("implicit $tt_l1") == 1
    lines[execute] = line.replace(", implicit $tt_l1", "")
elif mutation == "extra-read":
    lines[execute] = line.rstrip() + ", implicit $tt_l2\n"
elif mutation == "duplicate-read":
    lines[execute] = line.rstrip() + ", implicit $tt_l1\n"
elif mutation == "missing-def":
    assert line.count("implicit-def $tt_l0") == 1
    lines[execute] = line.replace(", implicit-def $tt_l0", "")
elif mutation == "duplicate-def":
    lines[execute] = line.rstrip() + ", implicit-def $tt_l0\n"
else:
    words = [i for i, line in enumerate(lines)
             if line.lstrip().startswith("PseudoTTReplayTemplateWord 301,")]
    assert len(words) == 1
    word = words[0]
    if mutation == "record-effects":
        lines[word] = lines[word].rstrip() + ", implicit-def $tt_l0\n"
    elif mutation == "unprepared":
        lines[word] = (
            "    $tt_l0 = TTSFPMOVAll $tt_l1, implicit $tt_config, "
            "implicit $tt_issue, implicit-def $tt_issue\n"
        )
    elif mutation == "orphan":
        begins = [i for i, line in enumerate(lines)
                  if line.lstrip().startswith("PseudoTTReplayTemplateBegin 301,")]
        assert len(begins) == 1 and begins[0] < word
        payload = lines.pop(word)
        lines.insert(begins[0], payload)
    else:
        raise AssertionError("unknown mutation: " + mutation)
pathlib.Path(destination).write_text("".join(lines))

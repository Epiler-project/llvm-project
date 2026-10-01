"""Observe changing capture epochs using the shared scalar/ISA test model."""
import importlib.util
import pathlib
import sys

owner = pathlib.Path(__file__).with_name("tensix-replay-dynamic-dst-observer.py")
spec = importlib.util.spec_from_file_location("dst_replay_observer", owner)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)
assembly = pathlib.Path(sys.argv[1]).read_text()
for mode in (0, 1):
    for conditional in (False, True):
        for base in (32, 1008):
            for count in (0, 1, 4):
                observer.observe(assembly, mode, base, 256, count,
                                 epochs=conditional)

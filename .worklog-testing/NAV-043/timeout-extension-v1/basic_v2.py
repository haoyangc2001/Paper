"""Correct synthetic displacement history so stagnation does not mask timeout."""
from pathlib import Path
source = Path(__file__).with_name('basic.py').read_text()
old = "mon.history[0] = [(deadline-1., initial.x, initial.y)]"
assert old in source
source = source.replace(old, "mon.history[0] = [(deadline-1., initial.x-1., initial.y)]")
exec(compile(source, str(Path(__file__)), 'exec'))

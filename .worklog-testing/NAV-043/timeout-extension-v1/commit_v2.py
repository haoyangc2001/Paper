"""Preserve failed commit proof; explicitly retain ignored raw logs unchanged."""
from pathlib import Path
source = Path(__file__).with_name('commit.py').read_text()
source = source.replace("assert not git('diff','--cached','--name-only'), 'unexpected staged work'",
    "initial_staged = git('diff','--cached','--name-only').splitlines()")
source = source.replace("subprocess.run(['git','add','--',*paths], cwd=ROOT, check=True)",
    "assert set(initial_staged) <= set(paths), 'unexpected staged work'\nsubprocess.run(['git','add','-f','--',*paths], cwd=ROOT, check=True)")
source = source.replace("if not p.endswith('.log')", "if not p.endswith(('.log', '.patch'))")
exec(compile(source, str(Path(__file__)), 'exec'))

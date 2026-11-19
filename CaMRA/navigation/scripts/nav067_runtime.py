"""NAV-067 local source binding and durable failure reporting.

This module never edits the shared Isaac environment or frozen pilot manifest.
"""
from datetime import datetime
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def bind_local_locomotion(nav):
    source = (Path(nav).parent/'locomotion/source/unitree_rl_lab').resolve()
    expected = source/'unitree_rl_lab/__init__.py'
    if not expected.is_file():
        raise ValueError('fixed local locomotion source is absent')
    existing = sys.modules.get('unitree_rl_lab')
    if existing is not None and Path(existing.__file__).resolve() != expected:
        raise ValueError('a different unitree_rl_lab is already imported')
    sys.path.insert(0,str(source))
    importlib.invalidate_caches()
    spec = importlib.util.find_spec('unitree_rl_lab')
    if spec is None or Path(spec.origin).resolve() != expected:
        raise ValueError('local locomotion source binding failed')
    return {'source':str(source),'origin':spec.origin}


def persist(path, value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def guarded_application(out, factory, work):
    """Persist errors before Kit cleanup, which can terminate the interpreter."""
    app = None
    try:
        app = factory()
        persist(Path(out)/'APP_READY.json',{'at':datetime.now().astimezone().isoformat()})
        return work()
    except BaseException as error:
        detail = traceback.format_exc()
        persist(Path(out)/'ERROR.json',{'at':datetime.now().astimezone().isoformat(),
            'error_type':type(error).__name__,'message':str(error),'traceback':detail})
        print(detail,file=sys.stderr,flush=True)
        raise
    finally:
        if app is not None:
            app.close()

"""Small startup/error-path checks; no Isaac app or physical episode."""
import json
from pathlib import Path
import sys

import pytest

import nav067_runtime as R


def test_exception_persisted_before_kit_close_can_exit(tmp_path):
    class App:
        def close(self):
            error=json.loads((tmp_path/'ERROR.json').read_text())
            assert error['error_type']=='ModuleNotFoundError'
            assert 'missing-local-module' in error['traceback']
            raise SystemExit(0)
    def broken():
        raise ModuleNotFoundError('missing-local-module')
    with pytest.raises(SystemExit):
        R.guarded_application(tmp_path,App,broken)
    assert (tmp_path/'APP_READY.json').is_file()
    assert not (tmp_path/'FINISHED.json').exists()


def test_app_startup_exception_persisted(tmp_path):
    def factory():
        raise RuntimeError('startup failed')
    with pytest.raises(RuntimeError,match='startup failed'):
        R.guarded_application(tmp_path,factory,lambda:None)
    assert json.loads((tmp_path/'ERROR.json').read_text())['message']=='startup failed'
    assert not (tmp_path/'APP_READY.json').exists()


def test_normal_work_closes_app_and_returns_result(tmp_path):
    closed=[]
    class App:
        def close(self): closed.append(True)
    assert R.guarded_application(tmp_path,App,lambda:7)==7
    assert closed==[True]
    assert not (tmp_path/'ERROR.json').exists()


def test_missing_local_source_refused(tmp_path):
    with pytest.raises(ValueError,match='absent'):
        R.bind_local_locomotion(tmp_path/'navigation')


def test_unapproved_recovery_refused_before_output_or_app(tmp_path,monkeypatch):
    import evaluate_navigation_v3 as E
    monkeypatch.setattr(E,'AMENDMENT',tmp_path/'not-approved.json')
    monkeypatch.setattr(E,'read_manifest',lambda:({},{},{}))
    with pytest.raises(ValueError,match='amendment is absent'):
        E.recovery_context('L1_empty')
    assert not list(tmp_path.iterdir())

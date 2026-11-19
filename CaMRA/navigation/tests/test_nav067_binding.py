"""Basic refusal checks for physical quota and frozen-source bindings."""
import hashlib
import json

import pytest
import numpy as np
import torch

import nav067_sources as S


def pin(path, text):
    path.write_text(text)
    path.with_suffix(path.suffix+'.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_legacy_quota_is_read_without_evaluating_other_fields(tmp_path):
    path=tmp_path/'protocol.json'
    pin(path,'{"total":4 * 6 * 6,"physical_pilot":{"clean_6_episodes":6,"note":"old"}}')
    assert S.original_pilot_quota(path)==6


@pytest.mark.parametrize('content',[
    '{"physical_pilot":{"clean_6_episodes":7,"note":"changed"}}',
    '{"physical_pilot":{"clean_6_episodes":6,"note":"a"},"physical_pilot":{"clean_6_episodes":6,"note":"b"}}',
    '{"clean_6_episodes":6}',
])
def test_changed_or_ambiguous_quota_is_refused(tmp_path,content):
    path=tmp_path/'protocol.json'
    pin(path,content)
    with pytest.raises(ValueError,match='not unique'):
        S.original_pilot_quota(path)


def test_changed_protocol_is_refused_even_if_quota_still_six(tmp_path):
    path=tmp_path/'protocol.json'
    pin(path,'{"physical_pilot":{"clean_6_episodes":6,"note":"old"}}')
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash mismatch'):
        S.original_pilot_quota(path)


def test_sealed_manifest_refuses_increased_episode_count(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'OUTPUT',tmp_path)
    monkeypatch.setattr(S,'read_sources',lambda:({'selected_candidates':[]},{},{},{'id':'fixed'}))
    manifest={'binding':{'id':'fixed'},'candidates':[],'new_sources':{},
              'limits':{'episodes':7,'episodes_per_level':2,'horizon_s':45.,
                        'warmup_s':1.,'optimizer_updates':0,'W_queries':0}}
    pin(tmp_path/'manifest.json',json.dumps(manifest))
    with pytest.raises(ValueError,match='quota/objective mismatch'):
        S.read_manifest()


def test_sealed_manifest_refuses_substituted_candidate(tmp_path,monkeypatch):
    monkeypatch.setattr(S,'OUTPUT',tmp_path)
    monkeypatch.setattr(S,'read_sources',lambda:({'selected_candidates':[{'id':'fixed'}]},{},{},{}))
    pin(tmp_path/'manifest.json',json.dumps({'binding':{},'candidates':[{'id':'replacement'}]}))
    with pytest.raises(ValueError,match='model binding mismatch'):
        S.read_manifest()


def test_physical_production_maps_actor_output_like_training():
    from evaluate_navigation_v2 import next_command

    class Monitor:
        def observation(self,i): return [0.]*41  # production encoder returns a list
        def set_commands(self,cmd): self.applied=cmd
    class Network:
        def act(self,obs,deterministic):
            assert obs.shape==(1,41) and deterministic
            return torch.tensor([[.5,1.,-1.]]),None,None

    monitor=Monitor()
    obs,action,command=next_command(Network(),monitor)
    assert obs.shape==(41,)
    np.testing.assert_allclose(command,[[.5,.4,-1.]])
    np.testing.assert_array_equal(monitor.applied,command)
    np.testing.assert_array_equal(action,[[.5,1.,-1.]])

"""Small CPU checks with synthetic critic outputs; no real W query/rollout."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
from nav067_sources import E, NAV, sha
from nav067_runtime import persist
from train_formal import verify_prereg
from prepare_nav065_sources import REGISTRATION
from nav043_budget import actual_start_env, decide, outcome
from camra_navigation.envs.physical_v2 import PhysicalState, SubstepNavigation
from camra_navigation.observation import field_slices
import numpy as np
import torch


class FakeNet:
    def __init__(self, vh=-.5): self.vh,self.observations=vh,[]
    def values(self,obs):
        self.observations.append(obs.numpy().copy())
        return torch.tensor([.4]),torch.tensor([self.vh])


class Basic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prereg,_=verify_prereg(REGISTRATION)
        cls.ctx=E.build_rollout_ctx('full')
        cls.spec=E.M.build('L2_single_detour',3001,cls.ctx['design'])
        cls.state=PhysicalState(cls.spec.start[0]+.02,cls.spec.start[1]-.01,.03,.04,-.02,.01,.3,
                                (math.cos(.015),0.,0.,math.sin(.015)))

    def test_actual_start_observation_matches_physical(self):
        for method in ('uncalibrated','full'):
            ctx=E.build_rollout_ctx(method)
            env=actual_start_env(self.spec,ctx,600.,self.prereg,self.state)
            monitor=SubstepNavigation(self.spec,ctx['geom'],ctx['cm'],[self.state],[600.],self.prereg)
            np.testing.assert_array_equal(env.observation[0],monitor.observation(0))
            self.assertEqual(env.pose[0].tolist(),[self.state.x,self.state.y,self.state.yaw])
            self.assertEqual(env._history[0],[(0.,self.state.x,self.state.y)])

    def test_decision_sealed_and_only_budget_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            trial=Path(directory); net=FakeNet()
            d=decide(net,self.spec,self.ctx,self.state,400.,self.prereg,trial)
            self.assertEqual(d['reason'],'selected')
            self.assertEqual(len(net.observations),5)
            zslice=field_slices('obs-v7')['z_ratio']
            mask=np.ones(net.observations[0].shape[1],dtype=bool);mask[zslice]=False
            for i,obs in enumerate(net.observations):
                np.testing.assert_array_equal(obs[:,mask],net.observations[0][:,mask])
                np.testing.assert_allclose(obs[:,zslice],d['trace'][i]['budget_J']/550,rtol=0,atol=1e-7)
            pin=json.loads((trial/'decision.pin.json').read_text())
            self.assertEqual(pin['sha256'],sha(trial/'decision.json'))
            with self.assertRaises(FileExistsError):
                persist(trial/'decision.json',{})
            self.assertEqual(pin['sha256'],sha(trial/'decision.json'))

    def test_rejection_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            d=decide(FakeNet(vh=.1),self.spec,self.ctx,self.state,400.,self.prereg,Path(directory))
            self.assertIsNone(d['selected_budget_J'])
            self.assertEqual(d['reason'],'no_feasible_grid_budget')

    def test_physical_budget_and_internal_cost_separate(self):
        r=outcome('goal_reached',0.,44.,510.,490.,500.)
        self.assertFalse(r['safe_timely_measured_budget_success'])
        self.assertTrue(r['internal_safe_budget_success'])
        self.assertEqual(r['measured_overspend_J'],10.)
        self.assertFalse(outcome('timeout',0.,45.,450.,440.,500.)['safe_timely_measured_budget_success'])
        self.assertFalse(outcome('goal_reached',.1,44.,450.,440.,500.)['safe_timely_measured_budget_success'])

    def test_protocol_and_upstream_preservation(self):
        reg=json.loads((NAV/'config/nav043_exploration_v1.json').read_text())
        self.assertEqual(sha(NAV/'config/nav043_exploration_v1.json'),
            (NAV/'config/nav043_exploration_v1.json.sha256').read_text().strip())
        self.assertEqual(reg['limits']['W_queries'],30)
        for path,digest in reg['sealed_hashes'].items():self.assertEqual(sha(path),digest,path)
        plan=json.loads((ROOT/'doc/plan/20261109_camra_navigation_implementation.json').read_text())
        self.assertEqual(sum(t['status'] for t in plan['tasks']),55)
        self.assertFalse(next(t for t in plan['tasks'] if t['id']=='NAV-043')['status'])


if __name__=='__main__':
    unittest.main(verbosity=2)

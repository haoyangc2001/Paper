"""Additional main-agent checks for preflight binding and one-shot launcher."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path('/home/caohy/repositories/Paper')
sys.path.insert(0,str(ROOT/'CaMRA/navigation/scripts'))
import nav043_matched_support_v5 as m


class Basic(unittest.TestCase):
    def test_missing_preflight_before_lock_or_child(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            with patch.object(m,'HERE',folder), patch.object(m,'approved',return_value={'output_root':str(folder/'output')}), \
                    patch.object(m.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaises(FileNotFoundError):
                    m.launch()
                self.assertEqual(list(folder.iterdir()),[])

    def test_successful_mock_launch_bound_to_preflight_then_refuses_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            here=folder/'here'
            here.mkdir()
            output=folder/'output'
            paths=[folder/name for name in ('proposal.json','approval.json','readiness.json')]
            for p in paths:
                p.write_text('{}\n')
            m.base.write_new(here/'preflight.json',dict(status='APPROVED_MATCHED_SUPPORT_PREFLIGHT_PASS',
                proposal_sha256=m.base.sha(paths[0]),approval_sha256=m.base.sha(paths[1]),readiness_sha256=m.base.sha(paths[2])))
            class Child:
                pid=98765
                def wait(self,timeout):
                    output.mkdir()
                    (output/'FINISHED.json').write_text('{}\n')
                    return 0
            with patch.object(m,'HERE',here), patch.object(m,'PROPOSAL',paths[0]), \
                    patch.object(m,'APPROVAL',paths[1]), patch.object(m,'READINESS',paths[2]), \
                    patch.object(m,'approved',return_value={'output_root':str(output)}), \
                    patch.object(m.subprocess,'Popen',return_value=Child()) as popen:
                self.assertEqual(m.launch(),0)
                self.assertEqual(m.base.load(here/'exit.json')['exit_code'],0)
                self.assertFalse(m.base.load(here/'exit.json')['retry_performed'])
                with self.assertRaisesRegex(ValueError,'already attempted'):
                    m.launch()
                popen.assert_called_once()

    def test_wrong_resource_approval_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            paths=[folder/name for name in ('proposal.json','readiness.json','approval.json')]
            def pinned(path,value):
                m.base.write_new(path,value)
                Path(str(path)+'.sha256').write_text(m.base.sha(path)+'\n')
            pinned(paths[0],dict(schema='nav043-matched-support-v5',status='PROPOSED',limits=m.LIMITS,
                                 output_root=str(folder/'output'),source_hashes={}))
            pinned(paths[1],dict(status='PREPARED_ONLY',proposal_sha256=m.base.sha(paths[0]),source_hashes={}))
            pinned(paths[2],dict(status='APPROVED',proposal_sha256=m.base.sha(paths[0]),
                readiness_sha256=m.base.sha(paths[1]),limits=dict(m.LIMITS,axis_solves=19),output_root=str(folder/'output')))
            with patch.object(m,'PROPOSAL',paths[0]),patch.object(m,'READINESS',paths[1]),patch.object(m,'APPROVAL',paths[2]):
                with self.assertRaisesRegex(ValueError,'exact new diagnostic'):
                    m.approved()

    def test_saved_wrong_coefficients_rejected_without_solving(self):
        x=np.vstack((np.eye(6),np.zeros((1,6))))
        y=np.zeros(7)
        record=dict(theta=[.1]*6,rows=7,rank=6,valid=True,
                    singular_values=np.linalg.svd(x,compute_uv=False).tolist(),residual_sum=[.06])
        with patch.object(np.linalg,'lstsq',side_effect=AssertionError('must not fit')):
            with self.assertRaisesRegex(ValueError,'normal equations'):
                m.verify_solution(x,y,record,'diagonal_v3')


if __name__=='__main__':
    m.context()
    unittest.main(verbosity=2)

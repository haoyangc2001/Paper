import copy
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from compute_metrics import ROOT,COLUMNS,advance,clearance,swept_clearance,calculate,read_json,read_raw,write_csv
from validate_data import validate,check_derived

class Consistency(unittest.TestCase):
    def setUp(self):
        self.s=read_json(ROOT/"data/scenario.json")
        self.s["obstacles"]=[{"center":[100,100],"radius_m":.2}]
        self.raw=np.array([[0,.3,0,0,.3,0,0],[.2,.3,0,0,.3,0,0]],float)
    def test_known_motion(self):
        np.testing.assert_allclose(advance([1,2,np.pi/2],[.5,0,.1],.2),[1,2.1,np.pi/2+.02],atol=1e-10)
        rows=calculate(self.raw,self.s,"rcppo")
        self.assertAlmostEqual(rows[-1]["x_exec_m"],.12)
        self.assertAlmostEqual(rows[-1]["J_exec"],.4*(3.6+3*.3**2))
    def test_geometry_and_sweep(self):
        self.s["obstacles"]=[{"center":[0,0],"radius_m":.3}]
        self.assertAlmostEqual(clearance([1,0,0],self.s),.5)
        self.assertAlmostEqual(swept_clearance([-1,0],[1,0],self.s),-.5)
    def test_budget_and_alpha(self):
        base=calculate(self.raw,self.s,"rcppo")
        self.s["initial_budget"]=base[-1]["J_exec"]
        self.assertEqual(calculate(self.raw,self.s,"rcppo")[-1]["budget_violation"],0)
        self.s["initial_budget"]-=.001
        self.assertEqual(calculate(self.raw,self.s,"rcppo")[-1]["budget_violation"],1)
        h=calculate(self.raw,self.s,"hd_mcra")[-1]
        self.assertAlmostEqual(h["J_charged"],1.18*h["J_nom"])
    def test_time_and_terminal(self):
        bad=self.raw.copy();bad[1,0]=-.2
        self.assertTrue(validate(bad,self.s,"rcppo",False)[0])
        self.s["goal_center"]=[.06,0];self.s["goal_radius_m"]=.001
        self.assertTrue(any("after goal" in x for x in validate(self.raw,self.s,"rcppo",False)[0]))
    def test_tracking_units(self):
        self.raw[:,6]=.2
        rows=calculate(self.raw,self.s,"rcppo")
        self.assertEqual(rows[0]["tracking_error_mps"],0)
        self.assertAlmostEqual(rows[0]["yaw_error_radps"],.2)
    def test_missing_data_and_derived_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"raw.csv"
            with open(p,"w",newline="") as f:
                w=csv.writer(f);w.writerow(COLUMNS);w.writerow([0,.3,0,0,"",0,0])
            with self.assertRaises(ValueError):read_raw(p)
            rows=calculate(self.raw,self.s,"rcppo")
            changed=copy.deepcopy(rows);changed[-1]["J_exec"]+=1
            p=Path(d)/"derived.csv";write_csv(p,changed)
            self.assertTrue(check_derived(p,rows))
    def test_lateral_and_acceleration(self):
        bad=self.raw.copy();bad[1,4:6]=[-.5,.4]
        self.assertTrue(any("acceleration" in x for x in validate(bad,self.s,"rcppo",False)[0]))

if __name__=="__main__":unittest.main(verbosity=2)


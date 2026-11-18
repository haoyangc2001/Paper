import copy
import math
import unittest

from camra_navigation.budget.guarded import GRID, select_guarded_budget, summarize_decisions


class GuardedBudgetTests(unittest.TestCase):
    def decision(self, scores):
        calls = []
        def query(z):
            i = len(calls)
            calls.append(z)
            return z/550-2, scores[i]
        result = select_guarded_budget(query, 700)
        self.assertEqual(len(calls),5)
        return result

    def test_select_next_budget_with_raw_J_above_reference_scale(self):
        r = self.decision([1, .1, -.01, -.1, -.2])
        self.assertEqual(r['selected_k'],1.15)
        self.assertAlmostEqual(r['selected_budget_J'],805)

    def test_no_room_for_reserve(self):
        self.assertEqual(self.decision([1,1,1,1,-1])['reason'],'no_reserve_within_grid')

    def test_refuse_non_monotonic(self):
        self.assertEqual(self.decision([1,-1,1,-1,-1])['reason'],'non_monotonic')

    def test_refuse_exact_zero(self):
        self.assertEqual(self.decision([1,0,-1,-1,-1])['reason'],'uncertain_zero')

    def test_refuse_nonfinite(self):
        for value in (math.nan, math.inf, -math.inf):
            self.assertEqual(self.decision([1,1,value,-1,-1])['reason'],'nonfinite')

    def test_no_raw_feasible_budget(self):
        self.assertEqual(self.decision([1]*5)['reason'],'no_feasible_grid_budget')

    def test_all_raw_feasible_still_adds_reserve(self):
        self.assertEqual(self.decision([-1]*5)['selected_k'],.85)

    def test_reject_bad_units(self):
        for ref in (0,-1,math.nan):
            with self.assertRaises(ValueError):
                select_guarded_budget(lambda z:(0,0),ref)
        with self.assertRaises(ValueError):
            select_guarded_budget(lambda z:(0,0),700,z_ref_J=1)

    def fixture(self):
        maps = [{'map_id':f'm{i}', 'level':f'L{i//3+1}'} for i in range(9)]
        ds, rows = [], []
        for method in ('uncalibrated','full'):
            for m in maps:
                ds.append({'method':method, 'map_id':m['map_id'], 'decision':self.decision([1,1,-1,-1,-1])})
                for i,k in enumerate(GRID):
                    rows.append({'method':method,'map_id':m['map_id'],'k':k,
                                 'episode':{'feasible':i>=2,'label':'goal_reached','t_s':40.,
                                            'z0_J':k*700,'cost_model_J':700.,'danger_steps':0}})
        return ds,rows,maps

    def test_finite_success_and_correct_denominators(self):
        result = summarize_decisions(*self.fixture())
        self.assertTrue(result['pass'])
        self.assertEqual(result['methods']['full']['correctly_accepted'],9)
        self.assertEqual(result['grid_episodes'],90)

    def test_all_rejected_does_not_pass(self):
        ds,rows,maps = self.fixture()
        for d in ds:
            d['decision'] = self.decision([1]*5)
        r = summarize_decisions(ds,rows,maps)
        self.assertFalse(r['pass'])
        self.assertEqual(r['methods']['full']['false_rejections'],9)

    def test_actual_failure_blocks_acceptance_even_if_raw_critic_accepts(self):
        ds,rows,maps = self.fixture()
        selected = next(e for e in rows if e['method']=='full' and e['map_id']=='m0' and e['k']==1.15)
        selected['episode'].update(feasible=False, danger_steps=1)
        self.assertFalse(summarize_decisions(ds,rows,maps)['methods']['full']['checks']['accepted_safety_time_budget'])

    def test_always_maximum_fails_efficiency(self):
        ds,rows,maps = self.fixture()
        for d in ds:
            d['decision'] = self.decision([1,1,1,-1,-1])
        self.assertFalse(summarize_decisions(ds,rows,maps)['methods']['full']['checks']['budget_efficiency'])

    def test_missing_duplicate_rows_do_not_pass(self):
        ds,rows,maps = self.fixture()
        with self.assertRaises(ValueError):
            summarize_decisions(ds,rows[:-1],maps)
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):
            summarize_decisions(ds,rows,maps)


if __name__ == '__main__':
    unittest.main()

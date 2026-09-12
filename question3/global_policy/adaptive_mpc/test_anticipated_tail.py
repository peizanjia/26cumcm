import unittest
import json
from pathlib import Path

import numpy as np

from ..model import World
from .anticipated_tail import anticipated_tail
from .test_planner import certified_target, small_parameters


class AnticipatedTailTests(unittest.TestCase):
    def test_future_known_stop_replaces_station_and_scan_is_charged(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        mask = (np.abs(world.coverage.centers[:,0]-600.) < 120.) & (np.abs(world.coverage.centers[:,1]) < 120.)
        world.coverage.covered[1,mask] = False
        certified_target(world,1,[600.,0.])
        before = world.coverage.covered.copy()
        value, route = anticipated_tail(world,{1:5.},np.zeros(2),None,np.zeros(2),[],[np.array([1200.,0.])])
        self.assertEqual(route, [[600.,0.]])
        self.assertAlmostEqual(value,600/5+5+6)
        np.testing.assert_array_equal(world.coverage.covered,before)
        self.assertFalse(world.coverage.complete(2))
        self.assertFalse(world.finished())
        self.assertEqual(world.targets[2].status,'unknown')
        self.assertEqual(world.targets[2].observations,[])

    def test_multiple_forecast_stops_do_not_pay_for_identical_coverage_twice(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        cell = int(np.argmin(np.linalg.norm(world.coverage.centers-np.array([600.,0.]),axis=1)))
        world.coverage.covered[2,cell] = False
        certified_target(world,1,[600.,0.])
        certified_target(world,2,[600.,0.])
        value,_ = anticipated_tail(world,{1:5.,2:5.},np.zeros(2),None,np.zeros(2),[],[])
        self.assertAlmostEqual(value,600/5+10+6)
        self.assertFalse(world.coverage.complete(3))

    def test_stale_station_list_is_repaired_without_marking_real_coverage(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        world.coverage.covered[0] = False
        before = world.coverage.covered.copy()
        value,route = anticipated_tail(world,{},np.zeros(2),None,np.zeros(2),[],[])
        self.assertGreater(value,0)
        self.assertTrue(route)
        np.testing.assert_array_equal(world.coverage.covered,before)
        for point in route:
            world.coverage.mark(1,point)
        self.assertTrue(world.coverage.complete(1))

    def test_scans_at_proposed_action_are_already_paid_and_not_recharged(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        world.coverage.covered[1,world.coverage.mask_at([600.,0.])] = False
        certified_target(world,1,[600.,0.])
        value,route = anticipated_tail(world,{1:5.},np.zeros(2),None,np.array([600.,0.]),[2],[])
        self.assertAlmostEqual(value,600/5+5)
        self.assertEqual(route,[[600.,0.]])

    def test_center_forecast_does_not_claim_twenty_meter_boundary_margin(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        center = np.array([600.,0.])
        cell = int(np.argmin(np.linalg.norm(world.coverage.centers-np.array([1530.,210.]),axis=1)))
        world.coverage.covered[1,cell] = False
        self.assertTrue(world.coverage.mask_at(center)[cell])
        far = np.abs(world.coverage.centers[cell]-center)+world.coverage.half
        self.assertGreater(np.linalg.norm(far),980.)
        certified_target(world,1,center)
        value,route = anticipated_tail(world,{1:5.},np.zeros(2),None,np.zeros(2),[],[np.array([1400.,200.])])
        self.assertEqual(len(route),2)
        self.assertGreater(value,600/5+5+6)
        self.assertFalse(world.coverage.complete(2))

    def test_real_20272318_zero_gain_repair_uses_finite_certificate_fallback(self):
        fixture = json.loads((Path(__file__).parent/'fixtures'/'anticipated_tail_20272318.json').read_text(encoding='utf8'))
        world = World(small_parameters())
        world.coverage.covered[:] = True
        world.position = np.asarray(fixture['position'])
        for channel,encoded in fixture['unknown'].items():
            packed = np.frombuffer(bytes.fromhex(encoded),dtype=np.uint8)
            world.coverage.covered[int(channel)-1] = np.unpackbits(packed)[:len(world.coverage.centers)].astype(bool)
        for known in fixture['known']:
            certified_target(world,known['channel'],known['center'],known['radius'])
        before = world.coverage.covered.copy()
        value,route = anticipated_tail(world,{int(c):v for c,v in fixture['residual'].items()},
                                      np.asarray(fixture['start']),fixture['removed'],np.asarray(fixture['y']),
                                      fixture['unknown_channels'],[np.asarray(q) for q in fixture['stations']],fixture['early'])
        self.assertTrue(np.isfinite(value))
        self.assertGreater(value,0.)
        np.testing.assert_array_equal(world.coverage.covered,before)
        for channel in fixture['unknown_channels']:
            world.coverage.mark(channel,fixture['y'])
        for point in route:
            for channel in fixture['unknown']:
                world.coverage.mark(int(channel),point)
        self.assertTrue(all(world.coverage.complete(int(channel)) for channel in fixture['unknown']))


if __name__ == '__main__':
    unittest.main()

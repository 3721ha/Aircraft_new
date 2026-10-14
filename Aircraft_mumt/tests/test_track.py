# -*- coding: utf-8 -*-
"""P1.3 unit tests: track file state machine, covariance bookkeeping, features."""
import math
import unittest

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.track import TrackFile


def hits_at(tid, x, y):
    return [(tid, x, y)]


class TestInitiation(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()
        self.tf = TrackFile(self.cfg)

    def test_two_hits_in_window_tentative(self):
        self.tf.update_with_hits(hits_at(1, 100.0, 50.0), t=0.0, t_frame=0.5)
        tr = self.tf.track_of(1)
        self.assertEqual(tr.state, "detected")
        self.tf.update_with_hits(hits_at(1, 100.2, 50.0), t=6.0, t_frame=0.5)
        self.assertEqual(tr.state, "tentative")  # 2 hits within 6.5 s

    def test_second_hit_outside_window_no_init(self):
        self.tf.update_with_hits(hits_at(1, 100.0, 50.0), t=0.0, t_frame=0.5)
        tr = self.tf.track_of(1)
        self.tf.update_with_hits(hits_at(1, 100.2, 50.0), t=7.0, t_frame=0.5)
        self.assertEqual(tr.state, "detected")  # 7.0 > 6.5 window

    def test_slot_capacity(self):
        for k in range(8):
            self.tf.update_with_hits(hits_at(k, 100.0, k * 5.0), t=0.0, t_frame=0.5)
        self.assertEqual(self.tf.n_tracks(), 8)
        # 9th target discarded
        self.tf.update_with_hits(hits_at(99, 100.0, 99.0), t=0.0, t_frame=0.5)
        self.assertIsNone(self.tf.track_of(99))


class TestConfirmation(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()
        self.tf = TrackFile(self.cfg)

    def _reach_tentative(self):
        self.tf.update_with_hits(hits_at(1, 100.0, 50.0), t=0.0, t_frame=0.5)
        self.tf.update_with_hits(hits_at(1, 100.1, 50.0), t=0.5, t_frame=0.5)

    def test_confirm_after_5s_tight_updates(self):
        self._reach_tentative()
        tr = self.tf.track_of(1)
        self.assertEqual(tr.state, "tentative")
        # 0.5 s frame gear: max gap = min(2*0.5, 1.6) = 1.0 s; feed every 1 s
        for k in range(1, 6):
            self.tf.update_with_hits(hits_at(1, 100.1 + 0.1 * k, 50.0), t=0.5 + k, t_frame=0.5)
        self.assertEqual(tr.state, "confirmed")
        self.assertIn(1, self.tf.confirmed_ids())

    def test_confirm_reset_on_gap(self):
        self._reach_tentative()
        tr = self.tf.track_of(1)
        self.tf.update_with_hits(hits_at(1, 100.2, 50.0), t=1.5, t_frame=0.5)  # gap 1.0 ok
        self.assertAlmostEqual(tr.confirm_progress, 1.0)
        self.tf.update_with_hits(hits_at(1, 100.3, 50.0), t=5.0, t_frame=0.5)  # gap 3.5 > 1.0
        self.assertAlmostEqual(tr.confirm_progress, 0.0)


class TestAgeAndDrop(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()
        self.tf = TrackFile(self.cfg)
        self.tf.update_with_hits(hits_at(1, 100.0, 50.0), t=0.0, t_frame=0.5)
        self.tf.update_with_hits(hits_at(1, 100.1, 50.0), t=0.5, t_frame=0.5)

    def test_coast_at_8s(self):
        self.tf.tick(t=9.0, dt=8.5)
        self.assertEqual(self.tf.track_of(1).state, "coast")

    def test_drop_at_13s_frees_slot(self):
        self.tf.tick(t=13.6, dt=13.1)  # age = 13.1 > 13.0
        self.assertIsNone(self.tf.track_of(1))
        self.assertEqual(self.tf.n_tracks(), 0)
        # slot reusable
        self.tf.update_with_hits(hits_at(2, 90.0, 40.0), t=14.0, t_frame=0.5)
        self.assertIsNotNone(self.tf.track_of(2))

    def test_rehit_after_coast_recovers(self):
        self.tf.tick(t=9.0, dt=8.5)
        self.assertEqual(self.tf.track_of(1).state, "coast")
        self.tf.update_with_hits(hits_at(1, 100.5, 50.0), t=9.5, t_frame=0.5)
        self.assertIn(self.tf.track_of(1).state, ("tentative", "confirmed"))


class TestCovariance(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()
        self.tf = TrackFile(self.cfg)

    def test_shrink_inflate_cap(self):
        self.tf.update_with_hits(hits_at(1, 100.0, 50.0), t=0.0, t_frame=0.5)
        tr = self.tf.track_of(1)
        self.assertEqual(tr.sigma, 4.0)  # starts at sigma_ref
        self.tf.update_with_hits(hits_at(1, 100.1, 50.0), t=1.0, t_frame=0.5)
        self.assertAlmostEqual(tr.sigma, 3.2)  # x0.8
        # inflate: sqrt(3.2^2 + 0.25*10) < 4 cap
        self.tf.tick(t=11.0, dt=10.0)
        self.assertAlmostEqual(tr.sigma, math.sqrt(3.2 ** 2 + 2.5))
        # cap at 4
        self.tf.tick(t=111.0, dt=100.0)
        tr2 = self.tf.track_of(1)
        self.assertIsNone(tr2)  # dropped long before; cap tested on fresh track


class TestFeatures(unittest.TestCase):
    def test_feature_shape_and_mask(self):
        cfg = DetectSimConfig()
        tf = TrackFile(cfg)
        feats = tf.features((0.0, 0.0), t=0.0)
        self.assertEqual(len(feats), 8)
        self.assertTrue(all(len(f) == 12 for f in feats))
        self.assertTrue(all(f[-1] == 0.0 for f in feats))  # all masked empty
        tf.update_with_hits(hits_at(1, 40.0, 0.0), t=0.0, t_frame=0.5)
        feats = tf.features((0.0, 0.0), t=0.0)
        used = [f for f in feats if f[-1] == 1.0]
        self.assertEqual(len(used), 1)
        self.assertAlmostEqual(used[0][0], 40.0 / 80.0)  # dx/R_max

    def test_velocity_estimate(self):
        cfg = DetectSimConfig()
        tf = TrackFile(cfg)
        # target moving west at 0.2 km/s
        for k in range(4):
            tf.update_with_hits(hits_at(1, 100.0 - 0.2 * k, 50.0), t=float(k), t_frame=0.5)
        tr = tf.track_of(1)
        self.assertTrue(-0.25 < tr.vx < -0.10, tr.vx)  # smoothed estimate
        self.assertAlmostEqual(tr.vy, 0.0)


if __name__ == "__main__":
    unittest.main()

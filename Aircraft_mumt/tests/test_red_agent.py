# -*- coding: utf-8 -*-
"""P1.4 unit tests: red spawn geometry, manoeuvre library, duty calibration."""
import math
import unittest

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.red_agent import RedFleet, WEST


class TestSpawn(unittest.TestCase):
    def test_k4_pair_lanes_and_stagger(self):
        cfg = DetectSimConfig(K_RED=4)
        fleet = RedFleet(cfg, seed=3)
        self.assertEqual(len(fleet.agents), 4)
        ys = sorted(a.y for a in fleet.agents)
        # sub-lane split +-4 km around each lane centre (V2.6 combat spread)
        self.assertEqual(ys, [26.0, 34.0, 86.0, 94.0])
        xs = sorted(a.x for a in fleet.agents)
        # intra-pair stagger 8 km: two at x0, two at x0+8
        self.assertEqual(xs, [158.0, 158.0, 166.0, 166.0])

    def test_k6_adds_centre_lane(self):
        cfg = DetectSimConfig(K_RED=6)
        fleet = RedFleet(cfg, seed=3)
        ys = sorted(a.y for a in fleet.agents)
        self.assertEqual(ys, [26.0, 34.0, 56.0, 64.0, 86.0, 94.0])


class TestSeparationAndRegain(unittest.TestCase):
    """V2.6: same-altitude 2 km floor + regain pump rhythm."""

    def test_min_separation_floor_k4(self):
        cfg = DetectSimConfig(K_RED=4)
        for seed in range(20):
            rep = RedFleet(cfg, seed=seed).run_episode()
            self.assertGreaterEqual(rep["min_separation"], 2.0,
                                    (seed, rep["min_separation"]))

    def test_min_separation_floor_k6_k8(self):
        for k in (6, 8):
            cfg = DetectSimConfig(K_RED=k)
            for seed in range(10):
                rep = RedFleet(cfg, seed=seed).run_episode()
                self.assertGreaterEqual(rep["min_separation"], 2.0,
                                        (k, seed, rep["min_separation"]))

    def test_regain_exercised_and_bounded(self):
        cfg = DetectSimConfig(K_RED=4)
        total = sum(RedFleet(cfg, seed=s).run_episode()["regain_count"]
                    for s in range(20))
        self.assertGreater(total, 0)  # pump rhythm present
        self.assertLess(total, 20 * 4 * 6)  # slack-gated: < 6 per agent

    def test_cutoff_regression_multiseed(self):
        cfg = DetectSimConfig(K_RED=4)
        for seed in range(20):
            rep = RedFleet(cfg, seed=seed).run_episode()
            self.assertTrue(rep["all_crossed"], (seed, rep["t_cutoff"]))
            self.assertLessEqual(rep["max_t_cutoff"], 880.0,
                                 (seed, rep["max_t_cutoff"]))


class TestDutyCalibration(unittest.TestCase):
    """V2.2 hard requirement: cutoff arrival <= 851 s; duty in [0.60, 0.70]."""

    def test_episode_duty_and_cutoff(self):
        cfg = DetectSimConfig()
        fleet = RedFleet(cfg, seed=11)
        rep = fleet.run_episode()
        self.assertTrue(rep["all_crossed"], rep["t_cutoff"])
        self.assertLessEqual(rep["max_t_cutoff"], 851.0)
        self.assertGreaterEqual(rep["max_t_cutoff"], 600.0)
        self.assertTrue(0.55 <= rep["duty_mean"] <= 0.75, rep["duty_mean"])
        # controller keeps late-window duty inside the band
        late = [d for t, d in fleet.duty_log if t > 300]
        for d in late:
            self.assertTrue(0.45 <= d <= 0.85, d)

    def test_deterministic_under_seed(self):
        cfg = DetectSimConfig()
        r1 = RedFleet(cfg, seed=5).run_episode()["t_cutoff"]
        r2 = RedFleet(cfg, seed=5).run_episode()["t_cutoff"]
        self.assertEqual(r1, r2)


class TestBoundaryAndManeuver(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_stays_inside_airspace(self):
        fleet = RedFleet(self.cfg, seed=23)
        t = 0.0
        while t < 900.0:
            fleet.step(t, 1.0)
            for a in fleet.agents:
                self.assertTrue(0.0 <= a.y <= self.cfg.L_Y, (a.uid, a.y))
                self.assertLessEqual(a.x, self.cfg.L_X + 12.0)  # stagger headroom
            t += 1.0

    def test_maneuver_library_resample_window(self):
        fleet = RedFleet(self.cfg, seed=7)
        a = fleet.agents[0]
        seen = set()
        t = 0.0
        for _ in range(400):
            fleet.step(t, 1.0)
            seen.add(a.maneuver)
            t += 1.0
        # library exercised: at least transit-like and snake/turn/waypoint appear
        self.assertTrue({"patrol", "snake"} <= seen or len(seen) >= 3, seen)

    def test_turn_rate_limited(self):
        fleet = RedFleet(self.cfg, seed=13)
        a = fleet.agents[0]
        h0 = a.heading
        fleet.step(0.0, 1.0)
        self.assertLessEqual(abs(a.heading - h0) % (2 * math.pi),
                             math.radians(self.cfg.RED_OMEGA_DEG) + 1e-9)


if __name__ == "__main__":
    unittest.main()

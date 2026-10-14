# -*- coding: utf-8 -*-
"""P1.2 unit tests: radar scan mechanics, P_d ladder, off-boresight, C8 dwell."""
import math
import random
import unittest

from aircraft_sim.detect_config import DetectSimConfig
from aircraft_sim.radar import RadarState, beam_offsets, wrap_pi


class TestBeamGeometry(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_offsets_edges(self):
        """Outer beam edges reach +/- beta; count = 2beta/15."""
        for beta, n in ((7.5, 1), (15.0, 2), (30.0, 4), (45.0, 6), (60.0, 8)):
            offs = beam_offsets(beta, 7.5)
            self.assertEqual(len(offs), n)
            self.assertAlmostEqual(offs[0] - 7.5, -beta)
            self.assertAlmostEqual(offs[-1] + 7.5, beta)

    def test_wrap(self):
        self.assertAlmostEqual(wrap_pi(math.radians(179) - math.radians(-179)),
                               math.radians(-2))

    def test_frame_progression_covers_sector_once(self):
        """A stationary target inside the sector is swept exactly once per frame."""
        radar = RadarState(self.cfg, gear=30.0, axis_deg=0.0)
        # target dead ahead at 10 deg off nose, 50 km
        tgt_bearing = math.radians(10.0)
        swept_steps = []
        for _ in range(4):  # 4 steps x 2 dwells = 8 dwells = 2 frames of 4 positions
            bores = radar.dwell_boresights(0.0)
            hit = any(abs(wrap_pi(tgt_bearing - b)) <= math.radians(7.5) for b in bores)
            swept_steps.append(hit)
        self.assertEqual(sum(swept_steps), 2)  # once per 2 s frame


class TestPd(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()
        self.radar = RadarState(self.cfg)

    def test_ladder_boresight(self):
        # on nose: theta_off = 0 -> R_max_eff = R_max
        self.assertEqual(self.radar.pd(40.0, 0.0), 0.9)
        self.assertEqual(self.radar.pd(60.0, 0.0), 0.6)
        self.assertEqual(self.radar.pd(70.0, 0.0), 0.3)
        self.assertEqual(self.radar.pd(85.0, 0.0), 0.0)

    def test_off_boresight_attenuation(self):
        # 45 deg off nose: R_max_eff = 80*sqrt(cos45) ~ 67.3 km
        #   d=50 -> r=0.743 -> 0.6 tier; d=60 -> r=0.892 -> 0.3 tier
        self.assertEqual(self.radar.pd(50.0, math.radians(45)), 0.6)
        self.assertEqual(self.radar.pd(60.0, math.radians(45)), 0.3)
        # 60 deg off nose: R_max_eff ~ 56.6 -> d=60 out of range entirely
        self.assertEqual(self.radar.pd(60.0, math.radians(60)), 0.0)
        # 90 deg off: zero
        self.assertEqual(self.radar.pd(10.0, math.radians(90)), 0.0)

    def test_monotonic_in_range(self):
        pds = [self.radar.pd(d, 0.0) for d in (20, 40, 55, 64, 72, 80)]
        self.assertEqual(pds, sorted(pds, reverse=True))


class TestGearSwitchDwell(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_c8_min_dwell(self):
        radar = RadarState(self.cfg, gear=60.0)  # frame 4 s
        radar.last_switch_step = 0
        radar.frame_progress = 2.0  # mid-frame
        self.assertFalse(radar.switch_gear(15.0, step=10))   # would void frame
        radar.frame_progress = 4.0
        self.assertTrue(radar.switch_gear(15.0, step=10))
        self.assertEqual(radar.gear, 15.0)
        self.assertEqual(radar.frame_progress, 0.0)

    def test_invalid_gear_rejected(self):
        radar = RadarState(self.cfg, gear=60.0)
        radar.frame_progress = 4.0
        self.assertFalse(radar.switch_gear(20.0, step=0))


class TestScanStep(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_bernoulli_reproducible(self):
        """Seeded RNG: hit sequence deterministic; hit rate over many trials ~ P_d."""
        radar = RadarState(self.cfg, gear=60.0, axis_deg=0.0)
        targets = {0: (50.0, 0.0)}  # dead ahead 50 km, 0.6 tier
        hits = 0
        trials = 4000
        rng = random.Random(1)
        for _ in range(trials):
            hits += len(radar.scan_step(0.0, (0.0, 0.0), targets, rng))
        rate = hits / trials
        # swept exactly twice per 4 s frame -> effective per-step sweep prob 0.5,
        # hit only when swept: rate ~ 0.5 * 0.6 = 0.30
        self.assertTrue(0.27 <= rate <= 0.33, rate)

    def test_out_of_sector_never_hit(self):
        radar = RadarState(self.cfg, gear=7.5, axis_deg=0.0)
        targets = {0: (50.0, 40.0)}  # ~38.7 deg off nose: outside narrow gear
        rng = random.Random(0)
        hits = sum(len(radar.scan_step(0.0, (0.0, 0.0), targets, rng)) for _ in range(100))
        self.assertEqual(hits, 0)

    def test_axis_trim_clamp(self):
        radar = RadarState(self.cfg)
        for _ in range(20):
            radar.trim_axis(1.0)
        self.assertEqual(radar.axis_deg, 60.0)
        for _ in range(40):
            radar.trim_axis(-1.0)
        self.assertEqual(radar.axis_deg, -60.0)


if __name__ == "__main__":
    unittest.main()

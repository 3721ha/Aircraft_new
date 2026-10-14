# -*- coding: utf-8 -*-
"""P1.1 unit tests: DetectSimConfig against plan V2.4 §10 registry and
appendix-A timeline verification numbers."""
import math
import unittest

from aircraft_sim.detect_config import (
    A_MACH_KMS, ZONE_DETECTION, ZONE_TRACKING, ZONE_BEHIND,
    DetectSimConfig, ma_to_kms,
)


class TestZones(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_zone_boundaries(self):
        c = self.cfg
        self.assertEqual(c.zone_of(0.0), ZONE_BEHIND)
        self.assertEqual(c.zone_of(39.99), ZONE_BEHIND)
        self.assertEqual(c.zone_of(40.0), ZONE_TRACKING)   # cutoff line inclusive
        self.assertEqual(c.zone_of(110.0), ZONE_TRACKING)  # [40,110]
        self.assertEqual(c.zone_of(110.01), ZONE_DETECTION)
        self.assertEqual(c.zone_of(160.0), ZONE_DETECTION)

    def test_age_standard(self):
        self.assertEqual(self.cfg.age_standard(70.0), 2.0)
        self.assertEqual(self.cfg.age_standard(130.0), 5.0)

    def test_subzones_cover_width(self):
        c = self.cfg
        bounds = [c.subzone_bounds(i) for i in range(3)]
        self.assertEqual(bounds[0], (0.0, 40.0))
        self.assertEqual(bounds[2], (80.0, 120.0))
        # contiguous, exactly 120 km
        self.assertAlmostEqual(bounds[1][0], bounds[0][1])
        self.assertAlmostEqual(bounds[2][0], bounds[1][1])


class TestPdLadder(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_tiers(self):
        c = self.cfg
        self.assertEqual(c.pd_at(40.0), 0.9)    # r=0.5 boundary
        self.assertEqual(c.pd_at(40.01), 0.6)
        self.assertEqual(c.pd_at(64.0), 0.6)    # r=0.8 boundary
        self.assertEqual(c.pd_at(64.01), 0.3)
        self.assertEqual(c.pd_at(80.0), 0.3)    # r=1.0 boundary
        self.assertEqual(c.pd_at(80.01), 0.0)

    def test_mor_mar_riding(self):
        """plan §1.3: 0.9 tier inside MAR, 0.6 tier straddles MOR."""
        c = self.cfg
        self.assertLessEqual(c.MAR, 40.0)
        self.assertEqual(c.pd_at(c.MAR), 0.9)
        self.assertEqual(c.pd_at(c.MOR), 0.6)
        # penalty-free standoff band [MOR, 64] width 4 km
        self.assertAlmostEqual(0.8 * c.R_MAX - c.MOR, 4.0)


class TestRedTimeline(unittest.TestCase):
    """Appendix A: duty 0.60-0.70 -> cutoff arrival 730-851 s (< 900 timeout)."""

    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_cutoff_arrival_window(self):
        c = self.cfg
        v = ma_to_kms(c.RED_CRUISE_MA)
        t_lo = (c.RED_X0 - c.X_CUTOFF) / (v * c.RED_DUTY_MAX)
        t_hi = (c.RED_X0 - c.X_CUTOFF) / (v * c.RED_DUTY_MIN)
        # V2.7 band 0.70-0.80: arrival ~638-730 s, still < 900 s timeout
        self.assertTrue(630.0 <= t_lo <= 650.0, t_lo)
        self.assertTrue(720.0 <= t_hi <= 740.0, t_hi)

    def test_tracking_zone_dwell_gives_windows(self):
        c = self.cfg
        v = ma_to_kms(c.RED_CRUISE_MA) * c.RED_DUTY_MIN
        dwell = (c.RED_X0 - c.X_CUTOFF - (c.RED_X0 - c.X_TRACK)) / v
        n_windows = dwell / c.STABLE_WINDOW
        self.assertGreaterEqual(n_windows, 14.0)  # ~15-17 attempts, appendix A


class TestGearTable(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_frame_periods(self):
        self.assertEqual(self.cfg.GEAR_FRAME[7.5], 0.5)
        self.assertEqual(self.cfg.GEAR_FRAME[60.0], 4.0)

    def test_role_masks_partition(self):
        c = self.cfg
        self.assertEqual(set(c.GEARS_SEARCH) | set(c.GEARS_TRACK), set(c.GEAR_FRAME))
        self.assertFalse(set(c.GEARS_SEARCH) & set(c.GEARS_TRACK))

    def test_scan_rate(self):
        """4 s full sweep of ±60 deg at 0.5 s dwell => 8 beam positions of 15 deg."""
        c = self.cfg
        beams = c.GEAR_FRAME[60.0] / c.BEAM_DWELL
        self.assertEqual(beams, 8)
        self.assertAlmostEqual(beams * 2 * c.FOV_HALF_DEG, 120.0)


class TestTurnRatePhysics(unittest.TestCase):
    """Plan §2 physical note: omega = g*tan(phi)/V equivalence."""

    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_bank_equivalence(self):
        c = self.cfg
        v_ms = ma_to_kms(0.7) * 1000.0
        for omega_deg, phi_lo, phi_hi in ((3.0, 48.0, 50.0), (6.0, 66.0, 68.0)):
            omega = math.radians(omega_deg)
            phi = math.degrees(math.atan(omega * v_ms / 9.81))
            self.assertTrue(phi_lo <= phi <= phi_hi, (omega_deg, phi))
        # turn radius at cruise gear ~4.1 km
        r = v_ms / math.radians(3.0) / 1000.0
        self.assertTrue(4.0 <= r <= 4.2, r)


class TestRegistryConsistency(unittest.TestCase):
    def setUp(self):
        self.cfg = DetectSimConfig()

    def test_obs_dim_decomposition(self):
        c = self.cfg
        self.assertEqual(17 + c.TRACK_SLOTS * 12 + 2 * 6 + 4 + 6 + 3, c.OBS_DIM)

    def test_w8_layer_mapping(self):
        """V1.20 x2.5 scaling of layer weights 4.0/3.0/2.0/1.0/0.5."""
        c = self.cfg
        expected = (4.0 * 2.5, 3.0 * 2.5, 4.0 * 2.5, 2.0 * 2.5, 1.0 * 2.5, 0.5 * 2.5)
        self.assertEqual(c.W8_LAYER, expected)

    def test_reward_magnitude_budget(self):
        """plan §7.3: single-target 30 s window w1 ~ +0.6, MAR 30 s ~ -3."""
        c = self.cfg
        self.assertAlmostEqual(c.W1_TRACK_TICK * 30, 0.6)
        self.assertAlmostEqual(c.W6_MAR * 30, 3.0)
        total_win = 4 * (c.W3_STABLE) + c.W4_SUCCESS - 0  # w1 small
        self.assertGreater(total_win, 4 * 3.0)  # success >> MAR dips


if __name__ == "__main__":
    unittest.main()

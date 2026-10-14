import unittest

from aircraft_sim import (Action, AircraftEnv, BeliefSTLMonitor,
                          ConflictAwareQPSafetyShield, Mode, SimConfig)
from aircraft_sim.scenarios import boundary_scenario, protocol_conflict_scenario, sample_scenario


class BeliefQPTests(unittest.TestCase):
    def test_belief_monitor_uses_observation_only(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset(sample_scenario(3, seed=1, difficulty=0.0))
        monitor = BeliefSTLMonitor(env.cfg)
        report = monitor.evaluate(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertTrue(report.belief.agents)
        self.assertNotIn("truth_state", report.belief.history[-1])

    def test_belief_probability_increases_with_uncertainty(self):
        env = AircraftEnv(2, SimConfig(noise_position=20.0))
        observations = env.reset({"positions": [(0, 0, 5000), (820, 0, 5000)], "resources": [0.8, 0.8]})
        first = BeliefSTLMonitor(env.cfg, process_noise=1.0).evaluate(observations)
        second = BeliefSTLMonitor(env.cfg, process_noise=300.0).evaluate(observations)
        self.assertLessEqual(second.robust_lower["S02:0-1"], first.robust_lower["S02:0-1"])

    def test_qp_residual_stays_close_to_nominal_when_safe(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset(sample_scenario(3, seed=2, difficulty=0.0))
        shield = ConflictAwareQPSafetyShield(env.cfg)
        nominal = {0: Action(Mode.HOLD), 1: Action(Mode.COVER), 2: Action(Mode.HOLD)}
        safe, events = shield.filter(observations, nominal)
        self.assertEqual(shield.last_solution.status, "optimal")
        self.assertEqual(events, [])
        self.assertEqual(safe, nominal)

    def test_qp_changes_critical_action_under_low_confidence(self):
        env = AircraftEnv(3, SimConfig(support_response_deadline=0))
        observations = env.reset(boundary_scenario())
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {i: Action(Mode.CRITICAL) for i in observations})
        self.assertNotEqual(safe[0].mode, Mode.CRITICAL)
        self.assertIn("I02:0", shield.last_solution.belief_report.violations)

    def test_qp_gates_unauthorized_critical_action(self):
        env = AircraftEnv(2)
        observations = env.reset({"positions": [(0, 0, 5000), (3000, 0, 5000)], "resources": [0.8, 0.8], "authorization": {0: False}})
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {0: Action(Mode.CRITICAL), 1: Action(Mode.HOLD)})
        self.assertNotEqual(safe[0].mode, Mode.CRITICAL)
        self.assertIn("I01:0", shield.last_solution.belief_report.violations)

    def test_qp_handles_capability_loss(self):
        env = AircraftEnv(2, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0, 0, 5000), (3000, 0, 5000)],
            "resources": [0.1, 0.9],
            "capabilities": [0.3, 1.0],
        })
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {0: Action(Mode.COVER), 1: Action(Mode.HOLD)})
        self.assertIn("F03:0", shield.last_solution.belief_report.violations)
        self.assertTrue(any(key.startswith("F03:") for key in shield.last_solution.belief_report.violation_probability))
        self.assertEqual(safe[0].mode, Mode.RECOVER)

    def test_belief_reports_lost_link_task_continuity(self):
        env = AircraftEnv(2, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0, 0, 5000), (3000, 0, 5000)],
            "c2_lost": {0: True},
            "task_authority": {0: True, 1: False},
        })
        monitor = BeliefSTLMonitor(env.cfg)
        report = monitor.evaluate(observations, {0: Action(Mode.HOLD), 1: Action(Mode.HOLD)})
        self.assertIn("M05:0", report.violations)

    def test_belief_reports_overallocated_soft_preference(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset()
        monitor = BeliefSTLMonitor(env.cfg)
        report = monitor.evaluate(observations, {i: Action(Mode.CRITICAL) for i in observations})
        self.assertIn("E01", report.violations)
        self.assertTrue(report.hard_safe)

    def test_belief_reports_fast_role_change_as_soft_rule(self):
        env = AircraftEnv(2, SimConfig(noise_position=0.0, noise_speed=0.0, mode_dwell_steps=2))
        observations = env.reset()
        env.step({0: Action(Mode.COVER), 1: Action(Mode.HOLD)})
        observations = env.observe()
        monitor = BeliefSTLMonitor(env.cfg)
        report = monitor.evaluate(observations, {0: Action(Mode.HOLD), 1: Action(Mode.HOLD)})
        self.assertIn("E02:0", report.violations)
        self.assertTrue(report.hard_safe)

    def test_soft_dwell_rule_does_not_override_safety_action(self):
        env = AircraftEnv(2, SimConfig(noise_position=0.0, noise_speed=0.0, mode_dwell_steps=2))
        observations = env.reset()
        env.step({0: Action(Mode.COVER), 1: Action(Mode.HOLD)})
        observations = env.observe()
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {0: Action(Mode.HOLD), 1: Action(Mode.COVER)})
        self.assertEqual(safe[0].mode, Mode.HOLD)

    def test_protocol_conflict_routes_support_and_records_information_risk(self):
        env = AircraftEnv(3, SimConfig(support_response_deadline=0))
        observations = env.reset(protocol_conflict_scenario(3))
        shield = ConflictAwareQPSafetyShield(env.cfg)
        safe, _ = shield.filter(observations, {i: Action(Mode.CRITICAL) for i in observations})
        self.assertNotEqual(safe[0].mode, Mode.CRITICAL)
        report = shield.last_solution.belief_report
        self.assertIn("I04:0", report.violations)
        self.assertTrue(any(action.mode == Mode.SUPPORT and action.target == 0 for action in safe.values()))

    def test_qp_records_infeasible_fallback(self):
        env = AircraftEnv(3)
        observations = env.reset(boundary_scenario())
        shield = ConflictAwareQPSafetyShield(env.cfg)
        shield.filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertIn(shield.last_solution.status, {"optimal", "infeasible_fallback"})
        if shield.last_solution.status == "infeasible_fallback":
            self.assertTrue(shield.last_solution.infeasibility_reason)

    def test_qp_joint_separation_intervention(self):
        env = AircraftEnv(3, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset(boundary_scenario())
        shield = ConflictAwareQPSafetyShield(env.cfg)
        _, events = shield.filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertTrue(events)
        self.assertTrue(any("S02" in reason for event in events for reason in event.reasons))

    def test_sparse_active_set_excludes_far_aircraft(self):
        env = AircraftEnv(5, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0.0, 0.0, 5000.0), (10000.0, 20000.0, 5200.0),
                          (20000.0, 40000.0, 5400.0), (30000.0, 60000.0, 5600.0),
                          (40000.0, 80000.0, 5800.0)],
            "resources": [0.9] * 5,
        })
        shield = ConflictAwareQPSafetyShield(env.cfg, sparse_active_set=True)
        shield.filter(observations, {i: Action(Mode.COVER if i == 0 else Mode.HOLD) for i in observations})
        self.assertEqual(shield.last_solution.solver_agents, [])
        self.assertEqual(shield.last_solution.solver_components, [])

    def test_sparse_component_matches_global_on_single_conflict(self):
        scenario = {
            "positions": [(0.0, 0.0, 5000.0), (700.0, 0.0, 5000.0),
                          (20000.0, 20000.0, 5200.0), (30000.0, 30000.0, 5400.0),
                          (40000.0, 40000.0, 5600.0)],
            "resources": [0.8] * 5,
        }
        nominal = {i: Action(Mode.HOLD) for i in range(5)}
        env_sparse = AircraftEnv(5, SimConfig(noise_position=0.0, noise_speed=0.0))
        obs_sparse = env_sparse.reset(scenario)
        sparse = ConflictAwareQPSafetyShield(env_sparse.cfg, sparse_active_set=True)
        sparse.filter(obs_sparse, nominal)
        env_global = AircraftEnv(5, SimConfig(noise_position=0.0, noise_speed=0.0))
        obs_global = env_global.reset(scenario)
        global_shield = ConflictAwareQPSafetyShield(env_global.cfg, sparse_active_set=False)
        global_shield.filter(obs_global, nominal)
        self.assertEqual(sparse.last_solution.solver_components, [[0, 1]])
        self.assertEqual(global_shield.last_solution.solver_components, [[0, 1, 2, 3, 4]])
        self.assertEqual(sparse.last_solution.status, global_shield.last_solution.status)
        self.assertGreaterEqual(sparse.last_solution.cross_component_violations, 0)

    def test_cross_component_pair_is_merged_before_return(self):
        env = AircraftEnv(5, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0.0, 0.0, 5000.0), (700.0, 0.0, 5000.0),
                          (20000.0, 20000.0, 5200.0), (30000.0, 30000.0, 5400.0),
                          (40000.0, 40000.0, 5600.0)],
            "resources": [0.8] * 5,
        })
        shield = ConflictAwareQPSafetyShield(env.cfg, sparse_active_set=True)
        shield._active_solver_agents = lambda selected, report, ids: ([0, 1], [[0], [1]])
        shield.filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertGreaterEqual(shield.last_solution.cross_component_repairs, 1)
        self.assertIn([0, 1], shield.last_solution.solver_components)

    def test_post_solve_check_repairs_an_unsafe_pair_omitted_from_active_set(self):
        env = AircraftEnv(4, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0.0, 0.0, 5000.0), (700.0, 0.0, 5000.0),
                          (20000.0, 20000.0, 5200.0), (30000.0, 30000.0, 5400.0)],
            "resources": [0.8] * 4,
        })
        shield = ConflictAwareQPSafetyShield(env.cfg, sparse_active_set=True)
        shield._active_solver_agents = lambda selected, report, ids: ([], [])
        shield.filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertGreaterEqual(shield.last_solution.cross_component_repairs, 1)
        self.assertIn([0, 1], shield.last_solution.solver_components)

    def test_disabling_sparse_active_set_restores_global_component(self):
        env = AircraftEnv(5, SimConfig(noise_position=0.0, noise_speed=0.0))
        observations = env.reset({
            "positions": [(0.0, 0.0, 5000.0), (10000.0, 20000.0, 5200.0),
                          (20000.0, 40000.0, 5400.0), (30000.0, 60000.0, 5600.0),
                          (40000.0, 80000.0, 5800.0)],
            "resources": [0.9] * 5,
        })
        shield = ConflictAwareQPSafetyShield(env.cfg, sparse_active_set=False)
        shield.filter(observations, {i: Action(Mode.HOLD) for i in observations})
        self.assertEqual(shield.last_solution.solver_agents, [0, 1, 2, 3, 4])
        self.assertEqual(shield.last_solution.solver_components, [[0, 1, 2, 3, 4]])

"""Time budget in 48 GB-hours and the schedule that fits it."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs604.engine.budget import (
    budget_wall_seconds,
    cosine_lr_factor,
    effective_step,
    lr_schedule_factor,
    lr_schedule_from_config,
    training_limit_seconds,
)


class BudgetTests(unittest.TestCase):
    def test_wall_seconds_scale_with_gpu_memory(self) -> None:
        self.assertAlmostEqual(budget_wall_seconds(2.0, 48.0), 2 * 3600)
        self.assertAlmostEqual(budget_wall_seconds(2.0, 24.0), 4 * 3600)
        self.assertAlmostEqual(budget_wall_seconds(2.0, 96.0), 1 * 3600)
        self.assertAlmostEqual(budget_wall_seconds(2.0, 40.0), 2.4 * 3600)
        with self.assertRaises(ValueError):
            budget_wall_seconds(0.0, 40.0)

    def test_without_budget_schedule_follows_steps(self) -> None:
        self.assertEqual(effective_step(100, 1000, 9999.0, None), 100.0)

    def test_schedule_runs_ahead_when_time_does(self) -> None:
        # Half the steps done but 75% of the time used: the schedule is at 75%.
        self.assertEqual(effective_step(500, 1000, 750.0, 1000.0), 750.0)
        # Faster than planned: steps lead, time never pulls the schedule back.
        self.assertEqual(effective_step(800, 1000, 100.0, 1000.0), 800.0)
        # Out of time: the schedule is at its end.
        self.assertEqual(effective_step(10, 1000, 2000.0, 1000.0), 1000.0)

    def test_learning_rate_reaches_zero_at_the_time_limit(self) -> None:
        position = effective_step(300, 1000, 1000.0, 1000.0)
        self.assertAlmostEqual(cosine_lr_factor(position, 50, 1000), 0.0)
        self.assertAlmostEqual(cosine_lr_factor(49, 50, 1000), 1.0)
        self.assertGreater(cosine_lr_factor(525, 50, 1000), 0.49)

    def test_constant_schedule_keeps_warmup_then_holds_peak(self) -> None:
        for step in (0, 10, 49):
            self.assertEqual(
                lr_schedule_factor(step, 50, 1000, "constant"),
                cosine_lr_factor(step, 50, 1000),
            )
        for step in (50, 525, 999, 1000, 1500):
            self.assertEqual(lr_schedule_factor(step, 50, 1000, "constant"), 1.0)
        self.assertEqual(lr_schedule_factor(525, 50, 1000, "cosine"), cosine_lr_factor(525, 50, 1000))

    def test_schedule_key_defaults_to_cosine_and_rejects_unknown(self) -> None:
        self.assertEqual(lr_schedule_from_config({}), "cosine")
        self.assertEqual(lr_schedule_from_config({"lr_schedule": None}), "cosine")
        self.assertEqual(lr_schedule_from_config({"lr_schedule": "constant"}), "constant")
        with self.assertRaises(ValueError):
            lr_schedule_from_config({"lr_schedule": "step"})

    def test_training_stops_early_enough_for_final_evaluation(self) -> None:
        self.assertEqual(training_limit_seconds(8640.0, 60.0), 8580.0)
        self.assertEqual(training_limit_seconds(30.0, 60.0), 0.0)


if __name__ == "__main__":
    unittest.main()

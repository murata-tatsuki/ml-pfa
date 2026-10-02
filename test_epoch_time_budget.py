import json
from pathlib import Path
import tempfile
import unittest

from epoch_time_budget import EpochTimeBudget


class EpochTimeBudgetTests(unittest.TestCase):
    def test_stop_before_starting_an_epoch_that_will_not_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'stop.json'
            budget = EpochTimeBudget(48 * 3600 - 600, marker)
            for duration in (7200, 7100, 7300):
                budget.record(duration)
            self.assertFalse(budget.should_stop(21, now=44 * 3600))
            self.assertFalse(marker.exists())
            self.assertTrue(budget.should_stop(22, now=46 * 3600))
            state = json.loads(marker.read_text())
            self.assertEqual(state['next_epoch'], 22)
            self.assertEqual(state['predicted_seconds'], 7300 * 1.2)
            self.assertEqual(state['remaining_seconds'], 2 * 3600 - 600)

    def test_recent_slow_epoch_is_not_hidden_by_the_average(self):
        budget = EpochTimeBudget(300, 'unused')
        for duration in (10, 10, 100, 10, 10):
            budget.record(duration)
        self.assertFalse(budget.should_stop(5, now=179))
        # Old startup outliers leave the five-epoch window eventually.
        for _ in range(5):
            budget.record(10)
        self.assertEqual(budget.durations, [10] * 5)
        self.assertFalse(budget.should_stop(10, now=287))

    def test_first_epoch_without_measurements_starts(self):
        budget = EpochTimeBudget(300, 'unused')
        self.assertFalse(budget.should_stop(0, now=1))

    def test_disabled_budget_preserves_manual_training(self):
        budget = EpochTimeBudget()
        budget.record(10000)
        self.assertFalse(budget.should_stop(50, now=100000))

    def test_expired_deadline_stops_even_before_first_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = EpochTimeBudget(300, Path(directory) / 'stop.json')
            self.assertTrue(budget.should_stop(0, now=301))


if __name__ == '__main__':
    unittest.main()

"""Test recovery when live detection changes the robot's starting clearance."""

from pathlib import Path
import unittest

from live_tabletop import OverlayWalker
from navigation_learning import SETTINGS


class LiveNavigationTests(unittest.TestCase):
    def test_valid_tap_recovers_from_an_obstacle_over_the_spawn(self):
        checkpoint = Path(__file__).resolve().parents[1] / "runs/baseline-seed0/policy.zip"
        walker = OverlayWalker(checkpoint)
        self.addCleanup(walker.close)
        walker.replan([])
        walker.replan([(0.5, 2.1, 1.2, 2.8)])

        self.assertIn("clear starting point", walker.choose_destination((6.1, 3.6), SETTINGS[0]))
        for _ in range(900):
            if walker.advance(1) == "Arrived":
                break
        self.assertEqual(walker.status, "Arrived")

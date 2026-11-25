"""Guard against interruptions caused by unstable tabletop fits."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from table_vision import TableTracker


class TableTrackingTests(unittest.TestCase):
    def test_keeps_plane_when_detection_loses_far_edge(self):
        full_table = np.array([(210, 140), (480, 140), (610, 385), (70, 385)])
        partial_table = np.array([(176, 200), (515, 200), (610, 385), (70, 385)])
        frame = Image.new("RGB", (640, 400), "#4d5458")
        ImageDraw.Draw(frame).polygon([tuple(point) for point in full_table], fill="#dcdedc")
        hidden = Image.new("RGB", frame.size, "#4d5458")
        tracker = TableTracker()

        with patch("table_vision.find_table", side_effect=[
            *([full_table] * 3), *([partial_table] * 5), None, full_table
        ]):
            self.assertIsNone(tracker.update(frame))
            self.assertIsNone(tracker.update(frame))
            lock = tracker.update(frame)
            self.assertIsNotNone(lock)
            for _ in range(5):
                self.assertIs(tracker.update(frame), lock)
            self.assertIsNone(tracker.update(hidden))
            self.assertIs(tracker.update(frame), lock)

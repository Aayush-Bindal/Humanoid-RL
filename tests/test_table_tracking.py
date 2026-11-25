"""Guard against interruptions caused by unstable tabletop fits."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from table_vision import TablePlane, TableTracker, find_table


class TableTrackingTests(unittest.TestCase):
    def test_recovers_far_edge_behind_phone_and_bag(self):
        frame = Image.new("RGB", (960, 600), "#636767")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((615, 142, 780, 179), fill="#d1d2cf")
        draw.polygon(((419, 225), (891, 236), (960, 276), (960, 599), (71, 599)),
                     fill="#daddda")
        draw.rounded_rectangle((485, 287, 617, 315), radius=7, fill="#363b3d")
        draw.rounded_rectangle((482, 395, 538, 467), radius=8, fill="#383830")
        draw.polygon(((960, 244), (896, 264), (856, 309), (809, 350), (798, 387),
                      (825, 435), (798, 495), (867, 546), (960, 575)), fill="#292c2b")

        corners = find_table(frame)
        self.assertIsNotNone(corners)
        self.assertLess(max(corners[:2, 1]), 265)
        self.assertGreater(corners[1, 0], 850)
        plane = TablePlane(corners)
        self.assertTrue(plane.visible_in(frame))
        boxes = plane.obstacles(frame, 32)
        for center in ((550, 301), (512, 430), (900, 413)):
            x, y = plane.world_point(*center)
            self.assertTrue(any(x0 <= x <= x1 and y0 <= y <= y1
                                for x0, y0, x1, y1 in boxes), (center, boxes))

    def test_expands_stable_lock_only_after_repeated_far_edge_evidence(self):
        full_table = np.array([(210, 140), (480, 140), (610, 385), (70, 385)])
        partial_table = np.array([(176, 200), (515, 200), (610, 385), (70, 385)])
        frame = Image.new("RGB", (640, 400), "#4d5458")
        ImageDraw.Draw(frame).polygon([tuple(point) for point in full_table], fill="#dcdedc")
        tracker = TableTracker()

        with patch("table_vision.find_table", side_effect=[
            *([partial_table] * 3), *([full_table] * 3), partial_table
        ]):
            tracker.update(frame)
            tracker.update(frame)
            partial = tracker.update(frame)
            self.assertIsNotNone(partial)
            self.assertIs(tracker.update(frame), partial)
            self.assertIs(tracker.update(frame), partial)
            full = tracker.update(frame)
            self.assertTrue(tracker.refined)
            self.assertIsNot(full, partial)
            self.assertIs(tracker.update(frame), full)
            self.assertFalse(tracker.refined)

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

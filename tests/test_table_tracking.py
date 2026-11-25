"""Guard against interruptions caused by unstable tabletop fits."""

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from table_vision import TablePlane, TableTracker, find_table


class TableTrackingTests(unittest.TestCase):
    def test_finds_outer_edge_when_only_part_of_a_wide_table_is_bright(self):
        frame = Image.new("RGB", (960, 600), "#58545d")
        draw = ImageDraw.Draw(frame)
        draw.polygon(((0, 100), (165, 110), (215, 370), (34, 596), (0, 596)), fill="#842b39")
        draw.rectangle((480, 20, 675, 101), fill="#adabad")
        draw.polygon(((427, 128), (959, 148), (959, 599), (34, 599)), fill="#777a7a")
        draw.polygon(((440, 386), (896, 400), (850, 599), (246, 599)), fill="#d6d8d2")
        draw.polygon(((690, 195), (905, 199), (930, 455), (510, 420)), fill="#d6d8d2")
        draw.rounded_rectangle((580, 322, 660, 369), radius=10, fill="#686c6d")
        draw.polygon(((761, 318), (793, 312), (960, 423), (960, 448)), fill="#353534")
        for height in (540, 600, 720):
            with self.subTest(height=height):
                view = frame.resize((960, height))
                corners = find_table(view)
                self.assertIsNotNone(corners)
                self.assertLess(corners[0, 1] / height, 0.25)
                self.assertLess(corners[3, 0], 95)
                self.assertGreater(corners[2, 0], 925)
                target = TablePlane(corners).world_point(475, 230 * height / 600)
                self.assertGreater(target[0], 0)

    def test_accepts_a_table_close_to_all_four_camera_edges(self):
        frame = Image.new("RGB", (960, 540), "#d6d8d2")
        self.assertIsNone(find_table(frame))
        ImageDraw.Draw(frame).rectangle((0, 0, 959, 16), fill="#555957")
        corners = find_table(frame)
        self.assertIsNotNone(corners)
        self.assertLess(max(corners[:2, 1]), 38)
        self.assertGreater(corners[2, 0], 925)

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

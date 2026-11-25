"""Find the white tabletop, then detect obstacles in its rectified plane."""

from collections import deque
import numpy as np
from PIL import Image, ImageFilter

from tabletop_demo import HEIGHT, LENGTH, SPAN, WIDTH, obstacle_boxes


TABLE_SIZE = (640, 400)


def perspective(source, destination):
    """Return Pillow's inverse mapping from output points to input points."""
    rows, values = [], []
    for (x, y), (u, v) in zip(source, destination):
        rows.append((x, y, 1, 0, 0, 0, -u * x, -u * y))
        rows.append((0, 0, 0, x, y, 1, -v * x, -v * y))
        values.extend((u, v))
    return np.linalg.solve(rows, values)


def white_spans(row):
    """Bridge small gaps caused by objects and reflections."""
    indices = np.flatnonzero(row)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 29) + 1)
    return [(int(group[0]), int(group[-1])) for group in groups
            if len(group) > 28 and len(group) > 0.57 * (group[-1] - group[0] + 1)]


def table_edge(rows, column, clipped):
    """Fit a straight edge while ignoring frame clipping and short occlusions."""
    points = np.array([(row[0], row[column]) for row in rows if row[column] != clipped])
    if len(points) < 7 or np.ptp(points[:, 0]) < 10:
        return np.array([0, clipped])
    best = np.zeros(len(points), dtype=bool)
    for first in range(0, len(points), 3):
        for second in range(first + 5, len(points), 3):
            dy, dx = points[second] - points[first]
            if abs(dy) < 10:
                continue
            slope = dx / dy
            fitted = points[first, 1] + slope * (points[:, 0] - points[first, 0])
            inliers = np.abs(fitted - points[:, 1]) <= 3
            if inliers.sum() > best.sum():
                best = inliers
    if best.sum() < 5:
        return np.array([0, np.median(points[:, 1])])
    return np.polyfit(points[best, 0], points[best, 1], 1)


def find_table(frame):
    """Locate the largest continuous white surface extending toward the camera."""
    small = frame.resize((WIDTH, HEIGHT)).filter(ImageFilter.GaussianBlur(1))
    pixels = np.asarray(small, dtype=np.int16)
    neutral = (pixels.min(axis=2) >= 128) & (np.ptp(pixels, axis=2) < 51)
    spans = [white_spans(row) for row in neutral]
    seed = None
    for y in range(HEIGHT - 4, HEIGHT // 2, -2):
        candidates = [(left, right) for left, right in spans[y] if right - left > WIDTH * 0.28]
        if not candidates:
            continue
        left, right = max(candidates, key=lambda pair: pair[1] - pair[0])
        seed = (y, left, right)
        break
    if seed is None:
        return None

    rows = [seed]
    for direction in (-1, 1):
        last_y, left, right = seed
        for y in range(seed[0] + direction, HEIGHT - 1 if direction > 0 else HEIGHT // 8, direction):
            candidates = []
            width = right - left
            for start, end in spans[y]:
                overlap = min(end, right) - max(start, left)
                # A side can reappear abruptly when the scan passes an object
                # touching the edge. The opposite edge should stay aligned.
                aligned_side = min(abs(start - left), abs(end - right)) <= 5
                if overlap >= max(34, width * 0.70) and (aligned_side or end - start <= width * 1.25 + 8):
                    candidates.append((overlap, start, end))
            if candidates:
                _, left, right = max(candidates)
                rows.append((y, left, right))
                last_y = y
            elif abs(y - last_y) > 5:
                break
    top, bottom = min(row[0] for row in rows), max(row[0] for row in rows)
    if bottom - top < HEIGHT * 0.23:
        return None
    if top <= HEIGHT // 8 + 6 and all(row[1] < 8 and row[2] > WIDTH - 9 for row in rows):
        return None
    left = table_edge(rows, 1, 0)
    right = table_edge(rows, 2, WIDTH - 1)
    # Inset the fitted boundary to keep edge pixels and room background out.
    top += 3
    bottom -= 3
    corners = []
    for y, edge, inset in ((top, left, 4), (top, right, -4),
                           (bottom, right, -4), (bottom, left, 4)):
        x = np.clip(np.polyval(edge, y) + inset, 3, WIDTH - 4)
        corners.append((x, y))
    quad = np.array(corners)
    if min(quad[1, 0] - quad[0, 0], quad[2, 0] - quad[3, 0]) < WIDTH * 0.21:
        return None
    quad *= np.array([frame.width / WIDTH, frame.height / HEIGHT])
    return quad


class TablePlane:
    def __init__(self, corners):
        self.corners = corners
        width, height = TABLE_SIZE
        rectangle = ((0, 0), (width, 0), (width, height), (0, height))
        self.rectification = perspective(rectangle, corners)
        self.projection = perspective(corners, rectangle)

    def project(self, layer, size):
        return layer.transform(size, Image.Transform.PERSPECTIVE, self.projection,
                               resample=Image.Resampling.BILINEAR)

    def visible_in(self, frame):
        """Check whether the locked plane still covers a mostly white surface."""
        width, height = TABLE_SIZE
        crop = (self.image_point(40, 40), self.image_point(width - 40, 40),
                self.image_point(width - 40, height - 40), self.image_point(40, height - 40))
        samples = np.asarray(frame.transform((24, 16), Image.Transform.QUAD,
                            (*crop[0], *crop[3], *crop[2], *crop[1])), dtype=np.int16)
        white = (samples.min(axis=2) >= 128) & (np.ptp(samples, axis=2) < 51)
        return white.mean() >= 0.68 and white[-4:].mean() >= 0.60

    def image_point(self, x, y):
        """Map a point in rectified tabletop pixels to its webcam position."""
        return self.map_point(self.rectification, x, y)

    def world_point(self, x, y):
        """Convert a click in webcam pixels to a point on the walking plane."""
        u, v = self.map_point(self.projection, x, y)
        width, height = TABLE_SIZE
        if not (0 <= u <= width and 0 <= v <= height):
            raise ValueError("Click inside the detected tabletop.")
        return (u / width * LENGTH, v / height * SPAN)

    @staticmethod
    def map_point(transform, x, y):
        a, b, c, d, e, f, g, h = transform
        divisor = g * x + h * y + 1
        if abs(divisor) < 1e-8:
            raise ValueError("Point is outside the table projection.")
        return np.array([(a * x + b * y + c) / divisor, (d * x + e * y + f) / divisor])

    def obstacles(self, frame, threshold):
        rectified = frame.transform(TABLE_SIZE, Image.Transform.PERSPECTIVE, self.rectification,
                                    resample=Image.Resampling.BILINEAR)
        small = rectified.resize((WIDTH, HEIGHT)).filter(ImageFilter.GaussianBlur(1))
        pixels = np.asarray(small, dtype=np.int16)
        # Estimate the white level per strip so a lighting gradient does not
        # get classified as one large object.
        white = np.percentile(pixels[:, 8:-8], 75, axis=1)
        darker = np.mean(white[:, None] - pixels, axis=2)
        chroma = np.ptp(pixels, axis=2)
        foreground = (darker > threshold) | ((chroma > 42) & (darker > 6))
        mask = Image.fromarray(foreground.astype(np.uint8) * 255)
        mask = np.array(mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(5)),
                        dtype=bool)
        mask[:5, :] = mask[-5:, :] = False
        mask[:, :5] = mask[:, -5:] = False
        return obstacle_boxes(mask)


class TableTracker:
    """Keep a walking plane anchored when a fit changes due to occlusion."""

    def __init__(self):
        self.plane = None
        self.pending = deque(maxlen=3)
        self.misses = 0

    def update(self, frame):
        corners = find_table(frame)
        scale = frame.width / TABLE_SIZE[0]
        supported = self.plane is not None and self.plane.visible_in(frame)
        if corners is None:
            self.pending.clear()
            self.misses += 1
            return self.plane if supported and self.misses <= 3 else None

        self.misses = 0
        # A new fit can pick up nearby white objects or lose the far edge.
        # Keep the established map while its interior still shows the table.
        if supported:
            self.pending.clear()
            return self.plane

        if self.pending and np.max(np.abs(corners - self.pending[-1])) > 22 * scale:
            self.pending.clear()
        self.pending.append(corners)
        if len(self.pending) == self.pending.maxlen:
            candidate = TablePlane(np.median(self.pending, axis=0))
            self.pending.clear()
            if candidate.visible_in(frame):
                self.plane = candidate
                return self.plane
        return self.plane if supported else None

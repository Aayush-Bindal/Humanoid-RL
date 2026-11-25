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


def trace_upper_side(neutral, brightness, side, top):
    """Follow the left edge upward even when objects split the white rows."""
    recovered = []
    last_y = top
    for y in range(top - 1, 2, -1):
        expected = round(np.polyval(side, y))
        matches = []
        for x in range(max(7, expected - 6), min(WIDTH - 8, expected + 7)):
            inside = brightness[y, x + 2:x + 8]
            outside = brightness[y, x - 7:x - 1]
            contrast = inside.mean() - outside.mean()
            if neutral[y, x + 2:x + 8].sum() >= 5 and contrast >= 12:
                matches.append((contrast, x))
        if matches:
            _, x = max(matches)
            recovered.append((y, x))
            last_y = y
        elif last_y - y > 8:
            break
    return recovered


def outer_left_edge(neutral, brightness, bottom, inner_edge):
    """Find a long white-surface boundary independently of the starting span."""
    columns = np.arange(12, WIDTH - 12)
    white_sum = np.pad(np.cumsum(neutral, axis=1), ((0, 0), (1, 0)))
    light_sum = np.pad(np.cumsum(brightness, axis=1), ((0, 0), (1, 0)))
    inside = (white_sum[:, columns + 12] - white_sum[:, columns + 2]) / 10
    outside = (white_sum[:, columns - 2] - white_sum[:, columns - 12]) / 10
    contrast = (light_sum[:, columns + 12] - light_sum[:, columns + 2]
                - light_sum[:, columns - 2] + light_sum[:, columns - 12]) / 10
    strength = np.where((inside >= 0.72) & ((contrast >= 15) | (inside - outside >= 0.5)),
                        np.clip(contrast, 0, 50) + 28 * (inside - outside), 0)
    strength = np.pad(strength, ((0, 0), (12, 12)))
    # Allow a few pixels of error for rounding and for a partly rounded rim.
    strength = np.maximum.reduce((strength, np.roll(strength, 2, axis=1),
                                  np.roll(strength, -2, axis=1)))
    rows = np.arange(4, bottom + 1, 2)
    near_limit = min(WIDTH * 0.61, np.polyval(inner_edge, bottom) + 12)
    near_positions = np.arange(-60, near_limit + 1, 2)
    if not len(near_positions):
        return None
    best_score, best = 0.0, None
    for slope in np.arange(-2.1, 0.61, 0.05):
        x = np.rint(near_positions[None] + slope * (rows[:, None] - bottom)).astype(int)
        scores = strength[rows[:, None], np.clip(x, 0, WIDTH - 1)]
        scores[(x < 12) | (x >= WIDTH * 0.73)] = 0
        counts = (scores > 0).sum(axis=0)
        scores = scores.sum(axis=0)
        scores[counts < HEIGHT * 0.13] = 0
        index = int(scores.argmax())
        if scores[index] > best_score:
            best_score = scores[index]
            best = np.array([slope, near_positions[index] - slope * bottom])
    if best is None:
        return None

    xs = np.rint(np.polyval(best, rows)).astype(int)
    hit = (xs >= 12) & (xs < WIDTH * 0.73) & (strength[rows, np.clip(xs, 0, WIDTH - 1)] > 0)
    observed_rows = rows[hit]
    runs = np.split(observed_rows, np.flatnonzero(np.diff(observed_rows) > 14) + 1)
    edge = max(runs, key=len)
    if len(edge) < HEIGHT * 0.12 or edge[-1] < HEIGHT * 0.55:
        return None
    return best, int(edge[0])


def far_boundary(brightness, neutral, left, right, top, bottom, *, search_wider=False):
    """Find a visible cross-table edge supported by several separated columns."""
    right_x = float(np.polyval(right, top))
    # Average both sides of candidate edges and ignore isolated highlights.
    averaged = np.vstack((np.zeros(WIDTH), np.cumsum(brightness, axis=0)))
    best_score, best_edge = 0, (top, top, right_x, False)
    start = max(7, top - 9)
    end = min(HEIGHT - 10, bottom - 42)
    if search_wider:
        start = max(7, top + 9)
        end = min(end, top + 92)
    else:
        end = min(end, top + 9)
    stride = 2 if search_wider else 1
    for y in range(start, end, stride):
        left_x = float(np.polyval(left, y))
        right_x = float(np.polyval(right, y))
        columns = np.arange(max(4, round(left_x) + 7), WIDTH - 4, 3)
        if len(columns) < 16:
            continue
        expected_columns = (columns <= right_x).sum()
        for slope in np.arange(-0.16, 0.17, 0.025):
            rows = np.rint(y + (columns - left_x) * slope).astype(int)
            valid = (rows >= 6) & (rows < HEIGHT - 7)
            row, col = rows[valid], columns[valid]
            above = (averaged[row - 1, col] - averaged[row - 6, col]) / 5
            below = (averaged[row + 7, col] - averaged[row + 2, col]) / 5
            contrast = below - above
            hits = neutral[row + 2, col] & (contrast >= 14)
            supported = col[hits]
            if len(supported) < max(11, expected_columns * 0.47):
                continue
            # Avoid carrying the boundary across a large gap into the room.
            gaps = np.flatnonzero(np.diff(supported) > 25)
            if len(gaps):
                supported = supported[:gaps[0] + 1]
            if len(supported) < 11 or supported[0] - left_x > 32:
                continue
            if supported[-1] - supported[0] < (right_x - left_x) * 0.55:
                continue
            on_edge = hits & (col <= supported[-1])
            sharpness = brightness[row + 2, col] - brightness[row - 2, col]
            score = (np.minimum(contrast[on_edge], 45).sum()
                     + np.minimum(sharpness[on_edge], 55).sum()
                     - (1 if search_wider else 6) * abs(y - top))
            if score > best_score:
                best_score = score
                intercept = y - left_x * slope
                left_y = round((slope * left[1] + intercept) / (1 - slope * left[0]))
                right_y = round((slope * right[1] + intercept) / (1 - slope * right[0]))
                observed_right = supported[-1]
                fitted_right = np.polyval(right, right_y)
                if search_wider or observed_right > fitted_right + 5 or fitted_right > observed_right + 12:
                    # A side fit can continue across the background beyond the
                    # visible far edge, especially beside a bright wall or bag.
                    right_y = round(slope * observed_right + intercept)
                else:
                    observed_right = fitted_right
                best_edge = (left_y, right_y, observed_right, True)
    return best_edge


def near_right_boundary(neutral, left, right, bottom):
    """Use white surface visible below a right-hand occluder when available."""
    left_x = max(0, round(np.polyval(left, bottom)))
    right_x = float(np.polyval(right, bottom))
    observations = []
    for row in neutral[max(0, bottom - 7):bottom + 1]:
        for start, end in white_spans(row):
            if abs(start - left_x) <= 13 and end > right_x + 5:
                observations.append(end)
    return max(right_x, np.median(observations)) if len(observations) >= 4 else right_x


def table_corners(neutral, left, right, bottom, far):
    far_left, far_right, far_right_x, _ = far
    if min(bottom - far_left, bottom - far_right) < HEIGHT * 0.23:
        return None
    bottom -= 3
    quad = np.array([
        (np.polyval(left, far_left + 3) + 4, far_left + 3),
        (far_right_x - 4, far_right + 3),
        (near_right_boundary(neutral, left, right, bottom) - 4, bottom),
        (np.polyval(left, bottom) + 4, bottom),
    ])
    quad[:, 0] = np.clip(quad[:, 0], 3, WIDTH - 4)
    if min(quad[1, 0] - quad[0, 0], quad[2, 0] - quad[3, 0]) < WIDTH * 0.21:
        return None
    return quad


def quad_area(corners):
    x, y = corners.T
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2


def find_table(frame):
    """Locate the largest continuous white surface extending toward the camera."""
    small = frame.resize((WIDTH, HEIGHT)).filter(ImageFilter.GaussianBlur(1))
    pixels = np.asarray(small, dtype=np.int16)
    brightness = pixels.mean(axis=2)
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
        for y in range(seed[0] + direction, HEIGHT - 1 if direction > 0 else 2, direction):
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
    if bottom - top < HEIGHT * 0.09:
        return None
    if (top <= 7 and neutral[:3].mean() > 0.82
            and all(row[1] < 8 and row[2] > WIDTH - 9 for row in rows)):
        return None
    left = table_edge(rows, 1, 0)
    right = table_edge(rows, 2, WIDTH - 1)
    recovered = trace_upper_side(neutral, brightness, left, top)
    if len(recovered) >= 5:
        # A projected edge can extend well past a dark phone or bag that
        # divided the upper rows into spans too small for the initial scan.
        top = recovered[-1][0]
    size_scale = np.array([frame.width / WIDTH, frame.height / HEIGHT])
    far = far_boundary(brightness, neutral, left, right, top, bottom)
    # White walls can connect the row scan to the top of the video. Require
    # a visible cross-table edge before accepting that larger footprint.
    if not far[3]:
        recovered_far = far_boundary(brightness, neutral, left, right, top, bottom, search_wider=True)
        if recovered_far[3]:
            far = recovered_far
        elif top <= 7:
            return None
    quad = table_corners(neutral, left, right, bottom, far)
    paperlike = neutral | ((brightness >= 112) & (np.ptp(pixels, axis=2) < 31))
    outer = outer_left_edge(paperlike, brightness, bottom, left)
    if outer is not None:
        outer_edge, outer_top = outer
        far = far_boundary(brightness, paperlike, outer_edge, right, outer_top, bottom)
        expanded = table_corners(paperlike, outer_edge, right, bottom, far)
        if far[3] and expanded is not None and (quad is None or quad_area(expanded) >= 1.09 * quad_area(quad)):
            # The extra area must contain white surface in the camera.
            if TablePlane(expanded * size_scale).visible_in(frame):
                quad = expanded
    return quad * size_scale if quad is not None else None


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
        spread = np.ptp(samples, axis=2)
        white = ((samples.min(axis=2) >= 128) & (spread < 51)
                 | ((samples.mean(axis=2) >= 112) & (spread < 31)))
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
        self.refined = False

    def extends_surface(self, corners, scale):
        """Require a larger tabletop fit to contain the existing walking area."""
        locked = self.plane.corners
        if (abs(np.mean(corners[2:, 1]) - np.mean(locked[2:, 1])) > 16 * scale
                or quad_area(corners) < 1.18 * quad_area(locked)):
            return False
        # A changed fit should add surface, rather than trade one side for another.
        candidate = TablePlane(corners)
        width, height = TABLE_SIZE
        for x, y in ((35, 35), (width - 35, 35), (35, height - 35), (width - 35, height - 35)):
            try:
                candidate.world_point(*self.plane.image_point(x, y))
            except ValueError:
                return False
        return True

    def update(self, frame):
        corners = find_table(frame)
        scale = frame.width / TABLE_SIZE[0]
        supported = self.plane is not None and self.plane.visible_in(frame)
        self.refined = False
        if corners is None:
            self.pending.clear()
            self.misses += 1
            return self.plane if supported and self.misses <= 3 else None

        self.misses = 0
        # A new fit can pick up nearby white objects or lose the far edge.
        # Keep the established map while its interior still shows the table.
        if supported and not self.extends_surface(corners, scale):
            self.pending.clear()
            return self.plane

        if self.pending and np.max(np.abs(corners - self.pending[-1])) > 22 * scale:
            self.pending.clear()
        self.pending.append(corners)
        if len(self.pending) == self.pending.maxlen:
            candidate = TablePlane(np.median(self.pending, axis=0))
            self.pending.clear()
            if candidate.visible_in(frame):
                self.refined = supported
                self.plane = candidate
                return self.plane
        return self.plane if supported else None

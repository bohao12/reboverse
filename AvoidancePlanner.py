"""
AvoidancePlanner.py  –  VFH+ (Vector Field Histogram Plus)

Key upgrade over the previous version:
- Continuous heading selection (gap-finding), not 4-direction grid snap
- The drone never needs to rotate 90° then 180°; it just steers into the
  widest open gap in one smooth move.
- Outputs a preferred_heading_rad so the caller can also align yaw smoothly.

FIX: _find_best_gap now prefers the centre of the FOV when multiple gaps
     have similar width, so the drone avoids unnecessary turns to the sides.
"""

import numpy as np
import math


class AvoidancePlanner:
    def __init__(
        self,
        K,
        width,
        height,
        max_speed: float = 1.0,
        safe_distance: float = 2.5,
        critical_distance: float = 0.8,
        num_bins: int = 72,
        min_gap_bins: int = 4,
        smoothing_alpha: float = 0.5,
    ):
        self.fx = float(K[0, 0])
        self.cx = float(K[0, 2])
        self.width = width
        self.height = height

        self.max_speed = max_speed
        self.safe_distance = safe_distance
        self.critical_distance = critical_distance
        self.num_bins = num_bins
        self.min_gap_bins = min_gap_bins
        self.alpha = smoothing_alpha

        # Smoothing state
        self.prev_north: float | None = None
        self.prev_east: float | None = None
        self.prev_down: float | None = None
        self.prev_heading_rad: float = 0.0

    # ------------------------------------------------------------------
    # Depth → polar obstacle histogram
    # ------------------------------------------------------------------
    def _build_histogram(self, depth_map: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = depth_map.shape
        histogram  = np.zeros(self.num_bins, dtype=np.float32)
        angles     = np.zeros(self.num_bins, dtype=np.float32)
        distances  = np.full(self.num_bins, self.safe_distance * 2, dtype=np.float32)

        # THE BLINDERS: only look at the horizontal band from 30%–60% of height
        horizon = depth_map[int(h * 0.3) : int(h * 0.6), :]

        for i in range(self.num_bins):
            x0 = int(i * w / self.num_bins)
            x1 = int((i + 1) * w / self.num_bins)
            region = horizon[:, x0:x1]

            d = float(np.nanpercentile(region, 20))
            distances[i] = d

            if d <= self.critical_distance:
                histogram[i] = 1.0
            elif d < self.safe_distance:
                histogram[i] = (self.safe_distance - d) / (self.safe_distance - self.critical_distance)
            else:
                histogram[i] = 0.0

            u_center   = (x0 + x1) / 2.0
            angles[i]  = math.atan((u_center - self.cx) / self.fx)

        return histogram, angles, distances

    # ------------------------------------------------------------------
    # Gap-finding: pick the widest open corridor, biased toward centre
    # ------------------------------------------------------------------
    def _find_best_gap(
        self, histogram: np.ndarray, angles: np.ndarray
    ) -> tuple[float, bool]:
        """
        Scans for runs of free (cost < 0.5) bins and returns the angle at
        the centre of the best gap.

        FIX: Among gaps whose width is within 2 bins of the maximum, the one
        whose centre is closest to the optical axis (bin num_bins//2) is preferred.
        This stops the drone from gratuitously veering left or right when it
        could go straight.

        Returns (best_angle_rad, found_gap).
        """
        free = histogram < 0.5

        gaps: list[tuple[int, int]] = []
        in_gap = False
        g_start = 0
        for i, f in enumerate(free):
            if f and not in_gap:
                g_start = i
                in_gap = True
            elif not f and in_gap:
                gaps.append((g_start, i - 1))
                in_gap = False
        if in_gap:
            gaps.append((g_start, len(free) - 1))

        # Filter out gaps that are too narrow to fly through
        gaps = [(s, e) for s, e in gaps if (e - s + 1) >= self.min_gap_bins]

        if not gaps:
            best_idx = int(np.argmin(histogram))
            return float(angles[best_idx]), False

        max_width = max(e - s + 1 for s, e in gaps)
        # Keep gaps within 2 bins of the widest
        candidates = [(s, e) for s, e in gaps if (e - s + 1) >= max_width - 2]

        # Among candidates, prefer the one whose centre is closest to bin centre
        centre_bin = self.num_bins // 2
        best_gap = min(candidates, key=lambda g: abs((g[0] + g[1]) // 2 - centre_bin))

        mid_idx = (best_gap[0] + best_gap[1]) // 2
        return float(angles[mid_idx]), True

    def _check_obstacle_height(self, depth_map: np.ndarray) -> str:
        """
        Analyzes the vertical profile of the centre column.
        Returns 'BOX' if top is clear but middle is blocked.
        Returns 'WALL' if both are blocked.
        """
        h, w = depth_map.shape
        center_w_start = w // 3
        center_w_end   = 2 * w // 3

        top_region = depth_map[0 : h // 5, center_w_start : center_w_end]
        mid_region = depth_map[2 * h // 5 : 3 * h // 5, center_w_start : center_w_end]

        try:
            top_depth = float(np.nanpercentile(top_region, 20))
            mid_depth = float(np.nanpercentile(mid_region, 20))
        except (ValueError, TypeError):
            return "CLEAR"

        if mid_depth < self.critical_distance:
            return "BOX" if top_depth > self.safe_distance else "WALL"
        return "CLEAR"

    # ------------------------------------------------------------------
    # Clearance helpers
    # ------------------------------------------------------------------
    def _clearance(self, depth_map: np.ndarray) -> tuple[float, float, float]:
        h, w = depth_map.shape
        horizon = depth_map[int(h * 0.3) : int(h * 0.6), :]
        l = float(np.nanpercentile(horizon[:, : w // 3], 20))
        c = float(np.nanpercentile(horizon[:, w // 3 : 2 * w // 3], 20))
        r = float(np.nanpercentile(horizon[:, 2 * w // 3 :], 20))
        return l, c, r

    def _is_blocked(self, left: float, center: float, right: float) -> bool:
        return (
            center < self.critical_distance
            and left  < self.safe_distance
            and right < self.safe_distance
        )

    def _environment_label(self, left: float, center: float, right: float) -> str:
        if center > self.safe_distance and left > self.safe_distance and right > self.safe_distance:
            return "OPEN"
        if center > self.safe_distance:
            return "FORWARD_CLEAR"
        return "LEFT_OPEN" if left > right else "RIGHT_OPEN"

    # ------------------------------------------------------------------
    # Position smoothing
    # ------------------------------------------------------------------
    def _smooth_pos(self, n: float, e: float, d: float) -> tuple[float, float, float]:
        if self.prev_north is None:
            self.prev_north, self.prev_east, self.prev_down = n, e, d
            return n, e, d
        n = self.alpha * self.prev_north + (1 - self.alpha) * n
        e = self.alpha * self.prev_east  + (1 - self.alpha) * e
        d = self.alpha * self.prev_down  + (1 - self.alpha) * d
        self.prev_north, self.prev_east, self.prev_down = n, e, d
        return n, e, d

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------
    def compute_position_ned(
        self,
        depth_map: np.ndarray,
        pose: dict,
        step_size: float = 1.5,
    ) -> tuple[float, float, float, float, dict]:
        """
        Args:
            depth_map : HxW float32 depth image (metres)
            pose      : {'north', 'east', 'down', 'yaw'}  yaw in radians
            step_size : metres to move per decision step

        Returns:
            north, east, down  – NED target position
            preferred_heading  – desired yaw (rad) for smooth rotation
            info               – diagnostic dict
        """
        histogram, angles, distances = self._build_histogram(depth_map)
        left, center, right = self._clearance(depth_map)
        blocked       = self._is_blocked(left, center, right)
        obstacle_type = self._check_obstacle_height(depth_map)
        cam_angle, found_gap = self._find_best_gap(histogram, angles)

        if center < self.critical_distance:
            cam_angle = -math.pi / 2 if left > right else math.pi / 2

        vx_body = math.cos(cam_angle)
        vy_body = math.sin(cam_angle)

        yaw       = float(pose.get("yaw", 0.0))
        north_dir = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
        east_dir  = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)

        norm = math.sqrt(north_dir**2 + east_dir**2) + 1e-9
        north_dir /= norm
        east_dir  /= norm

        north = float(pose["north"]) + step_size * north_dir
        east  = float(pose["east"])  + step_size * east_dir
        down  = float(pose["down"])
        north, east, down = self._smooth_pos(north, east, down)

        raw_heading = math.atan2(east_dir, north_dir)
        delta = raw_heading - self.prev_heading_rad
        while delta >  math.pi: delta -= 2 * math.pi
        while delta < -math.pi: delta += 2 * math.pi
        preferred_heading = self.prev_heading_rad + (1 - self.alpha) * delta
        self.prev_heading_rad = preferred_heading

        info = {
            "blocked":              blocked,
            "obstacle_type":        obstacle_type,
            "environment":          self._environment_label(left, center, right),
            "clearance":            {"left": left, "center": center, "right": right},
            "selected_cam_angle_deg": math.degrees(cam_angle),
            "preferred_heading_deg":  math.degrees(preferred_heading),
        }

        return north, east, down, preferred_heading, info
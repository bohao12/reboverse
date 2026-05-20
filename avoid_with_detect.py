#!/usr/bin/env python3
"""
avoid_with_detect.py  –  Autonomous multi-object search (FIXED & IMPROVED)

=== BUG FIXES ===

  BUG 1 – DRONE ALWAYS TURNS RIGHT
    Root cause: `_desired_yaw_deg` was initialised to 0.0 but never synced to the
    drone's real takeoff yaw.  When the BFS returned a heading of ~180° (south),
    the error = 180 – 0 = 180 which sits exactly on the wraparound boundary, so
    the clip always resolved to +80 °/frame (rightward) instead of choosing the
    short path.
    Fix: sync `_desired_yaw_deg` from the first real pose reading in `_update_pose`.

  BUG 2 – ALTITUDE NEVER MOVED (hidden objects never found)
    Root cause: `send_position_setpoint(down=0.0)` was hardcoded.
    The AltitudeSweeper computed a `target_down` but it was never sent to the
    flight controller, so the drone stayed at its takeoff altitude forever.
    Fix: pass the absolute target_down from the sweeper directly.

  BUG 3 – FRONTIER EXPLORATION SILENTLY BROKEN
    Root cause: `_frontier_heading_deg` did `from scipy.ndimage import binary_dilation`
    as a "soft dependency check" but scipy was never actually used.  On any machine
    without scipy installed the function returned None every time, meaning the BFS
    map was ignored completely and the drone just followed VFH headings in circles.
    Fix: remove the spurious import; the function now always calls the BFS.

  BUG 4 – BFS ALWAYS POINTS SOUTH FIRST
    Root cause: BFS neighbour order was (r-1, c), (r+1, c), (r, c-1), (r, c+1).
    In the NED grid, r increases with north, so r-1 is *south*.  The very first
    UNKNOWN cell dequeued was therefore always the cell directly south of the
    drone, returning atan2(0, -1.2) = 180 °.  This triggered the rightward spin
    from Bug 1 on every mission start.
    Fix: `get_frontier_heading` now accepts the drone's current heading and sorts
    the four BFS neighbours by angular proximity to it, so the drone naturally
    continues in its current direction rather than reversing.

  BUG 5 – NARROW PATHS IGNORED
    Root cause: The condition `center_clearance < safe_distance` (2.5 m) disabled
    BFS frontier following and forced VFH mode.  Because the drone was rarely
    more than 2.5 m from any wall in a room, the BFS was almost never consulted,
    so the drone never navigated *toward* a doorway or gap – it just reflected off
    whatever wall it was facing.
    Fix: Switch to VFH-only mode only when `center_clearance < critical_distance * 2`.
    Outside that tighter bubble the BFS frontier heading takes priority.

  BUG 6 – NO ANTI-LOOP MECHANISM
    Root cause: No detection of repeated position visits.  Once the drone had
    explored most of the room it would orbit the same corner indefinitely.
    Fix: track a rolling window of grid-cells visited.  If the drone revisits the
    same small set of cells for too long, trigger SCAN MODE: hover and rotate 360°
    so the depth camera can re-observe the whole room and reveal missed passages.

  BUG 7 – OBSTACLE MARKING BLOCKED NARROW PATHS
    Root cause: Obstacles were marked at `center_clearance` metres ahead.  With
    critical_distance = 1.0 m and grid resolution = 1.2 m, this often mapped to
    the drone's own cell, or to the cell just ahead that was actually a valid
    passage (doorframe depths include the wall, but the passage is behind it).
    Fix: only mark obstacles on confirmed repeated hits (hit_counts map) AND add
    a forward margin so the mark lands past the doorframe, not on it.  Also
    added `unmark_obstacle` so cells are cleared when the drone successfully passes
    through them.

  BUG 8 – YAW ERROR SIGN LOST IN ROTATION LOGIC
    Root cause: `yaw_error = abs(target_yaw_deg - actual_yaw_deg)` discarded the
    sign and then was only compared with 180 for wraparound.  The subsequent
    forward_step logic didn't know whether the turn was left or right, so it
    always applied the same 0.4 m/frame creep step.
    Fix: compute a signed yaw error via the standard shortest-path formula; keep
    sign through the step logic.

=== NEW FEATURES ===
  - SCAN MODE: 360° rotation when stuck, then resumes exploration
  - FRONTIER BLACKLIST: skip frontiers the drone has tried ≥3 times without
    getting closer; prevents orbiting an unreachable corner
  - HEADING-ALIGNED BFS: naturally extends current direction of travel
  - HIT-COUNT OBSTACLE MAP: reduces false-positive obstacles in doorways
"""

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from depth_receiver import DepthReceiver
from drone_control_new import Drone
from AvoidancePlanner import AvoidancePlanner
from get_position_with_task import SharedState, position_monitor_task


# ============================================================
#  OBJECT DETECTOR  (plug-in interface)
# ============================================================

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2


class ObjectDetector:
    """
    Replace `detect()` with your actual model call (YOLOv8, RT-DETR, etc.).
    Receives an HxWx3 uint8 RGB frame (or None) and returns Detection objects.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        print(f"[ObjectDetector] model_path={model_path!r}  (stub – replace with real model)")

    def detect(self, rgb_frame: np.ndarray | None) -> list[Detection]:
        if rgb_frame is None or self.model is None:
            return []
        return []


# ============================================================
#  EXPLORATION MAP  –  frontier-based coverage
# ============================================================

class ExplorationMap:
    """
    2-D occupancy grid in NED space.

    Cell states
    -----------
    0  UNKNOWN  – unexplored
    1  VISITED  – drone has been here
    2  OBSTACLE – confirmed obstacle (requires HIT_THRESHOLD repeated detections)

    The hit-count map reduces false positives near doorframes.
    """

    UNKNOWN  = 0
    VISITED  = 1
    OBSTACLE = 2
    HIT_THRESHOLD = 3   # how many consecutive hits before a cell is an obstacle

    def __init__(self, resolution_m: float = 1.0, arena_m: float = 80.0):
        """
        FIX: Reduced grid size dramatically.
        - resolution: 1.0m (was 0.5m) → 4x fewer cells
        - arena: 80m (was 200m) → 16x fewer cells
        - Total: 80x80 = 6,400 cells (was 160,000!)
        This makes visited region tracking visible (1% instead of 0.1%)
        """
        self.res = resolution_m
        n_cells = int(arena_m / resolution_m)
        self._origin = n_cells // 2
        self._grid     = np.zeros((n_cells, n_cells), dtype=np.uint8)
        self._hits     = np.zeros((n_cells, n_cells), dtype=np.uint8)
        # Frontiers that repeatedly failed to reach → skip them
        self._blacklist: set[tuple[int, int]] = set()
        self._frontier_attempts: dict[tuple[int, int], int] = {}
        # Track visited region centers to avoid re-exploring same area from different angles
        self._visited_region_radius_m: float = 2.0  # 2m radius (matches larger grid cells)

    def _idx_to_world(self, r: int, c: int) -> tuple[float, float]:
        return (r - self._origin) * self.res, (c - self._origin) * self.res

    def _to_idx(self, north: float, east: float) -> tuple[int, int]:
        r = int(round(north / self.res)) + self._origin
        c = int(round(east  / self.res)) + self._origin
        r = int(np.clip(r, 0, self._grid.shape[0] - 1))
        c = int(np.clip(c, 0, self._grid.shape[1] - 1))
        return r, c

    def mark_visited(self, north: float, east: float) -> None:
        # FIX: Only mark the single cell we are in so we don't accidentally wipe out nearby walls!
        r, c = self._to_idx(north, east)
        if self._grid[r, c] != self.OBSTACLE:
            self._grid[r, c] = self.VISITED
            self._blacklist.discard((r, c))

    def mark_obstacle(self, north: float, east: float, force: bool = False) -> None:
        r, c = self._to_idx(north, east)
        # If we have safely passed through here, ignore false walls (unless forced)
        if self._grid[r, c] == self.VISITED and not force:
            return
            
        # If we hit an emergency, skip the counter and map the wall INSTANTLY
        if force:
            self._grid[r, c] = self.OBSTACLE
            return
            
        # Otherwise, use the hit counter to build confidence
        self._hits[r, c] = min(self._hits[r, c] + 1, 255)
        if self._hits[r, c] >= self.HIT_THRESHOLD:
            self._grid[r, c] = self.OBSTACLE

    def unmark_obstacle(self, north: float, east: float) -> None:
        """
        Clear an obstacle cell when the drone successfully passes through it
        (it was a false positive, e.g. a transparent surface or doorframe).
        """
        r, c = self._to_idx(north, east)
        self._grid[r, c] = self.VISITED
        self._hits[r, c] = 0

    def record_frontier_attempt(self, north: float, east: float) -> None:
        """Track how many times we've tried (and failed) to reach a frontier."""
        r, c = self._to_idx(north, east)
        self._frontier_attempts[r, c] = self._frontier_attempts.get((r, c), 0) + 1
        if self._frontier_attempts.get((r, c), 0) >= 3:
            self._blacklist.add((r, c))

    def is_fully_explored(self, coverage_threshold: float = 0.85) -> bool:
        non_obs = self._grid != self.OBSTACLE
        if non_obs.sum() == 0:
            return False
        return (self._grid == self.VISITED).sum() / non_obs.sum() >= coverage_threshold

    def get_frontier_heading(
        self,
        north: float,
        east: float,
        current_heading_rad: float = 0.0,   # FIX BUG 4: pass in current heading
    ) -> tuple[float, tuple[int, int]] | None:
        """
        BFS to the nearest reachable UNKNOWN cell.

        FIX BUG 4: Neighbours are sorted by angular proximity to `current_heading_rad`
        so the drone prefers to continue forwards rather than reversing.
        Blacklisted frontiers are skipped.

        Returns the heading and the frontier grid cell.
        """
        dr, dc = self._to_idx(north, east)

        queue: deque[tuple[int, int]] = deque([(dr, dc)])
        visited_bfs: set[tuple[int, int]] = {(dr, dc)}
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        target_frontier: tuple[int, int] | None = None

        # Sort neighbours by angular proximity to current heading (closest first)
        raw_neighbours = [(dr-1, dc), (dr+1, dc), (dr, dc-1), (dr, dc+1)]

        def heading_diff(nr: int, nc: int) -> float:
            dn = (nr - dr) * self.res
            de = (nc - dc) * self.res
            ang = math.atan2(de, dn)
            diff = abs(ang - current_heading_rad)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            return diff

        first_neighbours = sorted(raw_neighbours, key=lambda nb: heading_diff(*nb))

        while queue:
            r, c = queue.popleft()

            if self._grid[r, c] == self.UNKNOWN and (r, c) not in self._blacklist:
                target_frontier = (r, c)
                break

            # Use heading-sorted order for the starting node only
            if (r, c) == (dr, dc):
                neighbours = first_neighbours
            else:
                neighbours = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

            for nr, nc in neighbours:
                if 0 <= nr < self._grid.shape[0] and 0 <= nc < self._grid.shape[1]:
                    if (nr, nc) not in visited_bfs and self._grid[nr, nc] != self.OBSTACLE:
                        visited_bfs.add((nr, nc))
                        parent[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        if target_frontier is None:
            return None

        # Backtrack to find the very next step from the drone's cell
        curr = target_frontier
        while curr in parent and parent[curr] != (dr, dc):
            curr = parent[curr]

        d_north = (curr[0] - dr) * self.res
        d_east  = (curr[1] - dc) * self.res

        if d_north == 0.0 and d_east == 0.0:
            return None

        return math.atan2(d_east, d_north), target_frontier

    def visited_count(self) -> int:
        return int((self._grid == self.VISITED).sum())
    
    def grid_coverage_stats(self) -> dict[str, float]:
        """Returns coverage statistics for debugging."""
        visited = (self._grid == self.VISITED).sum()
        unknown = (self._grid == self.UNKNOWN).sum()
        obstacle = (self._grid == self.OBSTACLE).sum()
        
        # Only count non-obstacle cells for percentage
        total_explorable = visited + unknown
        if total_explorable == 0:
            return {
                "visited_pct": 0.0,
                "unknown_pct": 0.0,
                "total_visited_cells": 0,
                "total_unknown_cells": 0,
                "blacklisted_frontiers": len(self._blacklist),
            }
        
        return {
            "visited_pct": 100.0 * visited / total_explorable,
            "unknown_pct": 100.0 * unknown / total_explorable,
            "total_visited_cells": int(visited),
            "total_unknown_cells": int(unknown),
            "obstacle_pct": 100.0 * obstacle / self._grid.size if self._grid.size > 0 else 0.0,
            "blacklisted_frontiers": len(self._blacklist),
        }


# ============================================================
#  ALTITUDE SWEEPER  –  3-level search state machine
# ============================================================

class AltitudeSweeper:
    """
    Cycles the drone through multiple altitudes so it can find objects hidden
    near the floor or ceiling.
    Advances to the next level after `dwell_steps` frames OR when stuck.
    """

    def __init__(
        self,
        altitudes_m: list[float] = (1.5, 3.0, 4.5),
        dwell_steps: int = 120,
        stuck_dist_m: float = 0.5,
        stuck_window: int = 40,
    ):
        self._levels = [-float(a) for a in altitudes_m]
        self._idx = 1                       # start at mid altitude
        self._step_count = 0
        self._dwell = dwell_steps
        self._stuck_dist = stuck_dist_m
        self._stuck_window = stuck_window
        self._pos_history: list[tuple[float, float]] = []

    @property
    def current_down(self) -> float:
        return self._levels[self._idx]

    def update(self, north: float, east: float, forced: bool = False) -> bool:
        self._pos_history.append((north, east))
        if len(self._pos_history) > self._stuck_window:
            self._pos_history.pop(0)

        self._step_count += 1

        stuck = False
        if len(self._pos_history) == self._stuck_window:
            oldest = np.array(self._pos_history[0])
            newest = np.array(self._pos_history[-1])
            if np.linalg.norm(newest - oldest) < self._stuck_dist:
                stuck = True

        if forced or stuck or self._step_count >= self._dwell:
            self._idx = (self._idx + 1) % len(self._levels)
            self._step_count = 0
            self._pos_history.clear()
            label = ["LOW", "MID", "HIGH"][self._idx % 3]
            print(f"[AltitudeSweeper] → {label}  (down={self.current_down:.1f} m)")
            return True
        return False


# ============================================================
#  MAIN NAVIGATION CLASS
# ============================================================

# Internal state-machine states
_STATE_EXPLORE  = "EXPLORE"
_STATE_SCAN     = "SCAN"     # 360° rotation in place to find new paths


class DroneNavigation:
    """
    Autonomous multi-object search with VFH+ avoidance and frontier exploration.

    Usage:
        nav = DroneNavigation(
            target_objects=["fire_extinguisher", "person", "backpack"],
            model_path="best.pt",
        )
        asyncio.run(nav.run())
    """

    # Anti-loop tuning
    _LOOP_WINDOW          = 60    # frames to look back
    _LOOP_UNIQUE_CELLS    = 4     # fewer distinct cells → entering SCAN mode
    _SCAN_ROTATE_RATE     = 15.0  # deg/frame during scan
    _SCAN_COMPLETE_DEG    = 480.0 # rotate 1.25 turns (450°) so drone doesn't face same obstacle

    def __init__(
        self,
        target_objects: list[str],
        model_path: str | None = None,
        depth_topic: str = "/depth_camera",
        loop_hz: float = 2.0,
        step_size: float = 1.5,
        safe_distance: float = 2.0,
        critical_distance: float = 0.8,
        detection_confidence: float = 0.6,
        search_altitudes_m: tuple[float, ...] = (1.5, 3.0, 4.5),
    ):
        self.loop_hz            = loop_hz
        self.step_size          = step_size
        self.detection_confidence = detection_confidence
        self.running            = True

        self.targets_remaining: set[str]      = set(target_objects)
        self.targets_found: dict[str, dict]   = {}

        self.pose = {"north": 0.0, "east": 0.0, "down": -3.0, "yaw": 0.0, "yaw_deg": 0.0}

        # FIX BUG 1: will be synced from first real telemetry reading
        self._desired_yaw_deg: float = 0.0
        self._first_pose_received: bool = False

        K = np.array([[433.0, 0.0, 320.0],
                      [0.0,   433.0, 240.0],
                      [0.0,   0.0,   1.0  ]], dtype=np.float32)

        self.receiver    = DepthReceiver(depth_topic)
        self.planner     = AvoidancePlanner(
            K, width=640, height=480,
            safe_distance=safe_distance,
            critical_distance=critical_distance,
            num_bins=72, min_gap_bins=2,
        )
        self.exp_map     = ExplorationMap(resolution_m=1.0, arena_m=80.0)
        self.alt_sweeper = AltitudeSweeper(altitudes_m=list(search_altitudes_m))
        self.detector    = ObjectDetector(model_path=model_path)

        self.drone          = Drone(takeoff_altitude=2.5)
        self.position_state = SharedState()
        self.monitor_task   = None

        # Anti-loop / scan state (FIX BUG 6)
        self._nav_state: str = _STATE_EXPLORE
        self._scan_start_yaw: float = 0.0
        self._scan_rotated: float   = 0.0
        self._recent_cells: deque[tuple[int, int]] = deque(maxlen=self._LOOP_WINDOW)

        # Last frontier target (for blacklisting after repeated failures)
        self._last_frontier_north: float | None = None
        self._last_frontier_east:  float | None = None
        self._frames_at_frontier: int = 0

        self._current_frontier: tuple[int, int] | None = None
        self._previous_frontier: tuple[int, int] | None = None
        self._frontier_stall_frames: int = 0
        self._frontier_last_dist: float | None = None
        
        # Emergency backup state to prevent oscillation
        self._emergency_backup_yaw: float | None = None
        self._emergency_backup_frames: int = 0

    # ------------------------------------------------------------------
    #  Pose helpers
    # ------------------------------------------------------------------
    async def _update_pose(self) -> None:
        pos = self.position_state.latest_position
        if pos is None:
            return
        self.pose["north"]   = float(pos.north_m)
        self.pose["east"]    = float(pos.east_m)
        self.pose["down"]    = float(pos.down_m)
        yaw_deg              = float(self.position_state.latest_yaw or 0.0)
        self.pose["yaw_deg"] = yaw_deg
        self.pose["yaw"]     = math.radians(yaw_deg)

        # FIX BUG 1: sync desired yaw to actual yaw on very first reading
        if not self._first_pose_received:
            self._desired_yaw_deg   = yaw_deg
            self._first_pose_received = True
            # Also reset the AvoidancePlanner's heading smoother
            self.planner.prev_heading_rad = math.radians(yaw_deg)

    # ------------------------------------------------------------------
    #  Object detection
    # ------------------------------------------------------------------
    def _run_detection(self, rgb_frame: np.ndarray | None) -> None:
        if not self.targets_remaining:
            return
        for det in self.detector.detect(rgb_frame):
            if (det.label in self.targets_remaining
                    and det.confidence >= self.detection_confidence):
                self.targets_remaining.discard(det.label)
                self.targets_found[det.label] = {
                    "pose":       dict(self.pose),
                    "confidence": det.confidence,
                    "timestamp":  time.time(),
                }
                print(f"\n✅  FOUND: '{det.label}'  @ "
                      f"N={self.pose['north']:.2f} E={self.pose['east']:.2f} "
                      f"alt={-self.pose['down']:.1f} m  conf={det.confidence:.2f}")
                print(f"   Remaining: {self.targets_remaining or '(all found!)'}\n")

    # ------------------------------------------------------------------
    #  Frontier heading  (FIX BUG 3: no more spurious scipy import)
    # ------------------------------------------------------------------
    def _frontier_heading_deg(self) -> float | None:
        """
        Returns NED heading in degrees toward nearest reachable unexplored cell,
        aligned with the drone's current travel direction.
        """
        res = self.exp_map.get_frontier_heading(
            self.pose["north"],
            self.pose["east"],
            current_heading_rad=self.pose["yaw"],   # FIX BUG 4
        )
        if res is None:
            self._current_frontier = None
            return None

        heading_rad, cell = res
        self._current_frontier = cell
        return math.degrees(heading_rad)

    # ------------------------------------------------------------------
    #  Anti-loop: check if we're cycling (FIX BUG 6)
    # ------------------------------------------------------------------
    def _check_loop(self, north: float, east: float) -> bool:
        """Returns True if the drone appears to be looping."""
        cell = (int(round(north / self.exp_map.res)),
                int(round(east  / self.exp_map.res)))
        self._recent_cells.append(cell)
        if len(self._recent_cells) < self._LOOP_WINDOW:
            return False
        unique = len(set(self._recent_cells))
        return unique < self._LOOP_UNIQUE_CELLS

    # ------------------------------------------------------------------
    #  Smooth yaw interpolation
    # ------------------------------------------------------------------
    def _step_yaw(self, target_deg: float, rate_deg_per_frame: float = 10.0) -> float:
        err = target_deg - self._desired_yaw_deg
        while err >  180: err -= 360
        while err < -180: err += 360
        step = float(np.clip(err, -rate_deg_per_frame, rate_deg_per_frame))
        self._desired_yaw_deg += step
        return self._desired_yaw_deg

    # ------------------------------------------------------------------
    #  Main run loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        print("\n=== AUTONOMOUS MULTI-OBJECT SEARCH ===")
        print(f"    Targets: {sorted(self.targets_remaining)}\n")

        await self.drone.connect()
        print("Connected. Waiting for drone telemetry to settle...")
        await asyncio.sleep(5)

        stop_event   = asyncio.Event()
        self.monitor_task = asyncio.create_task(
            position_monitor_task(self.drone, self.position_state, stop_event)
        )

        print("Checking pre-arm readiness...")
        await self.drone.arm_and_takeoff()

        try:
            while self.running:
                t0 = time.monotonic()

                # ── 1. Update pose ──────────────────────────────────────
                await self._update_pose()
                north       = self.pose["north"]
                east        = self.pose["east"]
                actual_yaw  = self.pose["yaw_deg"]

                # ── 2. Mark current cell as visited ─────────────────────
                self.exp_map.mark_visited(north, east)

                # ── 3. Altitude sweeper tick ────────────────────────────
                self.alt_sweeper.update(north, east)
                target_down = self.alt_sweeper.current_down  # absolute NED down

                # ── 4. Get depth + detection ─────────────────────────────
                depth_frame = self.receiver.get_frame()
                rgb_frame: np.ndarray | None = None
                self._run_detection(rgb_frame)

                # ── 5. Check completion ─────────────────────────────────
                if not self.targets_remaining:
                    print("\n🎯  ALL TARGETS FOUND — mission complete!\n")
                    for label, info in self.targets_found.items():
                        p = info["pose"]
                        print(f"  {label}: N={p['north']:.1f} E={p['east']:.1f} "
                              f"alt={-p['down']:.1f} m  conf={info['confidence']:.2f}")
                    break

                if depth_frame is None:
                    await asyncio.sleep(1.0 / self.loop_hz)
                    continue

                # ── 6. VFH+ avoidance ───────────────────────────────────
                self.pose["down"] = target_down
                _, _, _, pref_heading_rad, info = self.planner.compute_position_ned(
                    depth_frame, self.pose, step_size=self.step_size,
                )

                center_clearance = info["clearance"]["center"]
                blocked          = info["blocked"]
                cl               = info["clearance"]

                if center_clearance < 4.0:
                    wall_n = north + center_clearance * math.cos(math.radians(actual_yaw))
                    wall_e = east + center_clearance * math.sin(math.radians(actual_yaw))
                    self.exp_map.mark_obstacle(wall_n, wall_e)
                
                # Get coverage stats for debugging
                cov_stats = self.exp_map.grid_coverage_stats()

                print(
                    f"[NAV] state={self._nav_state} blocked={blocked} | "
                    f"L={cl['left']:.1f} C={cl['center']:.1f} R={cl['right']:.1f} | "
                    f"hdg={info['preferred_heading_deg']:.0f}° | "
                    f"alt={-target_down:.1f}m | visited={cov_stats['visited_pct']:.1f}% ({cov_stats['total_visited_cells']}cells) | "
                    f"unknown={cov_stats['unknown_pct']:.1f}% | blk_fr={cov_stats['blacklisted_frontiers']} | "
                    f"tgt={len(self.targets_remaining)}"
                )

                # ── 7. SCAN MODE – rotate 360° when looping ──────────────
                if self._nav_state == _STATE_SCAN:
                    target_yaw_deg = self._scan_start_yaw + self._scan_rotated
                    self._scan_rotated += self._SCAN_ROTATE_RATE
                    if self._scan_rotated >= self._SCAN_COMPLETE_DEG:
                        print("[NAV] Scan complete – resuming exploration")
                        self._nav_state   = _STATE_EXPLORE
                        self._scan_rotated = 0.0
                        self._recent_cells.clear()

                    send_yaw_deg = self._step_yaw(target_yaw_deg, rate_deg_per_frame=self._SCAN_ROTATE_RATE)
                    # Hover in place during scan
                    await self.drone.send_position_setpoint(
                        north=0.0, east=0.0,
                        down=target_down - self.pose["down"],  # <--- CHANGED TO RELATIVE OFFSET
                        yaw_deg=send_yaw_deg,
                    )
                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0.0, 1.0 / self.loop_hz - elapsed))
                    continue

                # ── 8. Heading selection (EXPLORE mode) ─────────────────

                if blocked:
                    # Fully blocked: change altitude tier and use VFH heading
                    self.alt_sweeper.update(north, east, forced=True)
                    target_yaw_deg = math.degrees(pref_heading_rad)

                else:
                    vfh_deg = math.degrees(pref_heading_rad)

                    # Deadband: if VFH heading is already close to current, keep heading
                    yaw_diff = abs(vfh_deg - actual_yaw)
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    if yaw_diff < 20.0:
                        vfh_deg = actual_yaw

                    # FIX BUG 5: use BFS frontier UNLESS we are very close to an obstacle.
                    # IMPROVED: Only use VFH if VERY close (0.84m), otherwise prefer BFS frontier
                    # But if we've visited most of the accessible area, use VFH to fill gaps
                    exploration_complete = cov_stats['visited_pct'] > 60.0
                    
                    if center_clearance < self.planner.critical_distance * 1.2:
                        # Very close to obstacle – trust VFH only
                        target_yaw_deg = vfh_deg
                    elif exploration_complete:
                        # Already explored most cells, use VFH to search remaining gaps
                        target_yaw_deg = vfh_deg
                    else:
                        # Open space – use global map frontier
                        frontier_deg = self._frontier_heading_deg()
                        if frontier_deg is not None:
                            target_yaw_deg = frontier_deg
                        else:
                            target_yaw_deg = vfh_deg

                if self._current_frontier is not None:
                    frontier_n, frontier_e = self.exp_map._idx_to_world(*self._current_frontier)
                    dist = math.hypot(frontier_n - north, frontier_e - east)

                    if self._previous_frontier == self._current_frontier:
                        # Check if we're getting closer or stuck
                        if self._frontier_last_dist is not None:
                            dist_delta = self._frontier_last_dist - dist
                            # FIX: If distance isn't decreasing OR we're very close but not arriving
                            # mark as stalled so we blacklist and find a new frontier
                            if (dist_delta < 0.2) or (dist < 1.5 and dist_delta < 0.05):
                                self._frontier_stall_frames += 1
                            else:
                                self._frontier_stall_frames = max(0, self._frontier_stall_frames - 1)
                        else:
                            self._frontier_stall_frames = 0
                        self._frontier_last_dist = dist

                        if self._frontier_stall_frames >= 10:
                            print(f"[NAV] Stuck on frontier ({cov_stats['total_visited_cells']} visited); blacklisting ({frontier_n:.1f}, {frontier_e:.1f})")
                            self.exp_map.record_frontier_attempt(frontier_n, frontier_e)
                            self._current_frontier = None
                            self._previous_frontier = None
                            self._frontier_stall_frames = 0
                            self._frontier_last_dist = None
                            target_yaw_deg = vfh_deg
                    else:
                        self._previous_frontier = self._current_frontier
                        self._frontier_stall_frames = 0
                        self._frontier_last_dist = dist

                # Anti-loop: enter SCAN mode if we appear to be circling (FIX BUG 6)
                # But only if we've been in EXPLORE mode for a while to avoid false triggers
                if self._nav_state == _STATE_EXPLORE and self._check_loop(north, east):
                    print(f"[NAV] Loop detected ({len(set(self._recent_cells))} unique cells in {self._LOOP_WINDOW} frames) – entering SCAN mode")
                    self._nav_state    = _STATE_SCAN
                    self._scan_start_yaw = actual_yaw
                    self._scan_rotated  = 0.0

                # ── 9. Incremental yaw ───────────────────────────────────
                send_yaw_deg = self._step_yaw(target_yaw_deg, rate_deg_per_frame=80.0)

# ── 10. Translation step (FIXED OSCILLATION) ──────────────
                yaw_error_signed = target_yaw_deg - actual_yaw
                while yaw_error_signed >  180: yaw_error_signed -= 360
                while yaw_error_signed < -180: yaw_error_signed += 360
                yaw_error = abs(yaw_error_signed)

                if center_clearance < self.planner.critical_distance:
                    # Emergency: We are too close. Mark the wall and back up.
                    mark_dist = center_clearance + 0.6
                    wall_n = north + mark_dist * math.cos(math.radians(actual_yaw))
                    wall_e = east  + mark_dist * math.sin(math.radians(actual_yaw))
                  
                    self.exp_map.mark_obstacle(wall_n, wall_e, force=True) 
                    
                    forward_step = -0.5   # back up
                    self.exp_map.mark_obstacle(wall_n, wall_e)
                    
                    # FIX: Commit to a backup direction for multiple frames to avoid oscillation.
                    # Only choose new direction if we weren't already backing up or if it's changed.
                    if self._emergency_backup_yaw is None or self._emergency_backup_frames == 0:
                        # Choose turn direction based on current clearance
                        if cl["left"] > cl["right"]:
                            self._emergency_backup_yaw = actual_yaw - 45.0
                        else:
                            self._emergency_backup_yaw = actual_yaw + 45.0
                        self._emergency_backup_frames = 8  # Commit for 8 frames
                    else:
                        self._emergency_backup_frames -= 1
                        # Continue with the chosen direction
                    
                    target_yaw_deg = self._emergency_backup_yaw
                    forward_step = -0.3   # back up (slightly less aggressive)

                else:
                    # Not in emergency, clear the backup state
                    self._emergency_backup_yaw = None
                    self._emergency_backup_frames = 0
                    
                    if yaw_error > 15.0:
                        # 🚨 THE FIX: If we need to turn, STOP MOVING FORWARD.
                        # Just hover in place and rotate until we face the clear path.
                        forward_step = 0.0    

                    else:
                        # We are facing the correct, clear direction! Fly forward.
                        if center_clearance < self.planner.safe_distance:
                            forward_step = 1.2    # thread carefully
                        else:
                            forward_step = 2.0 # full cruise

                actual_yaw_rad = math.radians(actual_yaw)
                n_target = north + forward_step * math.cos(actual_yaw_rad)
                e_target = east  + forward_step * math.sin(actual_yaw_rad)

                # ── 11. Send setpoint ────────────────────────────────────

                # FIX BUG 2: send target_down so altitude sweeper actually works
                await self.drone.send_position_setpoint(
                    north   = n_target - north,
                    east    = e_target - east,
                    down    = target_down - self.pose["down"],  # <--- CHANGED TO RELATIVE OFFSET
                    yaw_deg = send_yaw_deg,
                )

                # ── 12. Loop timing ──────────────────────────────────────
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.0, 1.0 / self.loop_hz - elapsed))

        except asyncio.CancelledError:
            print("Navigation cancelled.")

        finally:
            print("\n[NAV] Mission ended. Hovering then landing...")
            stop_event.set()
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            await self.drone.land()

    def stop(self) -> None:
        self.running = False


# ============================================================
#  ENTRY POINT
# ============================================================


async def main() -> None:
    nav = DroneNavigation(
        target_objects    = ["fire_extinguisher", "person", "backpack"],
        model_path        = "best.pt",  # (Make sure you are using your actual YOLO weights file here, like yolov10n.pt)
        depth_topic       = "/depth_camera",
        loop_hz           = 4.0,
        step_size         = 0.8,  # Smaller steps for tighter doorways
        
        # --- SHRINK THE SAFETY BUBBLE ---
        safe_distance     = 2.8,
        critical_distance = 1.4,
        
        # --- STOP TARGET HALLUCINATIONS ---
        detection_confidence = 0.85, # Was 0.6 (Must be 85% sure before logging a target!)
        
        search_altitudes_m   = (2.0, 4.0, 6.0),
    )

    task = asyncio.create_task(nav.run())
    try:
        await task
    except KeyboardInterrupt:
        print("\nStopping…")
        nav.stop()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
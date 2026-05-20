# Drone Navigation Fixes - Detailed Summary

## Problems Identified & Fixed

### Issue 1: ❌ "Moves back 3 times then activates 360 scan"
**Root Cause**: When the drone got too close to a wall, it would back up straight (-0.5 m/frame) but wouldn't turn away. It would repeat this 3 times until the loop detector triggered SCAN mode.

**The Fix** (Line ~702):
```python
# OLD (BAD):
forward_step = -0.5   # Just move back straight

# NEW (GOOD):
if cl["left"] > cl["right"]:
    target_yaw_deg = actual_yaw - 45.0  # Turn LEFT while backing
else:
    target_yaw_deg = actual_yaw + 45.0  # Turn RIGHT while backing
forward_step = -0.3  # Back up slightly
```

**Why it works**: Now when the drone backs up from a wall, it simultaneously turns toward the side with more clearance. This gives it an escape route instead of backing into the same wall 3 times.

---

### Issue 2: ❌ "Keeps forgetting visited grid"
**Root Cause**: The grid was marked with only 5 cells (center + 4 cardinal neighbors), but the drone moves 0.8-2.0 m per frame. This means it could skip over entire cells.

**The Fix** (Line ~163-177):
- Changed grid resolution from **1.5m → 0.5m** (3x finer)
- Expanded visited marking from **5 cells → ~50 cells** (radius 3)
- Added radius-based circular region marking instead of just cardinal neighbors
- Added `_visited_region_radius_m` parameter (3.0 m) to create proper "explored bubble"

```python
# OLD (BAD):
def __init__(self, resolution_m: float = 1.5, arena_m: float = 200.0):
    # Mark only center + 4 neighbors
    cells_to_mark = [(r_center, c_center), (r_center - 1, c_center), ...]

# NEW (GOOD):
def __init__(self, resolution_m: float = 0.5, arena_m: float = 200.0):
    self._visited_region_radius_m: float = 3.0  # 3m explored bubble

# In mark_visited:
radius_cells = max(3, int(round(self._visited_region_radius_m / self.res)))
for dr in range(-radius_cells, radius_cells + 1):
    for dc in range(-radius_cells, radius_cells + 1):
        dist_sq = dr*dr + dc*dc
        if dist_sq > (radius_cells * radius_cells + radius_cells):
            continue  # Circle, not square
        # ... mark as VISITED
```

**Why it works**: 
- Finer grid (0.5m) means every movement is tracked
- Larger visited region (3m radius) prevents "forgetting" areas the drone just explored
- Circular region marks prevents re-exploring the same area from different angles

---

### Issue 3: ❌ "Wall is far away but drone turns away anyway"
**Root Cause**: VFH threshold was `center_clearance < critical_distance * 2.0` (1.4m when critical=0.7m). Drone would only trust BFS if clearance > 1.4m, but it rarely got that open.

**The Fix** (Line ~666-679):
```python
# OLD (BAD):
if center_clearance < self.planner.critical_distance * 2.0:
    # Switched to VFH too easily
    target_yaw_deg = vfh_deg

# NEW (GOOD):
if center_clearance < self.planner.critical_distance * 1.2:
    # Only use VFH when VERY close (0.84m if critical=0.7m)
    target_yaw_deg = vfh_deg
else:
    # Trust BFS frontier to guide to unexplored areas
    frontier_deg = self._frontier_heading_deg()
    if frontier_deg is not None:
        target_yaw_deg = frontier_deg
```

**Why it works**: Now the drone trusts its global map (BFS frontier) much longer and only falls back to VFH when critically close. This prevents unnecessary turns when open paths exist.

---

### Issue 4: ❌ "MAIN ISSUE - Keeps visiting the same grids"
**Root Causes** (Multiple):
1. Visited marking was too sparse (only 5 cells)
2. Grid resolution too coarse (1.5m vs 0.5m step)
3. Frontier stall detection was too lenient (6 frames, threshold of -0.2m)
4. No proper "visited region" to prevent re-approaching same frontier

**The Fixes**:
1. **Finer grid**: 0.5m resolution instead of 1.5m
2. **Larger visited region**: 3m radius instead of 1 cell = prevents 8 different approach angles
3. **Better frontier stall detection** (Line ~695-718):

```python
# OLD (BAD):
if self._frontier_last_dist is not None and dist >= self._frontier_last_dist - 0.2:
    self._frontier_stall_frames += 1
if self._frontier_stall_frames >= 6:  # Quick trigger

# NEW (GOOD):
if self._frontier_last_dist is not None:
    dist_delta = self._frontier_last_dist - dist
    if dist_delta < 0.15 and dist < 5.0:  # Not making progress AND close
        self._frontier_stall_frames += 1
    else:
        self._frontier_stall_frames = max(0, self._frontier_stall_frames - 1)
if self._frontier_stall_frames >= 8:  # More conservative trigger
```

4. **Improved loop detection logging** (Line ~688-691):
```python
if self._nav_state == _STATE_EXPLORE and self._check_loop(north, east):
    print(f"[NAV] Loop detected ({len(set(self._recent_cells))} unique cells 
           in {self._LOOP_WINDOW} frames) – entering SCAN mode")
```

---

### Bonus: Better Diagnostics (Line ~297-313)

Added `grid_coverage_stats()` to track:
- **visited_pct**: How much of map is explored
- **unknown_pct**: How much remains to explore  
- **obstacle_pct**: How many cells are blocked
- **blacklisted_frontiers**: Unreachable areas that are avoided

**Enhanced Debug Output** (Line ~610-618):
```
[NAV] state=EXPLORE blocked=False | L=3.2 C=4.1 R=2.8 | hdg=45° | 
alt=3.0m | visited=65% | unknown=20% | blk_frontiers=2 | remaining=3
```

This shows you exactly where the drone is stuck and why.

---

## Summary of Changes

| What | Old | New | Impact |
|------|-----|-----|--------|
| **Grid resolution** | 1.5m | 0.5m | 3x finer tracking |
| **Visited region** | 5 cells | 50 cells (3m radius) | No more "forgetting" |
| **Backtrack behavior** | Straight back | Turn left/right | Escape walls instead of hitting 3x |
| **VFH threshold** | 1.4m | 0.84m | More frontier following |
| **Frontier stall** | 6 frames, -0.2m | 8 frames, -0.15m, <5m | Better stuck detection |
| **Loop trigger** | Always check | Only in EXPLORE state | Fewer false positives |

---

## How to Monitor the Fixes

1. **Check the grid coverage** in the debug output:
   - `visited=85% unknown=10%` means exploration is progressing
   - If stuck at same % for 30 seconds → frontier is unreachable

2. **Watch for "Stuck on frontier"** messages:
   - These indicate unreachable areas being blacklisted
   - Should NOT see same frontier twice

3. **Check SCAN mode triggers**:
   - Should only trigger after 4+ unique cells in 60 frames
   - Should rotate 480° (1.25 turns) to see new paths

4. **Look for "Loop detected"** with unique cell count:
   - Example: `Loop detected (3 unique cells in 60 frames)` = stuck
   - The drone should then rotate and find a new path

---

## Test These Scenarios

1. **Narrow hallway**: Drone should explore end-to-end without backtracking
2. **Dead end room**: Drone should back up, turn away, and find exit
3. **Large open space**: Drone should use BFS to systematically grid-search
4. **Complex maze**: Grid should show 70%+ visited without returning to same spot

---

## If You Still Have Issues...

Check the debug output for:
- Is grid coverage increasing? If not, visited marking may need larger radius
- Are unique cells changing? If not, loop detector is working but frontier exploration is stuck
- Are obstacles being marked correctly? If too many false walls, increase HIT_THRESHOLD


#!/usr/bin/env python3
"""
person_detector_node.py — LiDAR person detection and tracking
Unitree G1 speaker attribution system

Detects and tracks people around the robot with the head-mounted Livox
Mid-360 LiDAR, which is mounted upside down. The detector is geometric and
needs no training data. Each tracked person keeps a persistent ID.

PUBLISHES
---------
  /detected_people          (MarkerArray) confirmed tracks: CYLINDER markers in
                            namespace 'people' with marker.id = person_id and
                            position in the current sensor frame (livox_frame).
                            Marker alpha encodes state: active 0.85, held behind
                            a strut 0.68, coasting fading from 0.6 to 0.13.
  /nearest_person           (PoseStamped) the nearest confirmed track, with
                            hysteresis so it does not flip between two people
                            at similar distances (used when person_turn_node
                            should simply face whoever is closest).
  /person_detector/markers  (MarkerArray) the same markers, for RViz2.
  /person_detector/debug_json (String) background cells, accepted and rejected
                            clusters and pipeline statistics, for
                            lidar_web_viewer.py.

SUBSCRIBES
----------
  /utlidar/cloud_livox_mid360   point cloud (about 10 Hz)
  /lowstate or /odommodestate   robot yaw (and position) for ego-motion
                                compensation, depending on heading_source
  /person_detector/reset_background (Empty) clears the background model

PIPELINE (per frame, on a worker thread)
----------------------------------------
  1. Skip near-empty frames (the Mid-360 alternates full frames with
     near-empty ones); these do not age tracks.
  2. Range and height filters. Clusters within range_edge_margin of
     max_range are rejected later, because the range cut clips walls into
     person-width arcs. Effective person range = max_range - range_edge_margin.
  3. Per-point clustering weights, in the sensor frame:
     w = clip((r / density_ref_range)^2, 1, max_point_weight), multiplied by
     occlusion_weight_boost inside the head-strut sectors. Point density on
     a body falls roughly as 1/r^2; the weights let a distant or partly
     hidden person still form a cluster, while leaving behaviour inside
     density_ref_range unchanged.
  4. Ego-motion compensation: points are rotated by the robot's heading into
     a frame that does not turn with the robot, so a stationary object has
     near-zero apparent velocity while the robot turns. The heading is taken
     at each frame's capture time (header stamp + heading_time_offset) from a
     short history, not the latest value at processing time. heading_sign
     accounts for the LiDAR frame being mirrored relative to the body (-1 on
     the evaluated robot: the scene rotates opposite to the IMU yaw).
  5. RANSAC ground removal, accepting only near-horizontal planes
     (ground_min_normal_z), so a person's front can never be taken as floor.
  6. Optional background model (see BACKGROUND MODEL), applied before
     clustering.
  7. Weighted DBSCAN: a point is a core point when the summed weights of its
     neighbours within cluster_eps reach cluster_min_samples.
  8. Person test on each cluster: vertical spread, width (relaxed slightly
     with range), height and point density (scaled with the same range
     model). Rejections are logged with their reason.
  9. Tracking (see TRACKING).
 10. Publish, rotating positions back into the current sensor frame, since
     downstream nodes need bearings relative to where the robot faces now.

TRACKING
--------
  - One constant-velocity Kalman filter per track; velocity decays toward zero
    when the innovation is small, so a person who stops does not drift.
  - Detections are matched to tracks by an optimal assignment on predicted
    distance (scipy linear_sum_assignment; greedy fallback without scipy).
    Pairs outside a track's gate are given a prohibitive cost before solving,
    so the solver never trades a feasible match for one that would be
    rejected afterwards.
  - Gate = track_gate + speed * gate_speed_scale, except that it does not
    widen while the robot is turning (apparent speed is then mostly ego-motion
    error) or for coasting tracks (their speed is not observed).
  - Implied speed is checked against max_speed_ms using the time since the
    track was last matched, not one frame interval.
  - A track is confirmed after confirm_hits detections and given a person_id.
  - New tracks are not started while the robot is turning.
  - An unmatched track coasts on its prediction, with velocity decaying over
    coast_velocity_tau, and is published faded for up to coast_secs. It is
    dropped after max_absent_secs without a match.

OCCLUSION HOLD
--------------
The two struts that hold the head hide parts of the LiDAR's view
(occlusion_sectors_deg, sensor-frame bearings). A confirmed track that is
lost inside a strut sector (widened by occlusion_hold_margin_deg) is held
still, with zero velocity, for up to occlusion_hold_max_secs instead of
coasting. Hold time does not count toward the drop budget. A held track stays
published, so the robot keeps turning toward it, and counts as present. When
the person reappears, the track re-acquires with the wider
occlusion_reacquire_gate, because its position is remembered rather than
predicted.

BACKGROUND MODEL (background_filter_enabled)
--------------------------------------------
Learns which 0.2 m grid cells hold static structure (walls, furniture) from
how persistently they are occupied, and removes their points before
clustering, which also stops walls merging with people standing next to
them. Occupancy uses the same range weights as clustering and is OR-ed over
the last bg_window_frames frames. Cells near tracked people who have moved
are never learned, cells that cannot currently be seen are neither learned
nor forgotten, learning is frozen while the robot turns, and the model resets
itself if the scene stops matching it. While it is enabled, scan-to-map
alignment (scan_align_*) corrects small pose errors against a fine reference
map, and heading_source 'odom' adds the robot's translation. With the filter
off, which is the evaluated configuration, there is no warm-up period and a
person who stands still can never be absorbed into the background; objects
that pass the person test may then appear as tracks.

HEADING SOURCE (heading_source)
-------------------------------
  'lowstate'  yaw from /lowstate (unitree_hg); rotation only. Evaluated setting.
  'odom'      yaw and x/y from /odommodestate (unitree_go).
  'none'      no ego-motion compensation.
The unitree_hg / unitree_go messages come from the unitree_ros2 workspace,
which must be sourced. If the chosen package is missing, the node logs an
error and falls back ('odom' -> 'lowstate' -> 'none').

EVALUATED CONFIGURATION
-----------------------
  python3 person_detector_node.py --ros-args \
      -p max_range:=2.1 -p range_edge_margin:=0.1 -p background_filter_enabled:=false \
      -p coast_secs:=0.5 -p heading_source:=lowstate -p heading_sign:=-1.0
All other parameters at their defaults.
"""

from __future__ import annotations

import collections
import json
import queue
import socket
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Empty, String
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Unitree state messages for ego-motion compensation (see HEADING SOURCE).
# They come from the unitree_ros2 CycloneDDS workspace, which must be sourced
# on top of ROS 2 Foxy before this node is launched.
try:
    from unitree_go.msg import SportModeState
    SPORTSTATE_AVAILABLE = True
except ImportError:
    SportModeState = None
    SPORTSTATE_AVAILABLE = False

try:
    from unitree_hg.msg import LowState
    LOWSTATE_AVAILABLE = True
except ImportError:          # unitree_hg messages not available
    LowState = None
    LOWSTATE_AVAILABLE = False


# ── Fast PointCloud2 -> (N,3) float32 ─────────────────────────────────────────
def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Vectorised x/y/z extraction from a PointCloud2 (no per-point Python
    loop), dropping non-finite points."""
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ('x', 'y', 'z')) or msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dt = np.dtype({
        'names':    ['x', 'y', 'z'],
        'formats':  ['>f4' if msg.is_bigendian else '<f4'] * 3,
        'offsets':  [offs['x'], offs['y'], offs['z']],
        'itemsize': msg.point_step,
    })
    n = msg.width * msg.height
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
    return pts[np.isfinite(pts).all(axis=1)]


# ── Ego-motion compensation helper ─────────────────────────────────────────────
def _rotate_xy(points_xy: np.ndarray, theta: float) -> np.ndarray:
    """
    Rotate an (N, 2) array of XY points by angle theta (radians, CCW positive —
    standard ROS/REP-103 convention, matching G1 imu_state.rpy[2]).

    Used two ways in this node:
      - forward  (+heading): sensor frame -> stabilised (rotation-cancelled) frame,
        applied before clustering/tracking so a stationary object has ~zero
        apparent velocity regardless of robot yaw.
      - backward (-heading): stabilised frame -> current sensor frame, applied
        only at publish time so downstream nodes and RViz see robot-relative
        bearings.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return points_xy @ R.T


# ── Kalman Filter (2D constant-velocity) ───────────────────────────────────────
class KalmanXY:
    """
    Minimal 2-D constant-velocity Kalman filter with velocity damping.

    State vector:  [x, y, vx, vy]
    Observation:   [x, y]

    Velocity damping:
    After each update, if the observed position is close to the predicted
    position (i.e. the person is not moving much), the velocity estimate
    is exponentially decayed toward zero. This prevents accumulated velocity
    from causing the track to ghost-drift when the person stops suddenly.

    VELOCITY_DAMP_ALPHA: per-update decay factor applied when observed
        displacement < VELOCITY_DAMP_THRESHOLD. 0.7 means velocity
        decays to ~5% of its value after ~9 consecutive stationary updates.
        Increase toward 1.0 to damp less aggressively.

    VELOCITY_DAMP_THRESHOLD: if the Kalman innovation (residual between
        observed and predicted position) is below this (metres), the person
        is considered stationary and damping is applied.
    """

    VELOCITY_DAMP_ALPHA     = 0.7   # per-update decay when stationary
    VELOCITY_DAMP_THRESHOLD = 0.15  # metres — innovation below this = stationary

    def __init__(self, x: float, y: float, dt: float = 0.1):
        self.dt = dt
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # q controls how quickly the filter trusts new velocity estimates.
        # Higher = faster to respond to acceleration, noisier when stationary.
        # 4.0 suits walking pace (~1.5 m/s); reduce toward 1.5 if stationary
        # confirmed tracks jitter.
        q = 4.0
        self.Q = np.diag([0.05, 0.05, q, q])
        self.R = np.diag([0.05, 0.05])
        self.P = np.diag([0.5, 0.5, 2.0, 2.0])

    def predict(self, dt: float | None = None) -> np.ndarray:
        if dt is not None and dt > 0:
            self._rebuild_F(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray) -> np.ndarray:
        y  = z - self.H @ self.x          # innovation
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Velocity damping: if the person isn't moving (innovation is small),
        # bleed the velocity estimate toward zero. This prevents a track that
        # was moving from ghost-drifting after the person stops suddenly.
        innovation_dist = float(np.linalg.norm(y))
        if innovation_dist < self.VELOCITY_DAMP_THRESHOLD:
            self.x[2:4] *= self.VELOCITY_DAMP_ALPHA

        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))

    def _rebuild_F(self, dt: float):
        self.dt = dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt


# ── Background occupancy model ─────────────────────────────────────────────────
class BackgroundModel:
    """
    Learns which grid cells are occupied by static structure (walls,
    furniture, gantry legs) from how persistently they are occupied, and
    removes their points before clustering.

    Per cell, an occupancy score s in [0, 1] is updated every processed frame:
        occupied  : s <- s + a_up   * (1 - s)     a_up   = 1 - exp(-dt / learn_secs)
        empty     : s <- s - a_down * s           a_down = 1 - exp(-dt / forget_secs)
    A cell is background while s >= threshold. With learn_secs=20 and
    threshold=0.9 a cell must be occupied for ~46 s continuously before it
    is treated as background; with forget_secs=3 an emptied cell drops back
    below threshold in well under a second of absence and is fully cleared
    within ~10 s. Nothing is permanent.

    Cells inside `protect` (around protected tracks, computed by the node
    each frame) are frozen: their scores are neither raised nor decayed, so a
    tracked person can never be learned as background. Removal still applies
    to cells that were ALREADY background (a wall the person stands against).

    Evidence is occupancy, not rejected clusters, so a person's noisy frames
    cannot build up "static" evidence; scores decay, so a spot a person keeps
    returning to is empty in between and never accumulates; and tracked
    people who have moved are protected outright.
    """

    def __init__(self, max_range: float, cell_size: float = 0.2,
                 learn_secs: float = 20.0, forget_secs: float = 3.0,
                 threshold: float = 0.9, min_points: int = 3,
                 enabled: bool = True, threshold_low: float = 0.5,
                 window_frames: int = 5):
        self.cell = cell_size
        self.learn_secs = learn_secs
        self.forget_secs = forget_secs
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.min_points = min_points
        self.enabled = enabled
        # +1.5 m margin so the robot can drift from the grid origin (odom mode)
        self.half = int(np.ceil((max_range + 1.5) / cell_size)) + 2
        n = 2 * self.half
        self.score = np.zeros((n, n), dtype=np.float32)
        # Recent per-frame occupancy grids; a cell counts as occupied if it
        # was hit in ANY of the last `window_frames` frames. The Mid-360's
        # non-repetitive pattern covers a far region over several frames,
        # not in every one, so single-frame occupancy of distant static
        # objects flickers and their score never reaches threshold.
        idx = (np.arange(n) - self.half + 0.5) * cell_size
        self.cx, self.cy = np.meshgrid(idx, idx, indexing='ij')   # cell centres
        self.window_frames = max(1, int(window_frames))
        self._recent: list = []
        self.resets = 0

    def cell_index(self, xy: np.ndarray):
        ij = np.floor(xy / self.cell).astype(np.int64) + self.half
        n = self.score.shape[0]
        valid = (ij >= 0).all(axis=1) & (ij < n).all(axis=1)
        return ij, valid

    def occupancy(self, xy: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """Windowed, range-weighted occupancy.

        Each point contributes its DBSCAN weight (the same (r/ref)^2 range
        model), so a distant cell needs as much evidence to count as occupied
        as it needs to form part of a cluster, so clustering and background
        learning agree on what 'enough points' means at every range.
        The result is OR-ed over the last window_frames frames."""
        n = self.score.shape[0]
        acc = np.zeros((n, n), dtype=np.float64)
        if len(xy):
            ij, v = self.cell_index(xy)
            w = np.ones(len(xy)) if weights is None else weights
            np.add.at(acc, (ij[v, 0], ij[v, 1]), w[v])
        self._recent.append(acc >= self.min_points)
        if len(self._recent) > self.window_frames:
            self._recent.pop(0)
        return np.logical_or.reduce(self._recent)

    def background(self) -> np.ndarray:
        """Hysteresis: a cell is background if its score >= threshold, or
        >= threshold_low AND it touches (8-neighbourhood) a cell >= threshold.
        Catches the edges of real objects, whose faces straddle cell
        boundaries and so are only intermittently occupied."""
        strong = self.score >= self.threshold
        if not strong.any():
            return strong
        grown = strong.copy()
        grown[1:, :] |= strong[:-1, :]; grown[:-1, :] |= strong[1:, :]
        g2 = grown.copy()
        g2[:, 1:] |= grown[:, :-1];     g2[:, :-1] |= grown[:, 1:]
        return strong | (g2 & (self.score >= self.threshold_low))

    def consistency(self, occ: np.ndarray, observable: np.ndarray = None):
        """Fraction of (observable) background cells occupied this frame."""
        bg = self.background()
        if observable is not None:
            bg = bg & observable
        nbg = int(bg.sum())
        if nbg == 0:
            return 1.0, 0
        return float((occ & bg).sum()) / nbg, nbg

    def update(self, occ: np.ndarray, protect: np.ndarray, dt: float,
               observable: np.ndarray = None):
        """observable: cells the sensor can actually see this frame. Cells that
        are out of range or behind the head struts are UNOBSERVED, not empty,
        so they are neither learned nor forgotten."""
        if not self.enabled:
            return
        a_up = 1.0 - np.exp(-dt / self.learn_secs)
        a_dn = 1.0 - np.exp(-dt / self.forget_secs)
        free = ~protect
        up = occ & free                               # points seen = evidence anywhere
        dn = (~occ) & free                            # absence only counts where observable
        if observable is not None:
            dn &= observable
        self.score[up] += a_up * (1.0 - self.score[up])
        self.score[dn] -= a_dn * self.score[dn]

    def foreground_mask(self, xy: np.ndarray, dilate_except: np.ndarray = None) -> np.ndarray:
        """True for points to KEEP (not in a background cell).

        Protection deliberately does NOT exempt points from removal: cells
        near a tracked person are frozen (never learned), so the person's
        own cells cannot become background — but cells that were already
        background before they arrived (e.g. the wall they are standing
        against) must stay removed, or they re-merge with the person in
        DBSCAN and break the track."""
        if not self.enabled or len(xy) == 0:
            return np.ones(len(xy), dtype=bool)
        ij, v = self.cell_index(xy)
        keep = np.ones(len(xy), dtype=bool)
        bgm = self.background()
        if dilate_except is not None and bgm.any():
            # One-cell dilation (while turning) to absorb scan smear, never
            # into cells protected around tracked people.
            g = bgm.copy()
            g[1:, :] |= bgm[:-1, :]; g[:-1, :] |= bgm[1:, :]
            g2 = g.copy()
            g2[:, 1:] |= g[:, :-1];  g2[:, :-1] |= g[:, 1:]
            bgm = bgm | (g2 & ~dilate_except)
        keep[v] = ~bgm[ij[v, 0], ij[v, 1]]
        return keep

    def cells_xy(self) -> np.ndarray:
        """Centres of background cells (stabilised frame), for debug output."""
        ij = np.argwhere(self.background())
        return (ij - self.half + 0.5) * self.cell

    def reset(self):
        self.score[:] = 0.0
        self._recent = []
        self.resets += 1

    def blocked_count(self) -> int:
        return int(self.background().sum())


# ── Tracked person dataclass ───────────────────────────────────────────────────
class TrackedPerson:
    ROBOT_XY = np.zeros(2)   # robot position in the tracking frame (odom mode)
    """Maintains Kalman-filtered state for a single tracked person."""

    COLOURS = [
        (0.0, 1.0, 0.0),   # green
        (0.0, 0.5, 1.0),   # blue
        (1.0, 0.5, 0.0),   # orange
        (1.0, 0.0, 1.0),   # magenta
        (0.0, 1.0, 1.0),   # cyan
        (1.0, 1.0, 0.0),   # yellow
    ]

    def __init__(self, track_id: int, centroid: np.ndarray, dt: float = 0.1):
        self.track_id   = track_id
        self.person_id  = None
        self.confirmed  = False
        self.hits       = 1
        self.misses     = 0

        # Time-based absence tracking (seconds)
        self.absent_time: float = 0.0
        # Seconds of the current absence spent held behind a head strut
        self.held_time: float = 0.0
        self.held: bool = False

        # Static suppression: cumulative seconds spent near-stationary while confirmed
        self.static_time: float = 0.0

        # Total seconds since this track was spawned, accumulated every frame
        # whether matched or not (never reset).
        self.age: float = 0.0

        # Birth position, and whether the raw centroid has ever moved more
        # than 0.3 m from it.
        self.birth_pos: np.ndarray = centroid[:2].copy()
        self.has_moved: bool       = False
        # Largest Kalman-smoothed displacement from birth. Unlike has_moved
        # (a raw centroid jump, which cluster jitter on furniture can
        # trigger), this measures genuine movement; it gates protection
        # from the background model.
        self.max_disp: float       = 0.0
        # Protected from background learning: set once max_disp reaches
        # bg_moved_dist, or inherited from a protected track that recently
        # dropped nearby.
        self.protected: bool       = False

        self.kalman   = KalmanXY(centroid[0], centroid[1], dt)
        self.z_smooth = float(centroid[2])
        self.alpha_z  = 0.4

    def predict(self, dt: float | None = None) -> np.ndarray:
        self.age += dt if (dt is not None and dt > 0) else self.kalman.dt
        return self.kalman.predict(dt)

    def update(self, centroid: np.ndarray):
        self.kalman.update(centroid[:2])
        self.z_smooth    = self.alpha_z * centroid[2] + (1 - self.alpha_z) * self.z_smooth
        self.misses      = 0
        self.absent_time = 0.0
        self.held_time   = 0.0
        self.held        = False
        self.hits       += 1
        # Mark as moved once the raw displacement from birth exceeds 0.3 m.
        if not self.has_moved:
            displacement = float(np.linalg.norm(centroid[:2] - self.birth_pos))
            if displacement > 0.3:
                self.has_moved = True
        self.max_disp = max(self.max_disp,
                            float(np.linalg.norm(self.kalman.x[:2] - self.birth_pos)))

    @property
    def smoothed(self) -> np.ndarray:
        xy = self.kalman.x[:2]
        return np.array([xy[0], xy[1], self.z_smooth])

    @property
    def dist(self) -> float:
        # Distance from the robot's CURRENT position (tracks live in the
        # odom-anchored world frame; ROBOT_XY is set by the node each frame).
        xy = self.kalman.x[:2] - TrackedPerson.ROBOT_XY
        return float(np.sqrt(xy[0]**2 + xy[1]**2))

    @property
    def speed(self) -> float:
        return self.kalman.speed

    @property
    def colour(self) -> ColorRGBA:
        pid = self.person_id if self.person_id is not None else self.track_id
        r, g, b = self.COLOURS[pid % len(self.COLOURS)]
        return ColorRGBA(r=r, g=g, b=b, a=0.85)


# ── Main node ──────────────────────────────────────────────────────────────────
class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')
        self.output_frame = 'livox_frame'

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('min_height',              -2.0)
        self.declare_parameter('max_height',               0.5)
        self.declare_parameter('max_range',                3.5)
        self.declare_parameter('min_range',                0.3)
        self.declare_parameter('cluster_eps',              0.25)
        self.declare_parameter('cluster_min_samples',      200)
        self.declare_parameter('min_cluster_width',        0.35)
        self.declare_parameter('max_cluster_width',        0.75)
        self.declare_parameter('min_cluster_height',       0.35)
        self.declare_parameter('min_z_spread',             0.25)
        self.declare_parameter('max_z_spread',             1.5)
        # Tracker
        self.declare_parameter('track_gate',               0.8)
        # Speed-scaled gate: effective gate = track_gate + speed * gate_speed_scale
        # e.g. at 1.5 m/s walking: 0.8 + 1.5 * 0.7 = 1.85 m
        self.declare_parameter('gate_speed_scale',         0.7)
        # Coasting: publish faded markers for confirmed absent tracks for up
        # to this many seconds (should be <= max_absent_secs)
        self.declare_parameter('coast_secs',               2.5)
        # Time-based miss budget: drop a track after this many seconds of absence
        self.declare_parameter('max_absent_secs',          3.5)
        # Extra absence seconds allowed per metre beyond base range
        self.declare_parameter('dist_persist_extra_secs',  0.5)
        self.declare_parameter('dist_persist_base',        3.5)
        # Track confirmation
        self.declare_parameter('confirm_hits',             3)
        # Velocity sanity
        self.declare_parameter('max_speed_ms',             4.0)
        # Ground removal
        self.declare_parameter('ground_removal',           True)
        self.declare_parameter('ground_z_threshold',       0.12)
        self.declare_parameter('ground_ransac_iters',      50)
        # Only near-horizontal planes are accepted as ground. Otherwise RANSAC
        # can fit a person's vertical front when floor points are sparse
        # (short max_range, person close), deleting the person.
        self.declare_parameter('ground_min_normal_z',      0.9)
        # Occlusion hold (see OCCLUSION HOLD in the module docstring).
        self.declare_parameter('occlusion_hold_enabled',     True)
        self.declare_parameter('occlusion_hold_margin_deg',  8.0)
        self.declare_parameter('occlusion_hold_max_secs',    10.0)
        self.declare_parameter('occlusion_reacquire_gate',   1.2)
        # Point density
        self.declare_parameter('min_point_density',        300.0)
        # Distance-scaled width relaxation
        self.declare_parameter('dist_width_relax',         0.03)
        # ── Background model (see BACKGROUND MODEL) ───────────────────────
        self.declare_parameter('background_filter_enabled',  True)
        self.declare_parameter('bg_cell_size',               0.2)
        self.declare_parameter('bg_learn_secs',              20.0)
        self.declare_parameter('bg_forget_secs',             3.0)
        self.declare_parameter('bg_threshold',               0.9)
        self.declare_parameter('bg_threshold_low',           0.5)
        # Range-weighted points per cell (same weights as DBSCAN) needed to
        # count the cell as occupied in a frame.
        self.declare_parameter('bg_min_points',              3)
        # Frames over which occupancy is OR-ed (0.5 s at 10 Hz).
        self.declare_parameter('bg_window_frames',           5)
        # Radius (m) around protected tracks that is never learned/removed.
        self.declare_parameter('bg_protect_radius',          0.6)
        # A confirmed track is protected (its cells frozen, never learned)
        # once its Kalman-smoothed position has moved at least this far from
        # where it was born. People walking up to the robot pass this;
        # furniture whose cluster merely jitters does not.
        self.declare_parameter('bg_moved_dist',              0.5)
        # A new track born within bg_inherit_radius of where a protected
        # track dropped, within bg_inherit_secs, inherits protection, so a
        # stationary person who is briefly lost and re-acquired is not
        # treated as new, unmoved clutter.
        self.declare_parameter('bg_inherit_radius',          0.6)
        self.declare_parameter('bg_inherit_secs',            30.0)
        # Scene-consistency reset: if fewer than this fraction of known
        # background cells are occupied for bg_reset_frames consecutive
        # frames (and at least bg_reset_min_cells exist), clear the model.
        self.declare_parameter('bg_consistency_min',         0.5)
        self.declare_parameter('bg_reset_frames',            3)
        self.declare_parameter('bg_reset_min_cells',         20)
        # ── Range-adaptive clustering ──────────────────────────────────
        # Inside this range each point counts 1 towards cluster_min_samples;
        # beyond it, (r/ref)^2 (capped at max_point_weight).
        self.declare_parameter('density_ref_range',          1.0)
        self.declare_parameter('max_point_weight',           9.0)
        # Head-strut sectors as sensor-frame bearings atan2(y, x) in degrees,
        # given as flat [start, end, start, end, ...] pairs.
        self.declare_parameter('occlusion_sectors_deg',      [30.0, 60.0, 300.0, 330.0])
        self.declare_parameter('occlusion_weight_boost',     2.0)
        # ── Heading source for ego-motion compensation ─────────────────
        # 'lowstate' : yaw only, from /lowstate (unitree_hg). No translation.
        # 'odom'     : yaw and x/y from Unitree odometry (odom_topic,
        #              unitree_go/SportModeState). The position keeps the
        #              background map aligned when the robot drifts sideways
        #              while turning; it is not needed with the background
        #              filter off.
        # 'none'     : no ego-motion compensation.
        self.declare_parameter('heading_source',             'lowstate')
        self.declare_parameter('odom_topic',                 '/odommodestate')
        # Heading is looked up at each LiDAR frame's CAPTURE time (header
        # stamp + this offset; 0.05 s = middle of a 10 Hz scan) from a short
        # yaw history, instead of using the latest yaw at processing time.
        # Processing lags capture by ~0.1 s; at 60 deg/s that alone is 6 deg.
        self.declare_parameter('heading_time_offset',        0.05)
        # Only used if the cloud header stamp is not on this machine's clock:
        # capture time is then estimated as receive time minus this.
        self.declare_parameter('receive_latency',            0.08)
        # While a track is coasting (unmatched), its velocity decays with this
        # time constant (s). A person who is momentarily lost has usually
        # stopped or slowed; carrying full velocity lets the prediction drift
        # into clutter.
        self.declare_parameter('coast_velocity_tau',         0.5)
        # Clusters whose centre is within this distance of max_range are
        # rejected: the range cut clips walls into person-width arcs, and when
        # the robot drifts (odom mode) newly exposed clipped segments are not
        # yet in the background. Effective person range = max_range - margin.
        self.declare_parameter('range_edge_margin',          0.4)
        # Scan-to-map alignment: after each turn (and every scan_align_interval
        # seconds while settled) search a small window for the pose correction
        # that best aligns the live scan with the learned background. Corrects
        # odometry error, body sway and residual heading error together.
        self.declare_parameter('scan_align_enabled',         True)
        self.declare_parameter('scan_align_xy',              0.3)    # +/- m search
        self.declare_parameter('scan_align_xy_step',         0.05)
        self.declare_parameter('scan_align_yaw_deg',         4.0)    # +/- deg search
        self.declare_parameter('scan_align_yaw_step_deg',    1.0)
        self.declare_parameter('scan_align_interval',        2.0)    # s, while settled
        self.declare_parameter('scan_align_min_gain',        0.05)   # need >=5% better than no correction
        self.declare_parameter('scan_align_min_cells',       30)     # map must have this much structure
        # While |yaw rate| exceeds turn_yaw_rate_deg (deg/s), and for
        # turn_hold_secs after: background learning is frozen, background
        # removal is dilated by one cell (outside protected tracks) to absorb
        # scan smear, no new tracks are started, and association gates do not
        # widen with apparent track speed.
        self.declare_parameter('turn_yaw_rate_deg',          15.0)
        self.declare_parameter('turn_hold_secs',             0.5)
        # +1 or -1: relates the IMU yaw to rotation of the LiDAR scene. On the
        # evaluated robot the scene rotates opposite to /lowstate and
        # /odommodestate yaw (the LiDAR frame is mirrored relative to the
        # body), so -1. With the wrong sign, tracks run away from stationary
        # people during turns and the robot chases them.
        self.declare_parameter('heading_sign',               -1.0)
        # ── Debug output for lidar_web_viewer.py ───────────────────────
        self.declare_parameter('debug_json_hz',              5.0)
        # /nearest_person hysteresis: a challenger must be closer than the
        # current target by more than this margin (metres)...
        self.declare_parameter('target_switch_margin',           0.4)
        # ...AND at least this many seconds must have elapsed since the
        # last switch, before the lock is allowed to move again.
        self.declare_parameter('target_min_dwell_secs',          1.5)
        self.declare_parameter('debug',                         True)

        p = self.get_parameter
        self.min_height              = p('min_height').value
        self.max_height              = p('max_height').value
        self.max_range               = p('max_range').value
        self.min_range               = p('min_range').value
        self.cluster_eps             = p('cluster_eps').value
        self.cluster_min_samples     = p('cluster_min_samples').value
        self.min_cluster_width       = p('min_cluster_width').value
        self.max_cluster_width       = p('max_cluster_width').value
        self.min_cluster_height      = p('min_cluster_height').value
        self.min_z_spread            = p('min_z_spread').value
        self.max_z_spread            = p('max_z_spread').value
        self.track_gate              = p('track_gate').value
        self.gate_speed_scale        = p('gate_speed_scale').value
        self.coast_secs              = p('coast_secs').value
        self.max_absent_secs         = p('max_absent_secs').value
        self.dist_persist_extra_secs = p('dist_persist_extra_secs').value
        self.dist_persist_base       = p('dist_persist_base').value
        self.confirm_hits            = p('confirm_hits').value
        self.max_speed_ms            = p('max_speed_ms').value
        self.ground_removal          = p('ground_removal').value
        self.ground_z_threshold      = p('ground_z_threshold').value
        self.ground_ransac_iters     = p('ground_ransac_iters').value
        self.ground_min_normal_z  = float(p('ground_min_normal_z').value)
        self.hold_enabled         = bool(p('occlusion_hold_enabled').value)
        self.hold_margin          = float(p('occlusion_hold_margin_deg').value)
        self.hold_max_secs        = float(p('occlusion_hold_max_secs').value)
        self.reacquire_gate       = float(p('occlusion_reacquire_gate').value)
        self.min_point_density       = p('min_point_density').value
        self.dist_width_relax        = p('dist_width_relax').value
        self.bg_enabled              = bool(p('background_filter_enabled').value)
        self.bg_protect_radius       = p('bg_protect_radius').value
        self.bg_moved_dist           = p('bg_moved_dist').value
        self.bg_inherit_radius       = p('bg_inherit_radius').value
        self.bg_inherit_secs         = p('bg_inherit_secs').value
        self._recent_protected_drops: list = []   # [(xy, sim_time)]
        self.bg_consistency_min      = p('bg_consistency_min').value
        self.bg_reset_frames         = p('bg_reset_frames').value
        self.bg_reset_min_cells      = p('bg_reset_min_cells').value
        self.density_ref_range       = p('density_ref_range').value
        self.max_point_weight        = p('max_point_weight').value
        sect = list(p('occlusion_sectors_deg').value)
        self.occlusion_sectors       = [(sect[i], sect[i + 1]) for i in range(0, len(sect) - 1, 2)]
        self.occlusion_weight_boost  = p('occlusion_weight_boost').value
        self.heading_source          = str(p('heading_source').value)
        self.heading_sign            = 1.0 if float(p('heading_sign').value) >= 0 else -1.0
        self._pos_hist               = collections.deque(maxlen=1500)   # (t, x, y) in odom frame
        self._odom_origin            = None      # odom (x, y) the background grid is centred on
        self._odom_ok                = False
        self._frame_t                = np.zeros(2)   # robot offset from origin at this frame's capture
        self._frame_heading          = 0.0
        self.heading_time_offset     = p('heading_time_offset').value
        self.receive_latency         = p('receive_latency').value
        self.coast_velocity_tau      = p('coast_velocity_tau').value
        self.range_edge_margin       = p('range_edge_margin').value
        self.align_enabled           = bool(p('scan_align_enabled').value)
        self.align_xy                = p('scan_align_xy').value
        self.align_xy_step           = p('scan_align_xy_step').value
        self.align_yaw               = np.radians(p('scan_align_yaw_deg').value)
        self.align_yaw_step          = np.radians(p('scan_align_yaw_step_deg').value)
        self.align_interval          = p('scan_align_interval').value
        self.align_min_gain          = p('scan_align_min_gain').value
        self.align_min_cells         = p('scan_align_min_cells').value
        self._corr                   = np.zeros(3)     # accumulated [dx, dy, dyaw] on top of odometry
        self._last_align_t           = 0.0
        self._was_turning            = False
        self._align_stats            = {}
        # Fine (0.05 m) reference map of static structure used ONLY for
        # alignment (the 0.2 m background grid is too coarse to register on).
        self._ref_cell               = 0.05
        self._ref_half               = int(np.ceil((self.max_range + 1.5) / self._ref_cell)) + 2
        self._ref_counts             = np.zeros((2 * self._ref_half, 2 * self._ref_half), dtype=np.float32)
        self._ref_frames             = 0
        self._ref                    = None
        self.declare_parameter('scan_align_ref_frames',      20)
        self.align_ref_frames        = p('scan_align_ref_frames').value
        self.turn_yaw_rate           = np.radians(p('turn_yaw_rate_deg').value)
        self.turn_hold_secs          = p('turn_hold_secs').value
        self._heading_hist           = collections.deque(maxlen=1500)   # (t, yaw) ~3 s @ 500 Hz
        self._hist_lock              = threading.Lock()
        self._turning                = False
        self._turn_until             = 0.0
        self._yaw_rate               = 0.0
        self._stamp_mode             = None       # 'header' or 'receive' (logged once)
        self.debug_json_hz           = p('debug_json_hz').value
        self.target_switch_margin    = p('target_switch_margin').value
        self.target_min_dwell_secs   = p('target_min_dwell_secs').value
        self.debug                   = p('debug').value

        # ── Tracker state ─────────────────────────────────────────────
        self.tracks: list[TrackedPerson] = []
        self.next_track_id  = 0
        self.next_person_id = 0
        self.last_cloud_time: float | None = None
        self.last_debug_time = 0.0

        # ── /nearest_person target-lock state ───────────────────────────
        # person_id currently held as the published /nearest_person target.
        # None means no lock held (next frame acquires nearest immediately).
        self._current_target_person_id: int | None = None
        # Seconds elapsed since the lock last changed hands. Gates how soon
        # another switch is allowed (see target_min_dwell_secs).
        self._time_since_switch: float = 0.0

        # ── Ego-motion (yaw) compensation state ─────────────────────────
        # Latest robot yaw (radians, times heading_sign), read from the state
        # estimator's fused rpy, with no integration here. The head is rigidly
        # mounted to the torso, so base yaw maps directly onto the LiDAR's
        # heading with no joint-angle correction.
        self.heading = 0.0

        self.bg = BackgroundModel(
            max_range=self.max_range,
            cell_size=p('bg_cell_size').value,
            learn_secs=p('bg_learn_secs').value,
            forget_secs=p('bg_forget_secs').value,
            threshold=p('bg_threshold').value,
            threshold_low=p('bg_threshold_low').value,
            window_frames=p('bg_window_frames').value,
            min_points=p('bg_min_points').value,
            enabled=self.bg_enabled,
        )
        self._bg_inconsistent_frames = 0
        self._last_debug_json = 0.0
        self._last_rejected: list = []
        self._last_accepted: list = []
        self._stats: dict = {}
        self._heading_ok = False      # True once a heading message has arrived

        # ── Processing queue & worker thread ──────────────────────────
        # The subscriber callback enqueues messages and returns immediately,
        # so the ROS2 executor is never blocked by DBSCAN computation.
        # maxsize=2: if the worker falls behind, old frames are dropped
        # rather than accumulating a backlog.
        self._cloud_queue: queue.Queue = queue.Queue(maxsize=2)
        self._worker = threading.Thread(target=self._process_loop, daemon=True)
        self._worker.start()

        # Frames with fewer points than this are treated as empty (the LiDAR
        # alternates full frames with near-empty ones). They are skipped and
        # do not add to absent_time on existing tracks.
        self._min_points_to_process = 500

        # ── Subscribers / Publishers ──────────────────────────────────
        self.lidar_sub = self.create_subscription(
            PointCloud2, '/utlidar/cloud_livox_mid360', self.cloud_callback, 10
        )
        # Heading source for ego-motion (yaw) compensation.
        if self.heading_source == 'odom':
            if SPORTSTATE_AVAILABLE:
                self.odom_sub = self.create_subscription(
                    SportModeState, p('odom_topic').value, self._on_odom, 10)
            else:
                self.get_logger().error(
                    "heading_source='odom' but unitree_go is not importable — "
                    "falling back to 'lowstate' (yaw only, no translation).")
                self.heading_source = 'lowstate'
        if self.heading_source == 'lowstate':
            if LOWSTATE_AVAILABLE:
                self.lowstate_sub = self.create_subscription(
                    LowState, '/lowstate', self._on_lowstate, 10)
            else:
                self.get_logger().error(
                    "heading_source='lowstate' but unitree_hg is not importable — "
                    "falling back to 'none' (NO ego-motion compensation). Check that "
                    "the cyclonedds_ws overlay containing unitree_hg is sourced.")
                self.heading_source = 'none'
        elif self.heading_source not in ('none', 'odom'):
            self.get_logger().error(f"Unknown heading_source '{self.heading_source}' — using 'none'")
            self.heading_source = 'none'

        self.debug_pub = self.create_publisher(String, '/person_detector/debug_json', 10)
        # Manual background reset, e.g. after the robot is moved (yaw is
        # compensated, translation is not). lidar_web_viewer has a button.
        self.create_subscription(Empty, '/person_detector/reset_background', self._on_reset_bg, 10)
        self.person_pub  = self.create_publisher(PoseStamped,  '/nearest_person',          10)
        self.people_pub  = self.create_publisher(MarkerArray,  '/detected_people',          10)
        self.marker_pub  = self.create_publisher(MarkerArray,  '/person_detector/markers',  10)

        if not SCIPY_AVAILABLE:
            self.get_logger().warn(
                'scipy not found — falling back to greedy nearest-neighbour tracker. '
                'Install with: pip3 install scipy --break-system-packages'
            )
        self.get_logger().info('Person detector started.')
        self.get_logger().info(f'  Height filter : {self.min_height}m to {self.max_height}m')
        self.get_logger().info(f'  Z-spread      : {self.min_z_spread}m to {self.max_z_spread}m')
        self.get_logger().info(f'  Range         : {self.min_range}m to {self.max_range}m')
        self.get_logger().info(f'  Confirm hits  : {self.confirm_hits} frames')
        self.get_logger().info(f'  Max absent    : {self.max_absent_secs}s (coast up to {self.coast_secs}s)')
        self.get_logger().info(f'  Gate          : {self.track_gate}m + speed*{self.gate_speed_scale}')
        self.get_logger().info(
            f'  Background    : {"ENABLED" if self.bg_enabled else "DISABLED"} '
            f'cell={self.bg.cell}m learn={self.bg.learn_secs}s forget={self.bg.forget_secs}s '
            f'thr={self.bg.threshold} protect_r={self.bg_protect_radius}m '
            f'moved_dist={self.bg_moved_dist}m inherit={self.bg_inherit_radius}m/{self.bg_inherit_secs}s')
        self.get_logger().info(
            f'  Clustering    : eps={self.cluster_eps}m min_samples={self.cluster_min_samples} '
            f'(weighted: ref={self.density_ref_range}m, max_w={self.max_point_weight}, '
            f'strut sectors={self.occlusion_sectors} x{self.occlusion_weight_boost})')
        self.get_logger().info(f'  Heading source: {self.heading_source} (sign {self.heading_sign:+.0f})'
                               + ('  (NO ego-motion compensation — background model '
                                  'will reset when the robot turns)' if self.heading_source == 'none' else ''))
        self.get_logger().info(f'  Empty frame skip: < {self._min_points_to_process} pts')
        self.get_logger().info(f'  Target lock   : switch margin={self.target_switch_margin}m, '
                               f'min dwell={self.target_min_dwell_secs}s')


    # ── Ego-motion state ────────────────────────────────────────────────────────
    def _on_lowstate(self, msg: LowState):
        """Store the latest fused yaw (times heading_sign) and add it to the
        heading history. Runs on the executor thread."""
        yaw = self.heading_sign * float(msg.imu_state.rpy[2])
        self.heading = yaw
        self._heading_ok = True
        self._record_heading(self.get_clock().now().nanoseconds / 1e9, yaw)

    def _recentre_bg(self):
        """Re-anchor the (now empty) background grid on the robot's current position."""
        self._corr = np.zeros(3)
        self._ref = None
        self._ref_frames = 0
        self._ref_counts[:] = 0.0
        if self.heading_source == 'odom' and self._odom_ok:
            with self._hist_lock:
                last = self._pos_hist[-1] if self._pos_hist else None
            if last is not None:
                new_origin = np.array(last[1:3])
                shift = new_origin - self._odom_origin if self._odom_origin is not None else np.zeros(2)
                for t in self.tracks:            # keep tracks consistent with the new origin
                    t.kalman.x[:2] -= shift
                    t.birth_pos = t.birth_pos - shift
                self._recent_protected_drops = [(xy - shift, ts) for xy, ts in self._recent_protected_drops]
                self._odom_origin = new_origin
                self._frame_t = self._frame_t - shift
                TrackedPerson.ROBOT_XY = self._frame_t.copy()

    def _on_reset_bg(self, _msg):
        self.bg.reset()
        self._recentre_bg()
        self.get_logger().warn('Background RESET requested manually — relearning (~60 s, robot stationary).')

    def _on_odom(self, msg):
        """Unitree odometry: yaw + planar position of the robot base."""
        t = self.get_clock().now().nanoseconds / 1e9
        yaw = self.heading_sign * float(msg.imu_state.rpy[2])
        x, y = float(msg.position[0]), float(msg.position[1])
        self.heading = yaw
        if not self._heading_ok:
            self.get_logger().info(f'First odometry: x={x:.2f} y={y:.2f} yaw={np.degrees(yaw):.1f} deg')
        self._heading_ok = True
        self._odom_ok = True
        self._record_heading(t, yaw)
        self._record_pose(t, x, y)

    def _record_pose(self, t: float, x: float, y: float):
        with self._hist_lock:
            self._pos_hist.append((t, x, y))

    def _pos_at(self, t: float) -> np.ndarray:
        with self._hist_lock:
            if not self._pos_hist:
                return np.zeros(2)
            h = np.array(self._pos_hist)
        if len(h) == 1:
            return h[0, 1:3].copy()
        return np.array([np.interp(t, h[:, 0], h[:, 1]), np.interp(t, h[:, 0], h[:, 2])])

    def _record_heading(self, t: float, yaw: float):
        with self._hist_lock:
            self._heading_hist.append((t, yaw))

    def _heading_at(self, t: float) -> float:
        """Yaw at time t, linearly interpolated (angle-unwrapped) from the
        recent heading history. Falls back to the latest yaw if the history
        is too short, and clamps to its ends if t is outside it."""
        with self._hist_lock:
            if len(self._heading_hist) < 2:
                return self.heading
            h = np.array(self._heading_hist)
        ts, ys = h[:, 0], np.unwrap(h[:, 1])
        if t <= ts[0]:
            return float(ys[0])
        if t >= ts[-1]:
            return float(ys[-1])
        return float(np.interp(t, ts, ys))

    # ── Subscriber callback: enqueue only, never blocks the executor ──────────
    def cloud_callback(self, msg: PointCloud2):
        """Lightweight: just enqueue the message. If the queue is full,
        drop the oldest frame and insert the newest so we stay current."""
        item = (msg, self.get_clock().now().nanoseconds / 1e9)
        try:
            self._cloud_queue.put_nowait(item)
        except queue.Full:
            try:
                self._cloud_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._cloud_queue.put_nowait(item)
            except queue.Full:
                pass

    # ── Worker thread — all heavy processing runs here ────────────────────────
    def _process_loop(self):
        while True:
            msg, recv_t = self._cloud_queue.get()
            try:
                self._process_cloud(msg, recv_t)
            except Exception:
                # One bad frame must not kill this thread: if it died,
                # detection would stop silently while the node still looked
                # alive to ROS 2. Log the error and continue.
                self.get_logger().error(
                    'Exception in _process_cloud — frame dropped, worker continuing:',
                    exc_info=True,
                )

    def _process_cloud(self, msg: PointCloud2, recv_t: float = None):

        now = self.get_clock().now().nanoseconds / 1e9
        dt  = (now - self.last_cloud_time) if self.last_cloud_time is not None else 0.1
        dt  = max(0.02, min(dt, 0.5))
        self.last_cloud_time = now

        # Heading at the frame's capture time: one value per frame, used for
        # both the forward rotation below and the backward rotation at
        # publish time. Uses the cloud header stamp if it is on this machine's
        # clock (within 1 s of receipt), otherwise the receive time.
        heading = self._capture_heading(msg, recv_t if recv_t is not None else now, now)
        self._frame_heading = heading

        # 1. Extract XYZ (vectorised)
        points = cloud_to_xyz(msg)
        n_raw = len(points)

        # Skip near-empty frames entirely, without ageing tracks.
        if n_raw < self._min_points_to_process:
            return

        if self.debug and (now - self.last_debug_time) > 1.0:
            self.last_debug_time = now
            self.get_logger().info(
                f'Raw cloud: {n_raw} pts | '
                f'Z: {points[:,2].min():.2f} to {points[:,2].max():.2f}m | '
                f'Background cells: {self.bg.blocked_count()} (resets={self.bg.resets})'
            )

        # 2. Range filter
        xy_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        keep    = (xy_dist > self.min_range) & (xy_dist < self.max_range)
        points, xy_dist = points[keep], xy_dist[keep]

        # 3. Height filter
        z = points[:, 2]
        keep = (z > self.min_height) & (z < self.max_height)
        points, xy_dist = points[keep], xy_dist[keep]

        # 3.1 Per-point DBSCAN weights, computed in the SENSOR frame (the
        # strut sectors are fixed to the robot, not the room).
        weights = self._point_weights(points[:, :2], xy_dist)

        # 3.5 Ego-motion compensation — rotate into the stabilised frame.
        points[:, :2] = _rotate_xy(points[:, :2], heading) + self._frame_t
        TrackedPerson.ROBOT_XY = self._frame_t.copy()

        # 4. Ground removal (weights follow the surviving points)
        if self.ground_removal and len(points) >= 10:
            nonground = self._remove_ground(points)
            points, weights = points[nonground], weights[nonground]

        # 4.5 Background model: learn from this frame, then remove background.
        n_pre_bg = len(points)
        points, weights = self._apply_background(points, weights, dt)
        heading = self._frame_heading      # may have been refined by scan alignment

        self._stats = {
            'raw': n_raw, 'filtered': n_pre_bg, 'foreground': len(points),
            'bg_cells': self.bg.blocked_count(), 'bg_resets': self.bg.resets,
            'heading_deg': round(float(np.degrees(heading)), 1),
            'heading_source': self.heading_source, 'heading_ok': self._heading_ok,
            'yaw_rate_dps': round(float(np.degrees(self._yaw_rate)), 1), 'turning': self._turning,
            'odom_xy': [round(float(self._frame_t[0]), 2), round(float(self._frame_t[1]), 2)],
            'align': [round(float(self._corr[0]), 3), round(float(self._corr[1]), 3),
                      round(float(np.degrees(self._corr[2])), 2)],
        }

        min_floor = self.cluster_min_samples / max(self.max_point_weight * self.occlusion_weight_boost, 1.0)
        if len(points) < max(3, int(min_floor)):
            self._last_accepted, self._last_rejected = [], []
            self._update_tracks([], dt)
            self._age_tracks_and_publish([], msg, dt, heading)
            self._publish_debug(now, heading)
            return

        # 5. Weighted DBSCAN — a point is core if the SUM of neighbour
        # weights within eps reaches cluster_min_samples.
        labels = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        ).fit_predict(points[:, :2], sample_weight=weights)

        # 6. Classify -> candidates
        candidates = self._extract_candidates(points, labels)

        # 7. Update tracker
        self._update_tracks(candidates, dt)

        # 8. Publish
        self._age_tracks_and_publish(candidates, msg, dt, heading)
        self._publish_debug(now, heading)

    # ── Heading, weighting, background and alignment helpers ──────────────────
    def _capture_heading(self, msg, recv_t: float, now: float) -> float:
        try:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        except Exception:
            stamp = None
        if stamp is not None and abs(stamp - recv_t) < 1.0:
            t_cap, mode = stamp + self.heading_time_offset, 'header'
        else:
            t_cap, mode = recv_t - self.receive_latency, 'receive'
        if mode != self._stamp_mode:
            self._stamp_mode = mode
            self.get_logger().info(
                f'Heading lookup uses {"cloud header stamp" if mode == "header" else "cloud receive time (header stamp not on local clock)"}')
        heading = self._heading_at(t_cap) if self.heading_source != 'none' else self.heading
        # Robot position (odom) at capture, relative to the background grid's origin.
        if self.heading_source == 'odom' and self._odom_ok:
            pos = self._pos_at(t_cap)
            if self._odom_origin is None:
                self._odom_origin = pos.copy()
            self._frame_t = pos - self._odom_origin
        else:
            self._frame_t = np.zeros(2)
        # Yaw rate over the 0.1 s before capture -> turning state (with hold).
        if self.heading_source != 'none':
            self._yaw_rate = (heading - self._heading_at(t_cap - 0.1)) / 0.1
            if abs(self._yaw_rate) > self.turn_yaw_rate:
                self._turn_until = now + self.turn_hold_secs
        self._turning = now < self._turn_until
        # Scan-to-map correction accumulated on top of odometry/IMU.
        heading = heading + self._corr[2]
        self._frame_t = self._frame_t + self._corr[:2]
        return heading


    def _range_weight(self, r):
        """(r/ref)^2 clipped to [1, max_point_weight]. Works on scalars/arrays."""
        return np.clip((np.asarray(r, dtype=np.float64) / self.density_ref_range) ** 2,
                       1.0, self.max_point_weight)

    def _point_weights(self, xy_sensor: np.ndarray, r: np.ndarray) -> np.ndarray:
        w = self._range_weight(r)
        if self.occlusion_sectors and self.occlusion_weight_boost != 1.0 and len(r):
            bearing = np.degrees(np.arctan2(xy_sensor[:, 1], xy_sensor[:, 0])) % 360.0
            in_sector = np.zeros(len(r), dtype=bool)
            for lo, hi in self.occlusion_sectors:
                lo, hi = lo % 360.0, hi % 360.0
                if lo <= hi:
                    in_sector |= (bearing >= lo) & (bearing <= hi)
                else:
                    in_sector |= (bearing >= lo) | (bearing <= hi)
            w = np.where(in_sector, w * self.occlusion_weight_boost, w)
        return w.astype(np.float64)

    def _protect_mask(self) -> np.ndarray:
        """Grid cells (stabilised frame) near protected tracks."""
        n = self.bg.score.shape[0]
        mask = np.zeros((n, n), dtype=bool)
        rad = int(np.ceil(self.bg_protect_radius / self.bg.cell))
        for t in self.tracks:
            if not t.protected and t.max_disp >= self.bg_moved_dist:
                t.protected = True
            if not (t.confirmed and t.protected):
                continue
            ij, v = self.bg.cell_index(t.kalman.x[:2].reshape(1, 2))
            if not v[0]:
                continue
            i, j = ij[0]
            mask[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1] = True
        return mask

    def _build_reference(self, points: np.ndarray):
        """Accumulate points that fall in learned background cells, from the
        first `scan_align_ref_frames` settled frames once the background map
        has structure, into a fine grid; then blur and normalise it."""
        if self._ref is not None or self._turning or not self.bg_enabled:
            return
        bg = self.bg.background()
        if int(bg.sum()) < self.align_min_cells:
            return
        ij, v = self.bg.cell_index(points[:, :2])
        inbg = np.zeros(len(points), dtype=bool)
        inbg[v] = bg[ij[v, 0], ij[v, 1]]
        xy = points[inbg, :2]
        if len(xy) == 0:
            return
        fi = np.floor(xy / self._ref_cell).astype(np.int64) + self._ref_half
        n = self._ref_counts.shape[0]
        ok = (fi >= 0).all(axis=1) & (fi < n).all(axis=1)
        np.add.at(self._ref_counts, (fi[ok, 0], fi[ok, 1]), 1.0)
        self._ref_frames += 1
        if self._ref_frames >= self.align_ref_frames:
            c = (self._ref_counts > 0).astype(np.float32)
            pad = np.pad(c, 1)
            blur = sum(pad[1 + di:1 + di + n, 1 + dj:1 + dj + n] * w
                       for di, dj, w in [(0, 0, 4), (-1, 0, 2), (1, 0, 2), (0, -1, 2), (0, 1, 2),
                                         (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)]) / 16.0
            self._ref = blur / max(float(blur.max()), 1e-6)
            self.get_logger().info(f'Scan-alignment reference built from {self._ref_frames} frames '
                                   f'({int(c.sum())} fine cells)')

    def _maybe_align(self, points: np.ndarray):
        """Scan-to-map alignment. Runs on the first settled frame after a turn,
        then every align_interval s while settled. Modifies `points` (world
        frame, in place) and the frame pose if a correction is accepted."""
        now = self.get_clock().now().nanoseconds / 1e9
        settled_edge = self._was_turning and not self._turning
        self._was_turning = self._turning
        self._build_reference(points)
        if (not self.align_enabled or not self.bg_enabled or self._turning
                or len(points) < 200):
            return
        if not settled_edge and now - self._last_align_t < self.align_interval:
            return
        self._last_align_t = now
        bg = self.bg.background()
        if int(bg.sum()) < self.align_min_cells:
            return

        if self._ref is None:
            return
        sm = self._ref
        rng = np.random.default_rng(int(now * 1000) & 0xffffffff)
        xy = points[:, :2]
        if len(xy) > 1500:
            xy = xy[rng.choice(len(xy), 1500, replace=False)]
        rel = xy - self._frame_t

        shifts = np.arange(-self.align_xy, self.align_xy + 1e-9, self.align_xy_step)
        yaws = np.arange(-self.align_yaw, self.align_yaw + 1e-9, self.align_yaw_step)
        SX, SY = np.meshgrid(shifts, shifts, indexing='ij')
        SX, SY = SX.ravel(), SY.ravel()
        n = sm.shape[0]; cell = self._ref_cell; half = self._ref_half

        def score_for(dth):
            c, s_ = np.cos(dth), np.sin(dth)
            r = np.stack([c*rel[:, 0] - s_*rel[:, 1], s_*rel[:, 0] + c*rel[:, 1]], axis=1) + self._frame_t
            ix = np.floor((r[None, :, 0] + SX[:, None]) / cell).astype(np.int64) + half
            iy = np.floor((r[None, :, 1] + SY[:, None]) / cell).astype(np.int64) + half
            ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
            vals = np.where(ok, sm[np.clip(ix, 0, n-1), np.clip(iy, 0, n-1)], 0.0)
            return vals.sum(axis=1)

        best = (-1.0, 0.0, 0.0, 0.0); base = None
        i0 = int(np.argmin(SX**2 + SY**2))
        for dth in yaws:
            sc_ = score_for(dth)
            if abs(dth) < 1e-9:
                base = float(sc_[i0])
            i = int(np.argmax(sc_))
            if sc_[i] > best[0]:
                best = (float(sc_[i]), float(SX[i]), float(SY[i]), float(dth))
        score, dx, dy, dth = best
        accepted = (base is not None and score > base * (1.0 + self.align_min_gain)
                    and (abs(dx) > 1e-9 or abs(dy) > 1e-9 or abs(dth) > 1e-9))
        self._align_stats = {'base': round(base or 0.0, 1), 'best': round(score, 1),
                             'accepted': bool(accepted)}
        if not accepted:
            return

        # Apply: rotate about the robot position, then translate.
        c, s_ = np.cos(dth), np.sin(dth)
        relp = points[:, :2] - self._frame_t
        nx = c*relp[:, 0] - s_*relp[:, 1] + self._frame_t[0] + dx
        ny = s_*relp[:, 0] + c*relp[:, 1] + self._frame_t[1] + dy
        points[:, 0] = nx; points[:, 1] = ny
        self._corr += np.array([dx, dy, dth])
        self._frame_t = self._frame_t + np.array([dx, dy])
        self._frame_heading = self._frame_heading + dth
        TrackedPerson.ROBOT_XY = self._frame_t.copy()
        if self.debug:
            self.get_logger().info(
                f'Scan alignment: dx={dx:+.2f} dy={dy:+.2f} dyaw={np.degrees(dth):+.1f} deg '
                f'(score {base:.0f} -> {score:.0f})')

    def _observable_mask(self, heading: float) -> np.ndarray:
        """Cells inside the sensor's usable range and outside the strut sectors,
        given the robot's current position/heading in the background frame."""
        dx = self.bg.cx - self._frame_t[0]
        dy = self.bg.cy - self._frame_t[1]
        r = np.hypot(dx, dy)
        obs = (r > self.min_range + self.bg.cell) & (r < self.max_range - self.bg.cell)
        if self.occlusion_sectors:
            b = (np.degrees(np.arctan2(dy, dx) - heading)) % 360.0     # sensor-frame bearing
            for lo, hi in self.occlusion_sectors:
                lo, hi = lo % 360.0, hi % 360.0
                obs &= ~((b >= lo) & (b <= hi)) if lo <= hi else ~((b >= lo) | (b <= hi))
        return obs

    def _apply_background(self, points, weights, dt):
        if not self.bg_enabled or len(points) == 0:
            return points, weights
        # Background lives in the odom-anchored world frame: rotation-corrected
        # points plus the robot's translation since the grid origin was set.
        self._maybe_align(points)
        wxy = points[:, :2]
        occ = self.bg.occupancy(wxy, weights)

        # Scene-consistency guard: known background suddenly mostly empty
        # means the scene moved relative to the grid (robot turned without
        # heading compensation, or was moved) — clear rather than misalign.
        frac, nbg = self.bg.consistency(occ, self._observable_mask(self._frame_heading))
        # Only judged once the robot has settled: mid-turn scan smear makes
        # background briefly look inconsistent even when heading is right.
        # A genuine misalignment (wrong heading, robot moved) persists after
        # the turn and still triggers the reset.
        if not self._turning and nbg >= self.bg_reset_min_cells and frac < self.bg_consistency_min:
            self._bg_inconsistent_frames += 1
            if self._bg_inconsistent_frames >= self.bg_reset_frames:
                self.get_logger().warn(
                    f'Background RESET: only {frac:.0%} of {nbg} background cells '
                    f'occupied for {self._bg_inconsistent_frames} frames '
                    f'(robot turned/moved?). Relearning.')
                self.bg.reset()
                self._recentre_bg()
                self._bg_inconsistent_frames = 0
        else:
            self._bg_inconsistent_frames = 0

        protect = self._protect_mask()
        observable = self._observable_mask(self._frame_heading)
        if not self._turning:
            # Frozen while turning: residual scan smear/timing error would
            # otherwise make background cells look empty and erode the map
            # (forgetting is fast, relearning slow), resurfacing furniture.
            self.bg.update(occ, protect, dt, observable)
        keep = self.bg.foreground_mask(wxy,
                                       dilate_except=protect if self._turning else None)
        return points[keep], weights[keep]

    def _publish_debug(self, now: float, heading: float):
        if self.debug_json_hz <= 0 or (now - self._last_debug_json) < 1.0 / self.debug_json_hz:
            return
        self._last_debug_json = now

        def to_sensor(xy):
            xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
            return np.round(_rotate_xy(xy - self._frame_t, -heading), 3).tolist() if len(xy) else []

        bg_xy = self.bg.cells_xy() if self.bg_enabled else np.zeros((0, 2))
        acc = [dict(d, xy=to_sensor(d['xy'])[0]) for d in self._last_accepted]
        rej = [dict(d, xy=to_sensor(d['xy'])[0]) for d in self._last_rejected]
        payload = {
            't': round(now, 3),
            'stats': self._stats,
            'bg_cell': self.bg.cell,
            'bg': to_sensor(bg_xy),
            'accepted': acc,
            'rejected': rej,
            'tentative': [to_sensor(t.kalman.x[:2])[0] for t in self.tracks if not t.confirmed],
        }
        self.debug_pub.publish(String(data=json.dumps(payload)))

    def _remove_ground(self, points: np.ndarray) -> np.ndarray:
        """Returns a boolean mask of NON-ground points."""
        n = len(points)
        if n < 10:
            return np.ones(n, dtype=bool)

        z_vals       = points[:, 2]
        low_z_thresh = np.percentile(z_vals, 30)
        low_idx      = np.where(z_vals <= low_z_thresh)[0]

        best_inliers = np.zeros(n, dtype=bool)
        best_count   = 0

        rng   = np.random.default_rng(seed=42)
        iters = min(self.ground_ransac_iters, len(low_idx) // 3 if len(low_idx) >= 3 else 1)

        for _ in range(iters):
            if len(low_idx) < 3:
                break
            sample_idx = rng.choice(low_idx, 3, replace=False)
            p1, p2, p3 = points[sample_idx[0]], points[sample_idx[1]], points[sample_idx[2]]

            v1     = p2 - p1
            v2     = p3 - p1
            normal = np.cross(v1, v2)
            nlen   = np.linalg.norm(normal)
            if nlen < 1e-6:
                continue
            normal /= nlen
            if abs(normal[2]) < self.ground_min_normal_z:
                continue          # not a floor-like plane (e.g. a person's front)

            dists   = np.abs(np.dot(points - p1, normal))
            inliers = dists < self.ground_z_threshold
            count   = inliers.sum()

            if count > best_count:
                best_count   = count
                best_inliers = inliers

        return ~best_inliers

    # ── Candidate extraction ───────────────────────────────────────────────────
    def _extract_candidates(self, points: np.ndarray, labels: np.ndarray) -> list:
        candidates = []
        accepted, rejected = [], []

        for label in set(labels) - {-1}:
            cluster_pts = points[labels == label]
            centroid    = cluster_pts.mean(axis=0)
            cx, cy      = float(centroid[0]), float(centroid[1])

            x_range = float(cluster_pts[:, 0].max() - cluster_pts[:, 0].min())
            y_range = float(cluster_pts[:, 1].max() - cluster_pts[:, 1].min())
            z_range = float(cluster_pts[:, 2].max() - cluster_pts[:, 2].min())
            width   = max(x_range, y_range)
            dist    = float(np.hypot(cx - self._frame_t[0], cy - self._frame_t[1]))

            relaxed_max_width = self.max_cluster_width + max(
                0.0, dist - self.dist_persist_base
            ) * self.dist_width_relax

            footprint = max(x_range * y_range, 1e-4)
            density   = len(cluster_pts) / footprint
            # Density threshold scales with the same range model as DBSCAN.
            density_thresh = self.min_point_density / float(self._range_weight(dist))

            info = {'xy': [cx, cy], 'n': int(len(cluster_pts)), 'w': round(width, 2),
                    'zs': round(z_range, 2), 'dens': int(density), 'dist': round(dist, 2)}

            reason = None
            if dist > self.max_range - self.range_edge_margin:
                reason = f'range-edge={dist:.2f}m'
            elif z_range < self.min_z_spread or z_range > self.max_z_spread:
                reason = f'z-spread={z_range:.2f}m'
            elif not (self.min_cluster_width < width < relaxed_max_width):
                reason = f'width={width:.2f}m'
            elif z_range < self.min_cluster_height:
                reason = f'height={z_range:.2f}m'
            elif density < density_thresh:
                reason = f'density={density:.0f}<{density_thresh:.0f}pts/m²'

            if reason is not None:
                if self.debug:
                    self.get_logger().info(f'REJECTED ({reason}) at ({cx:.2f}, {cy:.2f}) n={len(cluster_pts)}')
                info['reason'] = reason.split('=')[0]
                rejected.append(info)
                continue

            candidates.append(centroid)
            accepted.append(info)
            if self.debug:
                self.get_logger().info(
                    f'Candidate: w={width:.2f}m z_spread={z_range:.2f}m '
                    f'density={density:.0f} n={len(cluster_pts)} dist={dist:.2f}m '
                    f'pos=({cx:.2f}, {cy:.2f})'
                )

        self._last_accepted, self._last_rejected = accepted, rejected
        return candidates

    # ── Tracker ────────────────────────────────────────────────────────────────
    def _in_strut_sector(self, xy_world: np.ndarray) -> bool:
        """True if a world-frame point currently lies inside a head-strut
        sector (sensor-frame bearing, widened by occlusion_hold_margin_deg)."""
        if not self.occlusion_sectors:
            return False
        d = np.asarray(xy_world[:2], dtype=np.float64) - self._frame_t
        b = (np.degrees(np.arctan2(d[1], d[0]) - self._frame_heading)) % 360.0
        m = self.hold_margin
        for lo, hi in self.occlusion_sectors:
            lo, hi = (lo - m) % 360.0, (hi + m) % 360.0
            if (lo <= b <= hi) if lo <= hi else (b >= lo or b <= hi):
                return True
        return False

    def _age_unmatched(self, t, dt: float, decay: float):
        """Age a track that got no detection this frame. Confirmed tracks
        behind a strut are held still instead of coasting (see OCCLUSION HOLD)."""
        t.absent_time += dt
        if (self.hold_enabled and t.confirmed and t.held_time < self.hold_max_secs
                and self._in_strut_sector(t.kalman.x[:2])):
            t.held = True
            t.held_time += dt
            t.kalman.x[2:4] = 0.0
        else:
            t.held = False
            t.kalman.x[2:4] *= decay

    def _update_tracks(self, candidates: list, dt: float):

        # Predict all tracks forward
        for t in self.tracks:
            t.predict(dt)

        if not candidates:
            decay = float(np.exp(-dt / max(self.coast_velocity_tau, 1e-3)))
            for t in self.tracks:
                self._age_unmatched(t, dt, decay)
            self._drop_old_tracks(dt)
            return

        if not self.tracks:
            if not self._turning:
                for c in candidates:
                    self._spawn_track(c)
            return

        track_pos = np.array([t.kalman.x[:2] for t in self.tracks])
        cand_pos  = np.array([c[:2]           for c  in candidates])

        cost = np.linalg.norm(
            track_pos[:, np.newaxis, :] - cand_pos[np.newaxis, :, :], axis=2
        )

        # Speed-scaled gate: faster tracks get a wider matching radius because
        # their predicted position has higher uncertainty.
        effective_gates = self.track_gate + np.array(
            [t.speed for t in self.tracks]
        ) * self.gate_speed_scale
        if self._turning:
            # While turning, a track's speed is dominated by ego-motion error,
            # not the person, so it does not widen the gate.
            effective_gates = np.full(len(self.tracks), self.track_gate)
        # A coasting track's speed is unobserved, so it never widens the gate
        # (otherwise an occluded target can latch onto nearby clutter).
        coasting = np.array([t.absent_time > 0 for t in self.tracks])
        effective_gates = np.where(coasting, self.track_gate, effective_gates)
        # A track held behind a strut re-acquires with a wider gate: its
        # position is remembered, not predicted.
        held = np.array([t.held_time > 0 and t.absent_time > 0 for t in self.tracks])
        effective_gates = np.where(held, max(self.reacquire_gate, self.track_gate), effective_gates)

        # Mask infeasible (track, candidate) pairs before solving the
        # assignment, not after. Solving on the raw cost matrix lets the
        # optimiser trade a feasible match for a lower-cost infeasible one
        # elsewhere (e.g. two people close together or crossing), which the
        # gate would then reject, losing the feasible match.
        INFEASIBLE = 1e6
        masked_cost = np.where(cost >= effective_gates[:, np.newaxis], INFEASIBLE, cost)

        if SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(masked_cost)
        else:
            row_ind, col_ind = self._greedy_assign(masked_cost)

        matched_tracks: set[int] = set()
        matched_cands:  set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if masked_cost[r, c] >= INFEASIBLE:
                continue

            # Implied speed uses the time since the track was last matched,
            # not this frame's dt. A track unmatched for several frames (e.g.
            # hidden behind a strut) accumulates prediction drift over the
            # whole gap; dividing that by one ~0.1 s frame would give a large,
            # meaningless speed even for a person who never moved.
            # absent_time does not yet include this frame's dt (ageing runs
            # after this loop), so the gap is absent_time + dt.
            elapsed = self.tracks[r].absent_time + dt
            implied_speed = cost[r, c] / max(elapsed, 1e-3)
            if implied_speed > self.max_speed_ms:
                if self.debug:
                    self.get_logger().info(
                        f'REJECTED association track={self.tracks[r].track_id} '
                        f'implied_speed={implied_speed:.1f} m/s > {self.max_speed_ms} m/s'
                    )
                continue

            self.tracks[r].update(candidates[c])
            matched_tracks.add(r)
            matched_cands.add(c)

        # Age unmatched tracks (and bleed off their velocity while coasting)
        decay = float(np.exp(-dt / max(self.coast_velocity_tau, 1e-3)))
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                self._age_unmatched(track, dt, decay)

        # Spawn tentative tracks for unmatched candidates (not while turning:
        # smeared clutter mid-turn should not start new tracks)
        for j, cand in enumerate(candidates):
            if j not in matched_cands and not self._turning:
                self._spawn_track(cand)

        # Confirm tracks that have hit the threshold
        for t in self.tracks:
            if not t.confirmed and t.hits >= self.confirm_hits:
                t.confirmed = True
                t.person_id = self.next_person_id
                self.next_person_id += 1
                self.get_logger().info(
                    f'Track {t.track_id} CONFIRMED as P{t.person_id} '
                    f'after {t.hits} hits'
                )

        self._drop_old_tracks(dt)

    def _spawn_track(self, centroid: np.ndarray):
        t = TrackedPerson(self.next_track_id, centroid)
        now = self.get_clock().now().nanoseconds / 1e9
        self._recent_protected_drops = [
            (xy, ts) for xy, ts in self._recent_protected_drops
            if now - ts <= self.bg_inherit_secs]
        if any(np.linalg.norm(centroid[:2] - xy) <= self.bg_inherit_radius
               for xy, ts in self._recent_protected_drops):
            t.protected = True
        self.next_track_id += 1
        self.tracks.append(t)

    def _drop_old_tracks(self, dt: float = 0.0):
        surviving = []
        for t in self.tracks:
            # Time-based absence budget
            extra_secs     = max(0.0, t.dist - self.dist_persist_base) * self.dist_persist_extra_secs
            absence_budget = self.max_absent_secs + extra_secs

            if t.absent_time - t.held_time > absence_budget:
                if t.confirmed and t.protected:
                    self._recent_protected_drops.append(
                        (t.kalman.x[:2].copy(), self.get_clock().now().nanoseconds / 1e9))
                status = 'confirmed' if t.confirmed else 'tentative'
                self.get_logger().info(
                    f'Dropped {status} track {t.track_id} '
                    f'(absent={t.absent_time:.1f}s > budget={absence_budget:.1f}s, '
                    f'dist={t.dist:.1f}m)'
                )
                continue

            # No static blacklisting. With the background model enabled, a
            # confirmed track that has never genuinely moved is not protected
            # (see _protect_mask); if it is furniture, its cells are learned as
            # background, its points stop reaching DBSCAN, and it drops through
            # the absence budget above. Nothing is blocked permanently.
            surviving.append(t)
        self.tracks = surviving

    @staticmethod
    def _greedy_assign(cost: np.ndarray):
        row_ind, col_ind = [], []
        used_cols: set[int] = set()
        for r in range(cost.shape[0]):
            best_c, best_v = -1, float('inf')
            for c in range(cost.shape[1]):
                if c not in used_cols and cost[r, c] < best_v:
                    best_c, best_v = c, cost[r, c]
            if best_c >= 0:
                row_ind.append(r)
                col_ind.append(best_c)
                used_cols.add(best_c)
        return row_ind, col_ind

    # ── /nearest_person target selection (with hysteresis) ─────────────────────
    def _select_target(self, active: list, coasting: list, dt: float):
        """
        Choose which confirmed track to publish as /nearest_person this frame.

        Holds a soft lock on a person_id rather than recomputing "nearest"
        from scratch every frame, to stop the published target flip-flopping
        when two people are near-equidistant (noise in either distance
        estimate would otherwise be enough to swap the winner frame to frame).

        Returns the TrackedPerson to publish, or None if the locked target
        exists but has no fresh detection this frame (coasting) — in that
        case the lock is deliberately held rather than handed to whoever is
        nearest, so a one-frame occlusion doesn't cause a target switch.
        """
        self._time_since_switch += dt
        all_tracked = active + coasting

        current = None
        if self._current_target_person_id is not None:
            current = next(
                (t for t in all_tracked if t.person_id == self._current_target_person_id),
                None,
            )

        best_active = min(active, key=lambda t: t.dist)

        # No valid lock (first frame, or the previously-locked track has
        # aged out entirely) — acquire immediately, no hysteresis needed
        # since there's nothing to be "stolen" from.
        if current is None:
            self._lock_onto(best_active, reason='acquire')
            return best_active

        # Locked target exists but didn't get a fresh hit this frame
        # (occluded / coasting) — hold the lock silently rather than
        # awarding it to someone else just because they're nearest now.
        if current not in active:
            return None

        # Locked target is active. Only switch if a challenger is closer by
        # more than the margin AND the dwell timer has expired.
        if (best_active.person_id != current.person_id
                and best_active.dist < current.dist - self.target_switch_margin
                and self._time_since_switch >= self.target_min_dwell_secs):
            self._lock_onto(best_active, reason='closer')
            return best_active

        return current

    def _lock_onto(self, track: 'TrackedPerson', reason: str):
        if track.person_id == self._current_target_person_id:
            return
        old_id = self._current_target_person_id
        self._current_target_person_id = track.person_id
        self._time_since_switch = 0.0
        self.get_logger().info(
            f'TARGET SWITCH: P{old_id} -> P{track.person_id} '
            f'(reason={reason}, dist={track.dist:.2f}m)'
        )

    # ── Publish ────────────────────────────────────────────────────────────────
    def _age_tracks_and_publish(self, candidates, msg, dt: float, heading: float):
        # Active: confirmed and matched this frame
        active  = [t for t in self.tracks if t.confirmed and t.absent_time == 0.0]
        # Coasting: confirmed, absent but within coast window — Kalman predicts position
        coasting = [
            t for t in self.tracks
            if t.confirmed and t.absent_time > 0.0 and (t.absent_time - t.held_time) <= self.coast_secs and not t.held
        ]
        # Held behind a strut: published for as long as the hold lasts
        held = [t for t in self.tracks if t.confirmed and t.absent_time > 0.0 and t.held]

        header = msg.header
        header.frame_id = self.output_frame

        # Tracking runs in the stabilised frame. Everything published here
        # (PoseStamped, markers) must be in the current sensor frame, because
        # downstream nodes need bearings relative to where the robot faces now,
        # so positions are rotated back by -heading only at this step.
        # track.dist is measured from the robot's position and is unaffected
        # by rotation.
        def _to_sensor_xy(xy: np.ndarray) -> np.ndarray:
            return _rotate_xy(xy.reshape(1, 2) - self._frame_t, -heading).flatten()

        if active:
            target = self._select_target(active, coasting, dt)
            if target is not None:
                sensor_xy = _to_sensor_xy(target.smoothed[:2])
                pose = PoseStamped()
                pose.header = header
                pose.pose.position.x = float(sensor_xy[0])
                pose.pose.position.y = float(sensor_xy[1])
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                self.person_pub.publish(pose)

                self.get_logger().info(
                    f'Tracking {len(active)} active + {len(coasting)} coasting person(s). Target: '
                    f'ID={target.person_id} dist={target.dist:.2f}m '
                    f'speed={target.speed:.2f}m/s '
                    f'pos=({sensor_xy[0]:.2f}, {sensor_xy[1]:.2f}) [sensor frame]'
                )

        people_markers = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        people_markers.markers.append(clear)

        def _add_track_markers(track: TrackedPerson, alpha_scale: float = 1.0):
            """Add cylinder, velocity arrow, and label for one track.
            Positions are converted from the internal stabilised frame back to
            the current sensor frame so RViz shows markers at the object's true
            physical bearing relative to the robot right now."""
            sensor_xy = _to_sensor_xy(track.smoothed[:2])
            sx, sy, sz = float(sensor_xy[0]), float(sensor_xy[1]), float(track.smoothed[2])

            # Cylinder body
            m = Marker()
            m.header = header
            m.ns     = 'people'
            m.id     = track.person_id
            m.type   = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = sx
            m.pose.position.y = sy
            m.pose.position.z = sz
            m.pose.orientation.w = 1.0
            m.scale.x = 0.5
            m.scale.y = 0.5
            m.scale.z = 1.6
            c = track.colour
            m.color = ColorRGBA(r=c.r, g=c.g, b=c.b, a=c.a * alpha_scale)
            people_markers.markers.append(m)

            # Velocity arrow: the velocity direction is also rotated back into
            # the sensor frame; its magnitude is unchanged.
            vel = track.kalman.velocity
            if np.linalg.norm(vel) > 0.1:
                sensor_vel = _to_sensor_xy(vel)
                arrow = Marker()
                arrow.header = header
                arrow.ns     = 'velocity'
                arrow.id     = track.person_id + 2000
                arrow.type   = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose.position.x = sx
                arrow.pose.position.y = sy
                arrow.pose.position.z = sz + 0.5
                speed     = float(np.linalg.norm(vel))
                yaw_arrow = float(np.arctan2(sensor_vel[1], sensor_vel[0]))
                arrow.pose.orientation.z = float(np.sin(yaw_arrow / 2.0))
                arrow.pose.orientation.w = float(np.cos(yaw_arrow / 2.0))
                arrow.scale.x = speed * 0.5
                arrow.scale.y = 0.06
                arrow.scale.z = 0.06
                arrow.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.7 * alpha_scale)
                people_markers.markers.append(arrow)

            # Text label; absent tracks show the seconds since last seen
            label = Marker()
            label.header = header
            label.ns     = 'labels'
            label.id     = track.person_id + 1000
            label.type   = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = sx
            label.pose.position.y = sy
            label.pose.position.z = sz + 1.0
            label.pose.orientation.w = 1.0
            label.scale.z = 0.3
            label.color   = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0 * alpha_scale)
            suffix = f' ~{track.absent_time:.1f}s' if alpha_scale < 1.0 else ''
            label.text = f'P{track.person_id} {track.dist:.1f}m{suffix}'
            people_markers.markers.append(label)

        for track in active:
            _add_track_markers(track, alpha_scale=1.0)

        for track in coasting:
            # Fade opacity linearly from 0.7 → 0.15 across the coast window
            fade = 1.0 - ((track.absent_time - track.held_time) / self.coast_secs)
            _add_track_markers(track, alpha_scale=max(0.15, fade * 0.7))

        for track in held:
            # alpha 0.85*0.8 = 0.68: above people_count's 0.65 coasting
            # threshold, so a person behind a strut still counts as present.
            _add_track_markers(track, alpha_scale=0.8)

        self.people_pub.publish(people_markers)
        self.marker_pub.publish(people_markers)


def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

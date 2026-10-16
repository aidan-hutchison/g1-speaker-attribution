#!/usr/bin/env python3
"""
person_turn_node.py — turns the robot to face a target
Unitree G1 speaker attribution system

PURPOSE
-------
Subscribes to a target position (PoseStamped on input_topic; /turn_target
from sensor_fusion_node in the full system) and computes a yaw rate that
turns the robot on the spot until its forward axis (+X in livox_frame)
points at the target. The robot does not walk: only vyaw is ever non-zero.

WHY THIS IS SPLIT INTO TWO PROCESSES
------------------------------------
rclpy and unitree_sdk2py both initialise CycloneDDS in the process that
imports them. On this robot's CycloneDDS build, having both in one process
is unreliable: LocoClient's topic creation fails with
DDS_RETCODE_BAD_PARAMETER once rclpy owns a participant on the domain.

So this node uses only rclpy and sends its command over a localhost UDP
socket. loco_driver.py uses only unitree_sdk2py, receives the command and
calls LocoClient.Move().

  sensor_fusion_node  ->  /turn_target (PoseStamped)
                                  |
                       [person_turn_node]      rclpy only (this file)
                                  |   UDP: little-endian float64 vyaw
                                  |   to udp_host:udp_port (default 127.0.0.1:47921)
                       [loco_driver.py]        unitree_sdk2py only
                                  |
                       LocoClient.Move(0, 0, vyaw)

CONTROL LAW
-----------
    error = atan2(target_y, target_x)            (livox_frame)
    vyaw  = clamp(-Kp * error, -max_vyaw, +max_vyaw)
    vyaw  = 0 while |error| < dead_zone

The sign of -Kp suits this robot's inverted, mirrored LiDAR mounting and was
checked physically: the robot turns toward the target. The command is sent
at COMMAND_RATE_HZ using the latest bearing. If no target message arrives for
lost_timeout seconds, an optional search sweep runs for search_timeout
seconds (search_timeout 0 disables it), after which vyaw is 0 until the
target returns.

USAGE
-----
  python3 person_turn_node.py --ros-args -p input_topic:=/turn_target -p Kp:=2.0 \
      -p lost_timeout:=0.5 -p search_timeout:=0.0        # evaluated configuration
  python3 person_turn_node.py --ros-args -p dry_run:=true # log only, no UDP sent

PARAMETERS (override with --ros-args -p key:=value)
----------
  Kp               : proportional gain (default: 1.2; evaluated: 2.0)
  max_vyaw         : maximum yaw rate in rad/s (default: 0.8)
  dead_zone_deg    : angular dead zone in degrees (default: 8.0)
  lost_timeout     : seconds without a target before searching (default: 2.0;
                     evaluated: 0.5)
  command_rate_hz  : declared, but the loop runs at COMMAND_RATE_HZ (20 Hz)
  dry_run          : log instead of sending UDP (default: False)
  min_distance     : ignore targets closer than this, metres (default: 0.5)
  input_topic      : target topic (default: /nearest_person; evaluated:
                     /turn_target)
  udp_host         : loco_driver host (default: 127.0.0.1)
  udp_port         : loco_driver port (default: 47921)
  search_vyaw      : search sweep yaw rate, rad/s (default: 0.25)
  search_sweep_deg : sweep half-angle in degrees (default: 35.0)
  search_timeout   : seconds of searching before stopping (default: 8.0;
                     evaluated: 0.0, search off)

TUNING
------
  Kp too high          -> oscillation or overshoot.
  Kp too low           -> sluggish; a moving person stays off-centre.
  dead_zone too tight  -> jittery stepping; too wide -> stops off-axis.
  max_vyaw above ~0.9  -> can make the robot stumble while turning on the spot.
"""

import math
import socket
import struct
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class PersonTurnNode(Node):

    COMMAND_RATE_HZ = 20   # Hz at which a command is re-issued (constant watchdog feed)

    def __init__(self):
        super().__init__('person_turn_node')

        # ── Parameters ─────────────────────────────────────────────────
        self.declare_parameter('Kp',               1.2)
        self.declare_parameter('max_vyaw',          0.8)
        self.declare_parameter('dead_zone_deg',     8.0)
        self.declare_parameter('lost_timeout',      2.0)
        self.declare_parameter('command_rate_hz',  20.0)
        self.declare_parameter('dry_run',          False)
        self.declare_parameter('min_distance',      0.5)
        # Target topic. /nearest_person (the detector's nearest track) turns
        # toward whoever is closest; /turn_target (sensor_fusion_node) is the
        # full speaker-attribution system.
        self.declare_parameter('input_topic',      '/nearest_person')
        # UDP link to loco_driver.py
        self.declare_parameter('udp_host',        '127.0.0.1')
        self.declare_parameter('udp_port',         47921)
        # Search sweep after the target is lost (search_timeout 0 disables it)
        self.declare_parameter('search_vyaw',       0.25)   # rad/s — slow deliberate sweep
        self.declare_parameter('search_sweep_deg',  35.0)   # degrees each side of centre
        self.declare_parameter('search_timeout',    8.0)    # seconds before giving up search

        p = self.get_parameter
        self._Kp                = p('Kp').value
        self._max_vyaw          = p('max_vyaw').value
        self._dead_zone         = math.radians(p('dead_zone_deg').value)
        self._lost_timeout      = p('lost_timeout').value
        self._cmd_rate          = p('command_rate_hz').value
        self._dry_run           = p('dry_run').value
        self._min_distance      = p('min_distance').value
        self._input_topic       = p('input_topic').value
        self._udp_host          = p('udp_host').value
        self._udp_port          = int(p('udp_port').value)
        self._search_vyaw       = p('search_vyaw').value
        self._search_sweep      = math.radians(p('search_sweep_deg').value)
        self._search_timeout    = p('search_timeout').value

        # ── State ───────────────────────────────────────────────────────
        self._heading_error: float = 0.0   # radians, updated by subscriber
        self._last_detection: float = 0.0  # wall time of the last target message
        self._state_lock = threading.Lock()

        # Search state, used while the target is lost
        self._search_start: float     = 0.0
        self._search_yaw_accum: float = 0.0
        self._search_direction: int   = 1
        self._search_idle: bool       = False

        # ── UDP command socket (to loco_driver.py) ─────────────────────
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_addr = (self._udp_host, self._udp_port)

        # ── ROS2 subscriber ─────────────────────────────────────────────
        self._sub = self.create_subscription(
            PoseStamped,
            self._input_topic,
            self._person_callback,
            10,
        )

        # ── Control loop timer ──────────────────────────────────────────
        self._timer = self.create_timer(
            1.0 / self.COMMAND_RATE_HZ,
            self._control_loop,
        )

        self.get_logger().info('Person turn node started (UDP-out variant).')
        self.get_logger().info(f'  Kp={self._Kp}  max_vyaw={self._max_vyaw} rad/s  '
                               f'dead_zone={math.degrees(self._dead_zone):.1f}°')
        self.get_logger().info(f'  lost_timeout={self._lost_timeout}s  '
                               f'dry_run={self._dry_run}')
        self.get_logger().info(f'  sending vyaw → udp {self._udp_host}:{self._udp_port}')
        self.get_logger().info(f'  bearing source: {self._input_topic}')

    # ── Person subscriber ───────────────────────────────────────────────────────
    def _person_callback(self, msg: PoseStamped):
        """Store the heading error to the incoming target,
        error = atan2(y, x) in livox_frame (see CONTROL LAW)."""
        px = msg.pose.position.x
        py = msg.pose.position.y
        distance = math.sqrt(px**2 + py**2)

        if distance < self._min_distance:
            self.get_logger().debug(
                f'Person too close ({distance:.2f}m < {self._min_distance}m), ignoring'
            )
            return

        error = math.atan2(py, px)

        with self._state_lock:
            self._heading_error  = error
            self._last_detection = time.time()

        self.get_logger().info(
            f'Person detected at ({px:.2f}, {py:.2f})  dist={distance:.2f}m  '
            f'bearing={math.degrees(error):+.1f}°'
        )

    # ── Control loop ────────────────────────────────────────────────────────────
    def _control_loop(self):
        """
        Called at COMMAND_RATE_HZ. Three states:

        TRACKING  — target current: proportional control toward its bearing
        SEARCHING — target recently lost: slow sweep either side
        IDLE      — no target yet, or search finished: vyaw = 0
        """
        now = time.time()
        dt  = 1.0 / self.COMMAND_RATE_HZ

        with self._state_lock:
            error     = self._heading_error
            last_seen = self._last_detection

        person_visible = (last_seen > 0.0) and ((now - last_seen) <= self._lost_timeout)

        # ── TRACKING ─────────────────────────────────────────────────
        if person_visible:
            self._search_start     = 0.0
            self._search_yaw_accum = 0.0
            self._search_direction = 1
            self._search_idle      = False

            if abs(error) < self._dead_zone:
                self._send_move(0.0, reason=f'in dead zone ({math.degrees(error):.1f}°)')
                return

            vyaw = -self._Kp * error
            vyaw = max(-self._max_vyaw, min(self._max_vyaw, vyaw))
            self._send_move(vyaw, reason=f'tracking  error={math.degrees(error):+.1f}°')
            return

        # ── Person just lost — start search if not already running ────
        if self._search_start == 0.0:
            if last_seen == 0.0 or self._search_idle:
                self._send_move(0.0, reason='idle/no detection yet' if last_seen == 0.0 else 'idle/search exhausted')
                return
            self._search_start     = now
            self._search_yaw_accum = 0.0
            self._search_direction = 1
            self.get_logger().info('Person lost — starting blind-spot search sweep')

        # ── SEARCHING ────────────────────────────────────────────────
        elapsed = now - self._search_start
        if elapsed > self._search_timeout:
            self._send_move(0.0, reason='search timed out')
            self.get_logger().info('Search timed out — going idle until person reacquired')
            self._search_start = 0.0
            self._search_idle  = True
            return

        vyaw = self._search_direction * self._search_vyaw
        self._search_yaw_accum += vyaw * dt

        if self._search_yaw_accum >= self._search_sweep:
            self._search_direction  = -1
            self._search_yaw_accum  =  self._search_sweep
        elif self._search_yaw_accum <= -self._search_sweep:
            self._search_direction  =  1
            self._search_yaw_accum  = -self._search_sweep

        self._send_move(
            vyaw,
            reason=f'searching  elapsed={elapsed:.1f}s  '
                   f'accum={math.degrees(self._search_yaw_accum):+.1f}°'
        )

    def _send_move(self, vyaw: float, reason: str = ''):
        """Emit vyaw to loco_driver.py over UDP (vx=vy=0 implied)."""
        if self._dry_run:
            if abs(vyaw) > 0.01:
                self.get_logger().info(f'[DRY-RUN] vyaw={vyaw:+.3f}  {reason}')
            return
        try:
            self._sock.sendto(struct.pack('<d', float(vyaw)), self._udp_addr)
        except Exception as e:
            self.get_logger().warn(f'UDP send failed: {e}')


# ── Entry point ────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = PersonTurnNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Emit a final stop (vyaw = 0) before shutting down
        node._send_move(0.0, reason='shutdown')
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

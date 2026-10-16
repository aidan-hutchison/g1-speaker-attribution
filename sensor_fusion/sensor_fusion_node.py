#!/usr/bin/env python3
"""
sensor_fusion_node.py — target selection (the selection layer)
Unitree G1 speaker attribution system

PURPOSE
-------
Decides whom the robot should face, and is the only node that publishes
/turn_target. It combines who is tracked (person_detector_node), who most
recently spoke (speaker_attribution_node) and what was asked
(voice_command_node) into one decision. Because nothing else publishes
/turn_target, the selection logic can change here without touching
person_turn_node or loco_driver.

It also sends the follow state to led_driver.py over UDP (red while
following a live target, blue otherwise). led_driver.py runs the Unitree SDK
in its own process; this node uses only rclpy.

INPUTS
------
  /detected_people   (MarkerArray, from person_detector_node)
                     Tracked people: ns='people', marker.id = person_id,
                     position (x, y) in the current sensor frame.
  /doa_locked_person (std_msgs/Int32, from speaker_attribution_node)
                     The person_id most recently attributed as speaking,
                     published while that attribution is locked. Treated as
                     current for doa_freshness_secs after the last message.
  /voice_command     (std_msgs/String, from voice_command_node)
                     "follow", "follow_closest" or "stop".

OUTPUT
------
  /turn_target       (PoseStamped) position of the person to face. Published
                     only when the target is present in the current
                     /detected_people message. No position is cached: a
                     repeated stale position would look like fresh data to
                     person_turn_node and drive it past the target. Brief
                     gaps are bridged by the detector's own coasting and
                     occlusion hold instead.
  UDP -> led_driver  one byte (0x00 / 0x01), sent when the follow state
                     changes. 0x01 means FOLLOWING with the target currently
                     tracked.

STATE MACHINE
-------------
  IDLE (default)  Nothing is published, so the robot never turns just
                  because someone nearby is talking. DOA attributions are
                  still recorded, so one is available the moment a follow
                  command arrives.

  FOLLOWING       Entered on "follow" or "follow_closest". The matching
                  nomination strategy picks a person_id once, and that id
                  stays the target whatever DOA does next. While the target's
                  track is absent, nothing is published and person_turn_node
                  stops turning. The robot only turns; it does not walk
                  toward the target.

  "stop"          Returns to IDLE and clears the target from any state, with
                  no other condition.

NOMINATION STRATEGIES
---------------------
  follow          the person DOA most recently and freshly attributed as
                  speaking, if that track still exists
  follow_closest  the person nearest the robot now, without DOA

Strategies are entries in _nomination_strategies, keyed by command string,
separate from the state machine. A new command needs only a new entry and
the matching string from voice_command_node.py. Once FOLLOWING starts,
behaviour is the same whichever strategy chose the target.

WALKING
-------
Following means turning to face the target. Walking toward them would need
an approach controller producing vx as well as vyaw, an extended UDP message
to loco_driver, and a defined safe following distance.

USAGE
-----
  cd ~/aidan/sensor_fusion        # with the ROS 2 environment sourced
  python3 sensor_fusion_node.py

Run after person_detector_node, speaker_attribution_node and
voice_command_node. led_driver.py is needed to see the LED change; the UDP
send is fire-and-forget, so this node runs normally without it.

PARAMETERS (override with --ros-args -p key:=value)
----------
  tracks_topic       : person tracks (default: /detected_people)
  doa_topic          : DOA-attributed person id (default: /doa_locked_person)
  command_topic      : voice commands (default: /voice_command)
  output_topic       : target output (default: /turn_target)
  frame_id           : header.frame_id of published messages (default: livox_frame)
  doa_freshness_secs : how recent a /doa_locked_person message must be to
                       nominate from it (default: 2.0)
  led_udp_host       : led_driver.py host (default: 127.0.0.1)
  led_udp_port       : led_driver.py port (default: 47922)
  debug              : verbose state/target logging (default: True)
"""

from __future__ import annotations

import math
import socket
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32, String
from visualization_msgs.msg import MarkerArray


class SensorFusionNode(Node):

    MODE_IDLE      = 'IDLE'
    MODE_FOLLOWING = 'FOLLOWING'

    def __init__(self):
        super().__init__('sensor_fusion_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('tracks_topic',    '/detected_people')
        self.declare_parameter('doa_topic',    '/doa_locked_person')
        self.declare_parameter('command_topic',    '/voice_command')
        self.declare_parameter('output_topic',       '/turn_target')
        self.declare_parameter('frame_id',          'livox_frame')
        self.declare_parameter('doa_freshness_secs',          2.0)
        self.declare_parameter('led_udp_host',        '127.0.0.1')
        self.declare_parameter('led_udp_port',              47922)
        self.declare_parameter('debug',                      True)

        p = self.get_parameter
        self._tracks_topic     = p('tracks_topic').value
        self._doa_topic        = p('doa_topic').value
        self._command_topic    = p('command_topic').value
        self._output_topic     = p('output_topic').value
        self._frame_id         = p('frame_id').value
        self._doa_freshness    = float(p('doa_freshness_secs').value)
        self._led_udp_host     = p('led_udp_host').value
        self._led_udp_port     = int(p('led_udp_port').value)
        self._debug            = bool(p('debug').value)

        # ── State ─────────────────────────────────────────────────────
        self._tracks: dict[int, tuple[float, float]] = {}   # id -> (x, y)

        self._last_doa_id: int | None = None
        self._last_doa_time: float = 0.0

        self._mode = self.MODE_IDLE
        self._follow_target_id: int | None = None

        # LED state link to led_driver.py. Fire-and-forget: without a
        # receiver the node runs normally and the LED simply does not change.
        self._led_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._led_addr = (self._led_udp_host, self._led_udp_port)
        self._led_last_sent: bool | None = None  # None = never sent yet

        # Nomination strategies, keyed by the /voice_command string that
        # triggers them (see NOMINATION STRATEGIES in the module docstring).
        self._nomination_strategies = {
            'follow': self._nominate_by_doa,
            'follow_closest': self._nominate_closest,
        }

        # ── Pub/Sub ──────────────────────────────────────────────────────
        self._pub = self.create_publisher(PoseStamped, self._output_topic, 10)
        self.create_subscription(MarkerArray, self._tracks_topic, self._on_tracks, 10)
        self.create_subscription(Int32, self._doa_topic, self._on_doa, 10)
        self.create_subscription(String, self._command_topic, self._on_command, 10)

        self.get_logger().info('Sensor fusion node started.')
        self.get_logger().info(
            f'  tracks={self._tracks_topic}  doa={self._doa_topic}  '
            f'command={self._command_topic}  output={self._output_topic}'
        )
        self.get_logger().info(f'  doa_freshness_secs={self._doa_freshness}')
        self.get_logger().info(
            f'  LED state -> udp {self._led_udp_host}:{self._led_udp_port}'
        )

    # ── /doa_locked_person subscriber ─────────────────────────────────────
    def _on_doa(self, msg: Int32):
        self._last_doa_id = int(msg.data)
        self._last_doa_time = time.time()

    def _doa_is_fresh(self) -> bool:
        return (
            self._last_doa_id is not None
            and (time.time() - self._last_doa_time) <= self._doa_freshness
        )

    # ── LED state UDP send ────────────────────────────────────────────────
    def _send_led_state(self, tracking: bool):
        """Send the follow-state byte to led_driver.py when it changes.
        True means FOLLOWING with the target currently tracked."""
        if tracking == self._led_last_sent:
            return
        try:
            self._led_sock.sendto(bytes([1 if tracking else 0]), self._led_addr)
        except Exception as e:
            self.get_logger().warn(f'LED UDP send failed: {e}')
        self._led_last_sent = tracking

    # ── /voice_command subscriber ──────────────────────────────────────────
    def _on_command(self, msg: String):
        command = msg.data.strip().lower()

        # Stop is unconditional: clear state regardless of current mode.
        if command == 'stop':
            if self._mode != self.MODE_IDLE or self._follow_target_id is not None:
                self.get_logger().info('STOP received — clearing follow target, returning to IDLE.')
            self._mode = self.MODE_IDLE
            self._follow_target_id = None
            self._send_led_state(False)
            return

        strategy = self._nomination_strategies.get(command)
        if strategy is None:
            self.get_logger().warn(f'Unrecognised voice command {command!r} — ignoring.')
            return

        nominee = strategy()
        if nominee is None:
            self.get_logger().warn(
                f'Command {command!r} received, but nomination strategy found nobody '
                'to target — staying in current mode.'
            )
            return

        self._mode = self.MODE_FOLLOWING
        self._follow_target_id = nominee
        self.get_logger().info(f'Command {command!r} — now FOLLOWING P{nominee}.')

    # ── Nomination strategies ────────────────────────────────────────────
    def _nominate_by_doa(self) -> int | None:
        """Whoever DOA most recently, freshly attributed as speaking."""
        if not self._doa_is_fresh():
            return None
        if self._last_doa_id not in self._tracks:
            return None
        return self._last_doa_id

    def _nominate_closest(self) -> int | None:
        """Whoever is nearest the robot now, without DOA. Triggered by
        'follow_closest' ("<wake word>, closest")."""
        if not self._tracks:
            return None
        return min(
            self._tracks,
            key=lambda pid: math.hypot(*self._tracks[pid]),
        )

    # ── /detected_people subscriber (also publishes /turn_target) ─────────
    def _on_tracks(self, msg: MarkerArray):
        tracks = {}
        for m in msg.markers:
            if m.ns == 'people':
                tracks[m.id] = (m.pose.position.x, m.pose.position.y)
        self._tracks = tracks

        target_id = self._current_target_id()
        if target_id is None:
            self._send_led_state(False)
            return

        xy = tracks.get(target_id)
        if xy is None:
            # Target absent this frame: publish nothing rather than a held
            # position. The detector's coasting and occlusion hold bridge
            # short gaps.
            self._send_led_state(False)
            return

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.pose.position.x = float(xy[0])
        out.pose.position.y = float(xy[1])
        out.pose.position.z = 0.0
        out.pose.orientation.w = 1.0
        self._pub.publish(out)
        self._send_led_state(True)

        if self._debug:
            self.get_logger().debug(
                f'[{self._mode}] target=P{target_id} pos=({xy[0]:.2f}, {xy[1]:.2f})'
            )

    def _current_target_id(self) -> int | None:
        if self._mode == self.MODE_FOLLOWING:
            return self._follow_target_id
        # IDLE: stay stationary. The robot turns only after a follow
        # command; DOA is recorded meanwhile so _nominate_by_doa can use it.
        return None


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

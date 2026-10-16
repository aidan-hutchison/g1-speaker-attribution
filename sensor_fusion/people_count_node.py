#!/usr/bin/env python3
"""
people_count_node.py — answers "how many people do you see?"
Unitree G1 speaker attribution system

PURPOSE
-------
Speaks one status line, such as "I see 2 people. The closest is 1.3 metres
away.", only when asked by voice ("<wake word>, how many people do you see").
It has no timer and never speaks on its own.

The text is sent once over localhost UDP to audio_driver.py, which drives the
robot's loudspeaker through the Unitree SDK in a separate process. This node
uses only rclpy.

INPUTS
------
  /detected_people   (MarkerArray, from person_detector_node). CYLINDER
                     markers in namespace 'people' are tracks. A marker with
                     alpha below coasting_alpha_threshold is a track being
                     carried through an occlusion on its prediction, and is
                     not counted, so the spoken count matches who is visible.
  /voice_command     (std_msgs/String). Responds only to "count_people"; it
                     is independent of the follow/stop state machine in
                     sensor_fusion_node.py.

OUTPUT
------
  UDP -> audio_driver.py (default port 47923), once per "count_people"
  command.

USAGE
-----
  cd ~/aidan/sensor_fusion        # with the ROS 2 environment sourced
  python3 people_count_node.py

PARAMETERS
----------
  command_topic            : (default: /voice_command)
  tracks_topic             : (default: /detected_people)
  audio_driver_port        : (default: 47923)
  coasting_alpha_threshold : (default: 0.65) tracks drawn fainter than this
                             are coasting and are not counted
  presence_hold_sec        : (default: 1.0) if the latest frame has no
                             visible tracks, use the last non-empty frame
                             from within this many seconds. This covers a
                             track that misses one match cycle and briefly
                             reads as coasting at the moment of the question.
"""

from __future__ import annotations

import math
import socket
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class TextSpeaker:
    """Fire-and-forget UDP send to audio_driver.py."""

    def __init__(self, host="127.0.0.1", port=47923):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def say(self, text: str):
        self.sock.sendto(text.encode("utf-8"), self.addr)


class PeopleCountNode(Node):
    def __init__(self):
        super().__init__("people_count_node")

        self.declare_parameter("command_topic", "/voice_command")
        self.declare_parameter("tracks_topic", "/detected_people")
        self.declare_parameter("audio_driver_port", 47923)
        self.declare_parameter("coasting_alpha_threshold", 0.65)
        self.declare_parameter("presence_hold_sec", 1.0)

        p = self.get_parameter
        self._command_topic = p("command_topic").value
        self._tracks_topic = p("tracks_topic").value
        self._coasting_alpha_threshold = p("coasting_alpha_threshold").value
        self._presence_hold_sec = p("presence_hold_sec").value

        self.speech = TextSpeaker(port=p("audio_driver_port").value)

        self._latest_tracks: list[float] = []       # distances (m), this frame
        self._last_nonzero_tracks: list[float] = []
        self._last_nonzero_time = 0.0

        self.create_subscription(MarkerArray, self._tracks_topic, self._on_tracks, 10)
        self.create_subscription(String, self._command_topic, self._on_command, 10)

        self.get_logger().info(
            f"people_count_node up — tracks={self._tracks_topic} "
            f"command={self._command_topic} -> audio_driver udp port "
            f"{p('audio_driver_port').value}. Speaks ONLY on 'count_people' "
            f"command, never on a timer."
        )

    # ── /detected_people subscriber ─────────────────────────────────────
    def _on_tracks(self, msg: MarkerArray):
        distances = []
        for m in msg.markers:
            if m.ns != "people" or m.type != Marker.CYLINDER or m.action != Marker.ADD:
                continue
            if m.color.a < self._coasting_alpha_threshold:
                continue
            x = m.pose.position.x
            y = m.pose.position.y
            distances.append(math.sqrt(x * x + y * y))

        self._latest_tracks = distances
        if distances:
            self._last_nonzero_tracks = distances
            self._last_nonzero_time = time.time()

    def _effective_tracks(self) -> list[float]:
        """Tracks to count at the moment of a query, bridging a one-frame
        active/coasting flicker (see presence_hold_sec)."""
        if self._latest_tracks:
            return self._latest_tracks
        if (time.time() - self._last_nonzero_time) <= self._presence_hold_sec:
            return self._last_nonzero_tracks
        return []

    # ── /voice_command subscriber ────────────────────────────────────────
    def _on_command(self, msg: String):
        command = msg.data.strip().lower()
        if command != "count_people":
            return  # other commands are handled by sensor_fusion_node

        tracks = self._effective_tracks()
        n = len(tracks)

        if n == 0:
            message = "I don't currently see anyone."
        elif n == 1:
            nearest = tracks[0]
            message = f"I see one person, {nearest:.1f} metres away."
        else:
            nearest = min(tracks)
            message = f"I see {n} people. The closest is {nearest:.1f} metres away."

        self.speech.say(message)
        self.get_logger().info(f"Spoke (one-shot): {message}")


def main():
    rclpy.init()
    node = PeopleCountNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
voice_command_node.py — spoken command recognition
Unitree G1 speaker attribution system

Listens to the G1's built-in microphone stream, runs a Vosk recogniser
restricted to a closed grammar, and publishes recognised commands on
/voice_command (std_msgs/String): "follow", "follow_closest",
"count_people" or "stop".

The microphone stream is UDP multicast, which allows several subscribers, so
this node can run alongside the robot's own voice service (g1-voice.service)
without interfering with it.

This audio path is already beam-formed to a single channel on the robot's
audio board, so it carries no direction. This node reports only that a
command was heard; which person gave it is decided downstream, by
speaker_attribution_node and sensor_fusion_node.

Asymmetric safety:
  - "follow", "closest" and "how many" are accepted only within
    wake_window_s seconds of a wake word ("Stuart" or "Eloise"), so motion
    is hard to start by accident.
  - "stop" is accepted on its own, at any time, with no wake word and no
    dependence on any state.

Audio stream (measured on the robot):
  multicast group   239.168.123.161:5555   (audio board 192.168.123.161)
  format            PCM16LE / 16 kHz / mono, beam-formed to one channel
  cadence           5120-byte datagrams, one every 160 ms

Usage (from ~/aidan/sensor_fusion, with the ROS 2 environment sourced):
    python3 voice_command_node.py --ros-args -p wake_window_s:=5.0 \\
        -p model_path:=/path/to/vosk-model-small-en-us-0.15

Verify:
    ros2 topic echo /voice_command
"""
from __future__ import annotations

import json
import socket
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import vosk

DEFAULT_GROUP = "239.168.123.161"
DEFAULT_PORT = 5555
DEFAULT_INTERFACE_IP = "192.168.123.164"
SAMPLE_RATE = 16000
DEFAULT_MODEL_PATH = (
    "/home/unitree/Projects/G1Butler/models/vosk-model-small-en-us-0.15"
)

WAKE_WORDS = {"stewart", "stuart", "eloise"}
GRAMMAR = json.dumps([
    "stewart", "stuart", "eloise", "follow", "closest", "me", "stop",
    "how", "many", "people", "see", "[unk]"
])


def join_multicast(group: str, port: int, interface_ip: str) -> socket.socket:
    """Join the microphone multicast group on the given interface. A plain
    function, so other scripts can open the same audio stream without
    starting rclpy."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port))
    membership = socket.inet_aton(group) + socket.inet_aton(interface_ip)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership)
    sock.setblocking(True)
    sock.settimeout(2.0)
    return sock


class VoiceCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("voice_command_node")

        self.declare_parameter("group", DEFAULT_GROUP)
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("interface_ip", DEFAULT_INTERFACE_IP)
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("wake_window_s", 5.0)

        self.group = self.get_parameter("group").value
        self.port = self.get_parameter("port").value
        self.interface_ip = self.get_parameter("interface_ip").value
        self.model_path = self.get_parameter("model_path").value
        self.wake_window_s = self.get_parameter("wake_window_s").value

        self.publisher = self.create_publisher(String, "/voice_command", 10)

        self._stop_event = threading.Event()
        self._awake_until = 0.0

        self.get_logger().info(
            "Loading vosk model from {}...".format(self.model_path)
        )
        self.model = vosk.Model(self.model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE, GRAMMAR)

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            "voice_command_node started. group={} port={} interface={} "
            "wake_window_s={}".format(
                self.group, self.port, self.interface_ip, self.wake_window_s
            )
        )

    # ------------------------------------------------------------- socket

    def _join_multicast(self) -> socket.socket:
        return join_multicast(self.group, self.port, self.interface_ip)

    # ---------------------------------------------------------- publish

    def _publish_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info(">>> COMMAND: {}".format(command.upper()))

    # ------------------------------------------------------- listen loop

    def _listen_loop(self) -> None:
        try:
            sock = self._join_multicast()
        except OSError as exc:
            self.get_logger().error("Failed to join multicast: {}".format(exc))
            return

        self.get_logger().info(
            "Joined {}:{} on {}".format(self.group, self.port, self.interface_ip)
        )

        try:
            while not self._stop_event.is_set():
                try:
                    data, _addr = sock.recvfrom(8192)
                except socket.timeout:
                    continue

                if not self.recognizer.AcceptWaveform(data):
                    continue  # still mid-utterance

                text = json.loads(self.recognizer.Result()).get("text", "")
                if not text:
                    continue

                words = set(text.split())
                self.get_logger().debug("[final] {!r}".format(text))

                # Stop is unconditional: no wake word, no wake-state check.
                if "stop" in words:
                    self._publish_command("stop")
                    self._awake_until = 0.0
                    continue

                if words & WAKE_WORDS:
                    self._awake_until = time.time() + self.wake_window_s
                    self.get_logger().info("Heard wake word")

                if time.time() < self._awake_until and "follow" in words:
                    self._publish_command("follow")
                    self._awake_until = 0.0
                elif time.time() < self._awake_until and "closest" in words:
                    self._publish_command("follow_closest")
                    self._awake_until = 0.0
                elif time.time() < self._awake_until and "many" in words:
                    self._publish_command("count_people")
                    self._awake_until = 0.0
        finally:
            sock.close()

    # -------------------------------------------------------- shutdown

    def destroy_node(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=3.0)
        super().destroy_node()


def main() -> None:
    rclpy.init()
    node = VoiceCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
audio_driver.py — text-to-speech bridge (Unitree SDK only, no rclpy)
Unitree G1 speaker attribution system

PURPOSE
-------
Receives short text strings over localhost UDP and speaks them through the
robot's built-in loudspeaker using the SDK's AudioClient. Like loco_driver.py
and led_driver.py, this process uses only the SDK, so it can run alongside the
rclpy nodes without the two DDS stacks sharing a process.

SDK FIX
-------
The SDK's AudioClient.TtsMaker() increments its request index with
    self.tts_index += self.tts_index
which starts at 0 and stays 0. The robot's voice service acknowledges
requests with a repeated index (code 0) but does not play them.
FixedAudioClient increments the index by 1 instead.

PORTS
-----
47923 = speech text (this driver), 47922 = LED state (led_driver.py),
47921 = yaw rate (loco_driver.py). Each driver needs its own port.

USAGE
-----
  python3 audio_driver.py --network-interface eth0
  python3 audio_driver.py --network-interface eth0 --dry-run   # log only

PARAMETERS
----------
  --network-interface : DDS network interface (default: eth0)
  --port              : UDP port to listen on (default: 47923)
  --speaker-id        : TtsMaker speaker_id (default: 1). IDs 1-7 give clear
                        English on this robot; 0 gives a different voice.
  --volume            : speaker volume set once at startup, 0-100 (default: 90)
  --dry-run           : log received text instead of speaking it
"""

import argparse
import json
import socket
import sys

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

ROBOT_API_ID_AUDIO_TTS = 1001  # from g1_audio_api.py


class FixedAudioClient(AudioClient):
    """AudioClient whose TtsMaker advances the request index on every call
    (see SDK FIX in the module docstring)."""

    def TtsMaker(self, text: str, speaker_id: int):
        self.tts_index += 1  # advance the request index on every call (see SDK FIX)
        p = {"index": self.tts_index, "text": text, "speaker_id": speaker_id}
        parameter = json.dumps(p)
        code, data = self._Call(ROBOT_API_ID_AUDIO_TTS, parameter)
        return code, self.tts_index


def main():
    parser = argparse.ArgumentParser(description="G1 TTS UDP bridge")
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--port", type=int, default=47923)
    parser.add_argument("--speaker-id", type=int, default=1)
    parser.add_argument("--volume", type=int, default=90)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audio_client = None
    if not args.dry_run:
        try:
            ChannelFactoryInitialize(0, args.network_interface)
            audio_client = FixedAudioClient()
            audio_client.SetTimeout(10.0)
            audio_client.Init()
            ret = audio_client.SetVolume(args.volume)
            print(f"[audio_driver] AudioClient connected via {args.network_interface}, "
                  f"volume set (code={ret})")
        except Exception as e:
            print(f"[audio_driver] AudioClient init failed: {e}")
            sys.exit(1)
    else:
        print("[audio_driver] DRY RUN -- will log text instead of speaking")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    print(f"[audio_driver] listening on 127.0.0.1:{args.port}")

    while True:
        data, _ = sock.recvfrom(4096)
        try:
            text = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            print(f"[audio_driver] dropped undecodable packet ({len(data)} bytes)")
            continue

        if not text:
            continue

        if args.dry_run:
            print(f"[audio_driver] [DRY-RUN] would speak: {text!r}")
            continue

        try:
            code, idx = audio_client.TtsMaker(text, args.speaker_id)
            print(f"[audio_driver] spoke (index={idx}, code={code}): {text!r}")
        except Exception as e:
            print(f"[audio_driver] TtsMaker() failed: {e}")


if __name__ == "__main__":
    main()

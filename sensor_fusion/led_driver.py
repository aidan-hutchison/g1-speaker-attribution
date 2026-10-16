#!/usr/bin/env python3
"""
led_driver.py — head-LED bridge (Unitree SDK only, no rclpy)
Unitree G1 speaker attribution system

PURPOSE
-------
Receives a one-byte tracking-state flag (0x00 = not following, 0x01 =
following) from sensor_fusion_node.py over a localhost UDP socket, and calls
AudioClient.LedControl() whenever that state changes.

WHY THIS IS A SEPARATE PROCESS
-------------------------------
As with loco_driver.py: rclpy and unitree_sdk2py both initialise CycloneDDS
in the process that imports them, and on this robot the two cannot reliably
share one process. sensor_fusion_node.py stays rclpy-only and sends its state
here over UDP; this process uses only the SDK.

BEHAVIOUR
---------
- following     -> LedControl(255, 0, 0)   red
- not following -> LedControl(0, 0, 255)   blue (idle)
- LedControl is called only when the state changes, not on every datagram,
  which keeps well clear of the SDK's 200 ms minimum interval between calls.
- Starts blue, before any datagram arrives.
- On Ctrl-C / SIGTERM, sets blue before exiting, so the LED is always left
  in the idle state.

USAGE
-----
  python3 led_driver.py --network-interface eth0

PARAMETERS
----------
  --network-interface : DDS network interface (default: eth0)
  --udp-host          : listen host (default: 127.0.0.1)
  --udp-port          : listen port (default: 47922)
  --dry-run           : log state changes; skip SDK initialisation and LED calls
"""

import argparse
import signal
import socket
import sys


def describe(code: int) -> str:
    known = {
        0: "OK",
        3102: "RPC_ERR_CLIENT_SEND (DDS write failed - is the voice service up?)",
        3103: "RPC_ERR_CLIENT_API_NOT_REG (AudioClient.Init() was not called)",
        3104: "RPC_ERR_CLIENT_API_TIMEOUT (no response from the voice service)",
    }
    return f"{code} ({known.get(code, 'unknown')})"


def main():
    ap = argparse.ArgumentParser(description="G1 head-LED UDP bridge")
    ap.add_argument("--network-interface", default="eth0")
    ap.add_argument("--udp-host", default="127.0.0.1")
    ap.add_argument("--udp-port", type=int, default=47922)
    ap.add_argument("--dry-run", action="store_true",
                     help="log received state changes; do not init SDK or call LedControl")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.udp_host, args.udp_port))
    print(f"[led_driver] listening for tracking-state on {args.udp_host}:{args.udp_port}")

    client = None
    if not args.dry_run:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
            ChannelFactoryInitialize(0, args.network_interface)
            client = AudioClient()
            client.SetTimeout(5.0)
            client.Init()
            print(f"[led_driver] AudioClient connected via {args.network_interface}")
        except Exception as e:
            print(f"[led_driver] SDK init failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("[led_driver] DRY-RUN: SDK not initialised — logging only, no LED calls")

    def _set_led(tracking: bool, reason: str = ""):
        rgb = (255, 0, 0) if tracking else (0, 0, 255)
        label = "RED (tracking)" if tracking else "BLUE (idle)"
        if args.dry_run:
            print(f"[led_driver][DRY-RUN] would set {label}  {reason}")
            return
        try:
            code = client.LedControl(*rgb)
            if code != 0:
                print(f"[led_driver] LedControl failed: {describe(code)}", file=sys.stderr)
            else:
                print(f"[led_driver] set {label}  {reason}")
        except Exception as e:
            print(f"[led_driver] LedControl raised: {e}", file=sys.stderr)

    running = {"v": True}

    def _stop(*_):
        running["v"] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Start in the known-safe idle state.
    last_state = False
    _set_led(last_state, reason="startup default")

    sock.settimeout(0.5)
    try:
        while running["v"]:
            try:
                data, _ = sock.recvfrom(8)
            except socket.timeout:
                continue
            except Exception:
                continue
            if not data:
                continue
            new_state = bool(data[0])
            if new_state != last_state:
                _set_led(new_state, reason="state change")
                last_state = new_state
    finally:
        _set_led(False, reason="shutdown")
        print("[led_driver] stopped")


if __name__ == "__main__":
    main()

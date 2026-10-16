#!/usr/bin/env python3
"""
loco_driver.py — locomotion bridge (Unitree SDK only, no rclpy)
Unitree G1 speaker attribution system

PURPOSE
-------
Receives yaw-rate commands from person_turn_node.py as little-endian float64
datagrams on a localhost UDP socket, and issues LocoClient.Move(0, 0, vyaw) at
a fixed rate.

This process does not import rclpy. rclpy and unitree_sdk2py both initialise
CycloneDDS, and on this robot the two cannot reliably share one process, so
the SDK runs here on its own and receives commands over UDP.

SAFETY
------
  * Watchdog: if no command arrives within --watchdog seconds, vyaw is forced
    to 0. If an upstream node dies or the socket goes quiet, the robot stops.
  * --max-vyaw clamps every command as an independent safety limit.
  * On Ctrl-C / SIGTERM, a final Move(0, 0, 0) is sent before exit.
  * --dry-run skips SDK initialisation and only logs, to test the UDP link
    end to end with no risk of motion.

USAGE
-----
  # Link test, no SDK, no motion:
  python3 loco_driver.py --dry-run

  # Live (robot standing, in motion-control mode with walking enabled):
  python3 loco_driver.py --network-interface eth0
"""

import argparse
import signal
import socket
import struct
import sys
import time


def main():
    ap = argparse.ArgumentParser(description='G1 LocoClient UDP bridge')
    ap.add_argument('--network-interface', default='eth0',
                    help='DDS NIC for the Unitree SDK (default: eth0)')
    ap.add_argument('--udp-host', default='127.0.0.1')
    ap.add_argument('--udp-port', type=int, default=47921)
    ap.add_argument('--rate', type=float, default=20.0,
                    help='Hz at which Move() is issued (default: 20)')
    ap.add_argument('--watchdog', type=float, default=0.3,
                    help='seconds without a command before forcing stop (default: 0.3)')
    ap.add_argument('--max-vyaw', type=float, default=0.8,
                    help='safety clamp on |vyaw| in rad/s (default: 0.8)')
    ap.add_argument('--loco-service-name', default='sport',
                    help='service the robot actually listens on; this G1 uses "sport" '
                         '(the installed SDK defaults LocoClient to "loco", which has no '
                         'subscriber on this robot)')
    ap.add_argument('--dry-run', action='store_true',
                    help='log received commands; do not init SDK or move')
    args = ap.parse_args()

    # ── UDP socket (inbound) ────────────────────────────────────────────
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.udp_host, args.udp_port))
    sock.setblocking(False)
    print(f'[loco_driver] listening for vyaw on {args.udp_host}:{args.udp_port}')

    # ── SDK init (skipped in dry-run) ───────────────────────────────────
    client = None
    if not args.dry_run:
        try:
            # This robot's locomotion service listens on /api/sport/*. The SDK
            # defaults LocoClient to /api/loco/*, which has no subscriber here,
            # so Move() would fail ("send request error"). Set the service name
            # before building the client so requests reach the live service.
            import unitree_sdk2py.g1.loco.g1_loco_client as _lc
            _lc.LOCO_SERVICE_NAME = args.loco_service_name
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize
            from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
            ChannelFactoryInitialize(0, args.network_interface)
            client = LocoClient()
            client.SetTimeout(5.0)
            client.Init()
            print(f'[loco_driver] LocoClient connected via {args.network_interface} '
                  f'(service: {args.loco_service_name})')
        except ImportError as e:
            print(f'[loco_driver] unitree_sdk2py not found: {e}', file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f'[loco_driver] SDK init failed: {e}', file=sys.stderr)
            sys.exit(1)
    else:
        print('[loco_driver] DRY-RUN: SDK not initialised — logging only, no motion')

    # ── Clean shutdown flag ─────────────────────────────────────────────
    running = {'v': True}

    def _stop(*_):
        running['v'] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    def _move(vyaw: float):
        if args.dry_run:
            if abs(vyaw) > 0.01:
                print(f'[loco_driver][DRY-RUN] Move(0, 0, {vyaw:+.3f})')
            return
        try:
            client.Move(0.0, 0.0, float(vyaw))
        except Exception as e:
            print(f'[loco_driver] Move() failed: {e}', file=sys.stderr)

    period      = 1.0 / args.rate
    last_vyaw   = 0.0
    last_recv   = 0.0   # wall time of last valid datagram (0 = none yet → stopped)
    stale_logged = False

    print(f'[loco_driver] running at {args.rate:.0f} Hz, watchdog {args.watchdog}s, '
          f'max_vyaw {args.max_vyaw} rad/s')

    try:
        while running['v']:
            t0 = time.time()

            # Drain the socket so we always act on the freshest command
            while True:
                try:
                    data, _ = sock.recvfrom(64)
                except BlockingIOError:
                    break
                except Exception:
                    break
                if len(data) >= 8:
                    last_vyaw = struct.unpack('<d', data[:8])[0]
                    last_recv = time.time()
                    stale_logged = False

            now = time.time()
            if (now - last_recv) > args.watchdog:
                vyaw = 0.0   # stale / no commands → stop
                if not stale_logged and last_recv > 0.0:
                    print('[loco_driver] watchdog: no command — stopping')
                    stale_logged = True
            else:
                vyaw = last_vyaw

            # Safety clamp
            vyaw = max(-args.max_vyaw, min(args.max_vyaw, vyaw))
            _move(vyaw)

            # Pace the loop
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        # Final stop before exit
        if client is not None:
            try:
                client.Move(0.0, 0.0, 0.0)
            except Exception:
                pass
        print('[loco_driver] stopped')


if __name__ == '__main__':
    main()

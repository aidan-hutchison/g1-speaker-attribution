#!/usr/bin/env python3
"""
odom_check.py — quick check of the G1's heading and odometry
Unitree G1 speaker attribution system

Prints the yaw from /odommodestate (unitree_go) and /lowstate (unitree_hg),
and the odometry x/y, all relative to their values at startup.

Test:
  1. Start this with the robot standing still.  -> all deltas about 0
  2. Turn the robot 90 deg to its left (CCW).    -> both yaw deltas about +90
  3. Turn back, then step it about 0.5 m forward. -> |dx, dy| about 0.5 m
  During a pure turn, watch how far x/y moves: the G1 does not pivot about a
  fixed point, and drifts by 0.1-0.25 m.

This checks the IMU's yaw direction only. The detector's heading_sign also
depends on how the LiDAR frame is oriented relative to the body (on the
evaluated robot it is mirrored, so heading_sign is -1). Confirm heading_sign
with the detector itself: with a person standing still, turn the robot by
hand; their track should keep its identity and position. If it runs away
from the person, flip the sign.

Needs the ROS 2 environment plus the unitree_ros2 message workspace sourced
(g1env in scripts/g1_shortcuts.sh does both).
"""
from __future__ import annotations

import math
import rclpy
from rclpy.node import Node
from unitree_go.msg import SportModeState
from unitree_hg.msg import LowState


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class OdomCheck(Node):
    def __init__(self):
        super().__init__('odom_check')
        self.odom0 = self.low0 = None
        self.odom = self.low = None
        self.create_subscription(SportModeState, '/odommodestate', self._on_odom, 10)
        self.create_subscription(LowState, '/lowstate', self._on_low, 10)
        self.create_timer(0.5, self._print)

    def _on_odom(self, m):
        v = (math.degrees(m.imu_state.rpy[2]), m.position[0], m.position[1])
        self.odom = v
        if self.odom0 is None:
            self.odom0 = v

    def _on_low(self, m):
        y = math.degrees(m.imu_state.rpy[2])
        self.low = y
        if self.low0 is None:
            self.low0 = y

    def _print(self):
        if self.odom is None:
            print('waiting for /odommodestate ...')
            return
        oy, ox, oyy = self.odom
        dyaw_o = wrap(oy - self.odom0[0])
        dx, dy = ox - self.odom0[1], oyy - self.odom0[2]
        low = f'{wrap(self.low - self.low0):+7.1f}' if self.low is not None else '   n/a'
        print(f'odom yaw Δ {dyaw_o:+7.1f}°   lowstate yaw Δ {low}°   '
              f'odom Δx {dx:+6.2f} m  Δy {dy:+6.2f} m  (|Δ| {math.hypot(dx, dy):.2f} m)')


def main():
    rclpy.init()
    n = OdomCheck()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

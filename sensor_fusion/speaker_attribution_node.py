#!/usr/bin/env python3
"""
speaker_attribution_node.py — who spoke: matches voice direction to a tracked person
Unitree G1 speaker attribution system

PURPOSE
-------
The LiDAR tracker says where people are; the microphone array's direction of
arrival (DOA) says which way a voice came from. This node combines the two to
decide who spoke, and holds that person_id for a fixed time, whether or not
they are the nearest person.

ROLE IN THE SYSTEM
------------------
This node does not decide when the robot moves. Starting needs a wake word
plus a command (voice_command_node.py), "stop" works at any time, and
sensor_fusion_node.py owns the IDLE/FOLLOWING state. This node only answers
"who most recently spoke". sensor_fusion_node reads that once, when a
"follow" command arrives (_nominate_by_doa, within doa_freshness_secs). A
wrong attribution makes the robot follow the wrong person, or nobody; it
never changes when the robot starts or stops.

BEHAVIOUR
---------
  1. Polls the array for DOAANGLE and VOICEACTIVITY at poll_hz.
  2. Once VOICEACTIVITY has been high for vad_min_active_polls consecutive
     polls, collects DOA samples for the rest of the utterance (see CAPTURE
     WINDOW) and takes their circular median.
  3. Converts that angle to a bearing in the tracker's convention and finds
     the tracked person whose LiDAR bearing is closest. If the closest is
     within doa_match_tolerance_deg, locks that person_id for
     lock_duration_secs. If not, nothing is locked: the utterance is logged
     and ignored, with no fallback to the nearest person.
  4. While locked, publishes the id on /doa_locked_person each time that id
     appears in a /detected_people update.
  5. When the lock expires, returns to IDLE and publishes nothing until the
     next utterance.

CAPTURE WINDOW
--------------
The array's first DOA samples in an utterance are unreliable: over the first
syllables of "Hey Stuart" the estimate can point 20-40 degrees away, and it
settles on the speaker by "follow me". A single sample is therefore not used. After the debounce, the
node keeps collecting samples while voice activity continues, for at most
capture_window_secs, and takes the circular median. A silence shorter than
capture_silence_grace_secs (such as the pause after "Stuart") does not end
the window; a longer silence does.

The array's estimate also carries over part of the previous sound's
direction, so the early samples lean toward wherever the last sound came
from. The median over the whole phrase reduces, but does not remove, that
lean.

SINGLE PROCESS
--------------
The array is read over USB control transfers, with no DDS involved, so DOA
polling and rclpy safely share this process (unlike the SDK drivers).

NO CACHED POSITIONS
-------------------
This node has no publish timer and caches nothing. Publishing is driven by
/detected_people updates: if the locked id is not in the current message,
nothing is published. Gaps are bridged by the detector's coasting and
occlusion hold, not by repeating an old value here.

STATE MACHINE
-------------
  IDLE    not publishing; waiting for a sustained voice-activity trigger.
  LOCKED  an utterance was matched to a person_id; each /detected_people
          update containing that id publishes it. Further utterances are
          ignored unless reacquire_during_lock is True.
  -> after lock_duration_secs: back to IDLE.

CALIBRATION
-----------
DOAANGLE's zero direction and sense of rotation depend on how the array is
mounted. To calibrate:
  1. Start person_detector_node and this node with mic_yaw_offset_deg 0,
     doa_direction_sign 1 and debug true.
  2. Stand straight ahead of the robot and speak. Set mic_yaw_offset_deg to
     the logged raw angle.
  3. Stand at about 90 degrees to one side and speak. Choose
     doa_direction_sign so that the logged DOA bearing has the same sign as
     your track's bearing; with the wrong sign the match is off by twice
     your bearing.
  4. Restart with both values and confirm that utterances from several
     directions are attributed to you.
doa_calibration_check.py runs a 12-mark version of this check. Walk the
marks in both directions: because of the carry-over described above, a
one-way pass reads the lean as a fixed offset. The evaluated mount used
mic_yaw_offset_deg 0.0 and doa_direction_sign -1.

USAGE
-----
  cd ~/aidan/sensor_fusion        # must run from here: imports tuning.py
  python3 tuning.py GAMMAVAD_SR 3.5   # restore the array's VAD threshold
  python3 speaker_attribution_node.py --ros-args \
      -p mic_yaw_offset_deg:=0.0 -p doa_direction_sign:=-1

Needs person_detector_node.py running and publishing /detected_people.

PARAMETERS (override with --ros-args -p key:=value)
----------
  lock_duration_secs         : how long an attribution stays locked, seconds
                               (default: 20.0)
  poll_hz                    : USB poll rate for DOAANGLE / VOICEACTIVITY
                               (default: 15.0)
  vad_min_active_polls       : consecutive VOICEACTIVITY=1 polls needed to
                               start a capture, so one noisy poll does not
                               trigger (default: 2)
  doa_match_tolerance_deg    : largest DOA-to-track bearing difference
                               accepted as a match (default: 30.0)
  mic_yaw_offset_deg         : raw DOAANGLE for a source straight ahead
                               (default: 0.0)
  doa_direction_sign         : +1 or -1, so DOA bearings follow the tracker's
                               convention (default: 1; evaluated: -1)
  reacquire_during_lock      : if True, an utterance during a lock
                               re-attributes and restarts the lock
                               (default: False)
  capture_window_secs        : longest time to collect DOA samples after
                               the trigger (default: 2.5)
  capture_silence_grace_secs : silence allowed inside the window before it
                               ends early (default: 0.5)
  tracks_topic               : person tracks (default: /detected_people)
  output_topic               : locked person_id, std_msgs/Int32
                               (default: /doa_locked_person)
  debug                      : verbose match/trigger/state logging
                               (default: True)
"""

from __future__ import annotations

import math
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from visualization_msgs.msg import MarkerArray

# tuning.py (ReSpeaker USB access) sits next to this file in ~/aidan/sensor_fusion/.
try:
    from tuning import find as find_mic
except ImportError:
    sys.exit(
        "Can't import tuning.py — run this from inside ~/aidan/sensor_fusion/ "
        "(tuning.py must be in the same folder)."
    )


def _unwrap_relative(angles: list[float], ref: float) -> list[float]:
    """Shift each angle by a multiple of 360 so it lies within [-180, 180)
    of ref, so an ordinary median works even when the samples straddle the
    0/360 boundary."""
    out = []
    for a in angles:
        d = (a - ref + 180.0) % 360.0 - 180.0
        out.append(ref + d)
    return out


def _circular_median(angles: list[float]) -> float:
    """Median of a cluster of angular samples (degrees), robust to the
    0/360 wrap. Assumes the samples lie within 180 degrees of the first one,
    which holds for one utterance's samples. A few outlying samples (such as
    an unreliable onset) do not move it."""
    ref = angles[0]
    unwrapped = sorted(_unwrap_relative(angles, ref))
    n = len(unwrapped)
    if n % 2 == 1:
        med = unwrapped[n // 2]
    else:
        med = (unwrapped[n // 2 - 1] + unwrapped[n // 2]) / 2.0
    return med % 360.0


class _DoaVadPoller:
    """Background thread polling DOAANGLE and VOICEACTIVITY over the
    ReSpeaker's USB control interface, passing each pair to on_sample."""

    def __init__(self, mic, poll_hz: float, on_sample):
        self._mic = mic
        self._interval = 1.0 / poll_hz
        self._on_sample = on_sample
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                angle = self._mic.direction
                voice = self._mic.read('VOICEACTIVITY')
            except Exception as e:
                # USB read error: report it, skip this sample, keep polling.
                self._on_sample(None, None, error=e)
                time.sleep(self._interval)
                continue
            self._on_sample(angle, voice, error=None)
            time.sleep(self._interval)


class SpeakerAttributionNode(Node):

    STATE_IDLE   = 'IDLE'
    STATE_LOCKED = 'LOCKED'

    def __init__(self):
        super().__init__('speaker_attribution_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('lock_duration_secs',        20.0)
        self.declare_parameter('poll_hz',                    15.0)
        self.declare_parameter('vad_min_active_polls',          2)
        self.declare_parameter('doa_match_tolerance_deg',    30.0)
        self.declare_parameter('mic_yaw_offset_deg',           0.0)
        self.declare_parameter('doa_direction_sign',            1)
        self.declare_parameter('reacquire_during_lock',     False)
        self.declare_parameter('capture_window_secs',         2.5)
        self.declare_parameter('capture_silence_grace_secs',  0.5)
        self.declare_parameter('tracks_topic',    '/detected_people')
        self.declare_parameter('output_topic',  '/doa_locked_person')
        self.declare_parameter('debug',                      True)

        p = self.get_parameter
        self._lock_duration    = float(p('lock_duration_secs').value)
        self._poll_hz            = float(p('poll_hz').value)
        self._vad_min_polls       = int(p('vad_min_active_polls').value)
        self._tolerance_deg       = float(p('doa_match_tolerance_deg').value)
        self._offset_deg          = float(p('mic_yaw_offset_deg').value)
        self._sign                = 1 if float(p('doa_direction_sign').value) >= 0 else -1
        self._reacquire           = bool(p('reacquire_during_lock').value)
        self._capture_window_secs = float(p('capture_window_secs').value)
        self._capture_silence_grace_secs = float(p('capture_silence_grace_secs').value)
        self._tracks_topic        = p('tracks_topic').value
        self._output_topic        = p('output_topic').value
        self._debug               = bool(p('debug').value)

        # ── Track state (from /detected_people) ─────────────────────────
        self._tracks_lock = threading.Lock()
        self._tracks: dict[int, tuple[float, float]] = {}   # id -> (x, y)

        # ── Attribution / lock state ─────────────────────────────────────
        self._state_lock = threading.Lock()
        self._state = self.STATE_IDLE
        self._locked_id: int | None = None
        self._lock_until: float = 0.0

        # ── VAD debounce state (poller thread only) ──────────────────────
        self._consecutive_voice_polls = 0
        self._pending_angle: float | None = None   # latest raw angle while VAD is high

        # ── Capture-window state (poller thread only; see CAPTURE WINDOW) ─
        self._capturing = False
        self._capture_samples: list[float] = []
        self._capture_deadline: float = 0.0
        self._capture_silence_since: float | None = None

        # ── Pub/Sub ──────────────────────────────────────────────────────
        self._pub = self.create_publisher(Int32, self._output_topic, 10)
        self.create_subscription(MarkerArray, self._tracks_topic, self._on_tracks, 10)
        self.create_timer(0.1, self._check_lock_expiry)   # 10Hz, state/logging only

        # ── Mic DOA+VAD poller ─────────────────────────────────────────────
        mic = find_mic()
        if mic is None:
            self.get_logger().error(
                'ReSpeaker (2886:0018) not found. Check `lsusb | grep 2886` '
                'and udev permissions before retrying.'
            )
            sys.exit(1)

        self._mic = mic
        self._poller = _DoaVadPoller(self._mic, self._poll_hz, self._on_sample)
        self._poller.start()

        self.get_logger().info('Speaker attribution node started.')
        self.get_logger().info(
            f'  lock_duration_secs={self._lock_duration}  '
            f'doa_match_tolerance_deg={self._tolerance_deg}  '
            f'reacquire_during_lock={self._reacquire}'
        )
        self.get_logger().info(
            f'  mic_yaw_offset_deg={self._offset_deg}  doa_direction_sign={self._sign}  '
            f'{"(UNCALIBRATED — see module docstring)" if self._offset_deg == 0.0 and self._sign == 1 else ""}'
        )
        self.get_logger().info(f'  tracks from {self._tracks_topic}, publishing locked id to {self._output_topic}')

    # ── /detected_people subscriber ─────────────────────────────────────────
    def _on_tracks(self, msg: MarkerArray):
        """Rebuild the track table from each message. person_detector_node
        sends DELETEALL plus fresh markers every frame, so a dropped track
        simply disappears from it."""
        tracks = {}
        for m in msg.markers:
            if m.ns == 'people':
                tracks[m.id] = (m.pose.position.x, m.pose.position.y)
        with self._tracks_lock:
            self._tracks = tracks

        # Publish the locked id only if it is in this update (see NO CACHED
        # POSITIONS in the module docstring).
        with self._state_lock:
            locked = (self._state == self.STATE_LOCKED and time.time() < self._lock_until)
            locked_id = self._locked_id

        if not locked:
            return

        if locked_id not in tracks:
            return

        out = Int32()
        out.data = int(locked_id)
        self._pub.publish(out)

        if self._debug:
            xy = tracks[locked_id]
            self.get_logger().debug(f'Published locked id P{locked_id} pos=({xy[0]:.2f}, {xy[1]:.2f})')

    # ── Mic poller callback (runs on the poller thread) ──────────────────────
    def _on_sample(self, raw_angle, voice, error=None):
        if error is not None:
            self.get_logger().warn(f'USB read failed: {error}')
            return

        if self._capturing:
            now = time.time()
            if voice:
                self._capture_samples.append(raw_angle)
                self._capture_silence_since = None
            else:
                # A short silence (e.g. the pause after "Stuart") does not end
                # the window; only silence longer than
                # capture_silence_grace_secs does.
                if self._capture_silence_since is None:
                    self._capture_silence_since = now
                elif now - self._capture_silence_since >= self._capture_silence_grace_secs:
                    self._finish_capture()
                    return
            if now >= self._capture_deadline:
                # Hard cap, so continuous speech (or a stuck VAD) cannot hold
                # the window open.
                self._finish_capture()
            return

        if not voice:
            self._consecutive_voice_polls = 0
            self._pending_angle = None
            return

        self._pending_angle = raw_angle
        self._consecutive_voice_polls += 1
        if self._consecutive_voice_polls < self._vad_min_polls:
            return  # not sustained yet: ignore single-poll blips

        # Debounce satisfied: start a capture window rather than using this
        # single, possibly unreliable, sample. If already locked and not
        # reacquiring, the result would be ignored, so skip the capture.
        with self._state_lock:
            already_locked = (self._state == self.STATE_LOCKED)
        self._consecutive_voice_polls = 0
        if already_locked and not self._reacquire:
            self._pending_angle = None
            return

        self._capturing = True
        self._capture_samples = [self._pending_angle]
        self._capture_silence_since = None
        self._capture_deadline = time.time() + self._capture_window_secs
        self._pending_angle = None

    def _finish_capture(self):
        samples, self._capture_samples = self._capture_samples, []
        self._capturing = False
        self._capture_silence_since = None
        if not samples:
            return
        angle = _circular_median(samples)
        self._on_trigger(angle)

    def _to_bearing(self, raw_angle_deg: float) -> float:
        """Raw DOAANGLE (degrees) -> bearing in radians in [-pi, pi], with
        0 = straight ahead, in the same convention as the track bearings
        atan2(y, x) once calibrated (see CALIBRATION)."""
        adjusted = self._sign * (raw_angle_deg - self._offset_deg)
        rad = math.radians(adjusted)
        return math.atan2(math.sin(rad), math.cos(rad))

    def _on_trigger(self, raw_angle: float):
        with self._state_lock:
            already_locked = (self._state == self.STATE_LOCKED)
            if already_locked and not self._reacquire:
                return  # mid-lock, ignoring per reacquire_during_lock=False

        with self._tracks_lock:
            tracks_snapshot = dict(self._tracks)

        if not tracks_snapshot:
            if self._debug:
                self.get_logger().info('Sound heard, but no tracked people to attribute it to.')
            return

        doa_bearing = self._to_bearing(raw_angle)

        best_id, best_diff = None, float('inf')
        for pid, (x, y) in tracks_snapshot.items():
            track_bearing = math.atan2(y, x)
            diff = abs(math.atan2(math.sin(track_bearing - doa_bearing),
                                   math.cos(track_bearing - doa_bearing)))
            if diff < best_diff:
                best_id, best_diff = pid, diff

        best_diff_deg = math.degrees(best_diff)
        if best_id is None or best_diff_deg >= self._tolerance_deg:
            if self._debug:
                self.get_logger().info(
                    f'Sound at raw={raw_angle:.0f}° (bearing={math.degrees(doa_bearing):+.1f}°) '
                    f'matched no track within {self._tolerance_deg}° '
                    f'(closest was P{best_id} at {best_diff_deg:.1f}°)' if best_id is not None else
                    f'Sound at raw={raw_angle:.0f}° matched no track — none tracked.'
                )
            return  # nobody to attribute this to — stay idle, no fallback

        with self._state_lock:
            self._state       = self.STATE_LOCKED
            self._locked_id     = best_id
            self._lock_until    = time.time() + self._lock_duration

        verb = 'Re-attributed' if already_locked else 'Sound attributed'
        self.get_logger().info(
            f'{verb} to P{best_id} (Δ={best_diff_deg:.1f}° from DOA bearing '
            f'{math.degrees(doa_bearing):+.1f}°) — locking for {self._lock_duration:.1f}s'
        )

    # ── Lock-expiry watchdog (state transition + logging only) ──────────────
    def _check_lock_expiry(self):
        now = time.time()
        with self._state_lock:
            if self._state != self.STATE_LOCKED:
                return
            if now < self._lock_until:
                return
            expired_id = self._locked_id
            self._state      = self.STATE_IDLE
            self._locked_id   = None

        self.get_logger().info(
            f'Lock window expired for P{expired_id} — reverting to idle, '
            'waiting for next sound.'
        )

    def destroy_node(self):
        self._poller.stop()
        try:
            self._mic.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SpeakerAttributionNode()
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

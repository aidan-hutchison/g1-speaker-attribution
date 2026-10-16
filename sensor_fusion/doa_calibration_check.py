#!/usr/bin/env python3
"""
doa_calibration_check.py — 12-mark check of the microphone array's DOA calibration
Unitree G1 speaker attribution system

PURPOSE
-------
Walks through 12 floor marks at 30 degree spacing around the robot and
prints, for each, the bearing the calibrated attribution node would assign
to your voice next to the mark you are standing on. A wrong offset, a wrong
sign or a misplaced mark shows up immediately.

It uses the attribution node's own code, imported rather than copied:
  - find_mic()          ReSpeaker discovery (tuning.py)
  - _to_bearing()       the node's calibration formula, bound to a small
                        stand-in object carrying only _sign and _offset_deg
  - _circular_median()  the node's way of turning one utterance's samples
                        into one angle
The trigger debounce (vad_min_active_polls consecutive voice polls) is
re-implemented here in a few lines because, in the node, it is tied to ROS
callback state; it follows the node's logic and defaults.

This script uses the microphone array only. It needs no ROS environment and
no LiDAR tracks.

CAPTURE
-------
As in the node: after the debounce, samples are collected while voice
activity continues, for at most --capture-window-secs, and a silence shorter
than --silence-grace-secs (such as the pause after "Stuart") does not end the
window. The reported angle is the circular median of all collected samples.
Speak the full phrase "Hey Stuart, follow me", including the natural pause.

BEARING CONVENTION
------------------
Marks run clockwise from straight ahead, so 90 is the robot's right side
(checked physically). Each result is converted to the mark scale as

    mark_scale_deg = corrected_bearing_deg % 360

so with a correct calibration, mark_scale_deg matches the mark you stand on.

MEASUREMENT CAVEAT
------------------
The array's estimate carries over part of the previous sound's direction.
This script always visits the marks clockwise, so every reading leans the
same way, by roughly 5-7 degrees, toward the previous mark. A one-way run can
therefore show that lean as an apparent offset. Before changing
mic_yaw_offset_deg, confirm the offset with readings taken in the opposite
direction as well.

USAGE
-----
  cd ~/aidan/sensor_fusion     # next to tuning.py and speaker_attribution_node.py
  python3 doa_calibration_check.py --offset 0.0 --sign -1 --distance-m 1.0

Pass the same --offset and --sign as speaker_attribution_node's
mic_yaw_offset_deg and doa_direction_sign (evaluated mount: 0.0 and -1).

At each prompt: stand on the mark, press Enter, then speak. A live line shows
voice=0/1, the debounce streak and the raw angle. If voice never reaches 1,
the array is not hearing you at that position; if it flickers 1/0/1, speech
is dropping out between polls. During capture, a "(pause, Xs / 0.5s grace)"
line after "Stuart" is expected. Each result is printed and written to CSV;
Ctrl+C stops early and saves what has been collected.

Extra flags:
  --vad-min-active-polls N   trigger debounce (default matches the node;
                             change only to diagnose)
  --pre-speech-delay S       pause after Enter before polling (default: 0.4)
  --vad-threshold-db DB      set GAMMAVAD_SR before starting (lower = more
                             sensitive; the correct value is 3.5)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

try:
    from tuning import find as find_mic
except ImportError:
    sys.exit(
        "Can't import tuning.py — run this from inside ~/aidan/sensor_fusion/ "
        "(same requirement as speaker_attribution_node.py)."
    )

try:
    from speaker_attribution_node import _DoaVadPoller, SpeakerAttributionNode, _circular_median
except ImportError:
    sys.exit(
        "Can't import speaker_attribution_node.py — run this from inside "
        "~/aidan/sensor_fusion/, next to that file."
    )

MARKS_DEG = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
VAD_MIN_ACTIVE_POLLS = 2   # mirrors SpeakerAttributionNode's declared default
POLL_HZ = 15.0             # mirrors SpeakerAttributionNode's declared default


class _CalibrationOnly:
    """Stand-in carrying only what _to_bearing() reads from self, so the
    node's method can be called without rclpy or a node instance."""

    def __init__(self, sign: int, offset_deg: float):
        self._sign = sign
        self._offset_deg = offset_deg


def build_to_bearing(sign: int, offset_deg: float):
    stub = _CalibrationOnly(sign, offset_deg)
    return SpeakerAttributionNode._to_bearing.__get__(stub)


def capture_stable_angle(mic, timeout_s: float | None = 15.0,
                          vad_min_active_polls: int = VAD_MIN_ACTIVE_POLLS,
                          capture_window_s: float = 2.5,
                          silence_grace_s: float = 0.5,
                          verbose: bool = True):
    """Wait for a debounced voice trigger, then keep collecting raw angles
    while voice activity continues, for at most capture_window_s, allowing
    silences shorter than silence_grace_s. Returns (circular median of all
    collected angles, the samples), or (None, []) if no trigger arrives
    within timeout_s. timeout_s=None waits indefinitely.

    The median uses _circular_median imported from
    speaker_attribution_node.py, so this check and the node turn samples
    into an angle the same way."""
    interval = 1.0 / POLL_HZ
    consecutive = 0
    collected = []
    deadline = None if timeout_s is None else time.time() + timeout_s

    # Phase 1: wait for the debounce trigger, keeping every sample seen
    # once voice goes high.
    while deadline is None or time.time() < deadline:
        try:
            angle = mic.direction
            voice = mic.read('VOICEACTIVITY')
        except Exception as e:
            print(f'\n  [USB read error, retrying: {e}]')
            time.sleep(interval)
            continue

        if not voice:
            consecutive = 0
            collected = []
            if verbose:
                sys.stdout.write(f'\r  listening... voice=0  streak=0   angle=  --  ')
                sys.stdout.flush()
            time.sleep(interval)
            continue

        consecutive += 1
        collected.append(angle)
        if verbose:
            sys.stdout.write(f'\r  listening... voice=1  streak={consecutive}   angle={angle:6.1f}')
            sys.stdout.flush()
        if consecutive >= vad_min_active_polls:
            break
        time.sleep(interval)
    else:
        if verbose:
            print()
        return None, []

    # Phase 2: keep collecting while voice continues. Silences shorter than
    # silence_grace_s are allowed; capture_window_s is a hard cap.
    capture_deadline = time.time() + capture_window_s
    silence_since = None
    while time.time() < capture_deadline:
        try:
            angle = mic.direction
            voice = mic.read('VOICEACTIVITY')
        except Exception as e:
            if verbose:
                print(f'\n  [USB read error during capture, stopping capture: {e}]')
            break

        if voice:
            collected.append(angle)
            silence_since = None
            if verbose:
                sys.stdout.write(f'\r  capturing... n={len(collected)}   angle={angle:6.1f}')
                sys.stdout.flush()
        else:
            now = time.time()
            if silence_since is None:
                silence_since = now
            elif now - silence_since >= silence_grace_s:
                break  # end of utterance: silence outlasted the grace period
            if verbose:
                gap = now - silence_since
                sys.stdout.write(f'\r  (pause, {gap:.1f}s / {silence_grace_s:.1f}s grace) '
                                  f'n={len(collected)} so far  ')
                sys.stdout.flush()
        time.sleep(interval)

    if verbose:
        print()

    median_angle = _circular_median(collected)
    return median_angle, collected


def wait_for_trigger(mic, timeout_s: float = 15.0, vad_min_active_polls: int = VAD_MIN_ACTIVE_POLLS,
                      verbose: bool = True):
    """Single-sample trigger: poll until vad_min_active_polls consecutive
    VOICEACTIVITY=1 reads, and return the raw angle of the sample that
    completed the debounce (None on timeout). Not used by main(), which uses
    capture_stable_angle(); kept to compare against a single-sample reading.
    With verbose=True, prints the voice flag, streak and raw angle each poll."""
    consecutive = 0
    pending_angle = None
    interval = 1.0 / POLL_HZ
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        try:
            angle = mic.direction
            voice = mic.read('VOICEACTIVITY')
        except Exception as e:
            print(f'\n  [USB read error, retrying: {e}]')
            time.sleep(interval)
            continue

        if not voice:
            consecutive = 0
            pending_angle = None
            if verbose:
                sys.stdout.write(f'\r  listening... voice=0  streak=0   angle=  --  ')
                sys.stdout.flush()
            time.sleep(interval)
            continue

        pending_angle = angle
        consecutive += 1
        if verbose:
            sys.stdout.write(f'\r  listening... voice=1  streak={consecutive}   angle={angle:6.1f}')
            sys.stdout.flush()
        if consecutive >= vad_min_active_polls:
            if verbose:
                print()  # move off the carriage-return line
            return pending_angle
        time.sleep(interval)

    if verbose:
        print()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offset', type=float, default=11.9,
                     help='mic_yaw_offset_deg to test (default: 11.9; pass the value '
                          'speaker_attribution_node uses, 0.0 for the evaluated mount)')
    ap.add_argument('--sign', type=int, default=-1, choices=(1, -1),
                     help='doa_direction_sign to test (default: -1, the evaluated mount)')
    ap.add_argument('--distance-m', type=float, default=1.0,
                     help='radius of the ring you are testing at, in metres — recorded '
                          'in the CSV for reference only, no effect on the DOA math '
                          '(default: 1.0)')
    ap.add_argument('--out', type=str, default=None,
                     help='CSV output path (default: doa_calibration_check_<distance>m.csv)')
    ap.add_argument('--vad-min-active-polls', type=int, default=VAD_MIN_ACTIVE_POLLS,
                     help=f'consecutive VOICEACTIVITY=1 polls required to trigger '
                          f'(default: {VAD_MIN_ACTIVE_POLLS}, as in the node; change only '
                          f'to diagnose unreliable triggering)')
    ap.add_argument('--pre-speech-delay', type=float, default=0.4,
                     help='seconds to pause after pressing Enter before polling starts, '
                          'so you have a moment to actually begin speaking (default: 0.4)')
    ap.add_argument('--capture-window-secs', type=float, default=2.5,
                     help='after the trigger, collect samples for up to this long and use '
                          'their median, so unreliable onset samples do not decide the '
                          'angle (default: 2.5, as in the node)')
    ap.add_argument('--silence-grace-secs', type=float, default=0.5,
                     help='silence allowed inside the capture window before it ends, '
                          'for natural pauses such as after "Stuart" (default: 0.5, as in '
                          'the node)')
    ap.add_argument('--vad-threshold-db', type=float, default=None,
                     help='set GAMMAVAD_SR (the array\'s voice-activity threshold, dB) '
                          'before starting; lower = more sensitive. The correct value is 3.5; '
                          'it has been found at 15 after power cycles, which almost disables '
                          'voice detection. Not changed unless given.')
    args = ap.parse_args()
    if args.out is None:
        args.out = f'doa_calibration_check_{args.distance_m:g}m.csv'

    to_bearing = build_to_bearing(args.sign, args.offset)

    print(f'Using mic_yaw_offset_deg={args.offset}  doa_direction_sign={args.sign}  '
          f'distance={args.distance_m:g}m')
    mic = find_mic()
    if mic is None:
        sys.exit('ReSpeaker (2886:0018) not found. Check `lsusb | grep 2886` and udev permissions.')

    if args.vad_threshold_db is not None:
        try:
            current = mic.read('GAMMAVAD_SR')
            print(f'GAMMAVAD_SR was {current}dB — setting to {args.vad_threshold_db}dB '
                  f'(lower = more sensitive)')
            mic.set_vad_threshold(args.vad_threshold_db)
            confirmed = mic.read('GAMMAVAD_SR')
            print(f'GAMMAVAD_SR now reads {confirmed}dB')
        except Exception as e:
            print(f'[WARNING] Could not set VAD threshold: {e}')

    out_path = Path(args.out)
    rows = []

    print(f'\n{len(MARKS_DEG)} marks, clockwise from front '
          f'(= robot\'s right side). Ctrl+C to stop early.\n')

    try:
        for expected_mark in MARKS_DEG:
            input(f'--- Stand at {expected_mark}° mark ({args.distance_m:g}m, robot\'s right '
                  f'of front, clockwise), then press Enter and speak the FULL phrase '
                  f'"Hey Stuart, follow me" naturally, including the pause after "Stuart" ---')
            time.sleep(args.pre_speech_delay)
            raw_angle, samples = capture_stable_angle(
                mic, vad_min_active_polls=args.vad_min_active_polls,
                capture_window_s=args.capture_window_secs,
                silence_grace_s=args.silence_grace_secs)

            if raw_angle is None:
                print('  [TIMEOUT — no sustained voice detected, skipping]')
                rows.append({
                    'distance_m': args.distance_m,
                    'expected_mark_deg': expected_mark,
                    'raw_angle_deg': '',
                    'n_samples': 0,
                    'raw_samples': '',
                    'corrected_bearing_deg': '',
                    'corrected_mark_deg': '',
                    'error_deg': '',
                    'timestamp': time.time(),
                })
                continue

            corrected_bearing_rad = to_bearing(raw_angle)
            corrected_bearing_deg = _rad_to_deg_signed(corrected_bearing_rad)
            corrected_mark = corrected_bearing_deg % 360.0

            error = _wrap_180(corrected_mark - expected_mark)

            print(f'  median_raw={raw_angle:.1f}° (n={len(samples)} samples)  '
                  f'corrected_bearing={corrected_bearing_deg:+.1f}° '
                  f'(native mic calibration output)  mark_scale={corrected_mark:.1f}°  '
                  f'error={error:+.1f}°  [expected {expected_mark}°]')

            rows.append({
                'distance_m': args.distance_m,
                'expected_mark_deg': expected_mark,
                'raw_angle_deg': round(raw_angle, 2),
                'n_samples': len(samples),
                'raw_samples': ';'.join(f'{s:.1f}' for s in samples),
                'corrected_bearing_deg': round(corrected_bearing_deg, 2),
                'corrected_mark_deg': round(corrected_mark, 2),
                'error_deg': round(error, 2),
                'timestamp': time.time(),
            })

    except KeyboardInterrupt:
        print('\nStopped early — saving what was collected so far.')

    finally:
        with out_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'distance_m', 'expected_mark_deg', 'raw_angle_deg', 'n_samples', 'raw_samples',
                'corrected_bearing_deg',
                'corrected_mark_deg', 'error_deg', 'timestamp',
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nSaved {len(rows)} rows to {out_path.resolve()}')

        try:
            mic.close()
        except Exception:
            pass


def _rad_to_deg_signed(rad: float) -> float:
    import math
    return math.degrees(rad)


def _wrap_180(deg: float) -> float:
    """Wrap an angular difference to [-180, 180]."""
    return (deg + 180.0) % 360.0 - 180.0


if __name__ == '__main__':
    main()

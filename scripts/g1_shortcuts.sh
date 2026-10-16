# g1_shortcuts.sh — one-word launchers for the G1 speaker-following stack.
# Install on the ROBOT (once), with this repository cloned to ~/aidan:
#   echo 'source ~/aidan/scripts/g1_shortcuts.sh' >> ~/.bashrc
# Then in any robot terminal:  t1 … t12   (g1help lists them)
#
# Launch arguments below are the evaluated configuration (see README).
# Override the paths with environment variables if your robot differs.

G1_DIR=${G1_DIR:-~/aidan/sensor_fusion}
G1_BAGS=$G1_DIR/bags
# colcon workspace of unitree_ros2 that provides the unitree_hg / unitree_go messages
UNITREE_MSGS_WS=${UNITREE_MSGS_WS:-~/unitree_ros2_TESTCLONE/cyclonedds_ws}
# Vosk model directory (vosk-model-small-en-us-0.15)
VOSK_MODEL=${VOSK_MODEL:-/home/unitree/Projects/G1Butler/models/vosk-model-small-en-us-0.15}

g1env() {                     # ROS 2 environment + working directory
    conda deactivate 2>/dev/null || true
    source /opt/ros/foxy/setup.bash
    source /home/unitree/cyclonedds_ws/install/setup.bash
    source "$UNITREE_MSGS_WS/install/setup.bash"
    # Same DDS settings as the robot's own ROS 2 Foxy login environment
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=~/cyclonedds_ws/cyclonedds.xml
    case ":$LD_LIBRARY_PATH:" in *":/usr/local/lib:"*) ;; *) export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH;; esac
    cd "$G1_DIR" || return 1
}
g1plain() {                   # no ROS env (LED / audio drivers)
    conda deactivate 2>/dev/null || true
    cd "$G1_DIR" || return 1
}
_g1title() { echo -ne "\033]0;$1\007"; echo "=== $1 ==="; }

# ── Pre-flight ───────────────────────────────────────────────────────────────
g1pre() {
    _g1title "pre-flight"
    sudo systemctl stop g1-voice.service
    systemctl is-active g1-voice.service >/dev/null && echo "!! g1-voice still active" || echo "g1-voice stopped (mic free)"
    df -h ~ | tail -1 | awk '{print "disk free: " $4}'
}

# ── The stack ───────────────────────────────────────────────────────────────
t1()  { _g1title "T1 detector";   g1env && python3 person_detector_node.py --ros-args \
          -p max_range:=2.1 -p range_edge_margin:=0.1 -p background_filter_enabled:=false \
          -p coast_secs:=0.5 -p heading_source:=lowstate -p heading_sign:=-1.0; }
t2()  { _g1title "T2 attribution"; g1env && \
        { python3 tuning.py GAMMAVAD_SR 3.5 || echo "!! could not reset GAMMAVAD_SR (array not found?)"; } && \
        python3 speaker_attribution_node.py --ros-args \
          -p mic_yaw_offset_deg:=0.0 -p doa_direction_sign:=-1; }
t3()  { _g1title "T3 voice";      g1env && python3 voice_command_node.py --ros-args \
          -p wake_window_s:=5.0 -p model_path:="$VOSK_MODEL"; }
t4()  { _g1title "T4 fusion";     g1env && python3 sensor_fusion_node.py; }
t5()  { _g1title "T5 turn";       g1env && python3 person_turn_node.py --ros-args \
          -p input_topic:=/turn_target -p Kp:=2.0 -p lost_timeout:=0.5 -p search_timeout:=0.0; }
t6()  { _g1title "T6 loco"
        read -r -p "Robot standing with walking enabled (L2+B, L2+UP, R2+A, START)? [y/N] " a
        [ "$a" = "y" ] || { echo "Not started."; return 1; }
        g1env && python3 loco_driver.py --network-interface eth0; }
t7()  { _g1title "T7 LED";        g1plain && python3 led_driver.py --network-interface eth0; }
t8()  { _g1title "T8 audio";      g1plain && python3 audio_driver.py --network-interface eth0; }
t9()  { _g1title "T9 count";      g1env && python3 people_count_node.py; }
t10() { _g1title "T10 viewer";    g1env && echo "Open http://<robot-ip>:8090" && python3 lidar_web_viewer.py; }

# ── Recording ───────────────────────────────────────────────────────────────
G1_TOPICS="/detected_people /turn_target /doa_locked_person /voice_command /nearest_person \
/person_detector/debug_json /odommodestate /trial_marker"
G1_RAW="/utlidar/cloud_livox_mid360 /lowstate"

t11() {      # t11 <bag name>        standard bag (small)
    [ -n "$1" ] || { echo "usage: t11 <bag_name>   e.g. t11 follow_test_1"; return 1; }
    _g1title "T11 bag $1"; g1env && mkdir -p "$G1_BAGS" && cd "$G1_BAGS" && \
    ros2 bag record -o "$1" $G1_TOPICS
}
t11raw() {   # t11raw <bag name>     + raw LiDAR and IMU (large, ~0.5 GB/min) for offline replay
    [ -n "$1" ] || { echo "usage: t11raw <bag_name>"; return 1; }
    _g1title "T11 RAW bag $1"; g1env && mkdir -p "$G1_BAGS" && cd "$G1_BAGS" && \
    ros2 bag record -o "$1" $G1_TOPICS $G1_RAW
}
t12() {      # marker terminal
    _g1title "T12 markers"
    echo 'Type:  mark <text>     e.g.  mark trial 3 start'
}
mark() {   # works from any shell (sources the ROS env in a subshell)
    ( g1env >/dev/null && ros2 topic pub -1 /trial_marker std_msgs/msg/String "{data: '$*'}" >/dev/null \
      && echo "marked: $*  ($(date +%H:%M:%S))" )
}

# ── Checks ──────────────────────────────────────────────────────────────────
g1check() {  # run once everything is up
    g1env >/dev/null
    echo "--- nodes";  ros2 node list
    echo "--- detector params"
    ros2 param list 2>/dev/null | grep -E "heading_sign|heading_source|background_filter_enabled|range_edge_margin|max_range|coast_secs|occlusion_hold_enabled|ground_min_normal_z"
    echo "(if heading_sign / occlusion_hold_enabled / ground_min_normal_z are missing, the detector running is not this version)"
}

g1help() {
    cat <<'EOF'
g1pre      stop g1-voice (mic), show disk space
t1..t10    detector, attribution, voice, fusion, turn, loco, LED, audio, count, viewer
t11 NAME   start standard bag          t11raw NAME   bag incl. raw LiDAR + IMU
t12        marker shell  ->  mark <text>
g1check    list nodes + key detector parameters
Launch order: g1pre, t1, t2, t3, t4, (check logs), t5, stand-up, t6, t7, t8, t9, t10, g1check, t12, t11
EOF
}

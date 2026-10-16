# g1_laptop.sh — laptop-side shortcuts. Install (once, on the laptop):
#   cp g1_laptop.sh ~/g1_laptop.sh
#   echo 'export G1_HOST=unitree@<robot-ip>' >> ~/.bashrc
#   echo 'source ~/g1_laptop.sh' >> ~/.bashrc
#   ssh-copy-id unitree@<robot-ip>           # one-time: no more password prompts
# Needs g1_shortcuts.sh installed on the robot.

G1_HOST=${G1_HOST:-unitree@192.168.0.164}

# g1            -> plain ssh session on the robot
# g1 t3         -> ssh in, run t3; when it exits you stay in a robot shell (type t3 to rerun)
# g1 t11 NAME   -> start a bag called NAME
# G1_AUTO=1 is exported for the remote shell, so a robot ~/.bashrc that asks
# which ROS distribution to load can check it and choose Foxy without a prompt.
g1() {
    if [ $# -eq 0 ]; then ssh -t "$G1_HOST" "G1_AUTO=1 exec bash -il"; return; fi
    ssh -t "$G1_HOST" "G1_AUTO=1 bash -ic '$*; exec bash -i'"
}

# g1up -> opens one tab per terminal (GNOME Terminal) in launch order.
# T6 (loco) and T11 (bag) are left for you: T6 needs the stand-up first,
# T11 needs a bag name.
g1up() {
    command -v gnome-terminal >/dev/null || { echo "gnome-terminal not found — use g1 t1 … g1 t12 in separate tabs"; return 1; }
    local cmds=("g1pre" "t1" "t2" "t3" "t4" "t5" "t7" "t8" "t9" "t10" "t12")
    for c in "${cmds[@]}"; do
        gnome-terminal --tab --title "$c" -- bash -ic "g1 $c; exec bash" 2>/dev/null
        sleep 1.5
    done
    echo "Opened: ${cmds[*]}"
    echo "Next: check T1-T4 logs, stand the robot up, then in new tabs:  g1 t6   and   g1 t11 <bag_name>"
    echo "Viewer: http://${G1_HOST#*@}:8090"
}

g1pull() {   # g1pull <bag_name>  -> copy a bag to ~/bags on the laptop
    [ -n "$1" ] || { echo "usage: g1pull <bag_name>"; return 1; }
    mkdir -p ~/bags && scp -r "$G1_HOST:~/aidan/sensor_fusion/bags/$1" ~/bags/
}

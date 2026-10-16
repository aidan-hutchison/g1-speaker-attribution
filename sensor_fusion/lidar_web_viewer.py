#!/usr/bin/env python3
"""
lidar_web_viewer.py — live browser view of the LiDAR cloud and person tracks
Unitree G1 speaker attribution system

A diagnostic tool; nothing else depends on it. It runs on the robot (rclpy
only, no SDK) alongside person_detector_node.py. Raw LiDAR data (about
5 MB/s) does not stream usably over Wi-Fi, so this node thins the cloud on
the robot and sends a compact frame (about 30 KB) to the browser a few times
a second. It needs no extra packages: Python's built-in HTTP server with
Server-Sent Events, and a plain HTML canvas in the browser.

Open from the laptop:  http://<robot-ip>:8090
(Port 8080 is used by a Unitree service while the robot is in motion mode.)

Shows, top-down in the sensor frame (+x forward drawn up, +y drawn left):
  - the point cloud, coloured by height and thinned to max_points
  - confirmed tracks (solid) and coasting tracks (faded), with ID, range and
    bearing; bearings are atan2(y, x) in the sensor frame, the same
    convention as the floor marks, increasing toward the robot's right
  - the /turn_target crosshair and the /doa_locked_person highlight
  - from /person_detector/debug_json: background cells, accepted and
    rejected clusters with the reason for rejection, and tentative tracks
  - range rings, 30-degree bearing rays and the head-strut sectors
Buttons: light/dark theme, pause, "Reset background" (clears the detector's
background model when it is enabled, e.g. after moving the robot, since only
yaw is compensated), and "Save PNG" (high-resolution image with legend and
timestamp).

Usage:
  python3 lidar_web_viewer.py
  python3 lidar_web_viewer.py --ros-args -p port:=8090 -p max_points:=6000 -p rate_hz:=5.0
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, Int32, String
from visualization_msgs.msg import Marker, MarkerArray


def cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    offs = {f.name: f.offset for f in msg.fields}
    if not all(k in offs for k in ('x', 'y', 'z')) or msg.width * msg.height == 0:
        return np.zeros((0, 3), dtype=np.float32)
    dt = np.dtype({'names': ['x', 'y', 'z'],
                   'formats': ['>f4' if msg.is_bigendian else '<f4'] * 3,
                   'offsets': [offs['x'], offs['y'], offs['z']],
                   'itemsize': msg.point_step})
    arr = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    pts = np.stack([arr['x'], arr['y'], arr['z']], axis=1).astype(np.float32)
    return pts[np.isfinite(pts).all(axis=1)]


class ViewerNode(Node):
    def __init__(self):
        super().__init__('lidar_web_viewer')
        self.declare_parameter('port', 8090)
        self.declare_parameter('cloud_topic', '/utlidar/cloud_livox_mid360')
        self.declare_parameter('tracks_topic', '/detected_people')
        self.declare_parameter('max_points', 6000)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('rate_hz', 5.0)
        # Only used to draw the shaded strut sectors; keep in sync with the detector.
        self.declare_parameter('occlusion_sectors_deg', [30.0, 60.0, 300.0, 330.0])
        p = self.get_parameter
        self.port = int(p('port').value)
        self.max_points = int(p('max_points').value)
        self.max_range = float(p('max_range').value)
        sect = list(p('occlusion_sectors_deg').value)
        self.sectors = [[sect[i], sect[i + 1]] for i in range(0, len(sect) - 1, 2)]

        self._lock = threading.Lock()
        self._cloud = np.zeros((0, 3), dtype=np.float32)
        self._cloud_hz = _Rate()
        self._people: list = []
        self._people_t = 0.0
        self._debug: dict = {}
        self._debug_t = 0.0
        self._turn_target = None
        self._turn_t = 0.0
        self._doa_id = None
        self._doa_t = 0.0

        self.cond = threading.Condition()
        self.frame_json = b''
        self.seq = 0
        self._rng = np.random.default_rng(0)

        self.create_subscription(PointCloud2, p('cloud_topic').value, self._on_cloud, 5)
        self.create_subscription(MarkerArray, p('tracks_topic').value, self._on_people, 10)
        self.create_subscription(String, '/person_detector/debug_json', self._on_debug, 10)
        self.create_subscription(PoseStamped, '/turn_target', self._on_turn, 10)
        self.create_subscription(Int32, '/doa_locked_person', self._on_doa, 10)
        self.create_timer(1.0 / float(p('rate_hz').value), self._build_frame)
        self.reset_pub = self.create_publisher(Empty, '/person_detector/reset_background', 10)
        self.get_logger().info(f'LiDAR web viewer on http://0.0.0.0:{self.port}  '
                               f'(max_points={self.max_points})')

    # ── subscriptions ────────────────────────────────────────────────────
    def _on_cloud(self, msg):
        pts = cloud_to_xyz(msg)
        if len(pts) < 500:            # the Mid-360 alternates near-empty frames
            return
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[(r > 0.15) & (r < self.max_range)]
        if len(pts) > self.max_points:
            pts = pts[self._rng.choice(len(pts), self.max_points, replace=False)]
        with self._lock:
            self._cloud = pts
            self._cloud_hz.tick()

    def _on_people(self, msg):
        people = []
        for m in msg.markers:
            if m.ns == 'people' and m.type == Marker.CYLINDER and m.action == Marker.ADD:
                people.append({'id': int(m.id), 'x': round(float(m.pose.position.x), 3),
                               'y': round(float(m.pose.position.y), 3),
                               'a': round(float(m.color.a), 2),
                               'c': [round(m.color.r, 2), round(m.color.g, 2), round(m.color.b, 2)]})
        with self._lock:
            self._people, self._people_t = people, time.time()

    def _on_debug(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self._debug, self._debug_t = d, time.time()

    def _on_turn(self, msg):
        with self._lock:
            self._turn_target = [round(msg.pose.position.x, 3), round(msg.pose.position.y, 3)]
            self._turn_t = time.time()

    def _on_doa(self, msg):
        with self._lock:
            self._doa_id, self._doa_t = int(msg.data), time.time()

    # ── frame assembly (timer) ───────────────────────────────────────────
    def _build_frame(self):
        now = time.time()
        with self._lock:
            pts = self._cloud
            cm = np.clip(np.round(pts * 100.0), -32767, 32767).astype('<i2')
            frame = {
                'seq': self.seq, 't': now,
                'n': int(len(pts)),
                'cloud': base64.b64encode(cm.tobytes()).decode('ascii'),
                'cloud_hz': round(self._cloud_hz.hz(), 1),
                'people': self._people if now - self._people_t < 1.0 else [],
                'debug': self._debug if now - self._debug_t < 1.5 else {},
                'turn_target': self._turn_target if now - self._turn_t < 0.6 else None,
                'doa_id': self._doa_id if now - self._doa_t < 0.6 else None,
                'sectors': self.sectors,
            }
        data = json.dumps(frame, separators=(',', ':')).encode()
        with self.cond:
            self.frame_json = data
            self.seq += 1
            self.cond.notify_all()


class _Rate:
    def __init__(self):
        self.ts: list = []

    def tick(self):
        t = time.time()
        self.ts = [x for x in self.ts if t - x < 2.0] + [t]

    def hz(self):
        return (len(self.ts) - 1) / (self.ts[-1] - self.ts[0]) if len(self.ts) > 1 else 0.0


def make_handler(node: ViewerNode):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Connection', 'keep-alive')
                self.end_headers()
                last = -1
                try:
                    while True:
                        with node.cond:
                            node.cond.wait_for(lambda: node.seq != last, timeout=2.0)
                            data, last = node.frame_json, node.seq
                        if data:
                            self.wfile.write(b'data: ' + data + b'\n\n')
                        else:
                            self.wfile.write(b': keepalive\n\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == '/reset_background':
                node.reset_pub.publish(Empty())
                node.get_logger().info('Background reset requested from browser')
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)
    return Handler


def main(args=None):
    rclpy.init(args=args)
    node = ViewerNode()
    server = ThreadingHTTPServer(('0.0.0.0', node.port), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G1 LiDAR person tracking — live</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--fg:#e6e8ec;--muted:#8a90a0;--grid:#2a2f3a;--grid2:#39404e;--accent:#4cc2ff;--border:#262b35}
body.light{--bg:#ffffff;--panel:#f4f5f7;--fg:#15171c;--muted:#5b6170;--grid:#dde0e6;--grid2:#b9bec9;--accent:#0069b4;--border:#d8dbe1}
*{box-sizing:border-box}html,body{height:100%;margin:0}
body{background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;display:flex}
#wrap{flex:1;position:relative;min-width:0}canvas{display:block;width:100%;height:100%}
#side{width:270px;background:var(--panel);border-left:1px solid var(--border);padding:14px;overflow:auto}
h1{font-size:14px;margin:0 0 10px}h2{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:16px 0 6px}
label{display:flex;gap:8px;align-items:center;margin:3px 0;cursor:pointer}
button,select{font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:5px 9px;cursor:pointer}
button:hover{border-color:var(--accent)}.row{display:flex;gap:6px;flex-wrap:wrap}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td{padding:2px 0}td:last-child{text-align:right}
#status{font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:#e5484d}
.dot.ok{background:#30a46c}#people td:first-child{font-weight:600}
</style></head><body>
<div id="wrap"><canvas id="c"></canvas></div>
<div id="side">
<h1>G1 LiDAR person tracking</h1>
<div id="status"><span class="dot" id="dot"></span><span id="stxt">connecting…</span></div>
<h2>View</h2>
<div class="row"><select id="range"><option>2</option><option selected>3</option><option>4</option><option>5</option></select><span style="align-self:center">m radius</span></div>
<label><input type="checkbox" id="sPts" checked> Point cloud (height colour)</label>
<label><input type="checkbox" id="sBg" checked> Background cells</label>
<label><input type="checkbox" id="sRej" checked> Rejected clusters</label>
<label><input type="checkbox" id="sAcc"> Accepted clusters</label>
<label><input type="checkbox" id="sTent" checked> Tentative tracks</label>
<label><input type="checkbox" id="sRays" checked> 30° bearing rays</label>
<label><input type="checkbox" id="sSect" checked> Strut sectors</label>
<label><input type="checkbox" id="sLegend" checked> Legend on canvas</label>
<div class="row" style="margin-top:8px"><button id="theme">Light theme</button><button id="pause">Pause</button></div>
<div class="row" style="margin-top:6px"><button id="resetbg" title="Use after moving the robot. Keep it still ~60 s afterwards.">Reset background</button></div>
<div class="row" style="margin-top:6px"><button id="save">Save PNG</button><select id="scale"><option value="2">2×</option><option value="3" selected>3×</option><option value="4">4×</option></select></div>
<h2>People</h2><table id="people"></table>
<h2>Pipeline</h2><table id="stats"></table>
</div>
<script>
const $=id=>document.getElementById(id);
const cv=$('c'),ctx=cv.getContext('2d');
let F=null,paused=false;
const opt=()=>({R:+$('range').value,pts:$('sPts').checked,bg:$('sBg').checked,rej:$('sRej').checked,acc:$('sAcc').checked,
  tent:$('sTent').checked,rays:$('sRays').checked,sect:$('sSect').checked,legend:$('sLegend').checked,light:document.body.classList.contains('light')});
function css(v){return getComputedStyle(document.body).getPropertyValue(v).trim()}
function heightColour(z){ // z in metres (sensor frame, ~ -1.3 floor .. +0.6)
  const t=Math.min(1,Math.max(0,(z+1.3)/1.9));
  const s=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const f=t*(s.length-1),i=Math.min(s.length-2,Math.floor(f)),u=f-i;
  const c=s[i].map((v,k)=>Math.round(v+(s[i+1][k]-v)*u));return `rgb(${c[0]},${c[1]},${c[2]})`}
const REJ={'z-spread':'#e5484d','width':'#f76b15','height':'#ffc53d','density':'#8e4ec6'};
function bearingDeg(x,y){let b=Math.atan2(y,x)*180/Math.PI;return (b+360)%360}
function decode(b64){const s=atob(b64),n=s.length/2,a=new Int16Array(n);for(let i=0;i<n;i++){const lo=s.charCodeAt(2*i),hi=s.charCodeAt(2*i+1);let v=lo|(hi<<8);if(v&0x8000)v-=0x10000;a[i]=v}return a}

function draw(c,W,H,f,o,k){ // k = pixel scale factor (1 on screen, >1 for export)
  const bg=css('--bg'),fg=css('--fg'),muted=css('--muted'),grid=css('--grid'),grid2=css('--grid2'),acc=css('--accent');
  c.save();c.fillStyle=bg;c.fillRect(0,0,W,H);
  const cx=W/2,cy=H/2,s=Math.min(W,H)/2/o.R*0.92;
  const P=(x,y)=>[cx-y*s,cy-x*s]; // forward up, left = left
  // strut sectors
  if(o.sect&&f&&f.sectors){c.fillStyle=o.light?'rgba(229,72,77,.07)':'rgba(229,72,77,.10)';
    for(const [a0,a1] of f.sectors){c.beginPath();c.moveTo(cx,cy);
      // canvas angle for bearing b (CCW from forward/up): theta = -90deg - b
      c.arc(cx,cy,o.R*s,(-90-a0)*Math.PI/180,(-90-a1)*Math.PI/180,true);c.closePath();c.fill()}}
  // rings
  c.lineWidth=1*k;c.font=`${11*k}px system-ui,sans-serif`;c.fillStyle=muted;
  for(let r=0.5;r<=o.R+1e-6;r+=0.5){c.strokeStyle=(Math.abs(r%1)<1e-6)?grid2:grid;c.beginPath();c.arc(cx,cy,r*s,0,2*Math.PI);c.stroke();
    if(Math.abs(r%1)<1e-6){c.textBaseline='top';c.fillText(`${r} m`,cx+r*s*0.94+3*k,cy+r*s*0.34+2*k);c.textBaseline='alphabetic'}}
  // rays
  if(o.rays){c.strokeStyle=grid;c.setLineDash([3*k,4*k]);
    for(let b=0;b<360;b+=30){const t=b*Math.PI/180,[x,y]=P(o.R*Math.cos(t),o.R*Math.sin(t));c.beginPath();c.moveTo(cx,cy);c.lineTo(x,y);c.stroke();
      c.save();c.setLineDash([]);c.fillStyle=muted;c.textAlign='center';c.textBaseline='middle';
      const [tx,ty]=P(o.R*1.045*Math.cos(t),o.R*1.045*Math.sin(t));c.fillText(`${b}°`,tx,ty);c.restore()}
    c.setLineDash([])}
  if(!f){c.restore();return}
  const dbg=f.debug||{};
  // background cells
  if(o.bg&&dbg.bg){const h=(dbg.bg_cell||0.2)*s;c.fillStyle=o.light?'rgba(90,100,120,.28)':'rgba(140,150,170,.22)';
    for(const [x,y] of dbg.bg){if(Math.hypot(x,y)>o.R)continue;const [px,py]=P(x,y);c.fillRect(px-h/2,py-h/2,h,h)}}
  // cloud
  if(o.pts&&f.cloud){const a=decode(f.cloud),r=Math.max(1,1.3*k);
    for(let i=0;i<a.length;i+=3){const x=a[i]/100,y=a[i+1]/100,z=a[i+2]/100;if(Math.hypot(x,y)>o.R)continue;
      const [px,py]=P(x,y);c.fillStyle=heightColour(z);c.fillRect(px-r/2,py-r/2,r,r)}}
  // clusters
  const mark=(x,y,col,shape)=>{const [px,py]=P(x,y),r=5*k;c.strokeStyle=col;c.lineWidth=2*k;c.beginPath();
    if(shape==='x'){c.moveTo(px-r,py-r);c.lineTo(px+r,py+r);c.moveTo(px+r,py-r);c.lineTo(px-r,py+r)}else{c.rect(px-r,py-r,2*r,2*r)}c.stroke()};
  if(o.rej&&dbg.rejected)for(const d of dbg.rejected)mark(d.xy[0],d.xy[1],REJ[d.reason]||'#e5484d','x');
  if(o.acc&&dbg.accepted)for(const d of dbg.accepted)mark(d.xy[0],d.xy[1],'#30a46c','sq');
  if(o.tent&&dbg.tentative){c.setLineDash([2*k,3*k]);c.strokeStyle=muted;c.lineWidth=1.5*k;
    for(const [x,y] of dbg.tentative){const [px,py]=P(x,y);c.beginPath();c.arc(px,py,0.22*s,0,2*Math.PI);c.stroke()}c.setLineDash([])}
  // robot
  c.fillStyle=fg;c.beginPath();c.moveTo(cx,cy-10*k);c.lineTo(cx-7*k,cy+7*k);c.lineTo(cx+7*k,cy+7*k);c.closePath();c.fill();
  // people
  for(const p of f.people){const [px,py]=P(p.x,p.y),col=`rgba(${p.c.map(v=>Math.round(v*255)).join(',')},${p.a<0.65?0.35:0.9})`;
    const rr=0.25*s;c.fillStyle=col.replace(/[\d.]+\)$/,p.a<0.65?'0.15)':'0.28)');c.strokeStyle=col;c.lineWidth=2.5*k;
    c.beginPath();c.arc(px,py,rr,0,2*Math.PI);c.fill();c.stroke();
    if(f.doa_id===p.id){c.strokeStyle=acc;c.lineWidth=3*k;c.beginPath();c.arc(px,py,rr+6*k,0,2*Math.PI);c.stroke()}
    const d=Math.hypot(p.x,p.y),b=bearingDeg(p.x,p.y);
    c.fillStyle=fg;c.font=`600 ${12*k}px system-ui,sans-serif`;c.textAlign='left';c.textBaseline='middle';
    c.fillText(`P${p.id}${p.a<0.65?' (coast)':''}${f.doa_id===p.id?'  DOA':''}`,px+rr+6*k,py-7*k);
    c.font=`${11*k}px system-ui,sans-serif`;c.fillStyle=muted;c.fillText(`${d.toFixed(2)} m  ${b.toFixed(0)}°`,px+rr+6*k,py+8*k)}
  // turn target
  if(f.turn_target){const [px,py]=P(f.turn_target[0],f.turn_target[1]),r=11*k;c.strokeStyle=acc;c.lineWidth=2*k;
    c.beginPath();c.moveTo(px-r,py);c.lineTo(px+r,py);c.moveTo(px,py-r);c.lineTo(px,py+r);c.stroke();
    c.setLineDash([4*k,4*k]);c.beginPath();c.moveTo(cx,cy);c.lineTo(px,py);c.stroke();c.setLineDash([])}
  if(o.legend)legend(c,W,H,k,o,f);
  c.restore()}

function legend(c,W,H,k,o,f){
  const fg=css('--fg'),muted=css('--muted'),pan=css('--panel'),bd=css('--border');
  const items=[['circ','#30a46c','Confirmed track'],['circf','#30a46c','Coasting track'],['dash',muted,'Tentative track'],
    ['cross',css('--accent'),'/turn_target'],['sq',o.light?'rgba(90,100,120,.5)':'rgba(140,150,170,.5)','Background cell'],
    ['x',REJ['z-spread'],'Rejected: z-spread'],['x',REJ['width'],'Rejected: width'],['x',REJ['height'],'Rejected: height'],['x',REJ['density'],'Rejected: density'],
    ['sect','rgba(229,72,77,.35)','Head-strut sector']];
  const pad=10*k,lh=17*k,w=170*k,h=pad*2+lh*(items.length+1)+10*k;const x0=W-w-12*k,y0=12*k;
  c.fillStyle=pan;c.globalAlpha=.92;c.fillRect(x0,y0,w,h);c.globalAlpha=1;c.strokeStyle=bd;c.lineWidth=k;c.strokeRect(x0,y0,w,h);
  c.font=`${11*k}px system-ui,sans-serif`;c.textBaseline='middle';c.textAlign='left';
  items.forEach(([t,col,txt],i)=>{const y=y0+pad+lh*i+lh/2,x=x0+pad+7*k;c.strokeStyle=col;c.fillStyle=col;c.lineWidth=2*k;
    if(t==='circ'||t==='circf'){c.globalAlpha=t==='circ'?1:.4;c.beginPath();c.arc(x,y,6*k,0,2*Math.PI);c.stroke();c.globalAlpha=1}
    else if(t==='dash'){c.setLineDash([2*k,2*k]);c.beginPath();c.arc(x,y,6*k,0,2*Math.PI);c.stroke();c.setLineDash([])}
    else if(t==='cross'){c.beginPath();c.moveTo(x-6*k,y);c.lineTo(x+6*k,y);c.moveTo(x,y-6*k);c.lineTo(x,y+6*k);c.stroke()}
    else if(t==='sq'||t==='sect'){c.fillRect(x-6*k,y-6*k,12*k,12*k)}
    else{c.beginPath();c.moveTo(x-5*k,y-5*k);c.lineTo(x+5*k,y+5*k);c.moveTo(x+5*k,y-5*k);c.lineTo(x-5*k,y+5*k);c.stroke()}
    c.fillStyle=fg;c.fillText(txt,x+14*k,y)});
  // height ramp
  const y=y0+pad+lh*items.length+4*k,x=x0+pad;const gw=w-2*pad;
  for(let i=0;i<gw;i++){c.fillStyle=heightColour(-1.3+1.9*i/gw);c.fillRect(x+i,y,1,8*k)}
  c.fillStyle=muted;c.font=`${10*k}px system-ui,sans-serif`;c.textAlign='left';c.fillText('low',x,y+16*k);c.textAlign='right';c.fillText('high',x+gw,y+16*k);c.textAlign='left'}

function resize(){const r=cv.parentElement.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;render()}
function render(){const d=devicePixelRatio||1;draw(ctx,cv.width,cv.height,F,opt(),d)}
function panels(){if(!F)return;
  $('people').innerHTML=F.people.length?F.people.map(p=>`<tr><td>P${p.id}${p.a<0.65?' <span style="color:var(--muted)">coast</span>':''}${F.doa_id===p.id?' <span style="color:var(--accent)">DOA</span>':''}</td><td>${Math.hypot(p.x,p.y).toFixed(2)} m</td><td>${bearingDeg(p.x,p.y).toFixed(0)}°</td></tr>`).join(''):'<tr><td style="color:var(--muted)">none</td></tr>';
  const s=(F.debug||{}).stats||{};const rej=((F.debug||{}).rejected||[]).length,acc=((F.debug||{}).accepted||[]).length;
  const rows=[['LiDAR rate',`${F.cloud_hz} Hz`],['Raw points',s.raw??'–'],['After range/height/ground',s.filtered??'–'],['Foreground (after bg)',s.foreground??'–'],
    ['Background cells',s.bg_cells??'–'],['Background resets',s.bg_resets??'–'],['Clusters accepted / rejected',`${acc} / ${rej}`],
    ['Heading',s.heading_ok?`${s.heading_deg}° (${s.heading_source})`:(s.heading_source?`<span style="color:#e5484d">${s.heading_source} — no data</span>`:'–')],
    ['Yaw rate',s.yaw_rate_dps!==undefined?`${s.yaw_rate_dps}°/s${s.turning?' <span style="color:var(--accent)">TURNING</span>':''}`:'–'],
    ['Odometry offset',s.odom_xy?`${s.odom_xy[0].toFixed(2)}, ${s.odom_xy[1].toFixed(2)} m`:'–'],
    ['Scan alignment',s.align?`${(s.align[0]*100).toFixed(0)}, ${(s.align[1]*100).toFixed(0)} cm · ${s.align[2].toFixed(1)}°`:'–']];
  $('stats').innerHTML=rows.map(([a,b])=>`<tr><td>${a}</td><td>${b}</td></tr>`).join('')}

let lastMsg=0;
function connect(){const es=new EventSource('/stream');
  es.onmessage=e=>{lastMsg=Date.now();if(paused)return;F=JSON.parse(e.data);render();panels()};
  es.onerror=()=>{es.close();setTimeout(connect,1500)}}
setInterval(()=>{const ok=Date.now()-lastMsg<2000;$('dot').className='dot'+(ok?' ok':'');
  $('stxt').textContent=ok?(paused?'paused':`live · ${F?F.n:0} pts shown`):'no data — is the node running?'},500);
['range','sPts','sBg','sRej','sAcc','sTent','sRays','sSect','sLegend'].forEach(id=>$(id).addEventListener('change',render));
$('theme').onclick=()=>{document.body.classList.toggle('light');$('theme').textContent=document.body.classList.contains('light')?'Dark theme':'Light theme';render()};
$('resetbg').onclick=()=>{if(confirm('Reset the learned background? Keep the robot still for ~60 s afterwards.'))fetch('/reset_background',{method:'POST'})};
$('pause').onclick=()=>{paused=!paused;$('pause').textContent=paused?'Resume':'Pause'};
$('save').onclick=()=>{const k=+$('scale').value,W=1200*k,H=1200*k,off=document.createElement('canvas');off.width=W;off.height=H;
  const c=off.getContext('2d'),o=opt();draw(c,W,H,F,o,k);
  const ts=F?new Date(F.t*1000):new Date();c.fillStyle=css('--muted');c.font=`${11*k}px system-ui,sans-serif`;c.textBaseline='bottom';
  c.fillText(`G1 LiDAR person tracking · ${ts.toLocaleString()} · radius ${o.R} m · forward = up, bearings CCW`,12*k,H-10*k);
  const a=document.createElement('a');a.download=`g1_lidar_${ts.toISOString().replace(/[:.]/g,'-')}.png`;a.href=off.toDataURL('image/png');a.click()};
addEventListener('resize',resize);resize();connect();
</script></body></html>
"""


if __name__ == '__main__':
    main()

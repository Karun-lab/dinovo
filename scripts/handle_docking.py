#!/usr/bin/env python3
import math, os, sys, yaml, threading, rospy
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import Empty
from visualization_msgs.msg import Marker, MarkerArray


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


IDLE, STAGING, APPROACH = 'IDLE', 'STAGING', 'APPROACH'
WAITING_FOR_USER, RETREAT, RETURN = 'WAITING_FOR_USER', 'RETREAT', 'RETURN'   # --- NEW ---


class GoToHandle:
    def __init__(self):
        rospy.init_node('go_to_handle')
        P = rospy.get_param

        # Topics / files
        self.robot_topic = P('~robot_topic', '/vicon/dingo1')
        self.handle_topic = P('~handle_topic', '/vicon/handle')
        self.cmd_topic = P('~cmd_topic', '/dingo1/cmd_vel')
        self.start_topic = P('~start_topic', '/dingo1/go_to_handle/start')
        self.abort_topic = P('~abort_topic', '/dingo1/go_to_handle/abort')
        self.frame_id = P('~frame_id', 'world')
        self.config_path = os.path.expanduser(P('~config_path', '~/handle_dock.yaml'))

        # Motion
        self.staging_dist = P('~staging_distance', 0.5)
        self.v_max = P('~v_max', 0.4)
        self.v_final = P('~v_final', 0.15)
        self.v_min = P('~v_min', 0.04)
        self.katt = P('~katt', 1.0)
        self.ktheta = P('~ktheta', 1.5)
        self.w_max = P('~w_max', 1.0)
        self.eps_pos = P('~eps_pos', 0.03)
        self.eps_stage = P('~eps_stage', 0.10)
        self.eps_yaw = P('~eps_yaw', 0.05)
        self.settle_time = P('~settle_time', 1.0)
        self.yaw_offset = P('~yaw_offset', 0.0)
        self.ctrl_period = P('~ctrl_period', 0.05)
        self.vicon_timeout = P('~vicon_timeout', 0.5)

        # --- NEW: back-up distance for opening the drawer ---
        self.drawer_offset = P('~drawer_offset', 0.3)     # m to back up (and to return)
        self.v_retreat = P('~v_retreat', 0.15)            # m/s while backing up / returning

        # Obstacles
        self.ignore = set(P('~ignore', []))
        self.robot_radius = P('~robot_radius', 0.35)
        self.default_radius = P('~default_radius', 0.5)
        self.rho0 = P('~influence_margin', 0.8)
        self.kr = P('~kr', 0.1)
        self.swirl = P('~swirl', 0.3)
        self.obstacle_timeout = P('~obstacle_timeout', 1.0)
        self.radii = {}

        # State
        self.state = IDLE
        self.pos, self.yaw, self.last_vicon = None, 0.0, None
        self.handle, self.handle_time = None, None
        self.dock = self.stage = None
        self.settle_start = None
        self.obstacles, self.subs = {}, {}

        # --- NEW: retreat/return state ---
        self.dock_arrived = None       # latched pose of where we stopped after APPROACH
        self.retreat_target = None     # computed when 'k' is pressed the first time
        self.waiting_for_k = False     # true when we are in WAITING_FOR_USER
        self.dock_yaw = None           # keep current heading while backing up

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('~bubbles', MarkerArray, queue_size=1)
        rospy.Subscriber(self.robot_topic, PoseStamped, self.robot_cb)
        rospy.Subscriber(self.handle_topic, PoseStamped, self.handle_cb)
        rospy.Subscriber(self.start_topic, Empty, self.start_cb)
        rospy.Subscriber(self.abort_topic, Empty, self.abort_cb)

        rospy.Timer(rospy.Duration(self.ctrl_period), self.control_loop)
        rospy.Timer(rospy.Duration(2.0), self.discover_obstacles)
        rospy.Timer(rospy.Duration(0.5), self.refresh_params_and_markers)
        self.discover_obstacles(None)

        # --- NEW: keyboard thread ---
        self.kb_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.kb_thread.start()

        rospy.loginfo("Go-to-handle ready. Start with: rostopic pub -1 %s std_msgs/Empty '{}'", self.start_topic)
        if P('~autostart', False):
            rospy.sleep(1.0)
            self.start_cb(None)

    # ------------------------------------------------------------ callbacks
    def robot_cb(self, msg):
        self.pos = (msg.pose.position.x, msg.pose.position.y)
        self.yaw = wrap(yaw_of(msg.pose.orientation) + self.yaw_offset)
        self.last_vicon = rospy.Time.now()

    def handle_cb(self, msg):
        self.handle = (msg.pose.position.x, msg.pose.position.y, yaw_of(msg.pose.orientation))
        self.handle_time = rospy.Time.now()

    def obstacle_cb(self, msg, name):
        self.obstacles[name] = (msg.pose.position.x, msg.pose.position.y, rospy.Time.now())

    def fresh(self, t):
        return t is not None and (rospy.Time.now() - t).to_sec() <= self.vicon_timeout

    def start_cb(self, _):
        if self.state != IDLE:
            rospy.logwarn("[GOTO] Already running (%s).", self.state)
            return
        try:
            with open(self.config_path) as f:
                cfg = yaml.safe_load(f)
            d = cfg['dock']
            use_handle_yaw = bool(cfg['use_handle_yaw'])
            rx, ry, ryaw = float(d['x']), float(d['y']), float(d['yaw'])
        except Exception as e:
            rospy.logerr("[GOTO] Cannot load %s: %s (run capture_dock_pose.py first)", self.config_path, e)
            return
        if not (self.fresh(self.last_vicon) and self.fresh(self.handle_time)):
            rospy.logerr("[GOTO] No fresh Vicon data for dingo and/or handle. Not starting.")
            return

        hx, hy, hyaw = self.handle
        if use_handle_yaw:
            c, s = math.cos(hyaw), math.sin(hyaw)
            gx, gy, gyaw = hx + c * rx - s * ry, hy + s * rx + c * ry, wrap(hyaw + ryaw)
        else:
            gx, gy, gyaw = hx + rx, hy + ry, ryaw
        self.dock = (gx, gy, gyaw)
        self.stage = (gx - self.staging_dist * math.cos(gyaw), gy - self.staging_dist * math.sin(gyaw))
        self.settle_start = None
        self.dock_arrived = None
        self.retreat_target = None
        self.waiting_for_k = False
        self.state = STAGING if self.staging_dist > 0.0 else APPROACH
        rospy.loginfo("[GOTO] Started. Dock pose (%.2f, %.2f, %.0f deg).", gx, gy, math.degrees(gyaw))

    def abort_cb(self, _):
        if self.state != IDLE:
            rospy.logwarn("[GOTO] Aborted in state %s.", self.state)
        self.state = IDLE
        self.waiting_for_k = False
        self.cmd_pub.publish(Twist())

    # ------------------------------------------------------------ NEW: keyboard
    def keyboard_loop(self):
        while not rospy.is_shutdown():
            try:
                line = sys.stdin.readline()
            except Exception:
                continue
            if not line:
                continue
            key = line.strip().lower()
            if key == 'k':
                self.on_k_pressed()

    def on_k_pressed(self):
        # Only meaningful once we've arrived in front of the handle
        if self.state == WAITING_FOR_USER:
            # First press: back up by drawer_offset
            if self.dock_arrived is None:
                rospy.logwarn("[GOTO] No dock arrival pose latched; ignoring 'k'.")
                return
            x, y, yaw = self.dock_arrived
            # Back up along the handle's outward direction (i.e. away from dock)
            self.retreat_target = (x - self.drawer_offset * math.cos(yaw),
                                   y - self.drawer_offset * math.sin(yaw),
                                   yaw)
            self.state = RETREAT
            rospy.loginfo("[GOTO] 'k' pressed: backing up %.2f m to open the drawer.", self.drawer_offset)
        elif self.state == IDLE and self.dock_arrived is not None and self.retreat_target is not None:
            # Second press: return to dock pose
            x, y, yaw = self.dock_arrived
            self.retreat_target = (x, y, yaw)   # reuse variable as return target
            self.state = RETURN
            rospy.loginfo("[GOTO] 'k' pressed: returning %.2f m to close the drawer.", self.drawer_offset)
        else:
            rospy.loginfo("[GOTO] 'k' ignored in state %s.", self.state)

    # ------------------------------------------------------------ obstacles
    def discover_obstacles(self, _):
        try:
            topics = rospy.get_published_topics('/vicon')
        except Exception as e:
            rospy.logwarn_throttle(5, "[GOTO] Topic discovery failed: %s", e)
            return
        for topic, typ in topics:
            if typ != 'geometry_msgs/PoseStamped' or topic in (self.robot_topic, self.handle_topic):
                continue
            name = topic.split('/')[-1]
            if name in self.ignore or name in self.subs:
                continue
            self.subs[name] = rospy.Subscriber(topic, PoseStamped, self.obstacle_cb, callback_args=name)
            rospy.loginfo("[GOTO] Tracking obstacle '%s', bubble %.2f m", name, self.radius_of(name))

    def radius_of(self, name):
        return float(self.radii.get(name, self.default_radius))

    def refresh_params_and_markers(self, _):
        self.radii = rospy.get_param('~obstacle_radii', {})
        self.default_radius = rospy.get_param('~default_radius', self.default_radius)
        self.rho0 = rospy.get_param('~influence_margin', self.rho0)
        self.kr = rospy.get_param('~kr', self.kr)
        self.swirl = rospy.get_param('~swirl', self.swirl)
        arr, now = MarkerArray(), rospy.Time.now()
        for name, (x, y, _) in self.obstacles.items():
            r = self.radius_of(name)
            for j, (rad, col) in enumerate([(r, (1.0, 0.1, 0.1, 0.5)),
                                            (r + self.robot_radius + self.rho0, (1.0, 0.9, 0.1, 0.15))]):
                m = Marker()
                m.header.frame_id, m.header.stamp = self.frame_id, now
                m.ns, m.id, m.type, m.action = name, j, Marker.CYLINDER, Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.05
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 2.0 * rad
                m.scale.z = 0.1
                m.color.r, m.color.g, m.color.b, m.color.a = col
                m.lifetime = rospy.Duration(1.0)
                arr.markers.append(m)
        self.marker_pub.publish(arr)

    def repulsion(self):
        fx = fy = 0.0
        now = rospy.Time.now()
        for name, (ox, oy, t) in list(self.obstacles.items()):
            if (now - t).to_sec() > self.obstacle_timeout:
                continue
            vx, vy = self.pos[0] - ox, self.pos[1] - oy
            c = math.hypot(vx, vy)
            if c < 1e-6:
                continue
            d = c - self.radius_of(name) - self.robot_radius
            if d >= self.rho0:
                continue
            d = max(d, 0.02)
            mag = self.kr * (1.0 / d - 1.0 / self.rho0)
            ux, uy = vx / c, vy / c
            fx += mag * (ux - self.swirl * uy)
            fy += mag * (uy + self.swirl * ux)
        return fx, fy

    # ------------------------------------------------------------ control
    def drive(self, tx, ty, tyaw, vmax, tol):
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        dist = math.hypot(dx, dy)
        ax = ay = 0.0
        if dist > tol:
            speed = clamp(self.katt * dist, self.v_min, vmax)
            ax, ay = speed * dx / dist, speed * dy / dist
        rx, ry = self.repulsion()
        vx, vy = ax + rx, ay + ry
        n = math.hypot(vx, vy)
        if n > vmax:
            vx, vy = vx / n * vmax, vy / n * vmax

        cmd = Twist()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        cmd.linear.x = c * vx + s * vy
        cmd.linear.y = -s * vx + c * vy
        yaw_err = wrap(tyaw - self.yaw)
        if abs(yaw_err) > self.eps_yaw:
            cmd.angular.z = clamp(self.ktheta * yaw_err, -self.w_max, self.w_max)
        return cmd, dist, yaw_err

    def control_loop(self, _):
        if self.state == IDLE:
            return
        if not self.fresh(self.last_vicon):
            rospy.logwarn_throttle(1.0, "[GOTO] Vicon data stale - stopping.")
            self.cmd_pub.publish(Twist())
            return

        gyaw = self.dock[2]

        if self.state == STAGING:
            cmd, dist, _ = self.drive(self.stage[0], self.stage[1], gyaw, self.v_max, self.eps_stage)
            if dist <= self.eps_stage:
                rospy.loginfo("[GOTO] At staging point. Final approach.")
                self.state = APPROACH

        elif self.state == APPROACH:
            cmd, dist, yaw_err = self.drive(self.dock[0], self.dock[1], gyaw, self.v_final, self.eps_pos)
            if dist <= self.eps_pos and abs(yaw_err) <= self.eps_yaw:
                if self.settle_start is None:
                    self.settle_start = rospy.Time.now()
                elif (rospy.Time.now() - self.settle_start).to_sec() >= self.settle_time:
                    self.cmd_pub.publish(Twist())
                    rospy.loginfo("[GOTO] Arrived (err %.1f mm, %.1f deg). Stopped.",
                                  dist * 1000, math.degrees(yaw_err))
                    # --- NEW: latch arrival and wait for 'k' ---
                    self.dock_arrived = (self.pos[0], self.pos[1], self.yaw)
                    self.retreat_target = None
                    self.state = WAITING_FOR_USER
                    rospy.loginfo("[GOTO] Type 'k' and Enter to back up %.2f m (open drawer).", self.drawer_offset)
                    return
            else:
                self.settle_start = None

        elif self.state == RETREAT:
            # Back straight up along the latched dock heading
            cmd, dist, yaw_err = self.drive(self.retreat_target[0], self.retreat_target[1],
                                            self.retreat_target[2], self.v_retreat, self.eps_pos)
            if dist <= self.eps_pos and abs(yaw_err) <= self.eps_yaw:
                self.cmd_pub.publish(Twist())
                rospy.loginfo("[GOTO] Retreated %.2f m. Drawer should be open. Type 'k' to return.",
                              self.drawer_offset)
                self.state = IDLE   # allows a second 'k' to trigger RETURN
                return

        elif self.state == RETURN:
            cmd, dist, yaw_err = self.drive(self.retreat_target[0], self.retreat_target[1],
                                            self.retreat_target[2], self.v_retreat, self.eps_pos)
            if dist <= self.eps_pos and abs(yaw_err) <= self.eps_yaw:
                self.cmd_pub.publish(Twist())
                rospy.loginfo("[GOTO] Returned to dock. Drawer should be closed.")
                self.dock_arrived = None
                self.retreat_target = None
                self.state = IDLE
                return

        elif self.state == WAITING_FOR_USER:
            # Parked, waiting for terminal input; keep publishing zero to hold
            self.cmd_pub.publish(Twist())
            return

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        GoToHandle()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
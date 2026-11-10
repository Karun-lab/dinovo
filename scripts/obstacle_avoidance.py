#!/usr/bin/env python3
import math
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class PotentialFieldController:
    def __init__(self):
        rospy.init_node('vicon_pf_controller', anonymous=False)

        #Robot / topics
        self.robot_topic = rospy.get_param('~robot_topic', '/vicon/dingo1')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/dingo1/cmd_vel')
        self.frame_id = rospy.get_param('~frame_id', 'world')   # must match RViz Fixed Frame
        self.ignore = set(rospy.get_param('~ignore', []))       # vicon object names to NOT treat as obstacles

        #Goal
        self.v_max = rospy.get_param('~v_max', 0.5)           # m/s
        self.katt = rospy.get_param('~katt', 1.0)             # attractive gain (speed = katt * dist, capped)
        self.ktheta = rospy.get_param('~ktheta', 1.5)
        self.w_max = rospy.get_param('~w_max', 1.0)           # rad/s
        self.eps_pos = rospy.get_param('~eps_pos', 0.10)      # m
        self.eps_yaw = rospy.get_param('~eps_yaw', 0.08)      # rad
        self.yaw_offset = rospy.get_param('~yaw_offset', 0.0)
        self.ctrl_period = rospy.get_param('~ctrl_period', 0.05)
        self.vicon_timeout = rospy.get_param('~vicon_timeout', 0.5)

        #Obstacles
        self.robot_radius = rospy.get_param('~robot_radius', 0.35)   # m, Dingo footprint approx
        self.default_radius = rospy.get_param('~default_radius', 0.5)  # m, bubble if object not in obstacle_radii
        self.rho0 = rospy.get_param('~influence_margin', 0.8)        # m, repulsion reaches this far beyond bubble
        self.kr = rospy.get_param('~kr', 0.1)                        # repulsive gain
        self.swirl = rospy.get_param('~swirl', 0.3)                  # tangential fraction, helps escape local minima
        self.obstacle_timeout = rospy.get_param('~obstacle_timeout', 1.0)  # s, ignore stale obstacle data
        self.radii = {}   # per-object bubble radii, refreshed from the param server at runtime

        #State
        self.pos = None
        self.yaw = 0.0
        self.last_vicon = None
        self.goal = None          # (x, y, yaw)
        self.reached = False
        self.obstacles = {}       # name -> (x, y, stamp)
        self.subs = {}

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('~bubbles', MarkerArray, queue_size=1)
        rospy.Subscriber(self.robot_topic, PoseStamped, self.robot_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        rospy.Timer(rospy.Duration(self.ctrl_period), self.control_loop)
        rospy.Timer(rospy.Duration(2.0), self.discover_obstacles)
        rospy.Timer(rospy.Duration(0.5), self.refresh_params_and_markers)
        self.discover_obstacles(None)
        rospy.loginfo("Potential field controller ready. Bubbles are visible on ~bubbles.")

    #callbacks
    def robot_cb(self, msg):
        self.pos = (msg.pose.position.x, msg.pose.position.y)
        self.yaw = wrap_angle(quat_to_yaw(msg.pose.orientation) + self.yaw_offset)
        self.last_vicon = rospy.Time.now()

    def goal_cb(self, msg):
        gx, gy = msg.pose.position.x, msg.pose.position.y
        self.goal = (gx, gy, quat_to_yaw(msg.pose.orientation))
        self.reached = False
        rospy.loginfo("[PF] New goal (%.2f, %.2f), yaw %.1f deg (frame: %s)",
                      gx, gy, math.degrees(self.goal[2]), msg.header.frame_id)
        for name, (ox, oy, _) in self.obstacles.items():
            if math.hypot(gx - ox, gy - oy) < self.radius_of(name) + self.robot_radius:
                rospy.logwarn("[PF] Goal is inside the bubble of '%s' - robot will not reach it.", name)

    def obstacle_cb(self, msg, name):
        self.obstacles[name] = (msg.pose.position.x, msg.pose.position.y, rospy.Time.now())

    #obstacles
    def discover_obstacles(self, _):
        """Subscribe to every geometry_msgs/PoseStamped under /vicon except the robot itself."""
        try:
            topics = rospy.get_published_topics('/vicon')
        except Exception as e:
            rospy.logwarn_throttle(5, "[PF] Topic discovery failed: %s", e)
            return
        for topic, typ in topics:
            if typ != 'geometry_msgs/PoseStamped' or topic == self.robot_topic:
                continue
            name = topic.split('/')[-1]
            if name in self.ignore or name in self.subs:
                continue
            self.subs[name] = rospy.Subscriber(topic, PoseStamped, self.obstacle_cb, callback_args=name)
            rospy.loginfo("[PF] Tracking obstacle '%s' (%s), bubble radius %.2f m",
                          name, topic, self.radius_of(name))

    def radius_of(self, name):
        return float(self.radii.get(name, self.default_radius))

    def refresh_params_and_markers(self, _):
        self.radii = rospy.get_param('~obstacle_radii', {})
        self.default_radius = rospy.get_param('~default_radius', self.default_radius)
        self.rho0 = rospy.get_param('~influence_margin', self.rho0)
        self.kr = rospy.get_param('~kr', self.kr)
        self.swirl = rospy.get_param('~swirl', self.swirl)
        self.robot_radius = rospy.get_param('~robot_radius', self.robot_radius)

        arr = MarkerArray()
        now = rospy.Time.now()
        for i, (name, (x, y, t)) in enumerate(self.obstacles.items()):
            r = self.radius_of(name)
            for j, (rad, color) in enumerate([(r, (1.0, 0.1, 0.1, 0.5)),
                                              (r + self.robot_radius + self.rho0, (1.0, 0.9, 0.1, 0.15))]):
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = now
                m.ns = name
                m.id = j
                m.type = Marker.CYLINDER
                m.action = Marker.ADD
                m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.05
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = 2.0 * rad
                m.scale.z = 0.1
                m.color.r, m.color.g, m.color.b, m.color.a = color
                m.lifetime = rospy.Duration(1.0)
                arr.markers.append(m)
        self.marker_pub.publish(arr)

    #potential field
    def repulsion(self):
        """Sum of repulsive velocity vectors (world frame) from all fresh obstacles."""
        fx = fy = 0.0
        now = rospy.Time.now()
        for name, (ox, oy, t) in list(self.obstacles.items()):
            if (now - t).to_sec() > self.obstacle_timeout:
                continue
            vx, vy = self.pos[0] - ox, self.pos[1] - oy
            c = math.hypot(vx, vy)
            if c < 1e-6:
                continue
            # Distance from the robot's edge to the bubble surface
            d = c - self.radius_of(name) - self.robot_radius
            if d >= self.rho0:
                continue
            if d <= 0.0:
                rospy.logwarn_throttle(1.0, "[PF] Inside bubble of '%s'!", name)
            d = max(d, 0.02)
            mag = self.kr * (1.0 / d - 1.0 / self.rho0)       # m/s, grows as the bubble is approached
            ux, uy = vx / c, vy / c                            # unit vector pointing AWAY from obstacle
            fx += mag * (ux - self.swirl * uy)                 # radial + a bit of tangential swirl
            fy += mag * (uy + self.swirl * ux)
        return fx, fy

    def control_loop(self, _):
        cmd = Twist()
        stale = (self.pos is None or self.last_vicon is None or
                 (rospy.Time.now() - self.last_vicon).to_sec() > self.vicon_timeout)
        if stale or self.goal is None:
            self.cmd_pub.publish(cmd)
            return

        gx, gy, gyaw = self.goal
        dx, dy = gx - self.pos[0], gy - self.pos[1]
        dist = math.hypot(dx, dy)

        # Attractive term: ramps down near the goal, deadband at the goal
        ax = ay = 0.0
        if dist > self.eps_pos:
            speed = min(self.v_max, self.katt * dist)
            ax, ay = speed * dx / dist, speed * dy / dist

        # Repulsive term from every obstacle bubble
        rx, ry = self.repulsion()

        vx_w, vy_w = ax + rx, ay + ry
        n = math.hypot(vx_w, vy_w)
        if n > self.v_max:
            vx_w, vy_w = vx_w / n * self.v_max, vy_w / n * self.v_max

        # World frame -> robot body frame (cmd_vel is body frame)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        cmd.linear.x = c * vx_w + s * vy_w
        cmd.linear.y = -s * vx_w + c * vy_w

        # Heading: track the goal arrow's yaw while driving and when parked
        yaw_err = wrap_angle(gyaw - self.yaw)
        if abs(yaw_err) > self.eps_yaw:
            cmd.angular.z = clamp(self.ktheta * yaw_err, -self.w_max, self.w_max)

        if dist <= self.eps_pos and abs(yaw_err) <= self.eps_yaw and not self.reached:
            self.reached = True
            rospy.loginfo("[PF] Goal reached (position and heading).")

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        PotentialFieldController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
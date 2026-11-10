#!/usr/bin/env python3
import rospy
import math
from geometry_msgs.msg import Twist, PoseStamped


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class ViconGoalController:
    IDLE, TRANSLATE, ROTATE = 'IDLE', 'TRANSLATE', 'ROTATE'

    def __init__(self):
        rospy.init_node('vicon_goal_controller', anonymous=True)

        # Params
        self.v_max = rospy.get_param('~v_const', 0.5)            # max linear speed m/s
        self.kpos = rospy.get_param('~kpos', 1.0)                # position gain (slows near goal)
        self.ktheta = rospy.get_param('~ktheta', 1.5)            # yaw gain
        self.w_max = rospy.get_param('~w_max', 1.0)              # max angular speed rad/s
        self.eps_pos = rospy.get_param('~eps_control', 0.15)     # m, position tolerance
        self.eps_yaw = rospy.get_param('~eps_yaw', 0.08)         # rad, final heading tolerance
        self.yaw_offset = rospy.get_param('~yaw_offset', 0.0)    # rad, Vicon frame vs robot forward
        self.ctrl_period = rospy.get_param('~ctrl_period', 0.05)
        self.vicon_timeout = rospy.get_param('~vicon_timeout', 0.5)  # s, stop if data goes stale

        # Goal
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_yaw = 0.0
        self.state = self.IDLE

        # Robot state
        self.current_x = None
        self.current_y = None
        self.current_yaw = 0.0
        self.last_vicon_time = None

        self.cmd_pub = rospy.Publisher('/dingo1/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/vicon/dingo1', PoseStamped, self.vicon_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.nav_goal_cb)
        rospy.Timer(rospy.Duration(self.ctrl_period), self.control_loop)
        rospy.loginfo("Vicon Goal Controller initialized.")

    def nav_goal_cb(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        self.goal_yaw = quat_to_yaw(msg.pose.orientation)
        self.state = self.TRANSLATE
        rospy.loginfo("[CTRL] New goal: (%.2f, %.2f), yaw %.1f deg (frame: %s)",
                      self.goal_x, self.goal_y, math.degrees(self.goal_yaw),
                      msg.header.frame_id)

    def vicon_cb(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_yaw = wrap_angle(quat_to_yaw(msg.pose.orientation) + self.yaw_offset)
        self.last_vicon_time = rospy.Time.now()

    def control_loop(self, event):
        cmd = Twist()  # zeros by default

        # Safety: no pose or stale pose -> stop
        if self.current_x is None or self.last_vicon_time is None or \
           (rospy.Time.now() - self.last_vicon_time).to_sec() > self.vicon_timeout:
            self.cmd_pub.publish(cmd)
            return

        if self.state == self.IDLE:
            self.cmd_pub.publish(cmd)
            return

        dx = self.goal_x - self.current_x
        dy = self.goal_y - self.current_y
        dist = math.hypot(dx, dy)
        yaw_err = wrap_angle(self.goal_yaw - self.current_yaw)

        # State transitions (with hysteresis so it doesn't chatter at the boundary)
        if self.state == self.TRANSLATE and dist < self.eps_pos:
            self.state = self.ROTATE
        elif self.state == self.ROTATE and dist > 2.0 * self.eps_pos:
            self.state = self.TRANSLATE

        if self.state == self.TRANSLATE:
            # Speed ramps down near the goal, capped at v_max
            speed = min(self.v_max, self.kpos * dist)
            vx_w = speed * dx / dist
            vy_w = speed * dy / dist

            # cmd_vel is in the robot's body frame, so rotate the world-frame velocity
            c, s = math.cos(self.current_yaw), math.sin(self.current_yaw)
            cmd.linear.x = c * vx_w + s * vy_w
            cmd.linear.y = -s * vx_w + c * vy_w

            # Turn toward the final heading while driving
            cmd.angular.z = clamp(self.ktheta * yaw_err, -self.w_max, self.w_max)

        elif self.state == self.ROTATE:
            if abs(yaw_err) < self.eps_yaw:
                self.state = self.IDLE
                rospy.loginfo("[CTRL] Goal reached (pos and heading).")
            else:
                cmd.angular.z = clamp(self.ktheta * yaw_err, -self.w_max, self.w_max)

        self.cmd_pub.publish(cmd)


if __name__ == '__main__':
    try:
        ViconGoalController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
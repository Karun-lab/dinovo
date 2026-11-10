#!/usr/bin/env python3
import math, os, sys, yaml, rospy
from geometry_msgs.msg import PoseStamped


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def mean_pose(s):
    n = len(s)
    x = sum(p[0] for p in s) / n
    y = sum(p[1] for p in s) / n
    yaw = math.atan2(sum(math.sin(p[2]) for p in s), sum(math.cos(p[2]) for p in s))
    spread = max(math.hypot(p[0] - x, p[1] - y) for p in s)
    return x, y, yaw, spread


def main():
    rospy.init_node('capture_dock_pose')
    robot_topic = rospy.get_param('~robot_topic', '/vicon/dingo1')
    handle_topic = rospy.get_param('~handle_topic', '/vicon/handle')
    path = os.path.expanduser(rospy.get_param('~config_path', '~/handle_dock.yaml'))
    duration = rospy.get_param('~duration', 2.0)
    yaw_offset = rospy.get_param('~yaw_offset', 0.0)          # must match the docking node
    use_handle_yaw = rospy.get_param('~use_handle_yaw', True)  # store offset in handle frame (True) or world frame (False)

    robot, handle = [], []
    rospy.Subscriber(robot_topic, PoseStamped, lambda m: robot.append(
        (m.pose.position.x, m.pose.position.y, wrap(yaw_of(m.pose.orientation) + yaw_offset))))
    rospy.Subscriber(handle_topic, PoseStamped, lambda m: handle.append(
        (m.pose.position.x, m.pose.position.y, yaw_of(m.pose.orientation))))

    rospy.loginfo("Sampling for %.1f s - keep the robot and cabinet still...", duration)
    rospy.sleep(duration)

    if len(robot) < 5 or len(handle) < 5:
        rospy.logerr("Not enough Vicon data (robot: %d, handle: %d samples).", len(robot), len(handle))
        sys.exit(1)

    rx, ry, ryaw, rspread = mean_pose(robot)
    hx, hy, hyaw, hspread = mean_pose(handle)
    if max(rspread, hspread) > 0.01:
        rospy.logwarn("Noisy data (spread: robot %.1f mm, handle %.1f mm). Consider re-running.",
                      rspread * 1000, hspread * 1000)

    dx, dy = rx - hx, ry - hy
    if use_handle_yaw:
        c, s = math.cos(-hyaw), math.sin(-hyaw)
        rel = (c * dx - s * dy, s * dx + c * dy, wrap(ryaw - hyaw))
    else:
        rel = (dx, dy, ryaw)

    cfg = {'use_handle_yaw': bool(use_handle_yaw),
           'dock': {'x': float(rel[0]), 'y': float(rel[1]), 'yaw': float(rel[2])}}
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)

    rospy.loginfo("Saved %s: x=%.3f y=%.3f yaw=%.1f deg (%s frame); marker distance %.3f m",
                  path, rel[0], rel[1], math.degrees(rel[2]),
                  'handle' if use_handle_yaw else 'world', math.hypot(dx, dy))


if __name__ == '__main__':
    main()
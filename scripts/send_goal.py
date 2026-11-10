#!/usr/bin/env python3
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf.transformations as tft

def send_nav_goal(x, y, yaw):
    # Create an action client for move_base
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base server")

    # Create a goal message
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    
    # Set position
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.position.z = 0.0
    
    # Convert yaw (rotation around Z) to quaternion
    quaternion = tft.quaternion_from_euler(0, 0, yaw)
    goal.target_pose.pose.orientation.x = quaternion[0]
    goal.target_pose.pose.orientation.y = quaternion[1]
    goal.target_pose.pose.orientation.z = quaternion[2]
    goal.target_pose.pose.orientation.w = quaternion[3]

    rospy.loginfo("Sending goal: x=%.2f, y=%.2f, yaw=%.2f", x, y, yaw)
    client.send_goal(goal)
    
    # Wait for the result (optional, but good for feedback)
    client.wait_for_result()
    
    state = client.get_state()
    if state == 3: # 3 corresponds to SUCCEEDED
        rospy.loginfo("Goal reached successfully!")
    else:
        rospy.logwarn("Goal failed with state: %d", state)

if __name__ == '__main__':
    try:
        rospy.init_node('dinovo_nav_goal_sender')
        # --- EDIT THESE COORDINATES TO A FREE SPACE ON YOUR MAP ---
        send_nav_goal(1.0, 0.5, 0.0) 
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation goal sender stopped.")
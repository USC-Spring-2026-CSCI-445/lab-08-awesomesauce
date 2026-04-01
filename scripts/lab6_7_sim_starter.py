#!/usr/bin/env python3
from typing import Optional, Tuple, List, Dict
from argparse import ArgumentParser
from math import radians, inf, sqrt, atan2, pi, isinf, cos, sin, degrees, hypot
from time import sleep, time
import queue

import rospy
from geometry_msgs.msg import Twist, Point32, Vector3, Quaternion, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud, ChannelFloat32
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion
from std_msgs.msg import ColorRGBA

MAX_ROT_VEL = 2.84
MAX_LIN_VEL = 0.22


OBS_FREE_WAYPOINTS = [
    {"x": 1, "y": 1},
    {"x": 2, "y": 1},
    {"x": 1, "y": 0},
]

W_OBS_WAYPOINTS = [
    {"x": 2, "y": 2},
    {"x": 4, "y": 1},
    {"x": 0, "y": 3.0},
]


def angle_to_0_to_2pi(angle: float) -> float:
    while angle < 0:
        angle += 2 * pi
    while angle > 2 * pi:
        angle -= 2 * pi
    return angle


def map_to_new_range(x: float, a_low: float, a_high: float, b_low: float, b_high: float):
    """Helper function to map a value from range [a_low, a_high] to [b_low, b_high]"""
    y = (x - a_low) / (a_high - a_low) * (b_high - b_low) + b_low
    return y

# AW: obstacle avoidance copy
class PDController:
    """
    Generates control action taking into account instantaneous error (proportional action)
    and rate of change of error (derivative action).
    """

    def __init__(self, kP, kD, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        self.kP = kP
        self.kD = kD
        self.u_min = u_min
        self.u_max = u_max
        self.t_prev = None
        self.e_prev = 0.0

    def clamp(self, output):
        if output < self.u_min:
            return self.u_min
        elif output > self.u_max:
            return self.u_max
        else:
            return output

    def control(self, err, t):
        if (self.t_prev is None):
            self.t_prev = t
            return 0
        
        dt = t - self.t_prev
        self.t_prev = t

        if dt <= rospy.Duration.from_sec(1e-10):
            return 0

        de = err - self.e_prev
        output = (self.kP * err) + (self.kD * (de/dt.to_sec()))
        return self.clamp(output)

class PIDController:
    """
    Generates control action taking into account instantaneous error (proportional action),
    accumulated error (integral action) and rate of change of error (derivative action).
    """
    def __init__(self, kP, kD, kI, i_min, i_max, u_min, u_max):
        assert u_min < u_max, "u_min should be less than u_max"
        assert i_min < i_max, "i_min should be less than i_max"

        self.p = 0.0
        self.i = 0.0
        self.d = 0.0

        self.kP = kP
        self.kD = kD
        self.kI = kI

        self.i_min = i_min
        self.i_max = i_max
        self.u_min = u_min
        self.u_max = u_max

        self.t_prev = None
        self.e_prev = 0.0

    def clamp(self, raw, floor, ceil):
        return floor if raw < floor else (ceil if raw > ceil else raw)

    def control(self, err, t):
        if (self.t_prev is None):
            self.t_prev = t
            return 0

        dt = t - self.t_prev
        self.t_prev = t

        if dt <= rospy.Duration.from_sec(1e-10):
            return 0

        de = err - self.e_prev
        dt = dt.to_sec()
        self.e_prev = err

        self.p = self.kP * err
        self.i += self.kI * (err * dt)        
        self.i = self.clamp(self.i, self.i_min, self.i_max)
        self.d = self.kD * (de/dt)

        output = self.p + self.i + self.d
        return self.clamp(output, self.u_min, self.u_max)

def publish_waypoints(waypoints: List[Dict], publisher: rospy.Publisher):
    marker_array = MarkerArray()
    for i, waypoint in enumerate(waypoints):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "waypoints"
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position = Point(waypoint["x"], waypoint["y"], 0.0)
        marker.pose.orientation = Quaternion(0, 0, 0, 1)
        marker.scale = Vector3(0.1, 0.1, 0.1)
        marker.color = ColorRGBA(0.0, 1.0, 0.0, 0.5)
        marker_array.markers.append(marker)
    publisher.publish(marker_array)


class ObstacleFreeWaypointController:
    def __init__(self, waypoints: List[Dict]):
        rospy.init_node("waypoint_follower", anonymous=True)
        self.waypoints = waypoints
        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        publish_waypoints(self.waypoints, self.waypoint_pub)

        self.current_position = None
        self.angular_PID = PIDController(1, 0.2, 0.01, -1, 1, -1 * MAX_ROT_VEL, MAX_ROT_VEL)
        self.linear_PID = PIDController(1, 0.5, 0.00, -0.3, 0.3, -1 * MAX_LIN_VEL, MAX_LIN_VEL)

    def odom_callback(self, msg):
        # Extracting current position from Odometry message
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None
        
        ex = goal_position["x"] - self.current_position["x"]
        ey = goal_position["y"] - self.current_position["y"]
        distance_error = -1 * hypot(ex, ey)
        goal_angle = atan2(ey, ex)
        angle_error = -1 * atan2(sin(goal_angle - self.current_position["theta"]), cos(goal_angle - self.current_position["theta"]))

        if angle_error > pi:
            angle_error -= 2 * pi
        elif angle_error < -pi:
            angle_error += 2 * pi

        return distance_error, angle_error

    def control_robot(self):
        rate = rospy.Rate(20)  # 20 Hz
        ctrl_msg = Twist()

        # initialize first waypoint
        current_waypoint_idx = 0
        for waypoint in self.waypoints:
            print("NEXT ITER:")
            print('\t', waypoint)
            self.current_position = None
            while not rospy.is_shutdown():
                errs = self.calculate_error(waypoint)
                if (errs is not None):
                    distance_error, angle_error = errs
                    u = -1 * self.angular_PID.control(angle_error, rospy.get_rostime())
                    ctrl_msg.angular.z = u
                    print("ang", angle_error, u)

                    if (angle_error < 0.2 and angle_error > -0.2):
                        v = -1 * self.linear_PID.control(distance_error, rospy.get_rostime())
                        ctrl_msg.linear.x = v
                        print("lin", distance_error, v)
                    else:
                        ctrl_msg.linear.x = 0

                    if abs(distance_error) < 0.05:
                        ctrl_msg.linear.x = 0
                        ctrl_msg.angular.z = 0
                        print("\nWAYPOINT REACHED\n")
                        break

                self.robot_ctrl_pub.publish(ctrl_msg)
                rate.sleep()
        ctrl_msg.linear.x = 0
        ctrl_msg.angular.z = 0
        self.robot_ctrl_pub.publish(ctrl_msg)
        print("DONE")



class ObstacleAvoidingWaypointController:
    def __init__(self, waypoints: List[Dict]):
        rospy.init_node("waypoint_follower", anonymous=True)
        self.waypoints = waypoints

        self.current_waypoint_idx = 0
        self.current_position = None
        self.laserscan: Optional[LaserScan] = None
        self.laserscan_angles: Optional[List[float]] = None
        self.ir_distance = None
        self.wall_following_desired_distance = 0.7  # set this to whatever you want

        self.current_position = None
        self.angular_point_PID = PIDController(1, 0.2, 0.01, -1, 1, -1 * MAX_ROT_VEL, MAX_ROT_VEL)
        self.linear_point_PID = PIDController(1, 0.5, 0.00, -0.3, 0.3, -1 * MAX_LIN_VEL, MAX_LIN_VEL)

        self.angular_obstacle_PID = PIDController(0.5, 0.1, 0.3, -1, 1, -1 * MAX_ROT_VEL, MAX_ROT_VEL)
        self.linear_obstacle_PID = PIDController(1, 0.1, 0.5, -0.3, 0.3, -1 * MAX_LIN_VEL, MAX_LIN_VEL)

        # Subscriber to the robot's current position (assuming you have Odometry data)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.laserscan_sub = rospy.Subscriber("/scan", LaserScan, self.robot_laserscan_callback)

        self.robot_ctrl_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.waypoint_pub = rospy.Publisher("/waypoints", MarkerArray, queue_size=10)
        self.pointcloud_pub = rospy.Publisher("/scan_pointcloud", PointCloud, queue_size=10)

        sleep(0.5)  # sleep to give time for rviz to subscribe to /waypoints
        publish_waypoints(self.waypoints, self.waypoint_pub)

        # Add PID controllers here for obstacle avoidance and waypoint following
        ######### Your code starts here #########

        ######### Your code ends here #########

    def calculate_error(self, goal_position: Dict) -> Optional[Tuple]:
        """Return distance and angle error between the current position and the provided goal_position. Returns None if
        the current position is not available.
        """
        if self.current_position is None:
            return None
        
        ex = goal_position["x"] - self.current_position["x"]
        ey = goal_position["y"] - self.current_position["y"]
        distance_error = -1 * hypot(ex, ey)
        goal_angle = atan2(ey, ex)
        angle_error = -1 * atan2(sin(goal_angle - self.current_position["theta"]), cos(goal_angle - self.current_position["theta"]))

        if angle_error > pi:
            angle_error -= 2 * pi
        elif angle_error < -pi:
            angle_error += 2 * pi

        return distance_error, angle_error


    def robot_laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg
        if self.laserscan_angles is None:
            self.laserscan_angles = [
                self.laserscan.angle_min + i * self.laserscan.angle_increment for i in range(len(self.laserscan.ranges))
            ]
            # sanity check the angles
            assert (abs(self.laserscan.angle_min) < 1e-4) and (abs(self.laserscan_angles[0]) < 1e-4)
            assert abs(self.laserscan.angle_max - 2 * pi) < 1e-4 and (abs(self.laserscan_angles[-1] - 2 * pi) < 1e-4)

        left = msg.ranges[80:100]
        left = [x for x in left if x != inf]
        if len(left) > 0:
            self.ir_distance = sum(left) / len(left)
        else:
            self.ir_distance = None

    def odom_callback(self, msg):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        self.current_position = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

    def waypoint_tracking_control(self, goal_position: Dict):
        ctrl_msg = Twist()

        waypoint = goal_position
        errs = self.calculate_error(waypoint)
        if (errs is not None):
            distance_error, angle_error = errs
            u = -1 * self.angular_point_PID.control(angle_error, rospy.get_rostime())
            ctrl_msg.angular.z = u

            if (angle_error < 0.5 and angle_error > -0.5):
                v = -1 * self.linear_point_PID.control(distance_error, rospy.get_rostime())
                ctrl_msg.linear.x = v
                print("lin", distance_error, v)
            else:
                print("ang", angle_error, u)
                ctrl_msg.linear.x = 0

            if abs(distance_error) < 0.2:
                ctrl_msg.linear.x = 0
                ctrl_msg.angular.z = 0
                print("WAYPOINT REACHED")
                self.current_waypoint_idx += 1
        self.robot_ctrl_pub.publish(ctrl_msg)

        # rospy.loginfo(
        #     f"POINT: {distance_error:.2f}\tangle error: {angle_error:.2f}"
        # )

    def obstacle_avoiding_control(self, visualize: bool = True):

        ctrl_msg = Twist()

        ######### Your code starts here #########

        if self.ir_distance is None:
            print("Waiting for IR sensor readings")
            ctrl_msg.angular.z = -1
            self.robot_ctrl_pub.publish(ctrl_msg)
            sleep(0.1)
            return

        err = self.wall_following_desired_distance - self.ir_distance

        # using PD controller, compute and send motor commands
        u = self.angular_obstacle_PID.control(err, rospy.get_rostime())
        ctrl_msg.angular.z = -1 * u

        v = self.linear_obstacle_PID.control(abs(err), rospy.get_rostime())
        ctrl_msg.linear.x = MAX_LIN_VEL - u


        ######### Your code ends here #########

        self.robot_ctrl_pub.publish(ctrl_msg)
        print(
            f"AVOID: {round(self.ir_distance, 4)}\ttgt: {round(self.wall_following_desired_distance, 4)}\tu: {round(u, 4)}"
        )

    def laserscan_distances_to_point(self, point: Dict, cone_angle: float, visualize: bool = False):
        """Returns the laserscan distances within the cone of angle `cone_angle` centered about the line pointing from
        the robots current position to the given point. Angles are in radians.

        Notes:
            1. Distances that are outside of the laserscan's minimum and maximum range are filterered out
        """
        curr_pos = self.current_position
        # angle to point in the local frame. this is the same as the lidar frame. Not neccessarily in [-pi, pi] because
        # of the theta subtraction
        angle_to_point_local = angle_to_0_to_2pi(
            atan2(point["y"] - curr_pos["y"], point["x"] - curr_pos["x"]) - curr_pos["theta"]
        )
        angle_low = angle_to_0_to_2pi(angle_to_point_local - cone_angle)
        angle_high = angle_to_0_to_2pi(angle_to_point_local + cone_angle)

        # This is the so called 'danger zone', because either the high or low angle has wrapped around. For example,
        # when low = 355 deg, and high = 20 deg. The solution is to set the low to 0 and use the high when angle is > 0,
        # or set the high to 2*pi and use the low when angle is < 2*pi
        if angle_to_point_local < cone_angle or angle_to_point_local > 2 * pi - cone_angle:
            if angle_to_point_local < cone_angle:
                angle_low = 0
                idx_low = 0
                idx_high = int(
                    map_to_new_range(
                        angle_high, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                    )
                )
            elif angle_to_point_local > 2 * pi - cone_angle:
                angle_high = 2 * pi
                idx_high = len(self.laserscan.ranges) - 1
                idx_low = int(
                    map_to_new_range(
                        angle_low, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                    )
                )
            else:
                assert False, "should not reach here"
        else:
            idx_low = int(
                map_to_new_range(
                    angle_low, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                )
            )
            idx_high = int(
                map_to_new_range(
                    angle_high, self.laserscan.angle_min, self.laserscan.angle_max, 0, len(self.laserscan.ranges)
                )
            )
        assert angle_low < angle_high, f"angle_low: {angle_low}, angle_high: {angle_high}"
        if idx_low > idx_high:
            idx_low, idx_high = idx_high, idx_low
        assert idx_low < idx_high, f"idx_low: {idx_low}, idx_high: {idx_high}"

        raw = self.laserscan.ranges[idx_low:idx_high]
        filtered = [r for r in raw if (r > self.laserscan.range_min and r < self.laserscan.range_max)]

        if visualize:
            # raw should include all ranges, even if they are inf, in the specified cone
            #   i.e. something like `raw = self.laserscan.ranges[idx_low:idx_high]`
            # `angle_low` and `angle_high` are the angles in the robots local frame
            pcd = PointCloud()
            pcd.header.frame_id = "odom"
            pcd.header.stamp = rospy.Time.now()
            for i, p in enumerate(raw):
                if isinf(p):
                    continue
                angle_local = map_to_new_range(i, 0, len(raw), angle_low, angle_high)
                angle = angle_local + curr_pos["theta"]
                x = p * cos(angle) + curr_pos["x"]
                y = p * sin(angle) + curr_pos["y"]
                z = 0.1
                pcd.points.append(Point32(x=x, y=y, z=z))
                pcd.channels.append(ChannelFloat32(name="rgb", values=(0.0, 1.0, 0.0)))
            self.pointcloud_pub.publish(pcd)
        return filtered

    def control_robot(self):
        rate = rospy.Rate(10)  # 20 Hz

        self.current_waypoint_idx = 0
        distance_from_wall_safety = 0.5
        cone_angle = radians(5)

        while not rospy.is_shutdown():

            if self.current_position is None or self.laserscan is None:
                sleep(0.01)
                continue

            # Travel through waypoints, checking if there is an obstacle in the way. Transition to obstacle avoidance if necessary
            ######### Your code starts here #########

            # laserscan_etcetcetc() returns a range of datapoints/distances... how to best handle this?
            dist = self.laserscan_distances_to_point(self.waypoints[self.current_waypoint_idx], pi/10) # pi/6 = 30 degrees
            
            # if any angle returns closer than acceptable distance, switch to object avoidance mode
            was_avoiding_shit = False
            avoid_shit = False
            for d in dist:
                if d < distance_from_wall_safety:
                    if not was_avoiding_shit:
                        ctrl_msg = Twist()
                        ctrl_msg.linear.x = 0
                        ctrl_msg.angular.z = 0
                        self.robot_ctrl_pub.publish(ctrl_msg)
                    avoid_shit = True
                    break

            
            if avoid_shit:
                was_avoiding_shit = True
                self.obstacle_avoiding_control()
            else:
                was_avoiding_shit = False
                self.waypoint_tracking_control(self.waypoints[self.current_waypoint_idx])

            ######### Your code ends here #########
            rate.sleep()


""" Example usage

rosrun development lab6_7_sim.py --mode obstacle_free
rosrun development lab6_7_sim.py --mode obstacle_avoiding
"""


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, help="Mode of operation: 'obstacle_free' or 'obstacle_avoiding'"
    )
    args = parser.parse_args()
    assert args.mode in {"obstacle_free", "obstacle_avoiding"}

    if args.mode == "obstacle_free":
        controller = ObstacleFreeWaypointController(OBS_FREE_WAYPOINTS)
    else:
        controller = ObstacleAvoidingWaypointController(W_OBS_WAYPOINTS)

    try:
        controller.control_robot()
    except rospy.ROSInterruptException:
        print("Shutting down...")

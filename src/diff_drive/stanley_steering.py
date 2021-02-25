"""
Path tracking simulation with Stanley steering control and PID speed control.
author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import cubic_spline_planner
from diff_drive.pose import Pose
from math import pi, sqrt, sin, cos, atan2

class StanleySteering:
    def __init__(self, ax = [0, 1], ay = [0, 1], target_speed = 2.0, k = 0.5, Kp = 1.0, dt = 0.5, L = 0.5842, max_steer = np.radians(30.0), show_animation = True):
        self.k = k  # control gain
        self.Kp = Kp  # speed proportional gain
        self.dt = dt  # [s] time difference
        self.L = L  # [m] Wheel base of vehicle
        self.max_steer = max_steer  # [rad] max steering angle

        self.show_animation = show_animation

        #  target course
        self.ax = ax
        self.ay = ay

        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            self.ax, self.ay, ds=0.1)

        self.target_speed = target_speed  # [m/s]

        # Initial state
        self.state = Pose()

        self.last_idx = len(self.cx) - 1
        self.time = 0.0
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.theta = [self.state.theta]
        self.v = [self.state.xVel]
        self.t = [0.0]
        self.target_idx, _ = self.calc_target_index()

        # Parameters
        self.max_linear_speed = 1.1
        self.min_linear_speed = 0.1
        self.max_angular_speed = 2.0
        self.min_angular_speed = 1.0
        self.max_linear_acceleration = 1E9
        self.max_angular_acceleration = 1E9
        self.linear_tolerance_outer = 0.3
        self.linear_tolerance_inner = 0.1
        self.angular_tolerance_outer = 0.2
        self.angular_tolerance_inner = 0.1
        self.ignore_angular_tolerance = False
        self.forward_movement_only = False
        self.end_of_path_stop = True
        self.within_linear_tolerance = False
        self.within_angular_tolerance = False

    def set_path(self, ax = [0,1], ay = [0,1], target_speed = 2.0, L = 0.5842, max_steer = np.radians(30.0), show_animation = True):
        self.k = 0  # control gain
        self.Kp = 0  # speed proportional gain
        self.dt = 0  # [s] time difference
        self.L = L  # [m] Wheel base of vehicle
        self.max_steer = max_steer  # [rad] max steering angle

        self.show_animation = show_animation

        #  target course
        self.ax = ax
        self.ay = ay

        self.cx, self.cy, self.cyaw, ck, s = cubic_spline_planner.calc_spline_course(
            self.ax, self.ay, ds=0.1)

        self.target_speed = target_speed  # [m/s]

         # Initial state
        self.state = Pose()

        self.last_idx = len(self.cx) - 1
        self.time = 0.0
        self.x = [self.state.x]
        self.y = [self.state.y]
        self.theta = [self.state.theta]
        self.v = [self.state.xVel]
        self.t = [0.0]
        self.target_idx, _ = self.calc_target_index()

    # Parameters
    def set_constants(self, Kp, k, blank):
        self.Kp = Kp
        self.k = k

    def set_max_linear_speed(self, speed):
        self.max_linear_speed = speed

    def set_min_linear_speed(self, speed):
        self.min_linear_speed = speed

    def set_max_angular_speed(self, speed):
        self.max_angular_speed = speed

    def set_min_angular_speed(self, speed):
        self.min_angular_speed = speed

    def set_max_linear_acceleration(self, accel):
        self.max_linear_acceleration = accel

    def set_max_angular_acceleration(self, accel):
        self.max_angular_acceleration = accel

    def set_linear_tolerance_outer(self, tolerance):
        self.linear_tolerance_outer = tolerance
    
    def set_linear_tolerance_inner(self, tolerance):
        self.linear_tolerance_inner = tolerance

    def set_angular_tolerance_outer(self, tolerance):
        self.angular_tolerance_outer = tolerance

    def set_angular_tolerance_inner(self, tolerance):
        self.angular_tolerance_inner = tolerance
    
    def set_ignore_angular_tolerance(self, ignore):
        self.ignore_angular_tolerance = ignore

    def set_forward_movement_only(self, forward_only):
        self.forward_movement_only = forward_only

    def set_end_of_path_stop(self, end_of_path_stop):
        self.end_of_path_stop = end_of_path_stop
    
    def reset_within_tolerance(self):
        self.within_linear_tolerance = False
        self.within_angular_tolerance = False



    def get_goal_distance(self):
        diffX = self.state.x - self.cx[len(self.cx) -1]
        diffY = self.state.y - self.cy[len(self.cy) -1]
        return sqrt(diffX*diffX + diffY*diffY)

    def at_goal(self):
        d = self.get_goal_distance()

        # Uses hysteresis to get closer to correct position
        if (not self.within_linear_tolerance):
            if(d < self.linear_tolerance_inner):
                self.within_linear_tolerance = True
            else:
                self.within_linear_tolerance = False
        
        # Checks for both linear and angular tolerance
        if (self.within_linear_tolerance):
            self.within_linear_tolerance = False
            self.within_angular_tolerance = False
            return True

        return False

    def pid_control(self, target, current):
        """
        Proportional control for the speed.
        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return self.Kp * (target - current)


    def stanley_control(self):
        """
        Stanley steering control.
        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index()

        if self.target_idx >= current_target_idx:
            current_target_idx = self.target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(self.cyaw[current_target_idx] - self.normalize_angle(self.state.theta))
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, self.state.xVel)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx


    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle


    def calc_target_index(self):
        """
        Compute index in the trajectory list of the target.
        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        # Calc front axle position
        fx = self.state.x + self.L * np.cos(self.normalize_angle(self.state.theta))
        fy = self.state.y + self.L * np.sin(self.normalize_angle(self.state.theta))

        # Search nearest point index
        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.normalize_angle(self.state.theta) + np.pi / 2),
                        -np.sin(self.normalize_angle(self.state.theta) + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle


    def get_velocity(self, pose, vel):
        """Plot an example of Stanley steering control on a cubic spline."""
        desired = Pose()
        self.state.x = pose.x 
        self.state.y = pose.y 
        self.state.theta = pose.theta
        self.state.xVel = vel

        if self.last_idx > self.target_idx:
            ai = self.pid_control(self.target_speed, self.state.xVel)
            di, self.target_idx = self.stanley_control()
            
            desired.xVel = self.state.xVel + ai
            desired.thetaVel = self.state.xVel / self.L * np.tan(di)

            print(desired.thetaVel)
            self.time += self.dt

            self.x.append(self.state.x)
            self.y.append(self.state.y)
            self.theta.append(self.state.theta)
            self.v.append(self.state.xVel)
            self.t.append(self.time)

            # if self.show_animation:  # pragma: no cover
            #     plt.cla()
            #     # for stopping simulation with the esc key.
            #     plt.gcf().canvas.mpl_connect('key_release_event',
            #             lambda event: [exit(0) if event.key == 'escape' else None])
            #     plt.plot(self.cx, self.cy, ".r", label="course")
            #     plt.plot(self.x, self.y, "-b", label="trajectory")
            #     plt.plot(self.cx[self.target_idx], self.cy[self.target_idx], "xg", label="target")
            #     plt.axis("equal")
            #     plt.grid(True)
            #     plt.title("Speed[km/h]:" + str(self.state.xVel * 3.6)[:4])
            #     plt.pause(0.001)
        return desired
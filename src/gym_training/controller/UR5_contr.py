import pathlib
import gymnasium as gym
import numpy as np
from simple_pid import PID
import cv2 as cv
import mujoco 
import math as m


class URController(object):
    """
    Class for controlling the UR5 robot
    """
    def __init__(self):
        path = pathlib.Path(__file__).parent.parent.resolve()
        path = str(path) + "/envs/mesh/ur5.xml"
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.goal_reached = False
        self.dof = 6

    def PID_values(self):
        self.PID_list = []
        # Values for training
        sample_time = 0.0001
        p_scale = 3
        i_scale = 0.0
        d_scale = 0.1

        i_gripper = 0
        self.PID_list.append(
            PID(
                7 * p_scale,
                0.0 * i_scale,
                1.1 * d_scale,
                setpoint=0,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Shoulder Pan Joint
        self.PID_list.append(
            PID(
                10 * p_scale,
                0.0 * i_scale,
                1.0 * d_scale,
                setpoint=-1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Shoulder Lift Joint
        self.PID_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.5 * d_scale,
                setpoint=1.57,
                output_limits=(-2, 2),
                sample_time=sample_time,
            )
        )  # Elbow Joint
        self.PID_list.append(
            PID(
                7 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 1 Joint
        self.PID_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 2 Joint
        self.PID_list.append(
            PID(
                5 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Wrist 3 Joint
        self.controller_list.append(
            PID(
                2.5 * p_scale,
                i_gripper,
                0.00 * d_scale,
                setpoint=0.0,
                output_limits=(-1, 1),
                sample_time=sample_time,
            )
        )  # Gripper joints

    def ur5_params(self):
        """
        Returns
        -------
        home_position : Home position of the UR5 with DH parameters
        screw : Screw matrix for UR5 with DH parameters
        """
        # UR5 link parameters (in meters)
        l1 = 0.425
        l2 = 0.392
        h1 = 0.160
        h2 = 0.09475
        w1 = 0.134
        w2 = 0.0815

        home_position = np.array([
        [-1, 0, 0, l1 + l2],
        [0, 0, 1, w1 + w2],
        [0, 1, 0, h1 - h2],
        [0, 0, 0, 1]
        ])

        screw = np.array([
        [0,0,1,0,0,0],
        [0,1,0,-h1,0,0],
        [0,1,0,-h1, 0,l1],
        [0,1,0, -h1, 0, l1+l2],
        [0,0,-1, -w1, l1+l2, 0],
        [0,1,0, h2-h1, 0, l1+l2]
        ])
        return home_position, screw
    
    def crossProductOperator(self, vector):
        w1,w2,w3 = vector
        W = [
            [  0, -w3,  w2],
            [ w3,   0, -w1],
            [-w2,  w1,   0]
            ]
        Ax = np.asarray(W)
        return Ax
    
    def exponential_map(self, action_axis, theta):
        action_axis = np.asarray(action_axis)
        linear = action_axis[3:]
        angular = action_axis[:3]

        exp_rot = self.exponential_form_rotation(angular, theta)
        exp_tras = self.exponential_form_traslation(linear, angular, theta)

        expMap = np.block([[exp_rot, exp_tras], [np.zeros( (1,3) ), 1]])
        return(expMap)

    def exponential_form_rotation(self, angular, theta):
        c = self.crossProductOperator(angular) @ self.crossProductOperator(angular)
        expRot = np.eye(3) + np.sin(theta) * self.crossProductOperator(angular) + ( 1-np.cos(theta) ) * c
        return expRot

    def exponential_form_traslation(self, linear, angular, theta):
        l1, l2, l3 = linear
        lin = np.array([[l1, l2, l3]])
        angular_mat = self.crossProductOperator(angular)
        c = angular_mat  @ angular_mat
        expTras = (theta * np.eye(3) + ( 1 - np.cos(theta) ) * angular_mat + ( theta - np.sin(theta) ) * c) @ (lin.transpose())
        return expTras
    
    def Ad(self, mat4):
        mat4  = np.asarray(mat4)
        rot = mat4[:3, :3]
        tras = mat4[0:3, 3]
        Ad = np.block([[rot, self.crossProductOperator(tras) @ rot],[np.zeros((3,3)), rot]])
        return(Ad)

    def jacobian(self, theta, screw):
        G_local = []
        expMap_local = []
        for i in range(self.dof):
            G_local.append(self.DH_transform(theta[i], i))
            expMap_local.append(G_local[i] @ self.exponential_map(screw[i], theta[i]))

        # Get G for the adjoint operator
        G = []
        for i in range(self.dof):
            if i == 0:
                g = np.eye(4)
            else:
                g = g @ G[i-1]
            G.append(g @ expMap_local[i])

        # Get space Jacobian
        J_s = []
        for i in range(6):
            J_s.append(self.Ad(G[i]) @ screw[i])

        # print('Space : \n', J_s)

        # Get location of end effector tip (p_tip)
        p_k = np.zeros((3,1))
        # p_k = np.array([[0],[0],[0.5]])

        p_0_extended = G[5] @ np.block([[p_k],[1]])
        p_0 = p_0_extended[:3]

        p_0_cpo = np.array([[0, -p_0[2], p_0[1]],[p_0[2], 0, -p_0[0]],[-p_0[1], p_0[0], 0]])

            # Geometric Jacobian
        """

        The geometric Jacobian is obtained from the spatial Jacobian and the vector p_tip

        p_tip : tip point of the end effector
        p_0^tip : p measured from the inertial reference frame
        p_k^tip : p measured from the frame k, if k is the end effector and the tip is at its origin then p_k = 0

        [p_0^tip; 1] = G_0^k [p_k^tip; 1]

        """
        J_g = np.block([[np.eye(3), -p_0_cpo],[np.zeros((3,3)), np.eye(3)]]) @ J_s

        # print('Geometric : \n', J_g)

        # Get Roll, Pitch, Yaw coordinates
        R = G[5][0:3][0:3]

        r_roll = m.atan2(R[2][1],R[2][2])
        r_pitch = m.atan2(-R[2][0],m.sqrt(R[2][2]*R[2][2] + R[2][1]*R[2][1]))
        r_yaw = m.atan2(R[1][0],R[0][0])

        # Build kinematic operator for Roll, Pitch, Yaw configuration
        # Taken from Olguin's formulaire book

        B = np.array(
            [
                [m.cos(r_pitch) * m.cos(r_yaw), -m.sin(r_yaw), 0],
                [m.cos(r_pitch) * m.cos(r_yaw), m.cos(r_yaw), 0],
                [- m.sin(r_pitch), 0, 1]
                ]
            )

        # print('Kinematic : \n', B)

        # Get Analytic Jacobian
        """

        Obtained from function

        J_a(q) = [[I 0],[0, B(alpha)^-1]] J_g(q)

        B(alpha) =    for roll, pitch, yaw

        """
        J_a = np.block(
            [
                [np.eye(3), np.zeros((3, 3))],
                [np.zeros((3,3)), np.linalg.inv(B)]
                ]
            )

        return J_a

    def DH_transform(self, theta, index):
        d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0523]
        alpha = [m.pi/2, 0, 0, m.pi/2, -m.pi/2, 0]
        r = [0, -0.425, -0.3922, 0, 0, 0]

        c_theta = m.cos(theta)
        s_theta = m.sin(theta)
        c_alpha = m.cos(alpha[index])
        s_alpha = m.sin(alpha[index])

        # print('DH Values: ', c_theta, s_theta, c_alpha, s_alpha)

        R_z = np.array([
                        [c_theta, -s_theta, 0, 0],
                        [s_theta, c_theta, 0, 0],
                        [0, 0, 1 ,0],
                        [0, 0, 0, 1]
                        ]) # DH rotation z-axis
        T_z = np.array([
                        [1, 0 ,0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1 , d[index]],
                        [0, 0, 0, 1]
                        ]) # DH translation z-axis
        T_x = np.array([
                        [1, 0, 0, r[index]],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                        ]) # DH translation x-axis
        R_x = np.array([
                        [1, 0, 0, 0],
                        [0, c_alpha, -s_alpha, 0],
                        [0, s_alpha, c_alpha, 0],
                        [0, 0, 0, 1]
                        ]) # DH rotation x-axis

        # print(R_z, T_z, T_x, R_x)

        G = R_z @ T_z @ T_x @ R_x

        return G

    def compute_e(self, theta_d, theta_0, home_position, screw):
        # Compute error
        T_0 = self.compute_T(screw, theta_0, home_position)
        T_d = self.compute_T(screw, theta_d, home_position)
        e = T_d - T_0

        # e = theta_d - theta_0
        return e

    def root_finding(self, theta_0, theta_d, tryMax, home_position, screw):
        n_try = 1; # Count number of iterations
        tol = 0.0001; # error tolerance
        theta = theta_0
        e = self.compute_e(theta_d, theta, home_position, screw) # Gets error from the transformation matrix

        while n_try < tryMax and np.linalg.norm(e) > tol :
            ja = self.jacobian(theta, screw)
            j_temp = np.zeros((6,6))
            for i in range(6):
                for j in range (6):
                    j_temp[i][j] = ja[i][j]

            inverse_jacobian = np.linalg.inv(j_temp)
            theta = theta + inverse_jacobian @ (theta_d - theta)
            e = self.compute_e(theta_d, theta, home_position, screw)
            n_try += 1
        return theta, e, n_try

    def compute_T(self, screw, theta,  M):
        # Compute new end effector position.
        expMap_local = []
        T = np.eye(4)
        for i in range(self.dof):
            expMap_local.append(self.exponential_map(screw[i], theta[i]))
            T = T @ expMap_local[i]
        T = T @ M
        return T
    
    def get_joint(self):
    # Get current coordinates of each joint
        UR5_q = self.data.qpos[:6]
        return UR5_q

    def move_joints(self, theta):
        # Return needed torque to reach goal
        tolerance = 100
        while self.goal_reached == False:
            self.data.ctrl[:6] = theta
            # if self.data.ctrl is not None:
            #     for i in range(theta.shape[i]):
            #         self.data.ctrl[i] = theta[i]
            #         #self.data.ctrl[i] = self.data.qpos[i] + action[i]
            if np.abs(np.sum(np.subtract(self.get_joint(), theta))) < tolerance:
                self.goal_reached == True
        return self.goal_reached

    def stay(self):
        pass


    def grasp(self, grasp=False):
        # Find current value of grippers
        # if grasp=true, a grasp must executed
        pass
    
    def home_pos(self):
        # Go to home pos

        pass

    def inverse_kinematics(self):
        # Convert end-effector pose to joint values
        home_position, screw = self.ur5_params() # Get UR5 parameters

        theta_0 = np.array([0, 0, 0, 0, 0, 0]) # Initial position
        theta_d = np.array([0,-m.pi/2, 0, 0, m.pi/2, 0]) # Desired position, converted to x_d in the solver

        T_0 = self.compute_T(screw, theta_0, home_position)
        T_d = self.compute_T(screw, theta_d, home_position)

        print("Home position: \n", T_0, "\n")
        print("Desired position: \n", T_d, "\n")

        # Find solution to the system
        theta, delta, n = self.root_finding(theta_0, theta_d, 20, home_position, screw)
        T_theta = self.compute_T(screw, theta, home_position)

        print('Solution : \n', theta, '\n', 'Transformation : \n', T_theta, '\n')

        R = T_theta[0:3][0:3] # Get RPY for the solution

        r_roll = m.atan2(R[2][1],R[2][2])
        r_pitch = m.atan2(-R[2][0],m.sqrt(R[2][2]*R[2][2] + R[2][1]*R[2][1]))
        r_yaw = m.atan2(R[1][0],R[0][0])

        self.get_joint()
        # Set new coordinates for the robot in Mujoco
        goal_success = self.move_joints(theta)

        return goal_success

#!/usr/bin/env python3

from collections import defaultdict
import os
from pathlib import Path
import mujoco
import time
import numpy as np
from simple_pid import PID
from termcolor import colored
from ikpy import chain
from pyquaternion import Quaternion
#import cv2 as cv
import matplotlib.pyplot as plt
import copy
#import cv2
import mujoco.viewer

class MJ_Controller(object):
    """
    Class for control of an robotic arm in MuJoCo .
    It can be used on its own, in which case a new model, simulation and Renderer will be created.
    It can also be passed these objects when creating an instance, in which case the class can be used
    to perform tasks on an already instantiated simulation.
    """

    def __init__(self, model=None, data=None, mujoco_renderer=None):
        path = os.path.realpath(__file__)
        path = str(Path(path).parent.parent.parent)
        if model is None:
            self.model = mujoco.MjModel.from_xml_path(path + "/gym_training/envs/mesh/ur5_no_noise.xml")
        else:
            self.model = model
        if data is None:
            self.data = mujoco.MjData(self.model)
        else:
            self.data = data
        if mujoco_renderer is None:
            self.mujoco_renderer = mujoco_renderer
        else:
            self.mujoco_renderer = mujoco_renderer

        self.create_lists()
        self.groups = defaultdict(list)
        self.groups["All"] = list(range(len(self.data.ctrl)))
        self.create_group("Arm", list(range(4)))
        self.actuated_joint_ids = np.array([i[2] for i in self.actuators])
        self.reached_target = False
        self.current_output = np.zeros(len(self.data.ctrl))
        self.image_counter = 0
        self.cam_matrix = None
        self.cam_init = False
        self.last_movement_steps = 0

    def create_group(self, group_name, idx_list):
        """
        Allows the user to create custom objects for controlling groups of joints.
        The method show_model_info can be used to get lists of joints and actuators.
        Args:
            group_name: String defining the désired name of the group.
            idx_list: List containing the IDs of the actuators that will belong to this group.
        """

        try:
            assert len(idx_list) <= len(self.data.ctrl), "Too many joints specified!"
            assert (
                group_name not in self.groups.keys()
            ), "A group with name {} already exists!".format(group_name)
            assert np.max(idx_list) <= len(
                self.data.ctrl
            ), "List contains invalid actuator ID (too high)"

            self.groups[group_name] = idx_list
            print("Created new control group '{}'.".format(group_name))

        except Exception as e:
            print(e)
            print("Could not create a new group.")

    def show_model_info(self):
        """
        Displays relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges.
        Also gives info on which actuators control which joints and which joints are included in the kinematic chain,
        as well as the PID controller info for each actuator.
        """

        print("\nNumber of bodies: {}".format(self.model.nbody))
        for i in range(self.model.nbody):
            print("Body ID: {}, Body Name: {}".format(i, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)))

        print("\nNumber of joints: {}".format(self.model.njnt))
        for i in range(self.model.njnt):
            print(
                "Joint ID: {}, Joint Name: {}, Limits[RAD]: {}".format(
                    i, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i), self.model.jnt_range[i]
                )
            )

        print("\nNumber of Actuators: {}".format(len(self.data.ctrl)))
        for i in range(len(self.data.ctrl)):
            print(
                "Actuator ID: {}, Actuator Name: {}, Controlled Joint: {}, Control Range[Nm]: {}".format(
                    i,
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                    self.actuators[i][3],
                    self.model.actuator_ctrlrange[i],
                )
            )

        print("\nJoints in kinematic chain: {}".format([i.name for i in self.ee_chain.links]))

        print("\nPID Info: \n")
        for i in range(len(self.actuators)):
            print(
                "{}: P: {}, I: {}, D: {}, setpoint: {}, sample_time: {}".format(
                    self.actuators[i][3],
                    self.actuators[i][4].tunings[0],
                    self.actuators[i][4].tunings[1],
                    self.actuators[i][4].tunings[2],
                    self.actuators[i][4].setpoint,
                    self.actuators[i][4].sample_time,
                )
            )

    def create_lists(self):
        """
        Creates some basic lists and fill them with initial values. This function is called in the class costructor.
        The following lists/dictionaries are created:
        - controller_list: Contains a controller for each of the actuated joints. This is done so that different gains may be
        specified for each controller.
        - current_joint_value_targets: Same as the current setpoints for all controllers, created for convenience.
        - current_output = A list containing the ouput values of all the controllers. This list is only initiated here, its
        values are overwritten at the first simulation step.
        - actuators: 2D list, each entry represents one actuator and contains:
            0 actuator ID
            1 actuator name
            2 joint ID of the joint controlled by this actuator
            3 joint name
            4 controller for controlling the actuator
        """

        self.controller_list = []

        # Values for training
        sample_time = 0.002
        # p_scale = 1
        p_scale = 150
        i_scale = 1
        d_scale = 3000

        # Shoulder Pan Joint
        self.controller_list.append(
            PID(
                20 * p_scale,
                0.0 * i_scale,
                0.7 * d_scale,
                setpoint=0,
                output_limits=(self.model.actuator_ctrlrange[0][0], self.model.actuator_ctrlrange[0][1]),
                sample_time=sample_time,
            )
        )
        # Shoulder Lift Joint
        self.controller_list.append(
            PID(
                29 * p_scale,
                0.0 * i_scale,
                0.7 * d_scale,
                setpoint=-1.57,
                output_limits=(self.model.actuator_ctrlrange[1][0], self.model.actuator_ctrlrange[1][1]),
                sample_time=sample_time,
            )
        )
        # Elbow Joint
        self.controller_list.append(
            PID(
                26 * p_scale,
                0.0 * i_scale,
                0.15 * d_scale,
                setpoint=1.57,
                output_limits=(self.model.actuator_ctrlrange[2][0], self.model.actuator_ctrlrange[2][1]),
                sample_time=sample_time,
            )
        )
        # Wrist 1 Joint
        self.controller_list.append(
            PID(
                27 * p_scale,
                0.0 * i_scale,
                0.1 * d_scale,
                setpoint=-1.57,
                output_limits=(self.model.actuator_ctrlrange[3][0], self.model.actuator_ctrlrange[3][1]),
                sample_time=sample_time,
            )
        )

        self.current_target_joint_values = [
            self.controller_list[i].setpoint for i in range(len(self.data.ctrl))
        ]

        self.current_target_joint_values = np.array(self.current_target_joint_values)

        self.current_output = [controller(0) for controller in self.controller_list]
        self.actuators = []
        for i in range(len(self.data.ctrl)):
            item = [i, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR , i)]
            item.append(self.model.actuator_trnid[i][0])
            item.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT , self.model.actuator_trnid[i][0]))
            item.append(self.controller_list[i])
            self.actuators.append(item)

    def actuate_joint_group(self, group, motor_values):
        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            assert len(motor_values) == len(
                self.groups[group]
            ), "Invalid number of actuator values!"
            for i, v in enumerate(self.groups[group]):
                self.data.ctrl[v] = motor_values[i]

        except Exception as e:
            print(e)
            print("Could not actuate requested joint group.")

    def move_group_to_joint_target(
        self,
        group="All",
        target=None,
        tolerance=0.02,
        max_steps=250,
        render=True,
        quiet=False
    ):
        """
        Moves the specified joint group to a joint target.
        Args:
            group: String specifying the group to move.
            target: List of target joint values for the group.
            tolerance: Threshold within which the error of each joint must be before the method finishes.
            max_steps: maximum number of steps to actuate before breaking
            plot: If True, a .png image of the group joint trajectories will be saved to the local directory.
                  This can be used for PID tuning in case of overshoot etc. The name of the file will be "Joint_angles_" + a number.
            marker: If True, a colored visual marker will be added into the scene to visualize the current
                    cartesian target.
        """

        try:
            assert group in self.groups.keys(), "No group with name {} exists!".format(group)
            if target is not None:
                assert len(target) == len(
                    self.groups[group]
                ), "Mismatching target dimensions for group {}! \n Provided joint target lenght: {}. Expected joint target lenght: {}".format(
                    group,
                    len(target),
                    len(self.groups[group]))
            ids = self.groups[group]
            steps = 1
            result = 0
            if plot:
                self.plot_list = defaultdict(list)
            self.reached_target = False
            deltas = np.zeros(len(self.data.ctrl))

            if target is not None:
                for i, v in enumerate(ids):
                    self.current_target_joint_values[v] = target[i]
                    # print('Target joint value: {}: {}'.format(v, self.current_target_joint_values[v]))

            for j in range(len(self.data.ctrl)):
                # Update the setpoints of the relevant controllers for the group
                self.actuators[j][4].setpoint = self.current_target_joint_values[j]
                # print('Setpoint {}: {}'.format(j, self.actuators[j][4].setpoint))

            while not self.reached_target:
                current_joint_values = self.data.qpos[self.actuated_joint_ids]

                # self.get_image_data(width=200, height=200, show=True)

                # We still want to actuate all motors towards their targets, otherwise the joints of non-controlled
                # groups will start to drift
                for j in range(len(self.data.ctrl)):
                    self.current_output[j] = self.actuators[j][4](current_joint_values[j])
                    self.data.ctrl[j] = self.current_output[j]
                for i in ids:
                    deltas[i] = abs(self.current_target_joint_values[i] - current_joint_values[i])
                

                if steps % 1000 == 0 and target is not None and not quiet:
                    print(
                        "Moving group {} to joint target! Max. delta: {}, Joint: {}".format(
                            group, max(deltas), self.actuators[np.argmax(deltas)][3]
                        )
                    )

                if max(deltas) < tolerance:
                    if target is not None and not quiet:
                        print(
                            colored(
                                "Joint values for group {} within requested tolerance! ({} steps)".format(
                                    group, steps
                                ),
                                color="green",
                                attrs=["bold"],
                            )
                        )
                    for delta in deltas:
                        result += 1
                    self.reached_target = True

                if steps > max_steps:
                    if not quiet:
                        print(
                            colored(
                                "Max number of steps reached: {}".format(max_steps),
                                color="red",
                                attrs=["bold"],
                            )
                        )
                        print("Deltas: ", deltas)
                        for delta in deltas:
                            if delta <= tolerance:
                                result += 1
                            else:
                                result -= 1
                    return result

                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                if render:
                    self.mujoco_renderer.render("human")
                steps += 1

            self.last_movement_steps = steps

            if plot:
                self.create_joint_angle_plot(group=group, tolerance=tolerance)

            return result

        except Exception as e:
            print(e)
            print("Could not move to requested joint target.")

    def set_group_joint_target(self, group, target):

        idx = self.groups[group]
        try:
            assert len(target) == len(
                idx
            ), "Length of the target must match the number of actuated joints in the group."
            self.current_target_joint_values[idx] = target

        except Exception as e:
            print(e)
            print(f"Could not set new group joint target for group {group}")

    def display_current_values(self):
        """
        Debug method, simply displays some relevant data at the time of the call.
        """

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ACTUATED)")
        print("################################################")
        for i in range(len(self.actuated_joint_ids)):
            print(
                "Current angle for joint {}: {}".format(
                    self.actuators[i][3], self.data.qpos[self.actuated_joint_ids][i]
                )
            )

        print("\n################################################")
        print("CURRENT JOINT POSITIONS (ALL)")
        print("################################################")
        for i in range(len(self.model.jnt_qposadr)):
            # for i in range(self.model.njnt):
            name = self.model.joint_id2name(i)
            print("Current angle for joint {}: {}".format(name, self.data.get_joint_qpos(name)))
            # print('Current angle for joint {}: {}'.format(self.model.joint_id2name(i), self.data.qpos[i]))

        print("\n################################################")
        print("CURRENT BODY POSITIONS")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current position for body {}: {}".format(
                    self.model.body_id2name(i), self.data.geom_xpos[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION MATRIZES")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.data.body_xmat[i]
                )
            )

        print("\n################################################")
        print("CURRENT BODY ROTATION QUATERNIONS (w,x,y,z)")
        print("################################################")
        for i in range(self.model.nbody):
            print(
                "Current rotation for body {}: {}".format(
                    self.model.body_id2name(i), self.data.body_xquat[i]
                )
            )

        print("\n################################################")
        print("CURRENT ACTUATOR CONTROLS")
        print("################################################")
        for i in range(len(self.data.ctrl)):
            print(
                "Current activation of actuator {}: {}".format(
                    self.actuators[i][1], self.data.ctrl[i]
                )
            )

    def stay(self, duration, render=True):
        """
        Holds the current position by actuating the joints towards their current target position.
        Args:
            duration: Time in ms to hold the position.
        """
        starting_time = time.time()
        elapsed = 0
        while elapsed < duration:
            self.move_group_to_joint_target(
                max_steps=10, tolerance=0.0000001, plot=False, quiet=True, render=render
            )
            elapsed = (time.time() - starting_time) * 1000

    @property
    def last_steps(self):
        return self.last_movement_steps
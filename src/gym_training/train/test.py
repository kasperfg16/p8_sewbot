from gym_training.controller.mujoco_controller import MJ_Controller
import numpy as np

# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

target = np.array([2.0371404, 0.5061614, -2.2295423, -2.5810633, 0.11319327, 0.41720057, 0.007])
controller.move_group_to_joint_target(target=target)

from gym_training.controller.mujoco_controller import MJ_Controller
import numpy as np

# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

target = np.array([ 0, 0,  -1.57, -1.57, -1.57,  0.  ,  0.  ])
controller.move_group_to_joint_target(target=target)

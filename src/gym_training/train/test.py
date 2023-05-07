from gym_training.controller.mujoco_py_controller import MJ_Controller

# create controller instance
controller = MJ_Controller()

# Display robot information
controller.show_model_info()

# Throw the object away
controller.toss_it_from_the_ellbow()

# Wait before finishing
controller.stay(2000)

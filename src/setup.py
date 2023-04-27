from setuptools import setup
setup(name='gym_training', 
      version='0.0.1',     
      description='A gym mujoco environment of ur3 w. 2 finger gripper',
      install_requires=['gymnasium ==0.28.1', 
                      'numpy',
                      'mujoco==2.3.3',
                      'skrl'])
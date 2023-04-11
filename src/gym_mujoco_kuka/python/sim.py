import mujoco 
import mujoco_py as mp
import mediapy as media
import matplotlib.pyplot as plt
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List

# def Kinematics(xml):
#     model = mujoco.MjModel.from_xml_string(xml)
#     data = mujoco.MjData(model)
#     # Propegate models
#     mujoco.mj_kinematics(model, data)
#     print('raw access:\n', data.geom_xpos)

#     # MjData also supports named access:
#     print('\nnamed access:\n', data.geom('green_sphere').xpos)

def PullModel(xml):
# Pull model from XML file
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    
    # renderer.update_scene(data)
    # media.show_image(renderer.render())


def SimulateDisplayVideo(model, data, renderer):
    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)

    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)
        media.show_video(frames, fps=framerate)



from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import sys 
sys.path.append("/home/cubox/ochansol/isaac_code/python/sim2real/cal_depth_width")
import Robot_task as Robot_task

from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni
import carb
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.sensors.camera import Camera

import omni.replicator.core as rep
import omni.timeline

import sys
import getpass
sys.path.append(f"/home/{getpass.getuser()}/ochansol/isaac_code/python/sim_4.5_ver/utils")
import json
import os
import scan_rep
import matplotlib.pyplot as plt
import cs_utils as cs
import cs_rep_utils as csr
import light_set as light
import sanjabu_Writer as SW
from pynput import keyboard



object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
root_path = "/nas/ochansol/isaac"
output_path =  "/nas/Dataset/VLA/UON/Isaacsim_OMY"





my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)

stage = omni.usd.get_context().get_stage()




my_robot_task = Robot_task.My_Robot_Task(name="robot_task" )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim

env_prim = stage.GetPrimAtPath(my_robot_task.prim_path)

light_list = csr.find_lights(env_prim)
Lights = light.Light(light_list)
# Lights.random_trans(0.2, [1])
Lights.set_all_exposure(val=1)



#### full = realsense D435i : 1920 * 1080
#### wrist = realsense D405 : 1280 * 720








obj_root_path = "/nas/ochansol/3d_model/scan_etc"
sampled_model_dict={
    "apple":{
        "name":"apple",
        "path": os.path.join(obj_root_path, "apple/edited/apple.usd"),
        "size_rank": 0,
    },
    "paprika":{
        "name":"paprika",
        "path": os.path.join(obj_root_path, "paprika/edited/paprika.usd"),
        "size_rank": 0,
    },
    "potato":{
        "name":"potato",
        "path": os.path.join(obj_root_path, "potato/edited/potato.usd"),
        "size_rank": 0,
    },    
}

obj_rep_all_list = []
for key in sampled_model_dict:
    model_attr = sampled_model_dict[key]
    print("model_attr : ", model_attr["name"])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                            class_name = model_attr["name"],
                            size = model_attr["size_rank"],)
    obj_rep_all_list.append(scan_obj)


for OBJ in obj_rep_all_list:
    print("set collider for : ", OBJ.class_name)
    OBJ.set_rigidbody_collider()
    OBJ.remove_collider()
    # OBJ.set_contact_sensor()
    # OBJ.set_physics_material(
    #     dynamic_friction=0.25,
    #     static_friction=0.4,
    #     restitution=0.0
    # )

physics_scene_conf={
    # 'physxScene:enableGPUDynamics': 1, # True
    # 'physxScene:broadphaseType' : "GPU",
    # 'physxScene:collisionSystem' : "PCM",
    
    # 'physxScene:timeStepsPerSecond' : 1000,
    'physxScene:minPositionIterationCount' : 30,
    'physxScene:minVelocityIterationCount' : 20,
    # "physics:gravityMagnitude":35,
    # "physxScene:updateType":"Asynchronous",
}
for key in physics_scene_conf.keys():
    stage.GetPrimAtPath("/physicsScene").GetAttribute(key).Set(physics_scene_conf[key])
        


platform_area_prims = csr.find_target_name(env_prim,["Mesh"],"platform_area")
platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == "demo"][0]

platform_path = platform_area_prims.GetPath().__str__()
platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])

my_world.reset()

platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_rep.set_tf(platform_tf)
platform_rep.set_scale(platform_scale)

# csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False)
scene_num = 0
with open( os.path.join(output_path, "action", f"{scene_num:04d}.json"), 'r') as f:
    action_data = json.load(f)




i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001
record_flag = False

action_list = []
config = {}

my_world.stop()






action_i = 0
while simulation_app.is_running():
    my_world.step(render=True)





    if my_world.is_stopped() and stop_flag:
        i=0
        state=0
        ik_first_flag=True
        obj_reset_flag = True
        stop_flag = False
        record_flag = False

        my_world.reset()
        my_world.pause()

    if my_world.is_playing():

        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset() 
        i += 1

        data = action_data[action_i]

        my_robot.apply_action(ArticulationAction(
                        joint_positions = data["robot"]["joint_positions"]) )
        for OBJ in obj_rep_all_list:
            pos_x,pos_y,pos_z = data["objects"][OBJ.class_name]["position"]
            rot_quat = data["objects"][OBJ.class_name]["orientation"]
            rot_euler = csr.rot_utils.quat_to_euler_angles(rot_quat, degrees= True)
            r,p,y = rot_euler
            w,x,y,z = rot_quat
            OBJ.prim.GetAttribute('xformOp:translate').Set(csr.Gf.Vec3d( tuple((pos_x, pos_y, pos_z))    ))
            if OBJ.prim.HasAttribute('xformOp:rotateXYZ'):
                OBJ.prim.GetAttribute('xformOp:rotateXYZ').Set(csr.Gf.Vec3d(r,p,y))
            elif OBJ.prim.HasAttribute('xformOp:orient'):
                OBJ.prim.GetAttribute('xformOp:orient').Set( csr.np_to_GfQuatf([w,x,y,z]) )

        action_i += 1
        if action_i >= len(action_data):
            action_i = 0



simulation_app.close()
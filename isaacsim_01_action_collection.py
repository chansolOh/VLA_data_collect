
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np
import sys 
sys.path.append("/home/uon/ochansol/isaac_code/python/VLA_data_collect")
import Robot_task as Robot_task

from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni
import carb
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.sensors.camera import Camera

import omni.replicator.core as rep
import omni.timeline

import json
import os
import matplotlib.pyplot as plt

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.kit.xr.core")
enable_extension("omni.kit.xr.system.openxr")
enable_extension("isaacsim.xr.openxr")

import sys
sys.path.append("/home/uon/ochansol/isaac_code/isaac_chansol")
import Utils.isaac_utils_51.rep_utils as csr
import Utils.isaac_utils_51.scan_rep as scan_rep
import Utils.isaac_utils_51.light_set as light
import Utils.isaac_utils_51.sanjabu_Writer as SW
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


from pxr import Usd, UsdShade, Gf, Sdf
record_viz_cube_path = f"{env_prim.GetPath().__str__()}/record_viz_cube"
save_viz_cube_path = f"{env_prim.GetPath().__str__()}/save_viz_cube"

record_viz_cube = stage.GetPrimAtPath(record_viz_cube_path)
save_viz_cube = stage.GetPrimAtPath(save_viz_cube_path)



record_shader_path = f"{record_viz_cube_path}/Material/Shader"  # 실제 shader 경로
record_shader = UsdShade.Shader.Get(stage, record_shader_path)

save_shader_path = f"{save_viz_cube_path}/Material/Shader"  # 실제 shader 경로
save_shader = UsdShade.Shader.Get(stage, save_shader_path)

record_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(0.0, 0.0, 0.0))
save_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(0.0, 1.0, 0.0))

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

box_path_list = [os.path.join(env_prim.GetPath().__str__(),i) for i in ["custom_box_12_12_08_blue", "custom_box_12_12_08_yellow","custom_box_12_12_08_magenta"]]
box_rep_list = []
for box_path in box_path_list:
    box_rep = scan_rep.Scan_Rep(
        prim_path = box_path,
        class_name = box_path.split("/")[-1],
        )
    box_rep_list.append(box_rep)



obj_rep_all_list = [] + box_rep_list
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
    # OBJ.set_contact_sensor()
    OBJ.set_physics_material(
        dynamic_friction=0.25,
        static_friction=0.4,
        restitution=0.0
    )

physics_scene_conf={
    # 'physxScene:enableGPUDynamics': 1, # True
    # 'physxScene:broadphaseType' : "GPU",
    # 'physxScene:collisionSystem' : "PCM",
    
    # 'physxScene:timeStepsPerSecond' : 1000,
    'physxScene:minPositionIterationCount' : 5,
    'physxScene:minVelocityIterationCount' : 5,
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

csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False, rotation=False)


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

# my_world.stop()

def check_eps_num(output_path):
    action_eps_list = sorted([int(i.strip(".json")) for i in os.listdir(os.path.join(output_path, 'action')) if i.endswith('.json')])
    for idx, num in enumerate(action_eps_list):
        if idx != num:
            return idx
    return len(action_eps_list)

os.makedirs( os.path.join( output_path, "action"), exist_ok=True)
episode_num = check_eps_num(output_path)
    
start_current_time = 0
index = 0


pressed = set()
key_flag = True
def on_press(key):
    try:
        pressed.add(key.char)
    except:
        pass

def on_release(key):
    global key_flag
    try:
        pressed.discard(key.char)
        key_flag = True
    except:
        pass

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()


def action_collect():
    global action_list, index 
    current_time = my_world.current_time - start_current_time
    # print("recording... time : ", round(current_time,5))
    obj_conf = {}

    for OBJ in obj_rep_all_list:
        pose = OBJ.get_world_pose()
        obj_conf[OBJ.class_name] = {
            "position" : pose["translation"],
            "orientation" : pose["rotation"],
        }

    action_list.append({
        "index" : index,
        "time": round(current_time,5),
        "robot":{
            "joint_positions": my_robot.get_joint_positions().tolist(),
            "joint_velocities": my_robot.get_joint_velocities().tolist(),
            "joint_names" : my_robot.dof_names,
        },
        "objects": obj_conf,
    })
    index += 1



def key_check():
    global record_flag, start_current_time, key_flag, episode_num, action_list, index
    if '`' in pressed and key_flag:
        record_flag = not record_flag
        if record_flag:
            record_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(1.0, 0.0, 0.0))
            save_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            record_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(0.0, 0.0, 0.0))

        start_current_time = my_world.current_time
        key_flag = False
        print("record state : ", record_flag)


    if "z" in pressed and key_flag:

        os.makedirs( os.path.join( output_path, "action"), exist_ok=True)
        with open( os.path.join(output_path, "action", f"{episode_num:04d}.json"), 'w') as f:
            json.dump(action_list, f, indent=4)

        record_flag = False
        print(f"Saved : {os.path.join(output_path, 'action', f'{episode_num:04d}.json')}")
        save_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(0.0, 1.0, 0.0))
        key_flag = False
        episode_num = check_eps_num(output_path)
        action_list = []
        index = 0

    if '/' in pressed and key_flag:
        my_world.stop()
        action_list = []
        index = 0
        key_flag = False
        print("record state : ", record_flag)


while simulation_app.is_running():
    my_world.step(render=True)
    key_check()
    current_time = my_world.current_time
    # print("current_time : ", current_time)



    if my_world.is_stopped() and stop_flag:
        i=0
        state=0
        ik_first_flag=True
        obj_reset_flag = True
        stop_flag = False
        record_flag = False

        my_world.reset()

        csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False, rotation=False)



    if my_world.is_playing():

        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset() 
        i += 1

        my_robot_task.teleop_step()
        if record_flag:
            action_collect()




simulation_app.close()

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


cam_model_conf_path = "/nas/ochansol/camera_params/azure_kinect_conf.json"
with open(cam_model_conf_path, 'r') as f:
    cam_model_conf = json.load(f)
cam_conf = {
    "name":"",
    "cam_model_conf_path" : cam_model_conf_path,
    "pixel_size" : cam_model_conf["pixel_size_RGB"]*1000,# 0.0000025, # 2.5um
    "output_size" : (1920,1080),# min object 1920*1280 = 96*54( 5% )
    "clipping_range" : (0.0001, 100000),
    "focus_distance" : 0,
    "f_stop" : 0,
    "cam_poses" : [],
}

# ((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]

# cam_conf["focal_length_isaac"] = (fx+fy)/2*cam_conf["pixel_size"]
# cam_conf["horizontal_aperture"] = cam_conf["output_size"][0]*cam_conf["pixel_size"]
# cam_conf["intrinsic_isaac"] = [[(fx+fy)/2, 0,cam_conf["output_size"][0]/2],
#                             [0, (fx+fy)/2, cam_conf["output_size"][1]/2],
#                             [0,0,1]]

# cam_pos = [0.032,0.2,2.0232]
# cam_tf = cs.trans(cam_pos[0],cam_pos[1],cam_pos[2]-1.03 )
# top_view_camera = rep.create.camera(
#     position = cam_pos,
#     rotation = [0,-90,-90],
#     # look_at =obj_rep_list[0].node,
#     focal_length = cam_conf["focal_length_isaac"], 
#     focus_distance =cam_conf["focus_distance"], 
#     f_stop = cam_conf["f_stop"], 
#     horizontal_aperture = cam_conf["horizontal_aperture"],
#     clipping_range = cam_conf["clipping_range"])
# cam_conf1 = cam_conf.copy()
# cam_conf1["name"] = "top_view_camera"


#### full = realsense D435i : 1920 * 1080
#### wrist = realsense D405 : 1280 * 720
full_res=(1920,1080)
wrist_res=(1280,720)

full_cam_path = f"{my_robot_task.prim_path}/demo/full_camera"
wrist_cam_path = f"{my_robot_task.prim_path}/OMY_custom/OMY/link6/wrist_camera"

full_camera = Camera(
    prim_path=wrist_cam_path,
    name="full_cam",
    frequency=25,
    resolution=full_res,)

wrist_camera = Camera(
    prim_path=full_cam_path,
    name="wrist_cam",
    frequency=25,
    resolution=wrist_res,)

full_camera.initialize()
wrist_camera.initialize()


# render_product_top = rep.create.render_product(top_view_camera, cam_conf["output_size"])
# depth_img_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
# depth_img_annotator.attach([render_product_top])
# rgb_img_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
# rgb_img_annotator.attach([render_product_top])

render_product_full = full_camera._render_product
render_product_wrist = wrist_camera._render_product
# render_product_side = rep.create.render_product(side_view_camera, cam_conf["output_size"])
writer = rep.WriterRegistry.get("SanjabuWriter")
writer.initialize(
    output_dir                      = output_path,
    rgb                             = True,
    distance_to_image_plane         = True,
)
writer.set_path(output_path,
                rgb_path = "rgb",
                bounding_box_path = "bbox",
                distance_to_image_plane_path = "depth",
                instance_segmentation_path = "inst_seg",
                pointcloud_path = "pointcloud",
                normals_path = "normals",)
writer.set_cam_name_list([full_camera.name, wrist_camera.name])# cam_conf2["name"]])
writer.attach([render_product_full, render_product_wrist])# render_product_side])
rep.orchestrator.pause()
rep.orchestrator.set_capture_on_play(False)








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
    # OBJ.set_rigidbody_collider()
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
    'physxScene:minPositionIterationCount' : 30,
    'physxScene:minVelocityIterationCount' : 20,
    "physics:gravityMagnitude":35,
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

csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False)


i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001
record_flag = False

my_world.stop()



pressed = set()
def on_press(key):
    try:
        pressed.add(key.char)
    except:
        pass

def on_release(key):
    try:
        pressed.discard(key.char)
    except:
        pass

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()


def key_check():
    global record_flag
    if '`' in pressed:
        record_flag = not record_flag

        if record_flag:
            rep.orchestrator.start()
        else:
            rep.orchestrator.pause()



while simulation_app.is_running():
    my_world.step(render=True)
    key_check()

    if my_world.is_stopped() and stop_flag:
        i=0
        state=0
        ik_first_flag=True
        obj_reset_flag = True
        stop_flag = False
        my_world.reset()
        my_world.pause()

    if my_world.is_playing():

        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset() 
        i += 1

        my_robot_task.teleop_step()



simulation_app.close()
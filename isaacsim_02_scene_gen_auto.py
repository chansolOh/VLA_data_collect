import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--start_num", type=int, default=0, help="starting episode number")
parser.add_argument("--end_num", type=int, default=100, help="ending episode number")
parser.add_argument("--augmentation", action="store_true", help="whether to apply augmentation during data collection")
parser.add_argument("--detection", action="store_true", help="whether to apply detection during data collection")
parser.add_argument("--output_path", type=str, default="", help="output path for collected data")
parser.add_argument("--dataset_path", type=str, default="", help="dataset path for loading action json files")

args = parser.parse_args()



from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

import Robot_task_scene_gen as Robot_task

from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni
import carb
from isaacsim.util.debug_draw import _debug_draw
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage

import omni.replicator.core as rep
import omni.timeline

import sys
import pathlib
sys.path.append(f"{pathlib.Path.home()}/ochansol/isaac_code/isaac_chansol")
# sys.path.append(f"{pathlib.Path.home()}/ochansol/isaac_chansol")
import Utils.isaac_utils_51.rep_utils as csr
import Utils.isaac_utils_51.scan_rep as scan_rep
import Utils.isaac_utils_51.light_set as light
import Utils.isaac_utils_51.sanjabu_Writer as SW
import Utils.isaac_utils_51.augmentation as aug
from Utils.Robot_45 import robot_configs, robot_policy
import json
import os
from tqdm import tqdm

carb.settings.get_settings().set("/rtx/post/motionblur/enabled", True)
# 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
carb.settings.get_settings().set("/rtx/post/aa/op", 2)
# (float): The fraction of the largest screen dimension to use as the maximum motion blur diameter.
carb.settings.get_settings().set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.3)
# (float): Exposure time fraction in frames (1.0 = one frame duration) to sample.
carb.settings.get_settings().set("/rtx/post/motionblur/exposureFraction", 2.0)
# (int): Number of samples to use in the filter. A higher number improves quality at the cost of performance.
carb.settings.get_settings().set("/rtx/post/motionblur/numSamples", 20)





object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
dataset_path = args.dataset_path
output_path =  args.output_path ## output을 다르게 쓸ㄸ
output_cache_path = os.path.join(output_path, "cache")

writer_dict = {
    "output_dir"                    : output_cache_path,
    "rgb"                           : True,
    "bounding_box_2d_loose"         : args.detection,
    "bounding_box_2d_tight"         : False,
    "bounding_box_3d"               : False,
    "distance_to_camera"            : False,
    "distance_to_image_plane"       : False,
    "instance_segmentation"         : args.detection,
    "normals"                       : False,
    "semantic_segmentation"         : False,
    "use_common_output_dir"         : False,
    "pointcloud_include_unlabelled" : False,
    "pointcloud"                    : False,
    "occlusion"                     : False,
}



my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.001,
                rendering_dt = 0.01)

stage = omni.usd.get_context().get_stage()




Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY_no_delay"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" ,
                idle_joint=np.array([0,-32,25,43,92,0,0,0,0,0])/180*np.pi 
                )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim

env_prim = add_reference_to_stage(prim_path = "/World/env", usd_path ="/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd")


vis_plane = csr.find_target_name(env_prim, ["Mesh"], "vis_plane")
# for plane in vis_plane:
#     aug.random_material(stage, plane, plane.GetChildren() )







light_list = csr.find_lights(env_prim)
Lights = light.Light(light_list)
# Lights.random_trans(0.2, [1])
Lights.set_all_exposure(val=1)



#### full = realsense D435i : 1920 * 1080
#### wrist = realsense D405 : 1280 * 720
full_res=(1280,720)
wrist_res=(848,480)

full_cam_path = f"{str(env_prim.GetPrimPath())}/demo/full_camera"
wrist_cam_path = f"{my_robot_task.prim_path}/OMY/link6/wrist_camera"

full_camera = Camera(
    prim_path=full_cam_path,
    name="cam_top",
    frequency=30,
    resolution=full_res,)

wrist_camera = Camera(
    prim_path=wrist_cam_path,
    name="cam_wrist",
    frequency=30,
    resolution=wrist_res,)

full_camera.initialize()
wrist_camera.initialize()


render_product_full = full_camera._render_product
render_product_wrist = wrist_camera._render_product
# render_product_side = rep.create.render_product(side_view_camera, cam_conf["output_size"])
writer = rep.WriterRegistry.get("SanjabuWriter")
writer.initialize(**writer_dict)
writer.set_path(output_cache_path,
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
        "path": "/nas/ochansol/3d_model/scan_etc/apple_test/apple.usd",
        "size_rank": 0,
        "scale" : [0.1,0.1,0.1]
    },
    "custom_box_12_12_08_magenta":{
        "name":"custom_box_12_12_08_magenta",
        "path": "/nas/ochansol/3d_model/VLA/custom_box_12_12_08_magenta/custom_box_12_12_08_magenta.usd",
        "size_rank": 0,
        "scale" : [1,1,1]
    }
}


obj_rep_all_list = []
for key in sampled_model_dict:
    model_attr = sampled_model_dict[key]
    print("model_attr : ", model_attr["name"])
    scan_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                            class_name = model_attr["name"],
                            size = model_attr["size_rank"],
                            scale = model_attr.get("scale", [0.1,0.1,0.1])
                            )
    sampled_model_dict[key]["rep"] = scan_obj
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
        


# platform_area_prims = csr.find_target_name(env_prim,["Mesh"],"platform_area")
# platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == "demo"][0]

# platform_path = platform_area_prims.GetPath().__str__()
# platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])

my_world.reset()

# platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
# platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
# platform_rep.set_tf(platform_tf)
# platform_rep.set_scale(platform_scale)

# csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = False)



i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.3
record_flag = False

action_list = []
config = {}

# my_world.stop()


# while True:
for episode_num in tqdm(range(args.start_num, args.end_num)):
    episode_num = f"{episode_num:04d}"

    # episode_list = sorted([i.strip(".json") for i in os.listdir( os.path.join(dataset_path, "action") ) if i.endswith('.json')])
    # for episode_num in episode_list:
    #     if os.path.exists( os.path.join(output_path,f"rgb/{episode_num}/{full_camera.name}")):
    #         with open( os.path.join(dataset_path, "action", f"{episode_num}.json"), 'r') as f:
    #             action_data = json.load(f)

    #         rgb_list = [i for i in os.listdir(os.path.join(output_path,f"rgb/{episode_num}/{full_camera.name}")) if i.endswith('.png')]
    #         if len(rgb_list) >= len(action_data)//4:
    #             print(f"Already exists : {episode_num} PNG , skip...")
    #             if episode_num == episode_list[-1]:
    #                 print("All episodes are loaded.")
    #                 simulation_app.close()
    #                 sys.exit()
    #             continue
    #         else:
    #             print("Load episode : ", episode_num)
    #             break
    #     else:
    #         with open( os.path.join(dataset_path, "action", f"{episode_num}.json"), 'r') as f:
    #             action_data = json.load(f)
    #         print("Load episode : ", episode_num)
    #         break
    if not os.path.exists( os.path.join(dataset_path, "action", f"{episode_num}.json") ):
        print(f"Episode {episode_num} does not exist, skip...")
        continue
    if args.augmentation:
        for plane in vis_plane:
            aug.random_material(stage, plane, plane.GetChildren() )
    with open( os.path.join(dataset_path, "action", f"{episode_num}.json"), 'r') as f:
        action_data = json.load(f)

    writer.set_path(output_path, 
                    rgb_path = f"rgb/{episode_num}",)
    writer.set_path(output_cache_path,
                rgb_path                        = f"rgb/{episode_num}",
                bounding_box_path               = f"bbox/{episode_num}",
                distance_to_image_plane_path    = f"depth/{episode_num}",
                instance_segmentation_path      = f"inst_seg/{episode_num}",
                pointcloud_path                 = f"pointcloud/{episode_num}",
                normals_path                    = f"normals/{episode_num}",)
    action_i = 0
    print("Start simulation...")

    while simulation_app.is_running():
        my_world.step(render=False)


        if my_world.is_stopped() and stop_flag:
            i=0
            state=0
            ik_first_flag=True
            obj_reset_flag = True
            stop_flag = False
            record_flag = False

            my_world.reset()
            # my_world.pause()

        my_world.play()

        if not record_flag:
            data = action_data[0]
            my_robot.apply_action(ArticulationAction(
                            joint_positions = data["robot"]["joint_positions"],
                            joint_velocities = data["robot"]["joint_velocities"],
                            ))
            
            if np.linalg.norm(np.array(my_robot.get_joint_positions()) - np.array(data["robot"]["joint_positions"])) < joint_err_th:
                record_flag = True
                action_i = 0
                writer.set_frame(frame_id=0)

                print("Start playing...")



        if my_world.is_playing() and record_flag:
            stop_flag=True
            # if my_world.current_time_step_index <= 1:
            #     my_world.reset() 
            # i += 1

            data = action_data[action_i]

            my_robot.apply_action(ArticulationAction(
                            joint_positions = data["robot"]["joint_positions"],
                            # joint_velocities = data["robot"]["joint_velocities"],
                            ))
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

            for i in range(5):   
                my_world.step(render=False)
            writer.set_frame(frame_id=data["index"])


            if action_i % 4 == 0:
                writer.dataset_path = output_path
                rep.orchestrator.step()
            # print(action_i)
            action_i += 1
            if action_i >= len(action_data):
                rep.orchestrator.pause()
                my_world.stop()
                # simulation_app.close()
                break



simulation_app.close()
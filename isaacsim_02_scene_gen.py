
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

import omni.replicator.core as rep
import omni.timeline

import sys
import pathlib
sys.path.append(f"{pathlib.Path.home()}/ochansol/isaac_chansol")
import Utils.isaac_utils_51.rep_utils as csr
import Utils.isaac_utils_51.scan_rep as scan_rep
import Utils.isaac_utils_51.light_set as light
import Utils.isaac_utils_51.sanjabu_Writer as SW

import json
import os






object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
root_path = "/nas/ochansol/isaac"
output_path =  "/nas/Dataset/VLA/UON/Isaacsim_OMY"
output_cache_path = os.path.join(output_path, "cache")




my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.001,
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
full_res=(1280,720)
wrist_res=(848,480)

full_cam_path = f"{my_robot_task.prim_path}/demo/full_camera"
wrist_cam_path = f"{my_robot_task.prim_path}/OMY_custom_no_delay/OMY/link6/wrist_camera"

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
writer.initialize(
    output_dir                      = output_cache_path,
    rgb                             = True,
    distance_to_image_plane         = False,
)
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
joint_err_th = 0.01
record_flag = False

action_list = []
config = {}

# my_world.stop()


episode_list = sorted([i.strip(".json") for i in os.listdir( os.path.join(output_path, "action") )])
for episode_num in episode_list:
    print("Load episode : ", episode_num)
    with open( os.path.join(output_path, "action", f"{episode_num}.json"), 'r') as f:
        action_data = json.load(f)

    if os.path.exists( os.path.join(output_path,f"rgb/{episode_num}/{full_camera.name}")):
        rgb_list = [i for i in os.listdir(os.path.join(output_path,f"rgb/{episode_num}/{full_camera.name}")) if i.endswith('.png')]
        if len(rgb_list) > len(action_data)//4:
            print(f"Already exists : {episode_num} PNG , skip...")
            continue


    writer.set_path(output_path, rgb_path = f"rgb/{episode_num}",)
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
            my_world.pause()

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
                            joint_velocities = data["robot"]["joint_velocities"],
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
                writer.output_path = output_path
                rep.orchestrator.step()
            print(action_i)
            action_i += 1
            if action_i >= len(action_data):
                rep.orchestrator.pause()
                my_world.stop()
                # simulation_app.close()
                break



simulation_app.close()
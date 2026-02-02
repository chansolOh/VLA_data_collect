
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



def debug_draw_obb(obb):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_lines(
        [carb.Float3(i) for i in obb[[0,1,2,2,0,4,5,5,7,6,7,6]]] , 
        [carb.Float3(i) for i in obb[[1,3,3,0,4,5,1,7,6,4,3,2]]] , 
        [carb.ColorRgba(1.0,0.0,0.0,1.0)]*12,
        [1]*12 )
    return draw

def debug_draw_points(points,size = 3, color=[1,0,0]):
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.draw_points(
        [carb.Float3(i) for i in points] , 
        [carb.ColorRgba(color[0],color[1],color[2],1.0)]*len(points),
        [size]*len(points) )
    return draw

def debug_draw_clear():
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()


object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
root_path = "/nas/ochansol/isaac"


gripper_info_path = "/nas/ochansol/gripper_info/gripper_info.json" 
with open(gripper_info_path, 'r') as f:
        gripper_info = json.load(f)
random_gripper_name = "Robotiq_2f_140_full"
gripper = gripper_info[random_gripper_name]


my_world = World(stage_units_in_meters=1.0,
                physics_dt  = 0.01,
                rendering_dt = 0.01)

stage = omni.usd.get_context().get_stage()
GroundPlane(prim_path="/World/GroundPlane", z_position=0)
light_1 = prim_utils.create_prim(
    "/World/Light_1",
    "SphereLight",
    position=np.array([0, 0, 20.0]),
    attributes={
        "inputs:radius": 0.01,
        "inputs:intensity": 5e3,
        "inputs:color": (255, 250, 245),
        "inputs:exposure" : 12,
    }
)
light_2 = prim_utils.create_prim(
    "/World/Light_2",
    "SphereLight",
    position=np.array([0, 0.79, 1.57]),
    attributes={
        "inputs:radius": 0.25,
        "inputs:intensity": 5e3,
        "inputs:color": (255, 250, 245),
        "inputs:exposure" : -4,
    }
)


my_robot_task = Robot_task.My_Robot_Task(name="robot_task" )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim




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

((fx,_,cx),(_,fy,cy),(_,_,_))= cam_model_conf["intrinsic_matrix"]

cam_conf["focal_length_isaac"] = (fx+fy)/2*cam_conf["pixel_size"]
cam_conf["horizontal_aperture"] = cam_conf["output_size"][0]*cam_conf["pixel_size"]
cam_conf["intrinsic_isaac"] = [[(fx+fy)/2, 0,cam_conf["output_size"][0]/2],
                            [0, (fx+fy)/2, cam_conf["output_size"][1]/2],
                            [0,0,1]]
cam_pos = [0.032,0.2,2.0232]
cam_tf = cs.trans(cam_pos[0],cam_pos[1],cam_pos[2]-1.03 )
top_view_camera = rep.create.camera(
    position = cam_pos,
    rotation = [0,-90,-90],
    # look_at =obj_rep_list[0].node,
    focal_length = cam_conf["focal_length_isaac"], 
    focus_distance =cam_conf["focus_distance"], 
    f_stop = cam_conf["f_stop"], 
    horizontal_aperture = cam_conf["horizontal_aperture"],
    clipping_range = cam_conf["clipping_range"])
cam_conf1 = cam_conf.copy()
cam_conf1["name"] = "top_view_camera"

render_product_top = rep.create.render_product(top_view_camera, cam_conf["output_size"])
depth_img_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
depth_img_annotator.attach([render_product_top])
rgb_img_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_img_annotator.attach([render_product_top])


model_list = []
for path in object_path_list:
    with open(os.path.join(path, "objects_conf.json"),'r'  ) as f:
        model_list += json.load(f)

sampled_model_dict = {}
for model_attr in model_list:
    sampled_model_dict[model_attr["name"]] = model_attr


target_obj_name = np.random.choice(list(sampled_model_dict.keys()),1,replace=False)
model_attr = sampled_model_dict[target_obj_name[0]]
target_obj = scan_rep.Scan_Rep(usd_path =  model_attr["path"],
                        class_name = model_attr["name"],
                        size = model_attr["size_rank"],)

target_obj.set_rigidbody_collider()
# target_obj.set_contact_sensor()
target_obj.set_physics_material(
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
        
        



env_prim = stage.GetPrimAtPath(my_robot_task.prim_path)
platform_area_prims = csr.find_target_name(env_prim,["Mesh"],"platform_area")
platform_area_prims = [i.GetParent() for i in platform_area_prims if i.GetParent().GetName() == "demo"][0]

platform_path = platform_area_prims.GetPath().__str__()
platform_rep = scan_rep.Scan_Rep_Platform(prim_path = platform_path,scale = [1,1,1], class_name = platform_path.split("/")[-1])

my_world.reset()

platform_tf = csr.find_parents_tf(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_scale = csr.find_parents_scale(stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
platform_rep.set_tf(platform_tf)
platform_rep.set_scale(platform_scale)

csr.scatter_in_platform_area(platform_rep,[target_obj],fixed_first = False)


i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001

close_r_joint = gripper["close_r_joint"]
close_l_joint = gripper["close_l_joint"]
my_world.stop()





# from grasp_sample_gen import InteractiveGraspRect

# fig, ax = plt.subplots()
# ax.imshow(rgb_img_cam00)
# inter = InteractiveGraspRect(ax, fixed_height=20)
# plt.show()


from grasp_sample_gen import InteractiveGraspRect
import cal_depth_width

while simulation_app.is_running():
    my_world.step(render=True)

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


        if state==0:
            if ik_first_flag:
                #target_pos = gamja_rep.get_position()[target_idx]
                target_pos = np.array([0.3,0.0,0.3])
                target_joint_positions,_ = my_robot_task.compute_ik(target_position = target_pos,
                                            target_orientation = [180,0,90], # x,y,z 순서로 회전
                                            frame_name = "OMY_link6",
                                            warm_start=my_robot.get_joint_positions()[:6]
                                            )
                target_joint_positions = np.hstack((target_joint_positions, 
                                                    np.array([0,0])))
                ik_first_flag =False

                print(target_pos)
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5,6,7] ,
                                  joint_positions = target_joint_positions) )
            joint_states = my_robot.get_joint_positions()[:8]
            joint_err = np.abs(joint_states - target_joint_positions)
            if np.mean(joint_err)<joint_err_th:
                ik_first_flag = True
                state+=1

    
        

        if i >= 200  :
            state+=1
            i=0
            ik_first_flag = True
            obj_reset_flag = True
        if state>=7:
            state=0

        # if target_idx >= gamja_rep.count:
        #     target_idx =0

simulation_app.close()
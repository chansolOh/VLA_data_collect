import os
import sys
from pathlib import Path

import carb
import numpy as np
import omni
import omni.isaac.core.utils.prims as prim_utils
from isaacsim.core.api import World
from isaacsim.core.api.objects.cuboid import VisualCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer
from isaacsim.robot_motion.motion_generation.lula import RRT


PROJECT_ROOT = Path(__file__).resolve().parents[4] / "isaac_chansol"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Utils.Robot_45 import robot_configs, robot_policy
from Utils.isaac_utils_51 import scan_rep
import Utils.isaac_utils_51.rep_utils as csr



def setup_environment(**kwargs):
    my_world = World(stage_units_in_meters=1.0, physics_dt=0.01, rendering_dt=0.01)

    robot_cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
    my_robot_task = robot_policy.My_Robot_Task(
        robot_config=robot_cfg,
        name="robot_task",
        idle_joint=np.array([0, -32, 25, 43, 92, 0, 0, 0, 0, 0]) / 180 * np.pi,
    )
    my_world.add_task(my_robot_task)
    my_world.reset()

    usd_stage = omni.usd.get_context().get_stage()
    my_robot = my_robot_task._robot
    env_prim = add_reference_to_stage(prim_path="/World/env", usd_path=kwargs["ENV_USD_PATH"])

    prim_utils.create_prim(
        "/World/Light_1",
        "SphereLight",
        position=np.array([0, 0, 20.0]),
        attributes={
            "inputs:radius": 0.01,
            "inputs:intensity": 5e3,
            "inputs:color": (255, 250, 245),
            "inputs:exposure": 12,
        },
    )

    sampled_model_dict = {
        "custom_box_12_12_08_magenta": {
            "name": "custom_box_12_12_08_magenta",
            "path": "/nas/ochansol/3d_model/VLA/custom_box_12_12_08_magenta/custom_box_12_12_08_magenta.usd",
            "size_rank": 0,
            "scale": [1, 1, 1],
            "position": [0.25, -0.015, 0.041],
        },
        "custom_box_12_12_08_blue": {
            "name": "custom_box_12_12_08_blue",
            "path": "/nas/ochansol/3d_model/VLA/custom_box_12_12_08_blue/custom_box_12_12_08_blue.usd",
            "size_rank": 0,
            "scale": [1, 1, 1],
            "position": [0.25, -0.015, 0.041],
        },
        "custom_box_12_12_08_yellow": {
            "name": "custom_box_12_12_08_yellow",
            "path": "/nas/ochansol/3d_model/VLA/custom_box_12_12_08_yellow/custom_box_12_12_08_yellow.usd",
            "size_rank": 0,
            "scale": [1, 1, 1],
            "position": [0.25, -0.015, 0.041],
        },
        "apple": {
            "name": "apple",
            "path": "/nas/ochansol/3d_model/scan_etc/apple/edited/apple.usd",
            "size_rank": 0,
            "scale": [0.1, 0.1, 0.1],
        },
        "paprika": {
            "name": "paprika",
            "path": "/nas/ochansol/3d_model/scan_etc/paprika/edited/paprika.usd",
            "size_rank": 0,
            "scale": [0.1, 0.1, 0.1],
        },
        "potato": {
            "name": "potato",
            "path": "/nas/ochansol/3d_model/scan_etc/potato/edited/potato.usd",
            "size_rank": 0,
            "scale": [0.1, 0.1, 0.1],
        },
    }

    obj_rep_all_list = []
    for key, model_attr in sampled_model_dict.items():
        print("model_attr : ", model_attr["name"])
        scan_obj = scan_rep.Scan_Rep(
            usd_path=model_attr["path"],
            class_name=model_attr["name"],
            size=model_attr["size_rank"],
            scale=model_attr.get("scale", [0.1, 0.1, 0.1]),
            position=model_attr.get("position", [0, 0, 0]),
        )
        sampled_model_dict[key]["rep"] = scan_obj
        obj_rep_all_list.append(scan_obj)

    for obj_rep in obj_rep_all_list:
        print("set collider for : ", obj_rep.class_name)
        obj_rep.set_rigidbody_collider()
        obj_rep.set_physics_material(
            dynamic_friction=0.25,
            static_friction=0.4,
            restitution=0.1,
        )

    platform_area_prims = csr.find_target_name(env_prim, ["Mesh"], "platform_area")
    platform_area_prim = [prim.GetParent() for prim in platform_area_prims if prim.GetParent().GetName() == "demo"][0]
    platform_path = str(platform_area_prim.GetPath())
    platform_rep = scan_rep.Scan_Rep_Platform(
        prim_path=platform_path,
        scale=[1, 1, 1],
        class_name=platform_path.split("/")[-1],
    )

    my_world.reset()

    platform_tf = csr.find_parents_tf(usd_stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_scale = csr.find_parents_scale(usd_stage.GetPrimAtPath(platform_path).GetPrim(), include_self=False)
    platform_rep.set_tf(platform_tf)
    platform_rep.set_scale(platform_scale)

    picking_rep = sampled_model_dict["apple"]["rep"]
    place_rep = sampled_model_dict["custom_box_12_12_08_magenta"]["rep"]
    box_obb = place_rep.get_init_obb()
    box_x, box_y, box_z = box_obb.max(0) - box_obb.min(0)
    obstacle = VisualCuboid(
        "/World/Wall",
        position=np.array([0, 0, 0]),
        size=1,
        scale=np.array([box_x, box_y, box_z]),
        visible=False,
    )

    rrt = RRT(
        robot_description_path=kwargs["ROBOT_DESCRIPTION_PATH"],
        urdf_path=kwargs["ROBOT_URDF_PATH"],
        rrt_config_path=kwargs["RRT_CONFIG_PATH"],
        end_effector_frame_name="OMY_grasp_joint",
    )
    rrt.add_obstacle(obstacle)
    rrt.set_max_iterations(8000)
    path_planner_visualizer = PathPlannerVisualizer(my_robot, rrt)

    os.makedirs(os.path.join(kwargs["OUTPUT_PATH"], "action"), exist_ok=True)

    return {
        "output_path": kwargs["OUTPUT_PATH"],
        "my_world": my_world,
        "usd_stage": usd_stage,
        "my_robot_task": my_robot_task,
        "my_robot": my_robot,
        "env_prim": env_prim,
        "sampled_model_dict": sampled_model_dict,
        "obj_rep_all_list": obj_rep_all_list,
        "platform_rep": platform_rep,
        "picking_rep": picking_rep,
        "place_rep": place_rep,
        "obstacle": obstacle,
        "rrt": rrt,
        "path_planner_visualizer": path_planner_visualizer,
    }

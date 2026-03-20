import json
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[4] / "isaac_chansol"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Utils.general_utils import mat_utils
import Utils.isaac_utils_51.rep_utils as csr


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("zero vector")
    return vector / norm


def rotation_between_vectors_to_quat(source_vec, target_vec):
    source_vec = normalize(source_vec)
    target_vec = normalize(target_vec)

    cross = np.cross(source_vec, target_vec)
    dot = np.dot(source_vec, target_vec)

    if np.isclose(dot, 1.0):
        return np.array([0.0, 0.0, 0.0, 1.0])

    if np.isclose(dot, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(source_vec[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - np.dot(axis, source_vec) * source_vec
        axis = normalize(axis)
        return R.from_rotvec(axis * np.pi).as_quat()

    axis = normalize(cross)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return R.from_rotvec(axis * angle).as_quat()


def check_eps_num(output_path):
    action_dir = os.path.join(output_path, "action")
    action_eps_list = sorted(int(file_name.strip(".json")) for file_name in os.listdir(action_dir) if file_name.endswith(".json"))
    for idx, num in enumerate(action_eps_list):
        if idx != num:
            return idx
    return len(action_eps_list)


def collect_action_snapshot(context, runtime_state):
    my_world = context["my_world"]
    my_robot_task = context["my_robot_task"]
    my_robot = context["my_robot"]
    obj_rep_all_list = context["obj_rep_all_list"]

    current_time = my_world.current_time - runtime_state["start_current_time"]
    ee_pos, ee_quat = my_robot_task.compute_fk("OMY_grasp_joint")
    obj_conf = {}

    for obj_rep in obj_rep_all_list:
        pose = obj_rep.get_world_pose()
        obj_conf[obj_rep.class_name] = {
            "position": pose["translation"],
            "orientation": pose["rotation"],
        }

    runtime_state["action_list"].append(
        {
            "index": runtime_state["index"],
            "time": round(current_time, 5),
            "stage": runtime_state["task_stage"],
            "robot": {
                "joint_positions": my_robot.get_joint_positions().tolist(),
                "joint_velocities": my_robot.get_joint_velocities().tolist(),
                "joint_names": my_robot.dof_names,
                "ee_position": ee_pos.tolist(),
                "ee_orientation": ee_quat.tolist(),
            },
            "objects": obj_conf,
        }
    )
    runtime_state["index"] += 1


def save_action_episode(output_path, runtime_state):
    os.makedirs(os.path.join(output_path, "action"), exist_ok=True)
    save_path = os.path.join(output_path, "action", f"{runtime_state['episode_num']:04d}.json")
    with open(save_path, "w") as file:
        json.dump(runtime_state["action_list"], file, indent=4)

    print(f"Saved : {save_path}")
    runtime_state["episode_num"] = check_eps_num(output_path)


def get_place_scatter_config(fixed_box_position):
    if fixed_box_position:
        return {
            "fixed_first": True,
            "initial_rotation": [[], ["x", "y", "z"]],
            "reset_rotation": [[], ["x", "y", "z"]],
        }

    return {
        "fixed_first": False,
        "initial_rotation": [["z"], ["x", "y", "z"]],
        "reset_rotation": [["z"], ["x", "y", "z"]],
    }


def _scatter_objects(platform_rep, obj_rep_all_list, fixed_first, rotation):
    csr.scatter_in_platform_area_spread(
        platform_rep,
        obj_rep_all_list,
        fixed_first=fixed_first,
        rotation=rotation,
    )


def _reset_episode(context, runtime_state, scatter_config):
    my_world = context["my_world"]
    platform_rep = context["platform_rep"]
    obj_rep_all_list = context["obj_rep_all_list"]

    runtime_state["task_stage"] = 0
    runtime_state["stop_flag"] = False
    runtime_state["plan"] = None
    runtime_state["rrt_flag"] = True
    runtime_state["rrt_attempt_count"] = 0
    runtime_state["frame_counter"] = 0
    runtime_state["compute_target_flag"] = True
    runtime_state["action_list"] = []
    runtime_state["index"] = 0
    runtime_state["apple_pos_buffer"] = []
    runtime_state["actions"] = []

    my_world.reset()
    _scatter_objects(
        platform_rep,
        obj_rep_all_list,
        fixed_first=scatter_config["fixed_first"],
        rotation=scatter_config["reset_rotation"],
    )


def run_action_collection(context, fixed_box_position=False, render=None):
    output_path = context["output_path"]
    my_world = context["my_world"]
    my_robot_task = context["my_robot_task"]
    my_robot = context["my_robot"]
    platform_rep = context["platform_rep"]
    obj_rep_all_list = context["obj_rep_all_list"]
    picking_rep = context["picking_rep"]
    place_rep = context["place_rep"]
    obstacle = context["obstacle"]
    rrt = context["rrt"]
    path_planner_visualizer = context["path_planner_visualizer"]

    if render is None:
        render = context.get("render", True)

    scatter_config = get_place_scatter_config(fixed_box_position)

    runtime_state = {
        "start_current_time": 0,
        "index": 0,
        "frame_counter": 0,
        "plan": None,
        "task_stage": 0,
        "rrt_flag": True,
        "rrt_attempt_count": 0,
        "compute_target_flag": True,
        "apple_pos_buffer": [],
        "action_list": [],
        "stop_flag": False,
        "target_pos": np.zeros(3),
        "target_quat": np.zeros(4),
        "view_pos": None,
        "view_quat": None,
        "actions": [],
        "episode_num": check_eps_num(output_path),
        "init_diff_grasp_gripper": None,
    }

    cam_vec = np.array([0, -1, 0])

    my_world.reset()
    _scatter_objects(
        platform_rep,
        obj_rep_all_list,
        fixed_first=scatter_config["fixed_first"],
        rotation=scatter_config["initial_rotation"],
    )

    while True:
        my_world.step(render=render)

        if runtime_state["rrt_attempt_count"] > 5:
            my_world.stop()

        if my_world.is_stopped() and runtime_state["stop_flag"]:
            _reset_episode(context, runtime_state, scatter_config)

        if not my_world.is_playing():
            continue

        runtime_state["stop_flag"] = True
        picking_pose = picking_rep.get_world_pose()
        place_pose = place_rep.get_world_pose()
        picking_pos = np.array(picking_pose["translation"])
        picking_quat = np.array(picking_pose["rotation"])
        place_pos = np.array(place_pose["translation"])
        place_quat = np.array(place_pose["rotation"])
        obstacle.set_world_pose(position=place_pos, orientation=place_quat)

        if runtime_state["task_stage"] == 0:
            apple_pos_buffer = runtime_state["apple_pos_buffer"]
            if len(apple_pos_buffer) < 30:
                apple_pos_buffer.append(picking_pos)
            else:
                apple_pos_buffer.pop(0)
                apple_pos_buffer.append(picking_pos)

                if np.std(apple_pos_buffer, axis=0).mean() < 0.0001:
                    runtime_state["task_stage"] += 1
                    runtime_state["rrt_flag"] = True
                    runtime_state["plan"] = None
                    runtime_state["compute_target_flag"] = True
                    runtime_state["apple_pos_buffer"] = []
                    runtime_state["rrt_attempt_count"] = 0
                    runtime_state["start_current_time"] = my_world.current_time

        if runtime_state["task_stage"] == 1:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            if runtime_state["compute_target_flag"]:
                target_pos = picking_pos
                target_ori_vec = target_pos - ee_pos
                pre_step_pos = ee_pos + target_ori_vec * 0.3

                ee_r = R.from_euler("xyz", ee_euler, degrees=True)
                cam_vec_rotated = ee_r.apply(cam_vec)
                pre_step_quat = rotation_between_vectors_to_quat(normalize(cam_vec_rotated), normalize(target_ori_vec))
                pre_step_quat = (R.from_quat(pre_step_quat) * ee_r).as_quat()[[3, 0, 1, 2]]

                runtime_state["compute_target_flag"] = False
                runtime_state["view_pos"] = pre_step_pos
                runtime_state["view_quat"] = pre_step_quat

            if runtime_state["rrt_flag"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[0, 0]))
                rrt.set_end_effector_target(runtime_state["view_pos"], runtime_state["view_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["view_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["view_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.05:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1

        if runtime_state["task_stage"] == 2:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")

            if runtime_state["compute_target_flag"]:
                target_pos = picking_pos
                target_quat = mat_utils.euler_to_quat(np.array([90, 0, 90]), degrees=True)
                plan_list = []
                candi_yaw_list = []
                rrt.set_max_iterations(80)
                for candi_yaw in range(0, 180, 5):
                    rrt.set_end_effector_target(target_pos, mat_utils.euler_to_quat(np.array([90, 0, candi_yaw]), degrees=True))
                    rrt.update_world()
                    plan = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                    if plan:
                        plan_list.append(plan)
                        candi_yaw_list.append(candi_yaw)
                if plan_list:
                    target_quat = mat_utils.euler_to_quat(
                        np.array([90, 0, np.random.choice(candi_yaw_list)]),
                        degrees=True,
                    )
                    target_pos = picking_pos + np.array([0, 0, 0.05])
                rrt.set_max_iterations(8000)
                runtime_state["target_pos"] = target_pos
                runtime_state["target_quat"] = target_quat
                runtime_state["compute_target_flag"] = False

            if runtime_state["rrt_flag"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[0, 0]))
                rrt.set_end_effector_target(runtime_state["target_pos"], runtime_state["target_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["target_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["target_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.03:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1

        if runtime_state["task_stage"] == 3:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")

            if runtime_state["compute_target_flag"]:
                runtime_state["target_pos"] = picking_pos

            if runtime_state["rrt_flag"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[0, 0]))
                rrt.set_end_effector_target(runtime_state["target_pos"], runtime_state["target_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["target_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["target_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.03:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1

        elif runtime_state["task_stage"] == 4:
            my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[np.pi / 4, np.pi / 4]))
            gripper_joint_idx = my_robot.get_dof_index("rh_r1_joint")
            gripper_joint_effort = my_robot.get_measured_joint_efforts(joint_indices=np.array([gripper_joint_idx]))
            if gripper_joint_effort > 0.5:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["init_diff_grasp_gripper"] = np.abs(picking_pos - ee_pos)

        elif runtime_state["task_stage"] == 5:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            diff_grasp_gripper = np.abs(picking_pos - ee_pos)
            if np.abs(runtime_state["init_diff_grasp_gripper"] - diff_grasp_gripper).sum() > 0.02:
                my_world.stop()
                print("init_diff_grasp_gripper : ", runtime_state["init_diff_grasp_gripper"], "diff_grasp_gripper : ", diff_grasp_gripper)
                print("grasp failed, retrying...")
                continue

            if runtime_state["compute_target_flag"]:
                runtime_state["target_pos"] = (place_pos + picking_pos) / 2 + np.array([0, 0, 0.1])
                runtime_state["compute_target_flag"] = False

            if runtime_state["rrt_flag"]:
                rrt.set_end_effector_target(runtime_state["target_pos"], runtime_state["target_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["target_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["target_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.03:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[np.pi / 4, np.pi / 4]))
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1
                print("rrt attempt count : ", runtime_state["rrt_attempt_count"])

        elif runtime_state["task_stage"] == 6:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            diff_grasp_gripper = np.abs(picking_pos - ee_pos)
            if np.abs(runtime_state["init_diff_grasp_gripper"] - diff_grasp_gripper).sum() > 0.02:
                my_world.stop()
                print("init_diff_grasp_gripper : ", runtime_state["init_diff_grasp_gripper"], "diff_grasp_gripper : ", diff_grasp_gripper)
                print("grasp failed, retrying...")
                continue

            if runtime_state["compute_target_flag"]:
                runtime_state["target_pos"] = place_pos + np.array([0, 0, 0.08])
                runtime_state["compute_target_flag"] = False

            if runtime_state["rrt_flag"]:
                rrt.set_end_effector_target(runtime_state["target_pos"], runtime_state["target_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["target_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["target_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.03:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[np.pi / 4, np.pi / 4]))
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1

        elif runtime_state["task_stage"] == 7:
            my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[0.0, 0.0]))

            gripper_joints = np.abs(my_robot_task.get_joint_positions()[[6, 7]])
            if np.sum(gripper_joints) < 0.001:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None

        elif runtime_state["task_stage"] == 8:
            ee_pos, ee_euler = my_robot_task.compute_fk("OMY_grasp_joint")
            if runtime_state["rrt_flag"]:
                my_robot.apply_action(ArticulationAction(joint_indices=[6, 7], joint_positions=[0, 0]))
                rrt.set_end_effector_target(runtime_state["view_pos"], runtime_state["view_quat"])
                rrt.update_world()
                runtime_state["plan"] = path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
                if runtime_state["plan"]:
                    runtime_state["actions"] = my_robot_task.rrt_plan_to_traj_actions(runtime_state["plan"], physics_dt=0.02)
                else:
                    runtime_state["actions"] = []
                runtime_state["rrt_flag"] = False

            pos_crit = np.abs(np.array(ee_pos) - runtime_state["view_pos"]).sum()
            ori_crit = np.abs(mat_utils.euler_to_quat(ee_euler, degrees=True) - runtime_state["view_quat"]).sum()
            if pos_crit < 0.005 and ori_crit < 0.03:
                runtime_state["task_stage"] += 1
                runtime_state["rrt_flag"] = True
                runtime_state["plan"] = None
                runtime_state["compute_target_flag"] = True
                runtime_state["rrt_attempt_count"] = 0

            if runtime_state["actions"]:
                my_robot.apply_action(runtime_state["actions"].pop(0))
            else:
                runtime_state["rrt_flag"] = True
                runtime_state["rrt_attempt_count"] += 1

        elif runtime_state["task_stage"] == 9:
            center_diff = np.abs(picking_pos - place_pos).sum()
            if center_diff < 0.03:
                print("success! center_diff : ", center_diff)
                save_action_episode(output_path, runtime_state)
                my_world.stop()
                continue
            else:
                print("failed... center_diff : ", center_diff)
                my_world.stop()
                continue

        if runtime_state["task_stage"] >= 1:
            collect_action_snapshot(context, runtime_state)

        runtime_state["frame_counter"] += 1

    return runtime_state

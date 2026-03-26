
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
import numpy as np

from isaacsim.core.api.objects.ground_plane import GroundPlane
import omni.isaac.core.utils.prims as prim_utils
import omni

import omni.replicator.core as rep
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrim

import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import colors



import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaac_chansol.Utils.isaac_utils_51 import scan_rep, rep_utils
from isaac_chansol.Utils.general_utils import mat_utils
from isaac_chansol.Utils.Robot_45 import robot_configs, robot_policy


import numpy as np
import matplotlib.pyplot as plt


def plot_fk_chunks_3d_single_arm(
    fk_points,
    wrapper_idx=0,
    title="FK 3D (Single Arm)",
):
    """
    fk_points:
        (N, 1, C, 3) or (N, C, 3)

    N: 데이터 개수
    C: 시간 순서의 action chunk 개수
    """

    fk_points = np.asarray(fk_points)

    if fk_points.ndim == 4:
        pts = fk_points[:, wrapper_idx, :, :]  # (N, C, 3)
    elif fk_points.ndim == 3:
        pts = fk_points
    else:
        raise ValueError(f"Unsupported shape: {fk_points.shape}")

    num_data, num_chunks, _ = pts.shape

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # chunk 시간 순서를 파랑 -> 초록 -> 노랑 -> 빨강으로 통일
    cmap = plt.cm.get_cmap("jet")
    chunk_values = np.linspace(0.0, 1.0, num_chunks)
    chunk_colors = cmap(chunk_values)
    norm = colors.Normalize(vmin=0, vmax=max(num_chunks - 1, 1))

    for data_idx in range(num_data):
        xyz = pts[data_idx]  # (C, 3)

        # 각 데이터의 chunk trajectory를 시간 순으로 연결
        if num_chunks > 1:
            segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
            line_collection = Line3DCollection(
                segments,
                colors=chunk_colors[1:],
                linewidths=2.0,
                alpha=0.65,
            )
            ax.add_collection3d(line_collection)

        # 점도 같은 chunk colormap으로 표시
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=chunk_values,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=32,
            alpha=0.9,
            depthshade=False,
        )

        # 시작/끝 chunk만 가볍게 강조
        ax.scatter(
            xyz[0, 0],
            xyz[0, 1],
            xyz[0, 2],
            color=chunk_colors[0],
            s=55,
            edgecolors="k",
            linewidths=0.4,
            depthshade=False,
            zorder=3,
        )
        ax.text(
            xyz[0, 0],
            xyz[0, 1],
            xyz[0, 2],
            f"{data_idx}",
            fontsize=8,
            color="navy",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
        )
        ax.scatter(
            xyz[-1, 0],
            xyz[-1, 1],
            xyz[-1, 2],
            color=chunk_colors[-1],
            s=70,
            marker="x",
            alpha=0.95,
            zorder=3,
        )
        ax.text(
            xyz[-1, 0],
            xyz[-1, 1],
            xyz[-1, 2],
            f"{data_idx}",
            fontsize=8,
            color="darkred",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.set_box_aspect(np.ptp(pts.reshape(-1, 3), axis=0) + 1e-6)
    ax.grid(True, alpha=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.08, shrink=0.75)
    cbar.set_label("Chunk index (time order)")

    plt.tight_layout()
    plt.show()


object_path_list = ["/nas/Dataset/Dataset_2025/sim2real"]
root_path = "/nas/ochansol/isaac"




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

# Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY_Dual_Arms"]()
Robot_Cfg = robot_configs.ROBOT_CONFIGS["Robotis_OMY"]()
my_robot_task = robot_policy.My_Robot_Task(robot_config=Robot_Cfg, name="robot_task" )
my_world.add_task(my_robot_task)
my_world.reset()
robot_name = my_robot_task.get_robot_name
# my_robot = my_world.scene.get_object(robot_name)
my_robot = my_robot_task._robot
my_robot_prim = my_robot_task.robot_prim








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
        
        
target_prim_path = "/World/target_xform2"
target_xprim = XFormPrim(
    prim_path=target_prim_path,
    name="my_xform2",
    position=np.array([0.1, 0.0, 2.0]),
    orientation=np.array([ 1.0, 0.0, 0.0, 0.0]),  # quat (w, x, y, z) 형태가 보통
)
my_world.scene.add(target_xprim)


# world_base_tf   = rep_utils.gf_mat_to_np( rep_utils.find_parents_tf(stage.GetPrimAtPath(f"{my_robot_task.prim_path}/world_base") , include_self=True)    )
# robot_tf        = rep_utils.gf_mat_to_np( rep_utils.find_parents_tf(stage.GetPrimAtPath(my_robot.prim_path)))
# robot_rot_tf_inv = np.linalg.inv( np.linalg.inv(world_base_tf).dot(robot_tf) )

i = 0
state = 0
target_idx = 0
ik_first_flag = True
obj_reset_flag = True
stop_flag = True
gpu_dynamic_flag = 0
joint_err_th = 0.001

my_world.stop()

joint_traj_path = "/nas/Dataset/dualarm/groot_action_infer_samples"
joint_traj_list = sorted(os.listdir(joint_traj_path))
joint_traj_arr = []
for data in joint_traj_list:
    arr = np.load(os.path.join(joint_traj_path, data))
    joint_traj_arr.append(arr)

joint_traj_arr = np.array(joint_traj_arr)
joint_traj_arr_new = np.asarray(joint_traj_arr)

action_idx_arr = np.where(joint_traj_arr==joint_traj_arr)
for ai, action_i in enumerate(joint_traj_arr):
    for wi, wrapper_i in enumerate(action_i):
        for ci, chunk_i in enumerate(wrapper_i):
            chunk_i[[6,7]]= 0
            joint_traj_arr_new[ai, wi, ci,:3] = my_robot_task.compute_fk(frame_name = "OMY_grasp_joint", joint_positions=chunk_i[:8])[0]

plot_fk_chunks_3d_single_arm(joint_traj_arr_new[...,:3])

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

        # import pdb; pdb.set_trace()
        stop_flag=True
        if my_world.current_time_step_index <= 1:
            my_world.reset() 
        i += 1


        if state==0:
            if ik_first_flag:
                target_pos, target_orientation = target_xprim.get_world_pose()
                target_orientation = mat_utils.quat_to_euler(np.array(target_orientation), degrees=True)
                # target_pos = np.linalg.inv(robot_tf).dot( mat_utils.trans(target_pos) )[:3,-1]

                # target_orientation = np.linalg.inv(robot_tf).dot( mat_utils.rotate(target_orientation) )
                # target_orientation = mat_utils.mat_to_euler(target_orientation, degrees=True)
                target_joint_positions = my_robot_task.compute_ik_traj(target_position = target_pos,
                                            target_orientation = target_orientation,
                                            frame_name = "OMY_grasp_joint",
                                            init_joint_state=my_robot_task.get_joint_positions()
                                            )

                
                target_joint_positions = np.hstack((target_joint_positions[:6], 
                                                    np.array([0,0])))
                ik_first_flag =False

                print(target_pos)

                # my_robot_task.action_traj_ik(ee_pos=target_pos,
                #                         ee_ori=target_orientation,
                #                         frame_name="OMY_grasp_joint"
                #                         )
            my_robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5,6,7] ,
                                  joint_positions = target_joint_positions) )
            joint_states = my_robot.get_joint_positions()[:8]
            joint_err = np.abs(joint_states - target_joint_positions)
            # if np.mean(joint_err)<joint_err_th:
            #     ik_first_flag = True
                # state+=1
 
        

        if i >= 300  :
            # state+=1
            i=0
            ik_first_flag = True
            # obj_reset_flag = True
        if state>=5:
            state=0

        # if target_idx >= gamja_rep.count:
        #     target_idx =0

simulation_app.close()

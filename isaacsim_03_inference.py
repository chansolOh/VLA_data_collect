import sys
from pathlib import Path
from copy import deepcopy

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fixed_box", action="store_true", help="whether to use fixed box position during data collection")


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

CURRENT_DIR = Path(__file__).resolve().parent.parent.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

from auto_rrt_modular.task_execution.run_apple_picking import run_action_collection
from auto_rrt_modular.environment_setup.setup_apple_picking import setup_environment

from isaac_chansol.socket_utils.vla_socket.vla_client import VLAClient
import isaac_chansol.Utils.isaac_utils_51.rep_utils as csr

args = parser.parse_args()
fixed_box_position = args.fixed_box



configs={
    "OUTPUT_PATH": "" ,
    "ENV_USD_PATH": "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd",
    "ROBOT_DESCRIPTION_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",
    "ROBOT_URDF_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
    "RRT_CONFIG_PATH": "/home/uon/ochansol/isaac_code/isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/omy/planner_config.yaml"
}



def main():
    environment_context = setup_environment(**configs)
    environment_context["render"] = True
    globals().update(environment_context)


    full_cam_path = f"{str(env_prim.GetPrimPath())}/demo/full_camera"
    wrist_cam_path = f"{my_robot_task.prim_path}/OMY/link6/wrist_camera"


    full_res=(1280,720)
    wrist_res=(848,480)
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
    my_world.reset()

    render_product_full = full_camera._render_product
    render_product_wrist = wrist_camera._render_product

    annotator_full = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator_full.attach([render_product_full])

    annotator_wrist = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator_wrist.attach([render_product_wrist])

    SERVER_IP = "127.0.0.1"
    PORT = 1823
    client = VLAClient(SERVER_IP, PORT)

    description = 'put the apple in the box'

    vla_flag = False
    action_flag = True
    reset_needed = False
    stop_flag = True

    ot = 0
    i=0

    import time
    SEND_HZ = 25
    send_period = 1.0 / SEND_HZ
    next_send_time = time.perf_counter()
    client.start_infer_thread(hz=SEND_HZ)

    while simulation_app.is_running():
        my_world.step(render=True)
        sim_t = my_world.current_time

        if my_world.is_stopped() and stop_flag:
            i=0
            state=0
            ik_first_flag=True
            obj_reset_flag = True
            stop_flag = False
            record_flag = False
            my_world.reset()

            csr.scatter_in_platform_area(platform_rep, obj_rep_all_list, fixed_first = True, rotation=False)




        if my_world.is_playing():
        
            stop_flag=True
            now = time.perf_counter()
            if now < next_send_time:
                continue
            next_send_time += send_period



            state = my_robot_task.get_joint_positions()[[0,1,2,3,4,5,7]].tolist()  # joint state + gripper state
            # state[-1]/=2
            # state = np.array(Robot_inst.get_state(action_type="joint"))[[0,1,2,3,4,5,-1]].tolist()  # joint state + gripper state
            full_rgb = annotator_full.get_data()
            wrist_rgb = annotator_wrist.get_data()

            client.push(
                images_bgr={
                    "full": full_rgb, 
                    "wrist": wrist_rgb
                    }, 

                obs={
                    "joint_state": state
                    }, 

                action_type="joint")  # 이미지 1장


            if client.action is not None:
                infer_action = deepcopy(client.action)
                infer_action[-1] *= 2

                my_robot_task.apply_action(
                    joint_indices=[0,1,2,3,4,5,7],
                    joint_positions=infer_action
                )
                # Robot_inst.set_action(client.action, action_type="joint", action_chunk=False)


            #### action
            if reset_needed:
                my_world.reset()
                # Robot_inst.reset()
                reset_needed = False

            # Robot_inst.action_step()




if __name__ == "__main__":
    
    main()


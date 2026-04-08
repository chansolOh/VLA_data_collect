import sys
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="whether to run in headless mode")
parser.add_argument("--fixed_box", action="store_true", help="whether to use fixed box position during data collection")
parser.add_argument("--output_path", type=str, default="/nas/Dataset/VLA/UON/Isaacsim/OMY_apple_picking/test",help="path to save collected data")
parser.add_argument("--start_num", type=int, default=0, help="start episode number")
parser.add_argument("--end_num", type=int, default=10, help="end episode number")
args = parser.parse_args()

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

from auto_rrt_modular.environment_setup.setup_paprika_picking import setup_environment
from auto_rrt_modular.task_execution.run_pick_and_place import run_action_collection


fixed_box_position = args.fixed_box
start_num = args.start_num
end_num = args.end_num



configs={
    "OUTPUT_PATH": args.output_path,
    "ENV_USD_PATH": "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd",
    "ROBOT_DESCRIPTION_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",
    "ROBOT_URDF_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
    "RRT_CONFIG_PATH": str(CURRENT_DIR.parent.parent / "isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/omy/planner_config.yaml")
}

def main():
    environment_context = setup_environment(**configs)
    environment_context["render"] = not args.headless

    run_action_collection(
        environment_context,
        fixed_box_position=fixed_box_position,
        start_num=start_num,
        end_num=end_num,
        stage_timeouts={
            0: 4.0,
            1: 4.0,
            2: 4.0,
            3: 4.0,
            4: 4.0,
            5: 4.0,
            6: 4.0,
            7: 4.0,
            8: 4.0,
            9: 4.0,
        },
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

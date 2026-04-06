import sys
from pathlib import Path

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--fixed_box", action="store_true", help="whether to use fixed box position during data collection")
parser.add_argument("--output_path", type=str, help="path to save collected data")

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.physx.ui")
enable_extension("omni.physx")

from auto_rrt_modular.task_execution.run_apple_picking import run_action_collection
from auto_rrt_modular.environment_setup.setup_apple_picking import setup_environment

args = parser.parse_args()
fixed_box_position = args.fixed_box



configs={
    "OUTPUT_PATH": args.output_path,
    "ENV_USD_PATH": "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_env.usd",
    "ROBOT_DESCRIPTION_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_RRT.yaml",
    "ROBOT_URDF_PATH": "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.urdf",
    "RRT_CONFIG_PATH": str(CURRENT_DIR.parent.parent / "isaac_chansol/Utils/Robot_45/basic_ik/motion_policy_configs/omy/planner_config.yaml")
}

def main():
    environment_context = setup_environment(**configs)
    environment_context["render"] = False

    run_action_collection(
        environment_context,
        fixed_box_position=fixed_box_position,
        max_episodes=400,
        stage_timeouts={
            0: 7.0,
            1: 7.0,
            2: 7.0,
            3: 7.0,
            4: 7.0,
            5: 7.0,
            6: 7.0,
            7: 7.0,
            8: 7.0,
            9: 7.0,
        },
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

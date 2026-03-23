import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from omni.isaac.core.utils.extensions import enable_extension

from auto_rrt_modular.environment_setup.setup_apple_picking_random_place import setup_environment
from auto_rrt_modular.task_execution.run_apple_picking import run_action_collection


def main():
    enable_extension("omni.physx.ui")
    enable_extension("omni.physx")

    environment_context = setup_environment()
    environment_context["render"] = False

    run_action_collection(
        environment_context,
        fixed_box_position=False,
        max_episodes=800,
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

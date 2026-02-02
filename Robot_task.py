# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
from typing import Optional

import numpy as np
import omni.isaac.core.tasks as tasks

from omni.isaac.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.core.prims import XFormPrim
import omni.usd
from pxr import UsdGeom, Gf
import torch
from isaacsim.core.utils.types import ArticulationAction



import roslibpy



# Inheriting from the base class Follow Target
class My_Robot_Task(tasks.BaseTask):
    def __init__(
        self,
        name: str = "Default_name",
        prim_path :str = "/World/Robot",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.BaseTask.__init__(self, name=name, offset=offset)
        self.prim_path = prim_path
        self.stage = omni.usd.get_context().get_stage()
    

        self.joint_states = None

        client = roslibpy.Ros(host='192.168.0.137', port=9090)
        client.run()


        topic = roslibpy.Topic(client, '/leader/joint_states', 'sensor_msgs/JointState')
        topic.subscribe(self.sub)

        return
    def sub(self,msg):
        self.joint_states = msg["position"]
        # self.joint_states[3] -=np.pi*2
        # print(self.joint_states)
        return
    
    def teleop_step(self):
        if self.joint_states is not None:
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[0,1,2,3,4,5] ,
                                    joint_positions = self.joint_states[:6]) )
            gripper_joint =63/180*np.pi- (self.joint_states[6]+0.6)/ 0.6 * 63/180*np.pi
            # print(self.joint_states[6], gripper_joint)
            self._robot.apply_action(ArticulationAction(
                                    joint_indices=[7] ,
                                    joint_positions = [ gripper_joint]) )
            
    
    def set_up_scene(self, scene: Scene) -> None:
        """[summary]

        Args:
            scene (Scene): [description]
        """
        super().set_up_scene(scene)
        #scene.add_default_ground_plane(z_position=0)


        self._robot = self.set_robot()
        self.kinematics_solver = self.set_solver()
        self.robot_prim = self.stage.GetPrimAtPath(self.prim_path)
        

        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._move_task_objects_to_their_frame()
        return
    
    def set_robot(self) -> SingleManipulator:

        asset_path = "/nas/ochansol/isaac/sim2real/uon_vla_demo_robotis_no_delay.usd"

        add_reference_to_stage(usd_path=asset_path, prim_path=self.prim_path)

        gripper = ParallelGripper(
            end_effector_prim_path=f"{self.prim_path}/OMY_custom_no_delay/OMY/link6",
            # end_effector_prim_path="/World/Doosan_M1013/robotiq_arg2f_base_link",
            joint_prim_names=["rh_r1_joint", "rh_l1"],
            joint_opened_positions=np.array([0, 0]),
            joint_closed_positions=np.array([1.04, 1.04]),
            action_deltas=np.array([-0., 0.]),
        )

        manipulator = SingleManipulator(
            prim_path=self.prim_path,
            name="doosan",
            end_effector_prim_path=f"{self.prim_path}/OMY_custom_no_delay/OMY/link6",
            gripper=gripper,
        )
        joints_default_positions = np.zeros(10)
        # joints_default_positions[6] = 0
        # joints_default_positions[7] = 0
        manipulator.set_joints_default_state(positions=torch.tensor(joints_default_positions, dtype=torch.float32))
        return manipulator

    def set_world_pose(self, position, rotation):
        if self.robot_prim.HasAttribute("xformOp:translate"):
            self.robot_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(position))
        
        
        return
    @property
    def get_robot_name(self):
        return self._robot.name
    



    def set_solver(self):
        kinematics_solver = LulaKinematicsSolver(
                            robot_description_path  = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom.yaml",
                            urdf_path               = "/nas/ochansol/isaac/USD/robots/manipulator/Robotis_OMY/config/OMY_custom_org.urdf"
        )

        return kinematics_solver


    def compute_ik(self,
        target_position : Optional[list],
        target_orientation : Optional[list], 
        frame_name : str = None,
        warm_start : np.ndarray = np.array([0.3,0.3,0.3,0.3,0.3,0.3])
    ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        if type(target_orientation) == list:
            target_orientation = np.array(target_orientation)
        ik = self.kinematics_solver.compute_inverse_kinematics(
            frame_name = frame_name,
            target_position = target_position ,
            target_orientation = euler_angles_to_quat( target_orientation/180*np.pi),
            warm_start=warm_start
            )
        return ik

    def compute_fk(self, 
        frame_name:str, 
        joint_positions : Optional[np.ndarray] 
        ):
        if frame_name == None : 
            frame_name = self.kinematics_solver.get_all_frame_names()[7]; print(frame_name)
        fk = self.kinematics_solver.compute_forward_kinematics(frame_name =frame_name, joint_positions = joint_positions)
        print("fk : ", fk)
        return fk


    def pre_step(self, current_time_step_index, current_time):
        return
    
    def post_reset(self):
        
        return
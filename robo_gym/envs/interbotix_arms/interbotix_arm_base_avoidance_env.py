#!/usr/bin/env python3
import copy
import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
from robo_gym.utils import utils
from typing import Tuple, Any
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.envs.interbotix_arms.interbotix_arm_base_env import InterbotixABaseEnv


DEBUG = True
# waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
JOINT_POSITIONS_6DOF = [0.0757, 0.0074, 0.0122, -0.00011, 0.0058, -0.00076]
JOINT_POSITIONS_5DOF = [0.1022, 0.0297, -0.00017, 0.00595, 0]
JOINT_POSITIONS_4DOF = [0.1056, 0.0913, 0.00178, 0.0008]


class InterbotixABaseAvoidanceEnv(InterbotixABaseEnv):
    """Interbotix avoidance base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_base (bool): Whether the base joint stays fixed or is movable. Defaults to False.
        fix_shoulder (bool): Whether the shoulder joint stays fixed or is movable. Defaults to False.
        fix_elbow (bool): Whether the elbow joint stays fixed or is movable. Defaults to False.
        fix_forearm_roll (bool): Whether the forearm roll joint stays fixed or is movable. Defaults to False.
        fix_wrist_angle (bool): Whether the wrist angle joint stays fixed or is movable. Defaults to False.
        fix_wrist_rotate (bool): Whether the wrist rotate joint stays fixed or is movable. Defaults to True.
        robot_model (str): determines which interbotix model will be used in the environment. Defaults to 'rx150'.
        include_polar_to_elbow (bool): determines whether the polar coordinates to the elbow joint are included in the
        state. Defaults to False.

    Attributes:
        interbotix (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_forearm_roll=False,
                 fix_wrist_angle=False, fix_wrist_rotate=True, robot_model='rx150', include_polar_to_elbow=False,
                 rs_state_to_info=True, **kwargs):
        self.include_polar_to_elbow = include_polar_to_elbow
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_forearm_roll, fix_wrist_angle,
                         fix_wrist_rotate, robot_model)
        self.elapsed_steps = 0
        if self.interbotix.dof == 4:
            self.joint_positions_list = JOINT_POSITIONS_4DOF
        elif self.interbotix.dof == 5:
            self.joint_positions_list = JOINT_POSITIONS_5DOF
        else:
            self.joint_positions_list = JOINT_POSITIONS_6DOF

        self.rs_state = None
        self.fixed_joints = []
        
    def _set_initial_robot_server_state(self, rs_state, fixed_object_position) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        # Set initial state of the Robot Server
        if fixed_object_position:
            # Object in a fixed position
            string_params = {"object_0_function": "fixed_position"}
            float_params = {"object_0_x": fixed_object_position[0], 
                            "object_0_y": fixed_object_position[1], 
                            "object_0_z": fixed_object_position[2]}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params, string_params=string_params,
                                           state_dict=rs_state)
        return state_msg

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Environment reset.

        Args:
            joint_positions (list[dof] or np.array[dof]): robot joint positions in radians.
            fixed_object_position (list[3]): x,y,z fixed position of object

        Returns:
            np.array: Environment state.

        """        
        super(InterbotixABaseEnv, self).reset(seed=seed)

        self.elapsed_steps = 0
        if options is None:
            options = {}
        joint_positions = (
            options["joint_positions"] if "joint_positions" in options else None
        )
        fixed_object_position = (
            options["fixed_object_position"]
            if "fixed_object_position" in options
            else None
        )

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Initialize desired joint positions
        if joint_positions: 
            assert len(joint_positions) == self.interbotix.dof
        else:
            joint_positions = self.joint_positions_list

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(
            rs_state, fixed_object_position)

        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state = self.client.get_state_msg().state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.observation_space.contains(state):
            raise InvalidStateError()

        # Check if current position is in the range of the initial joint positions
        for joint in self.joint_positions.keys():
            if not np.isclose(self.joint_positions[joint], rs_state[joint], atol=0.05):
                raise InvalidStateError('Reset joint positions are not within defined range')

        self.rs_state = rs_state

        return state, {}

    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Object polar coordinates
        # Transform cartesian coordinates of object to polar coordinates 
        # with respect to the end effector frame
        object_coord = np.array([
            rs_state['object_0_to_ref_translation_x'], 
            rs_state['object_0_to_ref_translation_y'],
            rs_state['object_0_to_ref_translation_z']])

        ee_to_ref_frame_translation = np.array([
            rs_state['ee_to_ref_translation_x'], 
            rs_state['ee_to_ref_translation_y'],
            rs_state['ee_to_ref_translation_z']])

        ee_to_ref_frame_quaternion = np.array([
            rs_state['ee_to_ref_rotation_x'], 
            rs_state['ee_to_ref_rotation_y'],
            rs_state['ee_to_ref_rotation_z'],
            rs_state['ee_to_ref_rotation_w']])

        ee_to_ref_frame_rotation = R.from_quat(ee_to_ref_frame_quaternion)
        ref_frame_to_ee_rotation = ee_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_ee_quaternion = ref_frame_to_ee_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_ee_translation = -ref_frame_to_ee_rotation.apply(ee_to_ref_frame_translation)

        object_coord_ee_frame = utils.change_reference_frame(object_coord,ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        object_polar = utils.cartesian_to_polar_3d(object_coord_ee_frame)


        # Joint positions 
        joint_positions = []
        joint_positions_keys = self.joint_list
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.interbotix.normalize_joint_values(joints=joint_positions)

        # joint positions at start
        starting_joints = self.interbotix.normalize_joint_values(self._get_joint_positions_as_array())
        # difference in position from start to current
        delta_joints = joint_positions - starting_joints
        
        # Transform cartesian coordinates of object to polar coordinates 
        # with respect to the forearm
        forearm_to_ref_frame_translation = np.array([
            rs_state['forearm_to_ref_translation_x'], 
            rs_state['forearm_to_ref_translation_y'],
            rs_state['forearm_to_ref_translation_z']])

        forearm_to_ref_frame_quaternion = np.array([
            rs_state['forearm_to_ref_rotation_x'], 
            rs_state['forearm_to_ref_rotation_y'],
            rs_state['forearm_to_ref_rotation_z'],
            rs_state['forearm_to_ref_rotation_w']])
        forearm_to_ref_frame_rotation = R.from_quat(forearm_to_ref_frame_quaternion)
        ref_frame_to_forearm_rotation = forearm_to_ref_frame_rotation.inv()
        # to invert the homogeneous transformation
        # R' = R^-1
        ref_frame_to_forearm_quaternion = ref_frame_to_forearm_rotation.as_quat()
        # t' = - R^-1 * t
        ref_frame_to_forearm_translation = -ref_frame_to_forearm_rotation.apply(forearm_to_ref_frame_translation)

        object_coord_forearm_frame = utils.change_reference_frame(object_coord, ref_frame_to_forearm_translation,
                                                                  ref_frame_to_forearm_quaternion)
        object_polar_forearm = utils.cartesian_to_polar_3d(object_coord_forearm_frame)

        # Compose environment state
        if self.include_polar_to_elbow:
            state = np.concatenate((object_polar, joint_positions, delta_joints, object_polar_forearm))
        else:
            state = np.concatenate((object_polar, joint_positions, delta_joints, np.zeros(3)))

        return state.astype(np.float32)

    def _get_observation_space(self) -> gym.spaces.Box:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        # Joint position range tolerance
        pos_tolerance = np.full(self.interbotix.dof, 0.1)
        # Joint positions range used to determine if there is an error in the sensor readings
        max_joint_positions = np.add(np.full(self.interbotix.dof, 1.0), pos_tolerance)
        min_joint_positions = np.subtract(np.full(self.interbotix.dof, -1.0), pos_tolerance)
        # Target coordinates range
        target_range = np.full(3, np.inf)
        
        max_delta_start_positions = np.add(np.full(self.interbotix.dof, 1.0), pos_tolerance)
        min_delta_start_positions = np.subtract(np.full(self.interbotix.dof, -1.0), pos_tolerance)

        # Definition of environment observation_space
        if self.include_polar_to_elbow:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, target_range))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, -target_range))
        else:
            max_obs = np.concatenate((target_range, max_joint_positions, max_delta_start_positions, np.zeros(3)))
            min_obs = np.concatenate((-target_range, min_joint_positions, min_delta_start_positions, np.zeros(3)))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def add_fixed_joints(self, action) -> np.ndarray:
        action = action.tolist()
        if self.interbotix.dof == 4:
            self.fixed_joints = np.array([self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_angle])

        elif self.interbotix.dof == 5:
            self.fixed_joints = np.array(
                [self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_wrist_angle, self.fix_wrist_rotate])
        else:
            self.fixed_joints = np.array(
                [self.fix_base, self.fix_shoulder, self.fix_elbow, self.fix_forearm_roll, self.fix_wrist_angle,
                 self.fix_wrist_rotate])

        fixed_joint_indices = np.where(self.fixed_joints)[0]

        temp = []
        for joint in range(len(self.fixed_joints)):
            if joint in fixed_joint_indices:
                temp.append(0)
            else:
                temp.append(action.pop(0))
        return np.array(temp)

    def env_action_to_rs_action(self, action) -> np.ndarray:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)

        joint_positions = self._get_joint_positions_as_array() + action

        rs_action = self.interbotix._interbotix_joint_list_to_ros_joint_list(joint_positions)

        return rs_action   

    def _get_robot_server_state_len(self) -> int:

        """Get length of the Robot Server state.

        Describes the composition of the Robot Server state and returns
        its length.

        Returns:
            int: Length of the Robot Server state.

        """
        return len(self.get_robot_server_composition())  
    
    def get_robot_server_composition(self) -> list:
        if self.interbotix.dof == 4:
            rs_state_keys = [
                'object_0_to_ref_translation_x',
                'object_0_to_ref_translation_y',
                'object_0_to_ref_translation_z',
                'object_0_to_ref_rotation_x',
                'object_0_to_ref_rotation_y',
                'object_0_to_ref_rotation_z',
                'object_0_to_ref_rotation_w',

                'base_joint_position',
                'shoulder_joint_position',
                'elbow_joint_position',
                'wrist_angle_joint_position',

                'base_joint_velocity',
                'shoulder_joint_velocity',
                'elbow_joint_velocity',
                'wrist_angle_joint_velocity',

                'ee_to_ref_translation_x',
                'ee_to_ref_translation_y',
                'ee_to_ref_translation_z',
                'ee_to_ref_rotation_x',
                'ee_to_ref_rotation_y',
                'ee_to_ref_rotation_z',
                'ee_to_ref_rotation_w',

                'forearm_to_ref_translation_x',
                'forearm_to_ref_translation_y',
                'forearm_to_ref_translation_z',
                'forearm_to_ref_rotation_x',
                'forearm_to_ref_rotation_y',
                'forearm_to_ref_rotation_z',
                'forearm_to_ref_rotation_w',

                'in_collision']
        elif self.interbotix.dof == 5:
            rs_state_keys = [
                'object_0_to_ref_translation_x',
                'object_0_to_ref_translation_y',
                'object_0_to_ref_translation_z',
                'object_0_to_ref_rotation_x',
                'object_0_to_ref_rotation_y',
                'object_0_to_ref_rotation_z',
                'object_0_to_ref_rotation_w',

                'base_joint_position',
                'shoulder_joint_position',
                'elbow_joint_position',
                'wrist_angle_joint_position',
                'wrist_rotate_joint_position',

                'base_joint_velocity',
                'shoulder_joint_velocity',
                'elbow_joint_velocity',
                'wrist_angle_joint_velocity',
                'wrist_rotate_joint_velocity',

                'ee_to_ref_translation_x',
                'ee_to_ref_translation_y',
                'ee_to_ref_translation_z',
                'ee_to_ref_rotation_x',
                'ee_to_ref_rotation_y',
                'ee_to_ref_rotation_z',
                'ee_to_ref_rotation_w',

                'forearm_to_ref_translation_x',
                'forearm_to_ref_translation_y',
                'forearm_to_ref_translation_z',
                'forearm_to_ref_rotation_x',
                'forearm_to_ref_rotation_y',
                'forearm_to_ref_rotation_z',
                'forearm_to_ref_rotation_w',

                'in_collision']
        else:
            rs_state_keys = [
                'object_0_to_ref_translation_x',
                'object_0_to_ref_translation_y',
                'object_0_to_ref_translation_z',
                'object_0_to_ref_rotation_x',
                'object_0_to_ref_rotation_y',
                'object_0_to_ref_rotation_z',
                'object_0_to_ref_rotation_w',

                'base_joint_position',
                'shoulder_joint_position',
                'elbow_joint_position',
                'forearm_roll_joint_position',
                'wrist_angle_joint_position',
                'wrist_rotate_joint_position',

                'base_joint_velocity',
                'shoulder_joint_velocity',
                'elbow_joint_velocity',
                'forearm_roll_joint_velocity',
                'wrist_angle_joint_velocity',
                'wrist_rotate_joint_velocity',

                'ee_to_ref_translation_x',
                'ee_to_ref_translation_y',
                'ee_to_ref_translation_z',
                'ee_to_ref_rotation_x',
                'ee_to_ref_rotation_y',
                'ee_to_ref_rotation_z',
                'ee_to_ref_rotation_w',

                'forearm_to_ref_translation_x',
                'forearm_to_ref_translation_y',
                'forearm_to_ref_translation_z',
                'forearm_to_ref_rotation_x',
                'forearm_to_ref_rotation_y',
                'forearm_to_ref_rotation_z',
                'forearm_to_ref_rotation_w',

                'in_collision']

        return rs_state_keys

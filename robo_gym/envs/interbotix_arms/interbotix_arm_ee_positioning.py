import copy
import numpy as np
import gym
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError
from robo_gym.utils import utils
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.envs.interbotix_arms.interbotix_arm_base_env import InterbotixABaseEnv

# waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
JOINT_POSITIONS_6DOF = [0.0757, 0.0074, 0.0122, -0.00011, 0.0058, -0.00076]
JOINT_POSITIONS_5DOF = [0.1022, 0.0297, -0.00017, 0.00595, 0]
JOINT_POSITIONS_4DOF = [0.1056, 0.0913, 0.00178, 0.0008]

RANDOM_JOINT_OFFSET_4DOF = [1.5, 0.25, 0.5, 1.0]
RANDOM_JOINT_OFFSET_5DOF = [1.5, 0.25, 0.5, 1.0, 0.4]
RANDOM_JOINT_OFFSET_6DOF = [1.5, 0.25, 0.5, 1.0, 0.4, 3.14]

# distance to target that need to be reached
DISTANCE_THRESHOLD = 0.1


class EndEffectorPositioningInterbotix(InterbotixABaseEnv):
    """Interbotix end effector positioning environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        fix_base (bool): Whether the base joint stays fixed or is movable. Defaults to False.
        fix_shoulder (bool): Whether the shoulder joint stays fixed or is movable. Defaults to False.
        fix_elbow (bool): Whether the elbow joint stays fixed or is movable. Defaults to False.
        fix_forearm_roll (bool): Whether the forearm roll joint stays fixed or is movable. Defaults to False.
        fix_wrist_angle (bool): Whether the wrist angle joint stays fixed or is movable. Defaults to False.
        fix_wrist_rotate (bool): Whether the wrist rotate joint stays fixed or is movable. Defaults to True.
        robot_model (str): determines which robot model will be used in the environment. Default to 'rx150'.

    Attributes:
        interbotix (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    def __init__(self, rs_address=None, fix_base=False, fix_shoulder=False, fix_elbow=False, fix_forearm_roll=False,
                 fix_wrist_rotate=False, fix_wrist_angle=False, robot_model='rx150', rs_state_to_info=True, **kwargs):
        super().__init__(rs_address, fix_base, fix_shoulder, fix_elbow, fix_forearm_roll, fix_wrist_angle,
                         fix_wrist_rotate, robot_model, rs_state_to_info)

        self.successful_ending = False
        self.last_position = np.zeros(self.interbotix.dof)
        self.elapsed_steps = 0
        self.previous_action = []

        if self.interbotix.dof == 4:
            self.joint_positions_list = JOINT_POSITIONS_4DOF
            self.random_joint_offset = RANDOM_JOINT_OFFSET_4DOF
        elif self.interbotix.dof == 5:
            self.joint_positions_list = JOINT_POSITIONS_5DOF
            self.random_joint_offset = RANDOM_JOINT_OFFSET_5DOF
        else:
            self.joint_positions_list = JOINT_POSITIONS_6DOF
            self.random_joint_offset = RANDOM_JOINT_OFFSET_6DOF

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
        # Joint velocities range 
        max_joint_velocities = np.array([np.inf] * self.interbotix.dof)
        min_joint_velocities = - np.array([np.inf] * self.interbotix.dof)
        # Cartesian coords of the target location
        max_target_coord = np.array([np.inf] * 3)
        min_target_coord = - np.array([np.inf] * 3)
        # Cartesian coords of the end effector
        max_ee_coord = np.array([np.inf] * 3)
        min_ee_coord = - np.array([np.inf] * 3)
        # Previous action
        max_action = np.array([1.01] * self.interbotix.dof)
        min_action = - np.array([1.01] * self.interbotix.dof)
        # Definition of environment observation_space
        max_obs = np.concatenate((target_range, max_joint_positions, max_joint_velocities, max_target_coord, max_ee_coord, max_action))
        min_obs = np.concatenate((-target_range, min_joint_positions, min_joint_velocities, min_target_coord, min_ee_coord, min_action))

        return gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def _set_initial_robot_server_state(self, rs_state, ee_target_pose) -> robot_server_pb2.State:
        string_params = {"object_0_function": "fixed_position"}
        float_params = {"object_0_x": ee_target_pose[0],
                        "object_0_y": ee_target_pose[1],
                        "object_0_z": ee_target_pose[2]}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params, string_params=string_params,
                                           state_dict=rs_state)
        return state_msg

    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        # Target polar coordinates
        # Transform cartesian coordinates of target to polar coordinates 
        # with respect to the end effector frame
        target_coord = np.array([
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

        target_coord_ee_frame = utils.change_reference_frame(target_coord, ref_frame_to_ee_translation,ref_frame_to_ee_quaternion)
        target_polar = utils.cartesian_to_polar_3d(target_coord_ee_frame)

        # Joint positions 
        joint_positions = []
        joint_positions_keys = self.joint_list
        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)
        # Normalize joint position values
        joint_positions = self.interbotix.normalize_joint_values(joints=joint_positions)

        # Joint Velocities
        joint_velocities = [] 
        joint_velocities_keys = self.joint_velocity_list
        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)

        # Compose environment state
        state = np.concatenate((target_polar, joint_positions, joint_velocities, target_coord,
                                ee_to_ref_frame_translation, self.previous_action))

        return state.astype(np.float32)

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

                'in_collision'
            ]
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

                'in_collision'
            ]
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

                'in_collision'
            ]
        return rs_state_keys

    def reset(self, joint_positions=None, ee_target_pose=None, randomize_start=False, continue_on_success=False) -> np.ndarray:
        """Environment reset.

        Args:
            joint_positions (list[dof] or np.array[dof]): robot joint positions in radians.
            ee_target_pose (list[6] or np.array[6]): [x,y,z,r,p,y] target end effector pose.
            randomize_start (bool): if True the starting position is randomized defined by the RANDOM_JOINT_OFFSET
            continue_on_success (bool): if True the next robot will continue from it current position when last episode was a success
        """

        if joint_positions: 
            assert len(joint_positions) == self.interbotix.dof
        else:
            joint_positions = self.joint_positions_list

        self.elapsed_steps = 0
        self.previous_action = np.zeros(self.interbotix.dof)

        # Initialize environment state
        state_len = self.observation_space.shape[0]
        state = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(), 0.0)

        # Randomize initial robot joint positions
        if randomize_start:
            joint_positions_low = np.array(joint_positions) - np.array(self.random_joint_offset)
            joint_positions_high = np.array(joint_positions) + np.array(self.random_joint_offset)
            joint_positions = np.random.default_rng().uniform(low=joint_positions_low, high=joint_positions_high)

        # Continue from last position if last episode was a success
        if self.successful_ending and continue_on_success:
            joint_positions = self.last_position

        # Set initial robot joint positions
        self._set_joint_positions(joint_positions)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)
        self.ee_target_pose = ee_target_pose

        # Set target End Effector pose
        if self.ee_target_pose:
            assert len(self.ee_target_pose) == 6
        else:
            self.ee_target_pose = self._get_target_pose()

        # Set initial state of the Robot Server

        state_msg = self._set_initial_robot_server_state(rs_state, self.ee_target_pose)

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
            
        return state

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        if type(action) == list:
            action = np.array(action)

        action = action.astype(np.float32)
        
        state, reward, done, info = super().step(action)
        self.previous_action = self.add_fixed_joints(action)

        if done:
            if info['final_status'] == 'success':
                self.successful_ending = True

                joint_positions = []
                joint_positions_keys = self.joint_list

                for position in joint_positions_keys:
                    joint_positions.append(self.rs_state[position])
                joint_positions = np.array(joint_positions)
                self.last_position = joint_positions

        return state, reward, done, info

    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        reward = 0
        done = False
        info = {}

        # Reward weight for reaching the goal position
        g_w = 2
        # Reward weight for collision (ground, table or self)
        c_w = -1
        # Reward weight according to the distance to the goal
        d_w = -0.005

        # Calculate distance to the target
        target_coord = np.array([rs_state['object_0_to_ref_translation_x'], rs_state['object_0_to_ref_translation_y'],
                                 rs_state['object_0_to_ref_translation_z']])
        ee_coord = np.array([rs_state['ee_to_ref_translation_x'], rs_state['ee_to_ref_translation_y'],
                             rs_state['ee_to_ref_translation_z']])
        euclidean_dist_3d = np.linalg.norm(target_coord - ee_coord)

        # Reward base
        reward += d_w * euclidean_dist_3d

        if euclidean_dist_3d <= DISTANCE_THRESHOLD:
            reward = g_w * 1
            done = True
            info['final_status'] = 'success'
            info['target_coord'] = target_coord

        if rs_state['in_collision']:
            reward = c_w * 1
            done = True
            info['final_status'] = 'collision'
            info['target_coord'] = target_coord

        elif self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'max_steps_exceeded'
            info['target_coord'] = target_coord
        
        return reward, done, info

    def _get_target_pose(self) -> np.ndarray:
        """Generate target End Effector pose.

        Returns:
            np.array: [x,y,z,alpha,theta,gamma] pose.

        """
        return self.interbotix.get_random_workspace_pose()

    def env_action_to_rs_action(self, action) -> np.array:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)
        rs_action = super().env_action_to_rs_action(rs_action)

        return rs_action

        
class EndEffectorPositioningInterbotixASim(EndEffectorPositioningInterbotix, Simulation):
    cmd = "roslaunch interbotix_arm_robot_server interbotix_arm_robot_server.launch \
        world_name:=tabletop_sphere50_no_collision.world \
        max_velocity_scale_factor:=0.1 \
        action_cycle_rate:=10 \
        rviz_gui:=false \
        gui:=true \
        gazebo_gui:=true \
        objects_controller:=true \
        rs_mode:=1object \
        n_objects:=1.0 \
        object_0_model_name:=sphere50_no_collision \
        object_0_frame:=target"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, robot_model='rx150', **kwargs):
        self.cmd = self.cmd + ' ' + 'robot_model:=' + robot_model + ' reference_frame:=' + robot_model + "/base_link"
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        EndEffectorPositioningInterbotix.__init__(self, rs_address=self.robot_server_ip, robot_model=robot_model, **kwargs)


class EndEffectorPositioningInterbotixARob(EndEffectorPositioningInterbotix):
    real_robot = True

# roslaunch interbotix_arm_robot_server interbotix_arm_robot_server.launch robot_model:=rx150 real_robot:=true rviz_gui:=true gui:=true reference_frame:=base max_velocity_scale_factor:=0.2 action_cycle_rate:=20 objects_controller:=true rs_mode:=1object n_objects:=1.0 object_0_frame:=target
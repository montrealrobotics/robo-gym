#!/usr/bin/env python3
import math
import copy
import numpy as np
import gymnasium as gym
from typing import Tuple, Any
from robo_gym.utils import interbotix_utils
from robo_gym.utils.exceptions import InvalidStateError, RobotServerError, InvalidActionError
import robo_gym_server_modules.robot_server.client as rs_client
from robo_gym_server_modules.robot_server.grpc_msgs.python import robot_server_pb2
from robo_gym.envs.simulation_wrapper import Simulation
from robo_gym.utils.camera import RoboGymCamera
# waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
JOINT_POSITIONS_6DOF = [0.0757, 0.0074, 0.0122, -0.00011, 0.0058, -0.00076]
JOINT_POSITIONS_5DOF = [0.1022, 0.0297, -0.00017, 0.00595, 0]
JOINT_POSITIONS_4DOF = [0.1056, 0.0913, 0.00178, 0.0008]

IMAGE_SHAPE = [120, 160, 3]


class InterbotixRBaseEnv(gym.Env):
    """Interbotix rover base environment.

    Args:
        rs_address (str): Robot Server address. Formatted as 'ip:port'. Defaults to None.
        robot_model (str): determines which interbotix rover model will be used in the environment. Default to 'locobot_wx250s'.

    Attributes:
        interbotix (:obj:): Robot utilities object.
        client (:obj:str): Robot Server client.
        real_robot (bool): True if the environment is controlling a real robot.

    """
    real_robot = False
    max_episode_steps = 100

    def __init__(self, rs_address=None, robot_model='locobot_wz250s', rs_state_to_info=True, with_camera=False, **kwargs):

        arm_model = robot_model.split('_')[1]
        self.interbotix = interbotix_utils.InterbotixArm(model=arm_model)
        if self.interbotix.dof == 4:
            self.joint_list = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                               'wrist_angle_joint_position']
            self.joint_velocity_list = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                                        'wrist_angle_joint_velocity']
        elif self.interbotix.dof == 5:
            self.joint_list = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                               'wrist_angle_joint_position', 'wrist_rotate_joint_position']
            self.joint_velocity_list = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                                        'wrist_angle_joint_velocity', 'wrist_rotate_joint_velocity']
        else:
            self.joint_list = ['base_joint_position', 'shoulder_joint_position', 'elbow_joint_position',
                               'forearm_roll_joint_position', 'wrist_angle_joint_position',
                               'wrist_rotate_joint_position']
            self.joint_velocity_list = ['base_joint_velocity', 'shoulder_joint_velocity', 'elbow_joint_velocity',
                                        'forearm_roll_joint_velocity', 'wrist_angle_joint_velocity',
                                        'wrist_rotate_joint_velocity']
            
        self.base_pose_list = ['base_position_x', 'base_position_y', 'base_position_z', 'base_orientation_x', 
                               'base_orientation_y', 'base_orientation_z', 'base_orientation_w']

        self.base_joint_wheel_list = ['right_wheel_joint_position', 'left_wheel_joint_position'] 
        self.base_joint_wheel_vel_list = ['right_wheel_joint_velocity', 'left_wheel_joint_velocity']

        self.elapsed_steps = 0
        self.camera = with_camera
        if self.camera:
            self.camera_config = RoboGymCamera(name='camera', image_shape=IMAGE_SHAPE,
                                        image_mode='temporal', context_size=3, num_cameras=1)

        self.rs_state_to_info = rs_state_to_info

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        self.abs_joint_pos_range = self.interbotix.get_max_joint_positions()

        self.rs_state = None
        
        # Connect to Robot Server
        if rs_address:
            self.client = rs_client.Client(rs_address)
        else:
            print("WARNING: No IP and Port passed. Simulation will not be started")
            print("WARNING: Use this only to get environment shape")
            
    def _set_initial_robot_server_state(self, rs_state) -> robot_server_pb2.State:
        string_params = {}
        float_params = {}
        state = {}

        state_msg = robot_server_pb2.State(state=state, float_params=float_params,
                                           string_params=string_params, state_dict=rs_state)
        return state_msg
    
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Environment reset.

        Args:
            joint_positions (list[6] or np.array[6]): robot joint positions in radians. Order is defined by 
        
        Returns:
            np.array: Environment state.

        """
        super().reset(seed=seed)

        if options is None:
            options = {}
        joint_positions = (
            options["joint_positions"] if "joint_positions" in options else None
        )

        if joint_positions: 
            assert len(joint_positions) == 6
        else:
            if self.interbotix.dof == 4:
                joint_positions = JOINT_POSITIONS_4DOF
            elif self.interbotix.dof == 5:
                joint_positions = JOINT_POSITIONS_5DOF
            else:
                joint_positions = JOINT_POSITIONS_6DOF

        self.elapsed_steps = 0

        # Initialize environment state
        state_len = self.observation_space['state'].shape[0]
        state = {}
        state['state'] = np.zeros(state_len)
        rs_state = dict.fromkeys(self.get_robot_server_composition(state=False), 0.0)

        # Set initial robot joint goals
        base_velocity = [0, 0]
        self._set_joint_goals(joint_positions, base_velocity)

        # Update joint positions in rs_state
        rs_state.update(self.joint_positions)

        # Set initial state of the Robot Server
        state_msg = self._set_initial_robot_server_state(rs_state)
        
        if not self.client.set_state_msg(state_msg):
            raise RobotServerError("set_state")

        # Get Robot Server state
        rs_state_all = self.client.get_state_msg()
        rs_state = rs_state_all.state_dict

        # Check if the length and keys of the Robot Server state received is correct
        self._check_rs_state_keys(rs_state)

        # Convert the initial state from Robot Server format to environment format
        state['state'] = self._robot_server_state_to_env_state(rs_state)

        # Check if the environment state is contained in the observation space
        if not self.camera and not self.observation_space['state'].contains(state['state']):
            raise InvalidStateError()

        # Check if current position is in the range of the initial joint positions
        for joint in self.joint_positions.keys():
            # do not check the commands for base velocity
            if not 'velocity' in joint:
                if not np.isclose(self.joint_positions[joint], rs_state[joint], atol=0.075):
                    raise InvalidStateError('Reset joint positions are not within defined range')

        self.rs_state = rs_state
        if self.camera:
             state['camera'] = self.camera_config.process_camera_images(rs_state_all.string_params)

        if self.camera:
            return state, {}
        else:
            return state['state'], {}
    
    def reward(self, rs_state, action) -> Tuple[float, bool, dict]:
        done = False
        info = {}

        if self.elapsed_steps >= self.max_episode_steps:
            done = True
            info['final_status'] = 'success'

        return 0, done, info
    
    def env_action_to_rs_action(self, action) -> np.ndarray:
        """Convert environment action to Robot Server action"""
        rs_action = copy.deepcopy(action)

        # Convert action indexing from interbotix to ros for arm joints
        rs_action_arm = self.interbotix._interbotix_joint_list_to_ros_joint_list(rs_action[:-2])

        # add back in the base velocity actions
        rs_action_all = np.concatenate((rs_action_arm, np.array(rs_action[-2:])))
    
        return rs_action_all        

    def step(self, action) -> Tuple[np.array, float, bool, dict]:
        # action should be a list/array with the following joint angles & velocities, depending on 
        # dof some arm joints can be ommitted.
        #  [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, base_velocity_liner_x, base_velocity_angular_z]
        
        if type(action) == list:
            action = np.array(action)

        rs_action = []
        self.elapsed_steps += 1
        state = {}

        action = action.astype(np.float32)

        # Check if the action is contained in the action space
        if not self.action_space.contains(action):
            raise InvalidActionError()

        # Convert environment action to robot server action
        rs_action = self.env_action_to_rs_action(action)

        # Send action to Robot Server and get state
        rs_state_all = self.client.send_action_get_state(rs_action.tolist())
        rs_state = rs_state_all.state_dict

        self._check_rs_state_keys(rs_state)

        # Convert the state from Robot Server format to environment format
        state['state'] = self._robot_server_state_to_env_state(rs_state)

        if self.camera:
             state['camera'] = self.camera_config.process_camera_images(rs_state_all.string_params)

            # cv2.imshow('Decoded Image', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        # Check if the environment state is contained in the observation space
        if not self.camera and not self.observation_space['state'].contains(state['state']):
            raise InvalidStateError()

        self.rs_state = rs_state

        reward = 0
        done = False
        reward, done, info = self.reward(rs_state=rs_state, action=action)
        if self.rs_state_to_info:
            info['rs_state'] = self.rs_state
        if self.camera:
            return state, reward, done, False, info
        else:
            return state['state'], reward, done, False, info

    def get_rs_state(self):
        return self.rs_state

    def render(self):
        pass
    
    def get_robot_server_composition(self, state=True) -> list:
        if self.interbotix.dof == 4:
            rs_state_keys = [
                'base_joint_position',
                'shoulder_joint_position',
                'elbow_joint_position',
                'wrist_angle_joint_position',
                
                'base_joint_velocity',
                'shoulder_joint_velocity',
                'elbow_joint_velocity',
                'wrist_angle_joint_velocity',
            ]
            
            
        elif self.interbotix.dof == 5:
            rs_state_keys = [
                'base_joint_position',
                'shoulder_joint_position',
                'elbow_joint_position',
                'wrist_angle_joint_position',
                'wrist_rotate_joint_position',

                'base_joint_velocity',
                'shoulder_joint_velocity',
                'elbow_joint_velocity',
                'wrist_angle_joint_velocity',
                'wrist_rotate_joint_velocity'
            ]
        else:
            rs_state_keys = [
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
                'wrist_rotate_joint_velocity'
            ]
            
        if state:
                rs_state_keys += self.base_joint_wheel_list
                rs_state_keys += self.base_joint_wheel_vel_list
                rs_state_keys += self.base_pose_list
        else:
                rs_state_keys += ['base_velocity_x', 'base_velocity_z']

        return rs_state_keys

    def _set_joint_goals(self, joint_positions, base_velocity) -> None:
        """Set desired robot joint positions/velocities with standard indexing."""
        # Set initial robot joint goals
        self.joint_positions = {}
        for i in range(self.interbotix.dof):
            self.joint_positions[self.joint_list[i]] = joint_positions[i]
            
        self.joint_positions['base_velocity_x'] = base_velocity[0]
        self.joint_positions['base_velocity_z'] = base_velocity[1]
        
    def _check_rs_state_keys(self, rs_state) -> None:
        keys = self.get_robot_server_composition()
        
        if not len(keys) == len(rs_state.keys()):
            raise InvalidStateError("Robot Server state keys to not match. Different lengths.")

        for key in keys:
            if key not in rs_state.keys():
                raise InvalidStateError("Robot Server state keys to not match")
            
    def _robot_server_state_to_env_state(self, rs_state) -> np.ndarray:
        """Transform state from Robot Server to environment format.

        Args:
            rs_state (list): State in Robot Server format.

        Returns:
            numpy.array: State in environment format.

        """
        joint_positions = []
        joint_positions_keys = self.joint_list + self.base_joint_wheel_list

        for position in joint_positions_keys:
            joint_positions.append(rs_state[position])
        joint_positions = np.array(joint_positions)

        joint_velocities = [] 
        joint_velocities_keys = self.joint_velocity_list + self.base_joint_wheel_vel_list

        for velocity in joint_velocities_keys:
            joint_velocities.append(rs_state[velocity])
        joint_velocities = np.array(joint_velocities)
        
        base_pose_keys = self.base_pose_list
        base_pose = []
        for p in base_pose_keys:
            base_pose.append(rs_state[p])
            
        base_pose = np.array(base_pose)

        # Compose environment state
        state = np.concatenate((joint_positions, joint_velocities, base_pose))

        return state.astype(np.float64)
    
    def _get_observation_space(self) -> gym.spaces.Dict:
        """Get environment observation space.

        Returns:
            gym.spaces: Gym observation space object.

        """
        # Joint position range tolerance
        dof = self.interbotix.dof
        pos_tolerance = np.full(dof, 0.1)
        
        max_joint_positions = np.add(self.interbotix.get_max_joint_positions(), pos_tolerance)
        min_joint_positions = np.subtract(self.interbotix.get_min_joint_positions(), pos_tolerance)
        max_joint_velocities = np.add(self.interbotix.get_max_joint_velocities(), pos_tolerance)
        min_joint_velocities = np.subtract(self.interbotix.get_min_joint_velocities(), pos_tolerance)
        
        max_wheel_positions = np.array([math.pi, math.pi])
        min_wheel_positions = np.array([-math.pi, -math.pi])
        max_wheel_velocity = np.array([np.inf, np.inf])
        min_wheel_velocity = np.array([-np.inf, -np.inf])
        
        max_pose = np.full(7, np.inf)
        min_pose = np.full(7, -np.inf)

        # Definition of environment observation_space
        max_obs = np.concatenate((max_joint_positions, max_wheel_positions, max_joint_velocities, 
                                  max_wheel_velocity, max_pose))
        min_obs = np.concatenate((min_joint_positions, min_wheel_positions, min_joint_velocities, 
                                  min_wheel_velocity, min_pose))
        base_obs_space = gym.spaces.Box(low=min_obs, high=max_obs, dtype=np.float64)

        if self.camera:
            camera_obs_shape = self.camera_config.observation_space
            return gym.spaces.Dict({
                'state': base_obs_space,
                'camera': gym.spaces.Box(
                    low=0, high=255,
                    shape=camera_obs_shape,
                    dtype=np.uint8
                )
            })
        else:
            return gym.spaces.Dict({
                'state': base_obs_space,
            })
    
    def _get_action_space(self) -> gym.spaces.Box:
        """Get environment action space.

        Returns:
            gym.spaces: Gym action space object.

        """
        max_joint_positions = self.interbotix.get_max_joint_positions()
        min_joint_positions = self.interbotix.get_min_joint_positions()
        
        max_base_velocity = np.array([np.inf, np.inf])
        min_base_velocity = np.array([-np.inf, -np.inf])
        
        max_action = np.concatenate((max_joint_positions, max_base_velocity))
        min_action = np.concatenate((min_joint_positions, min_base_velocity))

        return gym.spaces.Box(low=min_action, high=max_action, dtype=np.float64)


class EmptyEnvironmentInterbotixRSim(InterbotixRBaseEnv, Simulation):
    cmd = "ros2 launch interbotix_rover_robot_server interbotix_rover_robot_server.launch.py \
        world_name:=empty.world \
        reference_frame:=base_link \
        action_cycle_rate:=20.0 \
        rviz_gui:=false \
        gazebo_gui:=true"

    def __init__(self, ip=None, lower_bound_port=None, upper_bound_port=None, gui=False, robot_model='locobot_wx250s', **kwargs):
        self.cmd = self.cmd + ' ' + 'robot_model:=' + robot_model
        Simulation.__init__(self, self.cmd, ip, lower_bound_port, upper_bound_port, gui, **kwargs)
        InterbotixRBaseEnv.__init__(self, rs_address=self.robot_server_ip, robot_model=robot_model, **kwargs)


class EmptyEnvironmentInterbotixRRob(InterbotixRBaseEnv):
    real_robot = True

# ros2 launch interbotix_rover_robot_server interbotix_rover_robot_server.launch.py gui:=true reference_frame:=base
# action_cycle_rate:=20.0

from torch import jit
import torch
import robo_gym, gym
import numpy as np
import math
import time


def format_state(state_mag):

    # separate out the state into position, velocity, target and action
    polar_target = state[0:3]
    pos_start_idx = 3
    pos_end_idx = pos_start_idx + dof
    pos = state[pos_start_idx:pos_end_idx]
    vel_start_idx = pos_end_idx
    vel_end_idx = vel_start_idx + dof
    vel = state[vel_start_idx:vel_end_idx]
    targ_start_idx = vel_end_idx
    targ_end_idx = targ_start_idx + 3
    targ = state[targ_start_idx:targ_end_idx]

    act = state[-dof:]

    # the position, velocity and action received from the robot do not contain the gripper,
    # so we add them back in here as 0
    pos = np.concatenate(([0], pos, [0, 0]))
    vel = np.concatenate(([0], vel, [0, 0]))
    act = np.concatenate((act, [0]))

    #convert to float32, error thrown when not in this format
    pos = np.float32(pos)
    vel = np.float32(vel)
    targ = np.float32(targ)
    act = np.float32(act)

    # if using pwm, we need to convert the last action back to the
    # value before cos() was applied and it was scaled
    if control_type == 'pwm':
        laction = np.float32([math.acos(i / max_pwm) for i in act])
    elif control_type == 'delta_pos':
        # laction = last_action[:-1] + action
        laction = np.float32(np.concatenate((action, [0])))
    else:
        laction = act

    return pos, vel, targ, laction


real_robot = True
control_type = "delta_pos"
# fixed target for now, xyz and rotations
target = np.array([0.4, 0, 0.3, 0, 0, 0])
max_pwm = 885*2/8
# dof of robot
dof = 6

net = jit.load('policies/model_scripted_deltapos_mgsp.pt', map_location="cpu")

# depending on whether we are using the real robot or not, the robogym environment interface differs slightly
if real_robot:
    env = gym.make('EndEffectorPositioningInterbotixARob-v0', rs_address='192.168.1.30:50051', robot_model='wx250s')
else:
    env = gym.make('EndEffectorPositioningInterbotixASim-v0', ip='127.0.0.1', robot_model='wx250s', gui=True)

# reset and send the fixed target and the ee goal
state = env.reset(ee_target_pose=target)
#set last action at the start to 0s
last_action = np.zeros(dof+1)
last_action = np.float32(last_action)
action = last_action[:-1]
# format the state
positions, velocities, target, last_action = format_state(state)

done = False
# Define the desired rate (iterations per second)
rate = 60  # 60 iterations per second

# Calculate the delay between iterations
delay = 1.0 / 60
count = 0
while not done:
    # Record the start time of the iteration
    start_time = time.time()
    # we only take the first 'dof' number of values, the last one is the gripper
    action = np.array(net(torch.from_numpy(np.concatenate([positions, velocities, target, last_action]))).tolist()[:dof])
    print("action from policy: ", action)
    # apply cosine to action and scale for the pwm range
    if control_type == 'pwm':
        action = [math.cos(i)*max_pwm for i in action]
    elif control_type == 'delta_pos':
        action = action.clip(-0.1, 0.1)
    # set the wrist rotation to 0
    action[-1] = 0
    print("clipped action: ", action)
    print("last action: ", last_action)
    if control_type == 'delta_pos':
        # this shouldn't work - sometimes it does.
        action = action - last_action[:-1]
    # step with action
    print("Sending action: ", action)
    state, reward, done, info = env.step(action)

    if count == 10:
       t = 0

    # format the state
    positions, velocities, target, last_action = format_state(state)

    # handle the loop rate with timer
    # Calculate the time taken by this iteration
    iteration_time = time.time() - start_time
    sleep_time = max(0, delay - iteration_time)
    # Sleep for the remaining time to maintain the desired rate
    time.sleep(sleep_time)
    count += 1

    # if done:
    #     if control_type == 'pwm':
    #         action = [0] * dof
    #         state, reward, done, info = env.step(action)
    #     break

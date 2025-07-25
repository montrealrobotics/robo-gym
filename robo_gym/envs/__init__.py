# Example
from robo_gym.envs.example.example_env import ExampleEnvSim, ExampleEnvRob

# MiR100
from robo_gym.envs.mir100.mir100 import (
    NoObstacleNavigationMir100Sim,
    NoObstacleNavigationMir100Rob,
)
from robo_gym.envs.mir100.mir100 import (
    ObstacleAvoidanceMir100Sim,
    ObstacleAvoidanceMir100Rob,
)

# UR
from robo_gym.envs.ur.ur_base_env import EmptyEnvironmentURSim, EmptyEnvironmentURRob
from robo_gym.envs.ur.ur_ee_positioning import (
    EndEffectorPositioningURSim,
    EndEffectorPositioningURRob,
)
from robo_gym.envs.ur.ur_avoidance_basic import BasicAvoidanceURSim, BasicAvoidanceURRob
from robo_gym.envs.ur.ur_avoidance_raad import (
    AvoidanceRaad2022URSim,
    AvoidanceRaad2022URRob,
)
from robo_gym.envs.ur.ur_avoidance_raad import (
    AvoidanceRaad2022TestURSim,
    AvoidanceRaad2022TestURRob,
)

from robo_gym.envs.ur.ur_base import EmptyEnvironment2URSim, EmptyEnvironment2URRob
from robo_gym.envs.ur.ur_ee_pos import (
    EndEffectorPositioning2URSim,
    EndEffectorPositioning2URRob,
)
from robo_gym.envs.ur.ur_isaac_reach import IsaacReachURSim, IsaacReachURRob

# Panda
from robo_gym.envs.panda.panda_base import (
    EmptyEnvironmentPandaSim,
    EmptyEnvironmentPandaRob,
)
from robo_gym.envs.panda.panda_ee_pos import (
    EndEffectorPositioningPandaSim,
    EndEffectorPositioningPandaRob,
)
from robo_gym.envs.panda.panda_isaac_reach import IsaacReachPandaSim, IsaacReachPandaRob

# Interbotix Arms
from robo_gym.envs.interbotix_arms.interbotix_arm_base_env import (
    EmptyEnvironmentInterbotixASim,
    EmptyEnvironmentInterbotixARob
)
from robo_gym.envs.interbotix_arms.interbotix_arm_ee_positioning import (
    EndEffectorPositioningInterbotixASim,
    EndEffectorPositioningInterbotixARob
)
from robo_gym.envs.interbotix_arms.interbotix_arm_avoidance_basic import (
    BasicAvoidanceInterbotixASim,
    BasicAvoidanceInterbotixARob
)

# Interbotix Rover
from robo_gym.envs.interbotix_rover.interbotix_rover_base_env import EmptyEnvironmentInterbotixRSim, EmptyEnvironmentInterbotixRRob

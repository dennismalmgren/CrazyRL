from typing import Dict
import numpy as np
from crazy_rl.multi_agent.numpy.circle.circle import Circle
from pettingzoo import ParallelEnv
from learning.execution.exec_masac_torch import Actor
import torch
from pettingzoo.utils.env import AgentID

#Add some torchrl stuff
from torchrl.envs.libs import PettingZooWrapper
from torchrl.envs import RewardSum, TransformedEnv
from torch import nn
from torchrl.modules.models.multiagent import MultiAgentMLP
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs.utils import step_mdp

train_device = "cuda"
env_device = "cpu"
frames_per_batch = 100

#Define the environment and torchrl-ify it
env: ParallelEnv = Circle(
    drone_ids=np.array([0, 1]),
    render_mode=None,
    #render_mode="human",    # or real, or None
    init_flying_pos=np.array([[0, 0, 1], [2, 2, 1]]),
)
env.reset() # Hack to ensure env.agents is populated.

env = PettingZooWrapper(env,
                        group_map = {"agents": env.agents},)
env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

# Define a policy
actor_net = nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=2 * env.action_spec.shape[-1],
        n_agents=env.num_drones,
        centralised=False,
        share_params=True,
        device=train_device,
        depth=2,
        num_cells=256,
        activation_class=nn.Tanh,
    ),
    NormalParamExtractor(),
)
policy_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.action_spec.space.low,
        "max": env.action_spec.space.high,
    },
    return_log_prob=True,
)
episode = env.rollout(policy=policy, auto_cast_to_device=True, break_when_any_done=True, max_steps=1000)


print('Episode length: ', len(episode))
print('Episode reward: ', episode['next', 'agents', 'episode_reward'].sum()) #Note that each drone has its own reward returned in this setup.
print('done')
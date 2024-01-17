from typing import Dict
import numpy as np
from crazy_rl.multi_agent.numpy.circle.circle import Circle
from pettingzoo import ParallelEnv
from learning.execution.exec_masac_torch import Actor
import torch
from pettingzoo.utils.env import AgentID

def extract_agent_id(agent_str):
    """Extract agent id from agent string.

    Args:
        agent_str: Agent string in the format of "agent_{id}"

    Returns: (int) Agent id

    """
    return int(agent_str.split("_")[1])

def concat_id(local_obs: np.ndarray, id: AgentID) -> np.ndarray:
    """Concatenate the agent id to the local observation.

    Args:
        local_obs: the local observation
        id: the agent id to concatenate

    Returns: the concatenated observation

    """
    return np.concatenate([local_obs, np.array([extract_agent_id(id)], dtype=np.float32)])


env: ParallelEnv = Circle(
    drone_ids=np.array([0, 1]),
    render_mode="human",    # or real, or None
    init_flying_pos=np.array([[0, 0, 1], [2, 2, 1]]),
)

device = "cuda"
obs, info = env.reset()
actor = Actor(env)
actor = actor.to(device)

done = False
while not done:
    # Execute policy for each agent
    actions: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for agent_id in env.possible_agents:
            obs_with_id = torch.Tensor(concat_id(obs[agent_id], agent_id)).to(device)
            act, _, _ = actor.get_action(obs_with_id.unsqueeze(0))
            act = act.detach().cpu().numpy()
            actions[agent_id] = act.flatten()

    obs, _, terminated, truncated, info = env.step(actions)
    done = torch.any(torch.tensor(list(terminated.values()), dtype=torch.bool)) or torch.any(torch.tensor(list(truncated.values()), dtype=torch.bool))
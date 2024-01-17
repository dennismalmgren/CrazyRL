from typing import Dict
import numpy as np
from crazy_rl.multi_agent.numpy.circle.circle import Circle
from pettingzoo import ParallelEnv
from learning.execution.exec_masac_torch import Actor
import torch
from pettingzoo.utils.env import AgentID
import time

#Add some torchrl stuff
from torchrl.envs.libs import PettingZooWrapper
from torchrl.envs import RewardSum, TransformedEnv
from torch import nn
from torchrl.modules.models.multiagent import MultiAgentMLP
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from tensordict.utils import unravel_key
from torchrl.envs import Transform
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss, ValueEstimators

def swap_last(source, dest):
    source = unravel_key(source)
    dest = unravel_key(dest)
    if isinstance(source, str):
        if isinstance(dest, str):
            return dest
        return dest[-1]
    if isinstance(dest, str):
        return source[:-1] + (dest,)
    return source[:-1] + (dest[-1],)

class DoneTransform(Transform):
    """Expands the 'done' entries (incl. terminated) to match the reward shape.

    Can be appended to a replay buffer or a collector.
    """

    def __init__(self, reward_key, done_keys):
        super().__init__()
        self.reward_key = reward_key
        self.done_keys = done_keys

    def forward(self, tensordict):
        for done_key in self.done_keys:
            new_name = swap_last(self.reward_key, done_key)
            tensordict.set(
                ("next", new_name),
                tensordict.get(("next", done_key))
                .unsqueeze(-1)
                .expand(tensordict.get(("next", self.reward_key)).shape),
            )
        return tensordict
    
#config values (should come from yaml)
train_device = "cuda"
env_device = "cpu"
collector_frames_per_batch = 1000
collector_total_frames = 1000_000
train_minibatch_size = 100
buffer_memory_size = collector_frames_per_batch
train_lr = 3e-4
loss_gamma = 0.9
loss_lmbda = 0.9
loss_entropy_eps = 0
loss_clip_epsilon = 0.2
train_num_epochs = 10
train_max_grad_norm = 40.0
eval_evaluation_episodes = 5
eval_evaluation_interval = 100

def make_env(render_mode=None):
    #Define the environment and torchrl-ify it
    env: ParallelEnv = Circle(
        drone_ids=np.array([0, 1]),
        render_mode=render_mode,
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
    return env


env = make_env()

render_env = make_env(render_mode="human")
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

# Critic
module = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,
    n_agents=env.num_drones,
    centralised=True,
    share_params=True,
    device=train_device,
    depth=2,
    num_cells=256,
    activation_class=nn.Tanh,
)
value_module = ValueOperator(
    module=module,
    in_keys=[("agents", "observation")],
)

#collector
collector = SyncDataCollector(
    env,
    policy,
    device=env_device,
    storing_device=train_device,
    frames_per_batch=collector_frames_per_batch,
    total_frames=collector_total_frames,
#    postproc=DoneTransform(reward_key=env.reward_key, done_keys=env.done_keys),
)

#replay buffer (never really used for actual replay, only for storage)
replay_buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(buffer_memory_size, device=train_device),
    sampler=SamplerWithoutReplacement(),
    batch_size=train_minibatch_size,
)

#Loss
loss_module = ClipPPOLoss(
    actor=policy,
    critic=value_module,
    clip_epsilon=loss_clip_epsilon,
    entropy_coef=loss_entropy_eps,
    normalize_advantage=False,
)
loss_module.set_keys(
    reward=env.reward_key,
    action=env.action_key,
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=loss_gamma, lmbda=loss_lmbda
)

# Optimizer
optim = torch.optim.Adam(loss_module.parameters(), train_lr)

#skipping logging for now.
total_time = 0
total_frames = 0
sampling_start = time.time()
for i, tensordict_data in enumerate(collector):
    print(f"\nIteration {i}")
    sampling_time = time.time() - sampling_start

    with torch.no_grad():
        loss_module.value_estimator(
            tensordict_data,
            params=loss_module.critic_params,
            target_params=loss_module.target_critic_params,
        )

    current_frames = tensordict_data.numel()
    total_frames += current_frames
    data_view = tensordict_data.reshape(-1)
    replay_buffer.extend(data_view)

    training_tds = []
    training_start = time.time()
    for _ in range(train_num_epochs):
        for _ in range(collector_frames_per_batch // train_minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)
            training_tds.append(loss_vals.detach())

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), train_max_grad_norm
            )
            training_tds[-1].set("grad_norm", total_norm.mean())

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    training_time = time.time() - training_start

    iteration_time = sampling_time + training_time
    total_time += iteration_time
    training_tds = torch.stack(training_tds)
    #just eval all the time
    with torch.no_grad():
        rollout = env.rollout(policy=policy, max_steps=10000, auto_cast_to_device=True)
        print("Episode reward: ", rollout["next", "agents", "episode_reward"].sum())
    # if i % 10 == 0:
    #     with torch.no_grad():
    #         rollout = render_env.rollout(policy=policy, max_steps=10000, auto_cast_to_device=True)
    sampling_start = time.time()

print('done')
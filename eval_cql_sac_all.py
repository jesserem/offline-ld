# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# STRONG UNDER-PERFORMANCE ON PART OF ANTMAZE TASKS. BUT IN IQL PAPER IT WORKS SOMEHOW
# https://arxiv.org/pdf/2006.04779.pdf
import os
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from offline_jssp_rl.networks.mlp import MLP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.JSSP_Env import SJSSP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_jsp_fsp import parseAndMake
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from torchrl.modules import MaskedCategorical
from offline_jssp_rl.utils import (
    compute_mean_std_graph as compute_mean_std,
    eval_actor_l2d_original_dqn as eval_actor_l2d,
    masked_mean,
    wrap_env_l2d as wrap_env,
    wandb_init,
    minari_dataset_to_dict,
    write_results_checkpoint
)

import minari
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
from offline_jssp_rl.networks.gin import GraphCNN, g_pool_cal, aggr_obs

TensorBatch = List[torch.Tensor]

ENVS_WITH_GOAL = ("antmaze", "pen", "door", "hammer", "relocate")

def convert_to_preferred_format(sec):
   sec = sec % (24 * 3600)
   hour = sec // 3600
   sec %= 3600
   min = sec // 60
   sec %= 60
   return "%02d:%02d:%02d" % (hour, min, sec)


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cpu"

    n_jobs: int = 15 # Number of jobs
    n_machines: int = 15 # Number of machines
    eval_instances: str = f"" # Evaluation instances

    dataset: str = f"jsspl2d-norm_reward_05_noisy_prob_01-v0"
    eval_attributes: List[str] = ('last_time_step',)

    seed: int = 4  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 4  # Eval environment seed
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 1  # How many episodes run during evaluation
    train_epochs: int = 5000  # How many epochs to train
    eval_every_n_epochs: int = 100  # How often to evaluate
    use_epochs: bool = False  # Use epochs instead of steps
    checkpoints_path: Optional[str] = None  # Save path
    offline_iterations: int = int(2.5e5)  # Number of offline iterations
    load_model: str = ""  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 1_000_000  # Replay buffer size
    batch_size: int = 64  # Batch size for all networks
    discount: float = 1  # Discount factor
    alpha_multiplier: float = 1  # Multiplier for alpha in loss
    use_automatic_entropy_tuning: bool = True  # Tune entropy
    backup_entropy: bool = False  # Use backup entropy
    policy_lr: float = 2e-5  # Policy learning rate
    eps_policy: Optional[float] = 1e-4
    fe_lr: float = 2e-5  # Feature extractor learning rate
    qf_lr: float = 2e-5  # Critics learning rate
    eps_qf: Optional[float] = 1e-4
    alpha_lr: float = 2e-5  # Alpha learning rate
    # soft_target_update_rate: float = 2.5e-3 # Target network update rate
    soft_target_update_rate: float = 1 # Target network update rate

    bc_steps: int = int(0)  # Number of BC steps at start
    # target_update_period: int = 1
    target_update_period: int = 2500  # Frequency of target nets updates
    cql_alpha: float = 0.5 # CQL offline regularization parameter
    cql_alpha_online: float = 1 # CQL online regularization parameter
    cql_n_actions: int = 10  # Number of sampled actions
    cql_importance_sample: bool = False # Use importance sampling
    cql_lagrange: bool = False # Use Lagrange version of CQL
    cql_target_action_gap: float = 5 # Action gap
    cql_temp: float = 1 # CQL temperature
    cql_max_target_backup: bool = False # Use max target backup
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = False  # Orthogonal initialization
    normalize: bool = True # Normalize states
    normalize_reward: bool = False # Normalize reward
    q_n_hidden_layers: int = 1  # Number of hidden layers in Q networks
    p_n_hidden_layers: int = 1  # Number of hidden layers in policy networks
    hidden_dim_fe: int = 64
    hidden_dim_fe_mlp: int = 64
    n_dim_fe: int = 2
    n_dim_fe_mlp: int = 2
    q_hidden_size: int = 32 # Hidden size in Q networks
    p_hidden_size: int = 32
    activation_fn: str = "relu"  # Activation function
    graph_pool_type: str = "average"
    neighbor_pooling_type: str = "sum"
    target_entropy: float = 0.98 # Target entropy for entropy tuning
    reward_scale: float = 1 # Reward scale for normalization
    reward_bias: float = 0 # Reward bias for normalization
    q_dropout: float = 0.4 # Dropout in actor network
    q_gnn_dropout: float = 0 # Dropout in GNN
    actor_dropout: float = 0 # Dropout in actor network
    actor_gnn_dropout: float = 0 # Dropout in GNN
    use_next_action: bool = True  # Use next action in Q networks
    use_dueling: bool = False  # Use dueling networksnetworks
    # Cal-QL
    mixing_ratio: float = 0.5  # Data mixing ratio for online tuning
    is_sparse_reward: bool = False  # Use sparse reward
    use_cal_ql: bool = True  # Use Cal-QL
    shuffle_buffer: bool = True # Shuffle buffer
    decay_lr_steps: int = 1 # Decay steps for step size
    decay_gamma: float = 1 # Decay gamma for online tuning

    decay_lr_step: bool = False  # Decay learning rate
    # Wandb logging
    project: str = "JSSP-Offline-Final"
    group: str = f"{n_jobs}x{n_machines}"
    name: str = "CQL-" + "-" + dataset
    path_eval: str = ""
    save_path_checkpoint: str = ""
    save_folder_results: str = ""
    save_name: str = f"NJ{n_jobs}_NM{n_machines}"


    def __post_init__(self):
        self.name = f"{self.name}-{self.n_jobs}x{self.n_machines}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class ReplayBuffer:
    def __init__(
        self,
        adj_shape: np.ndarray,
        feature_shape: np.ndarray,
        omega_shape: np.ndarray,
        mask_shape: np.ndarray,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device
        device = torch.device("cpu")

        self._adj = torch.zeros(
            (buffer_size, *adj_shape), dtype=torch.bool, device=device
        )
        self._features = torch.zeros(
            (buffer_size, *feature_shape), dtype=torch.float32, device=device
        )
        self._omega = torch.zeros(
            (buffer_size, *omega_shape), dtype=torch.int64, device=device
        )

        self._actions = torch.zeros(
            (buffer_size, 1), dtype=torch.int64, device=device
        )
        self._action_masks = torch.ones(
            (buffer_size, *mask_shape), dtype=torch.bool, device=device
        )

        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_adj = torch.zeros(
            (buffer_size, *adj_shape), dtype=torch.bool, device=device
        )
        self._next_features = torch.zeros(
            (buffer_size, *feature_shape), dtype=torch.float32, device=device
        )
        self._next_omega = torch.zeros(
            (buffer_size, *omega_shape), dtype=torch.int64, device=device
        )
        self._next_action_masks = torch.ones(
            (buffer_size, *mask_shape), dtype=torch.bool, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)



    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device="cpu")

    def sample_c(self, batch_size: int) -> TensorBatch:
        if self._device == "cpu":
            return self.sample_c_cpu(batch_size)
        else:
            return self.sample_c_gpu(batch_size)

    def sample_c_gpu(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        adj_batch = self._adj[indices].to(torch.float32)
        adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1]).to(self._device)
        features_batch = self._features[indices]
        features_batch = features_batch.reshape(-1, features_batch.size(-1)).to(self._device)
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices].to(torch.float32)
        next_adj_batch = aggr_obs(next_adj_batch.to_sparse(), next_adj_batch.shape[1]).to(self._device)
        next_features_batch = self._next_features[indices]
        next_features_batch = next_features_batch.reshape(-1, next_features_batch.size(-1)).to(self._device)
        omega = self._omega[indices].to(self._device)
        next_omega = self._next_omega[indices].to(self._device)
        actions = self._actions[indices].to(self._device)
        action_masks = self._action_masks[indices].to(self._device)
        rewards = self._rewards[indices].to(self._device)
        # next_states = self._next_states[indices]
        next_action_masks = self._next_action_masks[indices].to(self._device)
        dones = self._dones[indices].to(self._device)
        return [adj_batch, features_batch, omega, actions, action_masks, rewards, next_adj_batch, next_features_batch, next_omega, next_action_masks, dones]

    def sample_c_cpu(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        adj_batch = self._adj[indices].to(torch.float32)
        adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1])
        features_batch = self._features[indices]
        features_batch = features_batch.reshape(-1, features_batch.size(-1))
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices].to(torch.float32)
        next_adj_batch = aggr_obs(next_adj_batch.to_sparse(), next_adj_batch.shape[1])
        next_features_batch = self._next_features[indices]
        next_features_batch = next_features_batch.reshape(-1, next_features_batch.size(-1))
        omega = self._omega[indices]
        next_omega = self._next_omega[indices]
        actions = self._actions[indices]
        action_masks = self._action_masks[indices]
        rewards = self._rewards[indices]
        # next_states = self._next_states[indices]
        next_action_masks = self._next_action_masks[indices]
        dones = self._dones[indices]
        return [adj_batch, features_batch, omega, actions, action_masks, rewards, next_adj_batch, next_features_batch, next_omega, next_action_masks, dones]


    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["adj"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._adj[:n_transitions] = self._to_tensor(data["adj"])
        # self._adj = self._adj.to_sparse()
        self._omega[:n_transitions] = self._to_tensor(data["omegas"])
        self._features[:n_transitions] = self._to_tensor(data["features"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"]).unsqueeze(-1)
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._action_masks[:n_transitions] = self._to_tensor(data["action_masks"])
        self._next_action_masks[:n_transitions] = self._to_tensor(data["next_action_masks"])
        # self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._next_adj[:n_transitions] = self._to_tensor(data["next_adj"])
        # self._next_adj = self._next_adj.to_sparse()
        self._next_omega[:n_transitions] = self._to_tensor(data["next_omegas"])
        self._next_features[:n_transitions] = self._to_tensor(data["next_features"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        adj_batch = self._adj[indices]
        features_batch = self._features[indices]
        # features_batch = features_batch.reshape(-1, features_batch.size(-1))
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices]
        next_features_batch = self._next_features[indices]
        # next_features_batch = next_features_batch.reshape(-1, next_features_batch.size(-1))
        omega = self._omega[indices]
        next_omega = self._next_omega[indices]
        actions = self._actions[indices]
        action_masks = self._action_masks[indices]
        rewards = self._rewards[indices]
        # next_states = self._next_states[indices]
        next_action_masks = self._next_action_masks[indices]
        dones = self._dones[indices]
        return [adj_batch, features_batch, omega, actions, action_masks, rewards, next_adj_batch, next_features_batch, next_omega, next_action_masks, dones]

    def train_epoch(self, batch_size: int, shuffle: bool = False) -> TensorBatch:
        if self._device == "cpu":
            return self.train_epoch_cpu(batch_size, shuffle)
        else:
            return self.train_epoch_gpu(batch_size, shuffle)



    def train_epoch_gpu(self, batch_size: int, shuffle: bool = False) -> TensorBatch:
        indices = torch.randperm(self._size) if shuffle else torch.arange(self._size)
        for start in range(0, self._size, batch_size):
            batch_indices = indices[start : start + batch_size]
            # adj_batch = aggr_obs(self._adj[batch_indices].to_sparse(), self._adj.shape[1]).to(self._device)
            adj_batch = self._adj[batch_indices].to(self._device)
            features_batch = self._features[batch_indices]
            features_batch = features_batch.reshape(-1, features_batch.size(-1)).to(self._device)
            next_adj_batch = aggr_obs(self._next_adj[batch_indices].to_sparse(), self._next_adj.shape[1]).to(self._device)
            next_features_batch = self._next_features[batch_indices]
            next_features_batch = next_features_batch.reshape(-1, next_features_batch.size(-1)).to(self._device)
            yield [adj_batch, features_batch, self._omega[batch_indices].to(self._device), self._actions[batch_indices].to(self._device),
                   self._action_masks[batch_indices].to(self._device), self._rewards[batch_indices].to(self._device), next_adj_batch, next_features_batch,
                   self._next_omega[batch_indices].to(self._device), self._next_action_masks[batch_indices].to(self._device), self._dones[batch_indices].to(self._device)]

    def train_epoch_cpu(self, batch_size: int, shuffle: bool = False) -> TensorBatch:
        indices = torch.randperm(self._size) if shuffle else torch.arange(self._size)
        for start in range(0, self._size, batch_size):
            batch_indices = indices[start: start + batch_size]
            # adj_batch = aggr_obs(self._adj[batch_indices].to_sparse(), self._adj.shape[1])
            # adj_batch = self._adj[indices]
            adj_batch = self._adj[batch_indices].to(torch.float32)
            adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1])
            features_batch = self._features[batch_indices]
            features_batch = features_batch.reshape(-1, features_batch.size(-1))
            # next_adj_batch = aggr_obs(self._next_adj[batch_indices].to_sparse(), self._next_adj.shape[1])
            next_adj_batch = self._next_adj[batch_indices].to(torch.float32)
            next_adj_batch = aggr_obs(next_adj_batch.to_sparse(), next_adj_batch.shape[1])
            next_features_batch = self._next_features[batch_indices]
            next_features_batch = next_features_batch.reshape(-1, next_features_batch.size(-1))
            yield [adj_batch, features_batch, self._omega[batch_indices],
                   self._actions[batch_indices],
                   self._action_masks[batch_indices],
                   self._rewards[batch_indices], next_adj_batch, next_features_batch,
                   self._next_omega[batch_indices],
                   self._next_action_masks[batch_indices],
                   self._dones[batch_indices]]


        # pass

    def add_transition(
            self,
            adj: np.ndarray,
            features: np.ndarray,
            omega: np.ndarray,
            action: np.ndarray,
            action_mask: np.ndarray,
            reward: float,
            next_adj: np.ndarray,
            next_features: np.ndarray,
            next_omega: np.ndarray,
            next_action_mask: np.ndarray,
            done: bool,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._adj[self._pointer] = self._to_tensor(adj)
        self._features[self._pointer] = self._to_tensor(features)
        self._omega[self._pointer] = self._to_tensor(omega)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_adj[self._pointer] = self._to_tensor(next_adj)
        self._next_features[self._pointer] = self._to_tensor(next_features)
        self._next_omega[self._pointer] = self._to_tensor(next_omega)
        self._dones[self._pointer] = self._to_tensor(done)
        self._action_masks[self._pointer] = self._to_tensor(action_mask)
        self._next_action_masks[self._pointer] = self._to_tensor(next_action_mask)

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def return_reward_range(dataset: Dict, max_episode_steps: int) -> Tuple[float, float, int]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns), max(lengths)


def collate_fn(batch):
    print(batch)
    exit()
    return batch


def get_episodes_indices(dataset: Dict) -> List[List[int]]:
    list_indices = []
    curr_indices = []
    for i, d in enumerate(dataset["terminals"]):
        curr_indices.append(i)
        if d:
            list_indices.append(curr_indices)
            curr_indices = []
    return list_indices


def modify_reward(
    dataset: Dict,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
) -> Dict:
    min_ret, max_ret, max_length = return_reward_range(dataset, max_episode_steps=1000000)
    modification_data = {}
    modification_data["min_ret"] = min_ret
    modification_data["max_ret"] = max_ret
    modification_data["max_length"] = max_length

    # dataset["rewards"] /= max_ret - min_ret
    # dataset["rewards"] *= max_length
    dataset["rewards"] = dataset["rewards"] * reward_scale + reward_bias

    return modification_data


def modify_reward_online(
    reward: float,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    **kwargs,
) -> float:
    # reward /= kwargs["max_ret"] - kwargs["min_ret"]
    # reward *= kwargs["max_length"]
    reward = reward * reward_scale + reward_bias
    return reward


def normalize_reward(
        dataset: Dict,
):
    mean_reward = np.mean(dataset["rewards"])
    std_reward = np.std(dataset["rewards"])
    dataset["rewards"] = (dataset["rewards"] - mean_reward) / std_reward
    modification_data = {"mean_reward": mean_reward, "std_reward": std_reward}
    return modification_data

def normalize_reward_online(reward: float, mean_reward: float, std_reward: float):
    return (reward - mean_reward) / std_reward


def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)


def init_module_weights(module: torch.nn.Module, orthogonal_init: bool = False):
    if isinstance(module, nn.Linear):
        if orthogonal_init:
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
        else:
            nn.init.xavier_uniform_(module.weight, gain=1e-2)


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        n_j: int,
        n_m: int,
        num_layers: int,
        hidden_dim_gnn: int,
        learn_eps: bool,
        input_dim: int,
        num_mlp_layers_feature_extract: int,
        neighbor_pooling_type: str,
        device: str,
        action_dim: int,
        hidden_dim_actor: int = 256,
        n_hidden: int = 3,
        orthogonal_init: bool = False,
        actor_dropout: float = 0.0,
        actor_gnn_dropout: float = 0.0,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.action_dim = action_dim

        self.orthogonal_init = orthogonal_init
        # self.dropout = dropout

        self.feature_extractor = FeatureExtractor(
            n_j=n_j,
            n_m=n_m,
            num_layers=num_layers,
            hidden_dim=hidden_dim_gnn,
            learn_eps=learn_eps,
            input_dim=input_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device,
            dropout=actor_gnn_dropout,
            orthogonal_init=orthogonal_init,
            activation_fn=activation_fn,
        )
        self.observation_dim = hidden_dim_gnn * 2
        self.base_network = MLP([self.observation_dim, *([hidden_dim_actor] * n_hidden), 1],
                                dropout=actor_dropout,
                                activation_fn=activation_fn)


        if orthogonal_init:
            self.base_network.apply(lambda m: init_module_weights(m, True))


    def log_prob(
        self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
            actions: torch.Tensor, action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        observations, _ = self.feature_extractor(adj, features, candidate, graph_pool)
        if action_masks is None:
            action_masks = torch.ones(observations.shape[0], self.action_dim, device=observations.device, dtype=torch.bool)
        base_network_output = self.base_network(observations).squeeze(-1)

        dist = MaskedCategorical(logits=base_network_output, mask=action_masks)
        log_probs = dist.log_prob(actions.squeeze(-1))
        return log_probs

    def forward(
        self,
        adj: torch.Tensor,
        features: torch.Tensor,
        candidate: torch.Tensor,
        graph_pool: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        repeat: bool = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observations, _ = self.feature_extractor(adj, features, candidate, graph_pool)
        if action_masks is None:
            action_masks = torch.ones(observations.shape[0], self.action_dim, device=observations.device, dtype=torch.bool)
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
            action_masks = extend_and_repeat(action_masks, 1, repeat)
        base_network_output = self.base_network(observations).squeeze(-1)
        # print(base_network_output.shape)
        # exit()
        dist = MaskedCategorical(logits=base_network_output, mask=action_masks)
        if deterministic:

            actions = dist.probs.argmax(dim=-1)
            if repeat is not None:
                log_probs = dist.log_prob(actions)
                return actions, log_probs

            # actions = actions.unsqueeze(-1)
        else:
            actions = dist.sample()
            if repeat is not None:
                log_probs = dist.log_prob(actions)
                return actions, log_probs
            # actions = actions.unsqueeze(-1)
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    @torch.no_grad()
    def act(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
            action_mask: Optional[np.ndarray], device: str = "cpu", deterministic: bool = False):
        # state, _ = self.feature_extractor(adj, features, candidate, graph_pool)
        # state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        if action_mask is not None:
            action_mask = torch.tensor(action_mask.reshape(1, -1), device=device, dtype=torch.bool)
        with torch.no_grad():
            actions, _ = self(adj, features, candidate, graph_pool, action_masks=action_mask, deterministic=deterministic)
        if actions.size(0) > 1:
            return actions.cpu().data.numpy().flatten()
        else:
            return actions.item()

    def get_action_probs(
            self,
            adj: torch.Tensor,
            features: torch.Tensor,
            candidate: torch.Tensor,
            graph_pool: torch.Tensor,
            action_masks: Optional[torch.Tensor] = None,
            deterministic: bool = False
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state, _ = self.feature_extractor(adj, features, candidate, graph_pool)
        if action_masks is None:
            action_masks = torch.ones(state.shape[0], self.action_dim, device=state.device, dtype=torch.bool)

        base_network_output = self.base_network(state).squeeze(-1)
        # print(base_network_output.shape)
        dist = MaskedCategorical(logits=base_network_output, mask=action_masks)
        action_probs = dist.probs
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_action = dist.log_prob(actions)
        return actions, action_probs, log_action_probs, log_action, dist.entropy()


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class FeatureExtractor(nn.Module):
    def __init__(
            self,
            n_j: int,
            n_m: int,
            num_layers: int,
            hidden_dim: int,
            learn_eps: bool,
            input_dim: int,
            num_mlp_layers_feature_extract: int,
            neighbor_pooling_type: str,
            device: str,
            dropout: float = 0.0,
            orthogonal_init: bool = False,
            activation_fn: nn.Module = nn.ReLU,
                 ):
        super().__init__()
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device


        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device,
                                        dropout=dropout,
                                        activation=activation_fn).to(device)
        # self.feature_extract.init_weights()

    def forward(self, adj, features, candidate, graph_pool):
        h_pooled, h_nodes = self.feature_extract(
            x=features,
            graph_pool=graph_pool,
            padded_nei=None,
            adj=adj)

        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)

        return concateFea, h_pooled


class DiscreteFullyConnectedQFunction(nn.Module):
    def __init__(
        self,
        n_j: int,
        n_m: int,
        num_layers: int,
        hidden_dim_gnn: int,
        learn_eps: bool,
        input_dim: int,
        num_mlp_layers_feature_extract: int,
        neighbor_pooling_type: str,
        device: str,
        orthogonal_init: bool = False,
        n_hidden_layers: int = 2,
        hidden_dim_critic: int = 256,
        activation_fn: nn.Module = nn.ReLU,
        use_dueling: bool = True,
        critic_dropout: float = 0.5,
        gnn_critic_dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            n_j=n_j,
            n_m=n_m,
            num_layers=num_layers,
            hidden_dim=hidden_dim_gnn,
            learn_eps=learn_eps,
            input_dim=input_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device,
            dropout=gnn_critic_dropout,
            orthogonal_init=orthogonal_init,
            activation_fn=activation_fn,
        )
        self.use_dueling = use_dueling
        self.observation_dim = hidden_dim_gnn * 2
        self.observation_dim_value = hidden_dim_gnn
        self.hidden_dim_critic = hidden_dim_critic
        self.orthogonal_init = orthogonal_init
        layers = []
        if critic_dropout > 0:
            layers.append(nn.Dropout(critic_dropout))
        # layers = [
        #     nn.Linear(self.observation_dim, self.hidden_dim_critic),
        #     activation_fn(),
        # ]
        layers.append(nn.Linear(self.observation_dim, self.hidden_dim_critic))
        layers.append(activation_fn())
        for _ in range(n_hidden_layers - 1):
            if critic_dropout > 0:
                layers.append(nn.Dropout(critic_dropout))
            layers.append(nn.Linear(self.hidden_dim_critic, self.hidden_dim_critic))
            layers.append(activation_fn())
        if critic_dropout > 0:
            layers.append(nn.Dropout(critic_dropout))
        layers.append(nn.Linear(self.hidden_dim_critic, 1))

        self.network = nn.Sequential(*layers)
        print(self.network)
        if orthogonal_init:
            self.network.apply(lambda m: init_module_weights(m, True))
        # else:
        #     for i in range(len(self.network)):
        #         activation_fn_name = (activation_fn).__name__.lower()
        #         gain_calc = torch.nn.init.calculate_gain(activation_fn_name)
        #         if isinstance(self.network[i], nn.Linear):
        #             if i == len(self.network) - 1:
        #                 # if activation_fn ==
        #                 # nn.init.xavier_uniform_(self.network_q[i].weight, gain=1)
        #                 if activation_fn_name == 'relu':
        #                     nn.init.kaiming_uniform_(self.network[i].weight, nonlinearity='relu')
        #                 else:
        #                     nn.init.xavier_uniform_(self.network[i].weight, gain=1)
        #             else:
        #                 if activation_fn_name == 'relu':
        #                     nn.init.kaiming_uniform_(self.network[i].weight, nonlinearity='relu')
        #                 else:
        #                     nn.init.xavier_uniform_(self.network[i].weight, gain=gain_calc)
        #                 # nn.init.kaiming_uniform_(self.network[i].weight, nonlinearity='relu')
        #             self.network[i].bias.data.fill_(0)
            # init_module_weights(self.network[-1], False)
        if self.use_dueling:
            layers = [
                nn.Linear(self.observation_dim_value, self.hidden_dim_critic),
                activation_fn(),
            ]
            if critic_dropout > 0:
                layers.append(nn.Dropout(critic_dropout))
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(self.hidden_dim_critic, self.hidden_dim_critic))
                layers.append(activation_fn())
                if critic_dropout > 0:
                    layers.append(nn.Dropout(critic_dropout))
            layers.append(nn.Linear(self.hidden_dim_critic, 1))
            self.value_network = nn.Sequential(*layers)
            if orthogonal_init:
                self.value_network.apply(lambda m: init_module_weights(m, True))
            # else:
            #     activation_fn_name = (activation_fn).__name__.lower()
            #     gain_calc = torch.nn.init.calculate_gain(activation_fn_name)
            #     for i in range(len(self.value_network)):
            #         if isinstance(self.value_network[i], nn.Linear):
            #             if i == len(self.value_network) - 1:
            #                 if activation_fn_name == "relu":
            #                     nn.init.kaiming_uniform_(self.value_network[i].weight, nonlinearity='relu')
            #                 else:
            #                     nn.init.xavier_uniform_(self.value_network[i].weight, gain=1)
            #             else:
            #                 if activation_fn_name == "relu":
            #                     nn.init.kaiming_uniform_(self.value_network[i].weight, nonlinearity='relu')
            #                 else:
            #                     nn.init.xavier_uniform_(self.value_network[i].weight, gain=gain_calc)


    def forward(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
                actions: torch.Tensor, action_masks: torch.Tensor) \
            -> torch.Tensor:
        observations, observation_v = self.feature_extractor(adj, features, candidate, graph_pool)
        # output_net = self.network(observations).squeeze(-1)
        # q_values = output_net.gather(1, actions)
        # q_values = torch.squeeze(q_values, dim=-1)
        output_q = self.network(observations).squeeze(-1)
        if self.use_dueling:
            output_v = self.value_network(observation_v)
            masked_q = masked_mean(output_q, action_masks, dim=-1, keepdim=True)

            output_q = output_v + output_q - masked_q
            # output_q = output_v + output_q - masked_mean(output_q, action_masks, dim=-1, keepdim=True)

        return torch.squeeze(output_q.gather(1, actions), dim=-1)

    def get_all(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
                action_masks: torch.Tensor) -> torch.Tensor:
        observations, observation_v = self.feature_extractor(adj, features, candidate, graph_pool)
        output_q = self.network(observations).squeeze(-1)
        if self.use_dueling:
            output_v = self.value_network(observation_v)
            masked_q = masked_mean(output_q, action_masks, dim=-1, keepdim=True)

            output_q = output_v + output_q - masked_q

        return output_q


def get_weights_copy(model):
    return {name: param.clone() for name, param in model.state_dict().items()}


class DiscreteCAL_QL:
    def __init__(
            self,
            critic_1,
            critic_1_optimizer,
            critic_2,
            critic_2_optimizer,
            actor,
            actor_optimizer,
            mb_graph_pool,
            n_tasks: int,
            graph_pool_type: str,
            target_entropy: float,
            discount: float = 0.99,
            alpha_multiplier: float = 1.0,
            use_automatic_entropy_tuning: bool = True,
            backup_entropy: bool = False,
            policy_lr: float = 3e-4,
            qf_lr: float = 3e-4,
            alpha_lr: float = 3e-4,
            soft_target_update_rate: float = 5e-3,
            bc_steps=100000,
            target_update_period: int = 1,
            cql_n_actions: int = 10,
            cql_importance_sample: bool = True,
            cql_lagrange: bool = False,
            use_next_action: bool = False,
            cql_target_action_gap: float = -1.0,
            cql_temp: float = 1.0,
            cql_alpha: float = 5.0,
            cql_max_target_backup: bool = False,
            cql_clip_diff_min: float = -np.inf,
            cql_clip_diff_max: float = np.inf,
            device: str = "cpu",
            action_dim: int = 1,
            decay_lr_steps: int = 500,
            decay_gamma: float = 0.9,
            max_steps: int = 1,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.graph_pool_type = graph_pool_type
        self.action_dim = action_dim
        self.discount = discount

        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.backup_entropy = backup_entropy
        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.alpha_lr = alpha_lr
        self.soft_target_update_rate = soft_target_update_rate
        self.bc_steps = bc_steps
        self.target_update_period = target_update_period
        self.cql_n_actions = cql_n_actions
        self.cql_importance_sample = cql_importance_sample
        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_max_target_backup = cql_max_target_backup
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device
        self.use_next_action = use_next_action
        self.mb_graph_pool = mb_graph_pool

        self.total_it = 0

        self.critic_1 = critic_1
        self.critic_2 = critic_2

        self.target_critic_1 = deepcopy(self.critic_1).to(device)
        self.target_critic_2 = deepcopy(self.critic_2).to(device)
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.actor = actor
        self.first_update = False

        # self.fe_optimizer = fe_optimizer
        self.actor_optimizer = actor_optimizer
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2_optimizer = critic_2_optimizer
        self.actor_decay_lr = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                              step_size=decay_lr_steps,
                                                              gamma=decay_gamma)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.alpha_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_critic_1, self.critic_1, soft_target_update_rate)
        soft_update(self.target_critic_2, self.critic_2, soft_target_update_rate)

    def step_lr(self):
        self.actor_decay_lr.step()

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled

    def _alpha_and_alpha_loss(self, observations: torch.Tensor, log_prob: torch.Tensor, action_masks: torch.Tensor):
        assert not log_prob.requires_grad
        if self.use_automatic_entropy_tuning:
            action_probabilities = torch.sum(action_masks, dim=1)
            target_entropy = (-torch.log(action_probabilities) * self.target_entropy).unsqueeze(-1)
            alpha_loss = -(
                    self.log_alpha() * (log_prob - target_entropy).detach()
            ).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = observations.new_tensor(0.0)
            alpha = observations.new_tensor(self.alpha_multiplier)
        return alpha, alpha_loss

    def _policy_loss(
            self,
            adj: torch.Tensor,
            features: torch.Tensor,
            candidate: torch.Tensor,
            graph_pool: torch.Tensor,
            actions: torch.Tensor,
            action_masks: torch.Tensor,
            action_probs: torch.Tensor,
            new_actions: torch.Tensor,
            log_action_probs: torch.Tensor,
            alpha: torch.Tensor,
            log_pi: torch.Tensor,
    ) -> torch.Tensor:
        if self.total_it <= self.bc_steps:
            log_probs = self.actor.log_prob(adj, features, candidate, graph_pool, actions, action_masks=action_masks)
            policy_loss = (alpha * log_pi - log_probs).mean()

        else:
            with torch.no_grad():
                q = torch.min(
                    self.critic_1.get_all(adj, features, candidate, graph_pool, action_masks),
                    self.critic_2.get_all(adj, features, candidate, graph_pool, action_masks),
                )
            policy_loss = (action_probs * (alpha * log_action_probs - q)).sum(1).mean()

        return policy_loss

    def _q_loss(
            self,
            adj: torch.Tensor,
            features: torch.Tensor,
            candidate: torch.Tensor,
            graph_pool: torch.Tensor,
            actions: torch.Tensor,
            action_masks: torch.Tensor,
            next_adj: torch.Tensor,
            next_features: torch.Tensor,
            next_candidate: torch.Tensor,
            next_action_masks: torch.Tensor,
            rewards: torch.Tensor,
            dones: torch.Tensor,
            alpha: torch.Tensor,
            log_dict: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():

            if self.cql_max_target_backup:

                q1_next_val = self.target_critic_1.get_all(next_adj, next_features, next_candidate, graph_pool,
                                                           next_action_masks)
                q2_next_val = self.target_critic_2.get_all(next_adj, next_features, next_candidate, graph_pool,
                                                           next_action_masks)

                q_fill_val = torch.full_like(q1_next_val, -np.inf)
                next_q1_values = torch.where(next_action_masks, q1_next_val, q_fill_val)
                next_q2_values = torch.where(next_action_masks, q2_next_val, q_fill_val)
                next_q1_values = next_q1_values.max(1)[0].view(-1, 1)
                next_q2_values = next_q2_values.max(1)[0].view(-1, 1)

                target_q_values = torch.min(next_q1_values, next_q2_values)

            else:
                _, next_action_probs, next_log_action_probs, _, _ = self.actor.get_action_probs(
                    next_adj,
                    next_features,
                    next_candidate,
                    graph_pool,
                    action_masks=next_action_masks
                )
                next_q_values = torch.min(
                    self.target_critic_1.get_all(next_adj, next_features, next_candidate, graph_pool,
                                                 next_action_masks),
                    self.target_critic_2.get_all(next_adj, next_features, next_candidate, graph_pool,
                                                 next_action_masks),
                )

                if self.backup_entropy:
                    next_q_values = next_q_values - alpha * next_log_action_probs

                target_q_values = (next_action_probs * next_q_values).sum(dim=1)
                target_q_values = target_q_values.unsqueeze(-1)

            td_target = rewards + (1.0 - dones) * self.discount * target_q_values.detach()

            td_target = td_target.squeeze(-1)
        q1 = self.critic_1.get_all(adj, features, candidate, graph_pool, action_masks)
        q2 = self.critic_2.get_all(adj, features, candidate, graph_pool, action_masks)

        q1_predicted = q1.gather(1, actions).squeeze(-1)
        q2_predicted = q2.gather(1, actions).squeeze(-1)

        qf1_loss = F.mse_loss(q1_predicted, td_target.detach())
        qf2_loss = F.mse_loss(q2_predicted, td_target.detach())

        q1_cql = torch.where(~action_masks, torch.full_like(q1, -np.inf), q1)
        q2_cql = torch.where(~action_masks, torch.full_like(q2, -np.inf), q2)

        cql_qf1_ood = torch.logsumexp(q1_cql / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(q2_cql / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_predicted,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = adj.new_tensor(0.0)
            alpha_prime = adj.new_tensor(0.0)
        cql_qf1_loss = qf1_loss + cql_min_qf1_loss
        cql_qf2_loss = qf2_loss + cql_min_qf2_loss

        log_dict.update(
            dict(
                qf1_loss=qf1_loss.item(),
                qf2_loss=qf2_loss.item(),
                alpha=alpha.item(),
                average_qf1=q1_predicted.mean().item(),
                std_qf1=q1_predicted.std().item(),
                average_qf2=q2_predicted.mean().item(),
                std_qf2=q2_predicted.std().item(),
                mean_diff_q=torch.abs(q1_predicted - q2_predicted).mean().item(),
                average_target_q=target_q_values.mean().item(),
                std_target_q=target_q_values.std().item(),
                target_td=td_target.mean().item(),
            )
        )

        log_dict.update(
            dict(
                cql_min_qf1_loss=cql_min_qf1_loss.mean().item(),
                cql_min_qf2_loss=cql_min_qf2_loss.mean().item(),
                cql_qf1_diff=cql_qf1_diff.mean().item(),
                cql_qf2_diff=cql_qf2_diff.mean().item(),
                alpha_prime_loss=alpha_prime_loss.item(),
                alpha_prime=alpha_prime.item(),

            )
        )

        return cql_qf1_loss, cql_qf2_loss, alpha_prime, alpha_prime_loss

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        (
            adj,
            features,
            candidates,
            actions,
            action_masks,
            rewards,
            next_adj,
            next_features,
            next_candidates,
            next_action_masks,
            dones,
        ) = batch
        self.total_it += 1
        num_batches = dones.shape[0]
        mb_g_pool = g_pool_cal(self.graph_pool_type,
                               (num_batches, self.n_tasks, self.n_tasks),
                               self.n_tasks, self._device)

        new_actions, action_probs, log_action_probs, log_pi, _ = self.actor.get_action_probs(
            adj,
            features,
            candidates,
            mb_g_pool,
            action_masks=action_masks
        )

        entropy = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)

        alpha, alpha_loss = self._alpha_and_alpha_loss(adj, log_pi.detach(), action_masks)

        """ Policy loss """
        policy_loss = self._policy_loss(
            adj, features, candidates, mb_g_pool, actions, action_masks, action_probs, new_actions, log_action_probs,
            alpha, log_pi
        )

        log_dict = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=alpha.item(),
        )

        """ Q function loss """
        qf1_loss, qf2_loss, alpha_prime, alpha_prime_loss = self._q_loss(
            adj,
            features,
            candidates,
            mb_g_pool,
            actions,
            action_masks,
            next_adj,
            next_features,
            next_candidates,
            next_action_masks,
            rewards,
            dones,
            alpha,
            log_dict,
        )

        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            log_dict["alpha_grad"] = self.log_alpha.constant.grad

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_1_optimizer.zero_grad()
        qf1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic_2_optimizer.step()

        log_dict["entropy"] = entropy.mean().detach()
        log_dict["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "actor_lr_scheduler": self.actor_decay_lr.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optim": self.alpha_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict=state_dict["actor"])
        self.critic_1.load_state_dict(state_dict=state_dict["critic1"])
        self.critic_2.load_state_dict(state_dict=state_dict["critic2"])
        self.actor_decay_lr.load_state_dict(state_dict=state_dict["actor_lr_scheduler"])

        self.target_critic_1.load_state_dict(state_dict=state_dict["critic1_target"])
        self.target_critic_2.load_state_dict(state_dict=state_dict["critic2_target"])

        self.critic_1_optimizer.load_state_dict(
            state_dict=state_dict["critic_1_optimizer"]
        )
        self.critic_2_optimizer.load_state_dict(
            state_dict=state_dict["critic_2_optimizer"]
        )
        self.actor_optimizer.load_state_dict(state_dict=state_dict["actor_optim"])

        self.log_alpha = state_dict["sac_log_alpha"]
        self.alpha_optimizer.load_state_dict(
            state_dict=state_dict["sac_log_alpha_optim"]
        )

        self.log_alpha_prime = state_dict["cql_log_alpha"]
        self.alpha_prime_optimizer.load_state_dict(
            state_dict=state_dict["cql_log_alpha_optim"]
        )
        self.total_it = state_dict["total_it"]


def minari_dataset_to_dict(minari_list: List[minari.MinariDataset], switch_mask: bool=True) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dict = {
        "adj": [],
        "features": [],
        "omegas": [],
        "actions": [],
        "action_masks": [],
        "rewards": [],
        "next_adj": [],
        "next_features": [],
        "next_omegas": [],
        "next_actions": [],
        "next_action_masks": [],
        "terminals": [],
    }
    adj_shape = minari_list[0][0].observations["adj"].shape[1:]
    features_shape = minari_list[0][0].observations["fea"].shape[1:]
    omegas_shape = minari_list[0][0].observations["omega"].shape[1:]
    mask_shape = minari_list[0][0].observations["mask"].shape[1:]
    for dataset in minari_list:
        for epi in dataset:
            adj = epi.observations["adj"]
            features = epi.observations["fea"]
            omegas = epi.observations["omega"]
            # print(epi.observations)
            mask = epi.observations["mask"]
            if switch_mask:
                mask = ~mask
                mask[-1] = ~mask[-1]

            for i in range(epi.total_timesteps):
                # omega = omegas[1]
                fake_action = epi.actions[i]

                real_action = np.where(omegas[i] == fake_action)[0][0]

                dataset_dict["adj"].append(adj[i])
                dataset_dict["features"].append(features[i])
                dataset_dict["omegas"].append(omegas[i])
                dataset_dict["actions"].append(real_action)
                dataset_dict["action_masks"].append(mask[i])
                dataset_dict["rewards"].append(epi.rewards[i])
                dataset_dict["next_adj"].append(adj[i + 1])
                dataset_dict["next_features"].append(features[i + 1])
                dataset_dict["next_omegas"].append(omegas[i + 1])
                dataset_dict["next_action_masks"].append(mask[i + 1])
                done = epi.terminations[i] or epi.truncations[i]
                dataset_dict["terminals"].append(done)
    dataset_dict["adj"] = np.array(dataset_dict["adj"])
    dataset_dict["features"] = np.array(dataset_dict["features"])
    dataset_dict["omegas"] = np.array(dataset_dict["omegas"])
    dataset_dict["actions"] = np.array(dataset_dict["actions"])
    dataset_dict["action_masks"] = np.array(dataset_dict["action_masks"])
    dataset_dict["rewards"] = np.array(dataset_dict["rewards"])
    dataset_dict["next_adj"] = np.array(dataset_dict["next_adj"])
    dataset_dict["next_features"] = np.array(dataset_dict["next_features"])
    dataset_dict["next_omegas"] = np.array(dataset_dict["next_omegas"])
    dataset_dict["next_action_masks"] = np.array(dataset_dict["next_action_masks"])
    dataset_dict["terminals"] = np.array(dataset_dict["terminals"])

    return dataset_dict, adj_shape, features_shape, omegas_shape, mask_shape


# @pyrallis.wrap()
def eval(config: TrainConfig, eval_instance_arr: np.ndarray, state_dict_path: str):
    datasets = [config.dataset]
    dataset_min = [minari.load_dataset(d) for d in datasets]
    dataset, adj_shape, features_shape, omega_shape, mask_shape = minari_dataset_to_dict(dataset_min)
    state_mean, state_std = compute_mean_std(dataset["features"], eps=1e-3)
    state_dict = torch.load(state_dict_path, map_location=config.device)

    config.n_jobs, config.n_machines = eval_instance_arr.shape[2], eval_instance_arr.shape[3]
    if config.activation_fn == "relu":
        activation_fn = nn.ReLU
    elif config.activation_fn == "tanh":
        activation_fn = nn.Tanh
    else:
        raise ValueError(f"Unknown activation function: {config.activation_fn}")

    critic_1 = DiscreteFullyConnectedQFunction(
        n_j=config.n_jobs,
        n_m=config.n_machines,
        input_dim=2,
        neighbor_pooling_type=config.neighbor_pooling_type,
        hidden_dim_gnn=config.hidden_dim_fe,
        num_layers=config.n_dim_fe,
        num_mlp_layers_feature_extract=config.n_dim_fe_mlp,
        learn_eps=False,
        device=config.device,
        orthogonal_init=config.orthogonal_init,
        hidden_dim_critic=config.q_hidden_size,
        n_hidden_layers=config.q_n_hidden_layers,
        activation_fn=activation_fn,
        use_dueling=config.use_dueling,
        critic_dropout=config.q_dropout,
        gnn_critic_dropout=config.q_gnn_dropout,
    ).to(config.device)
    critic_2 = DiscreteFullyConnectedQFunction(
        n_j=config.n_jobs,
        n_m=config.n_machines,
        input_dim=2,
        neighbor_pooling_type=config.neighbor_pooling_type,
        hidden_dim_gnn=config.hidden_dim_fe,
        num_layers=config.n_dim_fe,
        num_mlp_layers_feature_extract=config.n_dim_fe_mlp,
        learn_eps=False,
        device=config.device,
        orthogonal_init=config.orthogonal_init,
        hidden_dim_critic=config.q_hidden_size,
        n_hidden_layers=config.q_n_hidden_layers,
        activation_fn=activation_fn,
        use_dueling=config.use_dueling,
        critic_dropout=config.q_dropout,
        gnn_critic_dropout=config.q_gnn_dropout,
    ).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), config.qf_lr)

    actor = CategoricalPolicy(
        n_j=config.n_jobs,
        n_m=config.n_machines,
        input_dim=2,
        neighbor_pooling_type=config.neighbor_pooling_type,
        hidden_dim_gnn=config.hidden_dim_fe,
        num_layers=config.n_dim_fe,
        num_mlp_layers_feature_extract=config.n_dim_fe_mlp,
        learn_eps=False,
        device=config.device,
        action_dim=config.n_jobs,
        orthogonal_init=config.orthogonal_init,
        n_hidden=config.p_n_hidden_layers,
        # dropout=config.actor_dropout,
        hidden_dim_actor=config.p_hidden_size,
        activation_fn=activation_fn,
        actor_dropout=config.actor_dropout,
        actor_gnn_dropout=config.actor_gnn_dropout,
    ).to(config.device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr, eps=1e-4)
    mb_g_pool = g_pool_cal(config.graph_pool_type,
                           (config.batch_size, config.n_jobs * config.n_machines, config.n_jobs * config.n_machines),
                           config.n_jobs * config.n_machines, config.device)
    num_steps_decay = 1
    kwargs = {
        "mb_graph_pool": mb_g_pool.clone(),
        "graph_pool_type": config.graph_pool_type,
        "n_tasks": config.n_jobs * config.n_machines,
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": config.target_entropy,
        "use_next_action": config.use_next_action,
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "alpha_lr": config.alpha_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        "max_steps": num_steps_decay,
        "decay_lr_steps": config.decay_lr_steps,
        "decay_gamma": config.decay_gamma,
    }

    print("---------------------------------------")
    print(f"Training CAL-QL, Env: n_jobs {config.n_jobs}, n_machines {config.n_machines}, EVAL")
    print("---------------------------------------")

    trainer = DiscreteCAL_QL(**kwargs)

    trainer.load_state_dict(state_dict)
    eval_env = SJSSP(config.n_jobs, config.n_machines)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, min_max_normalize=False)
    g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
                             batch_size=torch.Size(
                                 [1, config.n_jobs * config.n_machines, config.n_jobs * config.n_machines]),
                             n_nodes=config.n_jobs * config.n_machines, device=config.device)
    eval_scores, success_rate, make_spans, run_times_instances = eval_actor_l2d(
        env=eval_env,
        actor=actor,
        readArrFunc=parseAndMake,
        eval_instances=eval_instance_arr,
        device=config.device,
        g_pool=g_pool_step,
        deterministic=True,
    )
    results_dict = {
        "index": np.arange(len(eval_scores), dtype=int),
        "eval_scores": eval_scores,
        "make_spans": make_spans,
        "run_times_instances": run_times_instances,
    }
    results_df = pd.DataFrame(results_dict, index=results_dict["index"])
    if not os.path.exists(config.save_folder_results):
        os.makedirs(config.save_folder_results)

    return results_df

@pyrallis.wrap()
def eval_all(config: TrainConfig):
    checkpoint = 50000
    print(config.save_folder_results)
    instance_size = config.save_path_checkpoint.strip().split("/")[-2]
    type_exp = config.save_path_checkpoint.strip().split("/")[-1]
    exp_folder = os.path.join(config.save_folder_results, instance_size, type_exp)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    seed_folders_list = os.listdir(config.save_path_checkpoint)
    seed_folders_list = sorted(seed_folders_list, key=lambda s: int(s.split("_")[-1]))

    for seed_folder in seed_folders_list:
        seed = int(seed_folder.split("_")[-1])
        curr_folder_path = os.path.join(config.save_path_checkpoint, seed_folder)
        file_in_folder = os.listdir(curr_folder_path)
        if len(file_in_folder) == 0 or len(file_in_folder) > 1:
            raise ValueError(f"Folder {curr_folder_path} has {len(file_in_folder)} files")
        curr_folder_path = os.path.join(curr_folder_path, file_in_folder[0])
        checkpoint_path = os.path.join(curr_folder_path, f"checkpoint_{checkpoint}.pt")
        for instance_files in os.listdir(config.path_eval):
            instance_name = instance_files.split(".")[0]
            save_folder_instance = os.path.join(exp_folder, instance_name)
            if not os.path.exists(save_folder_instance):
                os.makedirs(save_folder_instance)
            instance_path = os.path.join(config.path_eval, instance_files)
            instance_arr = np.load(instance_path)
            print("Evaluating instance", instance_name)
            df = eval(config, instance_arr, checkpoint_path)
            save_path = os.path.join(save_folder_instance, f"seed_{seed}.csv")
            df.to_csv(save_path, index=False)


if __name__ == "__main__":
    eval_all()

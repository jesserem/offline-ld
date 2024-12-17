# source: https://github.com/young-geng/CQL/tree/934b0e8354ca431d6c083c4e3a29df88d4b0a24d
# STRONG UNDER-PERFORMANCE ON PART OF ANTMAZE TASKS. BUT IN IQL PAPER IT WORKS SOMEHOW
# https://arxiv.org/pdf/2006.04779.pdf
import os
import time
import uuid
from dataclasses import asdict, dataclass
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.JSSP_Env import SJSSP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_jsp_fsp import parseAndMake

from typing import Any, Dict, List, Optional, Tuple, Callable
from offline_jssp_rl.utils import (
    compute_mean_std_graph as compute_mean_std,
    eval_actor_l2d_original_dqn as eval_actor_l2d,
    normalize_states,
    set_env_seed,
    set_seed,
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
import wandb
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

    n_jobs: int = 6 # Number of jobs
    n_machines: int = 6 # Number of machines
    eval_instances: str = f"./eval_generated_data/generatedData6_6_Seed300.npy"
    dataset: str = f"L2D/6_6/Small_Dataset-v0"
    # dataset: str = f"jsspl2d-norm_reward_05_noisy_prob_01-v0"

    eval_attributes: List[str] = ('last_time_step',)

    seed: int = 4  # Sets Gym, PyTorch and Numpy seeds
    eval_seed: int = 4  # Eval environment seed
    eval_freq: int = int(1e3)  # How often (time steps) we evaluate
    n_episodes: int = 1  # How many episodes run during evaluation
    train_epochs: int = 5000  # How many epochs to train
    eval_every_n_epochs: int = 10  # How often to evaluate
    use_epochs: bool = False
    checkpoints_path: Optional[str] = None  # Save path
    offline_iterations: int = int(2.5e5)  # Number of offline iterations
    load_model: str = ""  # Model load file name, "" doesn't load
    # CQL
    buffer_size: int = 1_000_000  # Replay buffer size
    batch_size: int = 64 # Batch size for all networks
    discount: float = 1  # Discount factor
    dqn_lr: float = 2e-5  # Critics learning rate
    dqn_dueling: bool = False # Use dueling DQN
    # soft_target_update_rate: float = 5e-3 # Target network update rate
    soft_target_update_rate: float = 1 # Target network update rate
    # target_update_period: int = 1
    target_update_period: int = 1000  # Frequency of target nets updates
    cql_alpha: float = 1 # CQL offline regularization parameter
    use_per: bool = False  # Use PER
    use_munchausen: bool = False # Use Munchausen
    munchausen_alpha: float = 0.9  # Munchausen alpha
    munchausen_tau: float = 0.03  # Munchausen tau
    beta_per: float = 0.6  # PER beta
    cql_n_actions: int = 1  # Number of sampled actions
    cql_lagrange: bool = False # Use Lagrange version of CQL
    cql_target_action_gap: float = 1 # Action gap
    cql_temp: float = 1 # CQL temperature
    cql_clip_diff_min: float = -np.inf  # Q-function lower loss clipping
    cql_clip_diff_max: float = np.inf  # Q-function upper loss clipping
    orthogonal_init: bool = False  # Orthogonal initialization
    normalize: bool = True # Normalize states
    normalize_reward: bool = False # Normalize reward
    q_n_hidden_layers: int = 1  # Number of hidden layers in Q networks
    hidden_dim_fe: int = 64
    hidden_dim_fe_mlp: int = 64
    n_dim_fe: int = 2
    n_dim_fe_mlp: int = 2
    q_hidden_size: int = 32  # Hidden size in Q networks
    loss_func: str = "smooth_l1_loss"
    activation_fn: str = "relu"  # Activation function
    graph_pool_type: str = "average"
    neighbor_pooling_type: str = "sum"
    reward_scale: float = 0.01  # Reward scale for normalization
    reward_bias: float = 0 # Reward bias for normalization
    q_dropout: float = 0  # Dropout in actor network
    gnn_dropout: float = 0  # Dropout in GNN
    use_next_action: bool = True  # Use next action in Q networks
    # Cal-QL
    mixing_ratio: float = 0.5  # Data mixing ratio for online tuning
    is_sparse_reward: bool = False  # Use sparse reward
    use_cal_ql: bool = True  # Use Cal-QL
    shuffle_buffer: bool = True # Shuffle buffer
    decay_lr_steps: int = 1 # Decay steps for step size
    decay_gamma: float = 1 # Decay gamma for online tuning

    decay_lr_step: bool = True  # Decay learning rate
    # Wandb logging
    project: str = "JSSP-Offline"
    group: str = f"{n_jobs}x{n_machines}"
    name: str = "QRDQN-CQL-" + "-" + dataset

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
        is_discrete: bool = False,
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
        adj_batch = self._adj[indices].to(torch.float32)
        features_batch = self._features[indices]
        # features_batch = features_batch.reshape(-1, features_batch.size(-1))
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices].to(torch.float32)
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
            adj_batch = aggr_obs(self._adj[batch_indices].to_sparse(), self._adj.shape[1]).to(self._device)
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
            adj_batch = aggr_obs(self._adj[batch_indices].to_sparse(), self._adj.shape[1])
            features_batch = self._features[batch_indices]
            features_batch = features_batch.reshape(-1, features_batch.size(-1))
            next_adj_batch = aggr_obs(self._next_adj[batch_indices].to_sparse(), self._next_adj.shape[1])
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
            orthogonal_init: bool = False
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
                                        dropout=dropout).to(device)

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


class QRDQN_net(nn.Module):
    def __init__(self,
             n_j: int,
             n_m: int,

             num_layers: int,
             hidden_dim: int,
             learn_eps: bool,
             input_dim: int,
             action_dim: int,
             num_mlp_layers_feature_extract: int,
             neighbor_pooling_type: str,
             device: str,
             N: int = 32,
             q_dropout: float = 0.0,
             gnn_dropout: float = 0.0,
             orthogonal_init: bool = False,
             hidden_dim_q: int = 256,
             n_hidden_q: int = 3,
             activation_fn: nn.Module = nn.ReLU,
             dueling: bool = True,
             ):
        super().__init__()
        self.N = N
        self.feature_extractor = FeatureExtractor(
            n_j=n_j,
            n_m=n_m,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            learn_eps=learn_eps,
            input_dim=input_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            neighbor_pooling_type=neighbor_pooling_type,
            device=device,
            dropout=gnn_dropout,
            orthogonal_init=orthogonal_init
        )
        self.dueling = dueling

        layers = []
        if q_dropout > 0:
            layers.append(nn.Dropout(p=q_dropout))
        layers.append(nn.Linear(hidden_dim * 2, hidden_dim_q))
        layers.append(activation_fn())

        for _ in range(n_hidden_q - 1):
            if q_dropout > 0:
                layers.append(nn.Dropout(p=q_dropout))
            layers.append(nn.Linear(hidden_dim_q, hidden_dim_q))
            layers.append(activation_fn())
        if q_dropout > 0:
            layers.append(nn.Dropout(p=q_dropout))
        layers.append(nn.Linear(hidden_dim_q, self.N))

        self.network = nn.Sequential(*layers)


        self.action_dim = action_dim


    def forward(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
                action_mask: torch.Tensor) -> torch.Tensor:
        out_net, h_pooled = self.feature_extractor(adj, features, candidate, graph_pool)

        q_values = self.network(out_net)

        return q_values

    def get_q_values(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
                     action_mask: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(adj, features, candidate, graph_pool, action_mask)
        q_values = q_values.mean(dim=2)
        q_values = q_values.masked_fill(~action_mask, -np.inf)
        return q_values

    @torch.no_grad()
    def act(self, adj: torch.Tensor, features: torch.Tensor, candidate: torch.Tensor, graph_pool: torch.Tensor,
            action_mask: Optional[np.ndarray], device: str = "cpu",
            deterministic: bool = False):
        if action_mask is not None:
            action_mask = torch.tensor(action_mask.reshape(1, -1), device=device, dtype=torch.bool)
        with torch.no_grad():
            q_values = self.get_q_values(adj, features, candidate, graph_pool, action_mask)
            actions = q_values.argmax(dim=-1)
            return actions.item()


def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    # print(loss.shape)
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss


class DiscreteCQL_DQN:
    def __init__(
        self,
        dqn_net: QRDQN_net,
        target_dqn_net: QRDQN_net,
        dqn_optimizer: torch.optim.Optimizer,
        mb_graph_pool,
        n_tasks: int,
        graph_pool_type: str,
        discount: float = 0.99,
        use_munchausen: bool = True,
        entropy_tau: float = 0.03,
        dqn_lr: float = 3e-4,
        soft_target_update_rate: float = 5e-3,
        target_update_period: int = 1,
        cql_lagrange: bool = False,
        cql_target_action_gap: float = -1.0,
        cql_temp: float = 1.0,
        cql_alpha: float = 5.0,
        N: int = 32,
        cql_clip_diff_min: float = -np.inf,
        cql_clip_diff_max: float = np.inf,
        device: str = "cpu",
        action_dim: int = 1,
        munchausen_alpha: float = 0.9,
        loss_func: Callable = F.huber_loss,
        decay_lr_steps: int = 500,
        decay_gamma: float = 0.9,
        max_steps: int = 1,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.graph_pool_type = graph_pool_type
        self.action_dim = action_dim
        self.discount = discount
        self.entropy_tau = entropy_tau
        self.munchausen_alpha = munchausen_alpha
        self.loss_func = loss_func
        self.N = N
        self.quantile_tau = torch.FloatTensor([i / self.N for i in range(1, self.N + 1)]).to(device)
        self.soft_target_update_rate = soft_target_update_rate


        self.target_update_period = target_update_period

        self.cql_lagrange = cql_lagrange
        self.cql_target_action_gap = cql_target_action_gap
        self.cql_temp = cql_temp
        self.cql_alpha = cql_alpha
        self.cql_clip_diff_min = cql_clip_diff_min
        self.cql_clip_diff_max = cql_clip_diff_max
        self._device = device
        self.mb_graph_pool = mb_graph_pool

        self.total_it = 0

        self.dqn_net = dqn_net
        self.target_dqn_net = target_dqn_net
        self.target_dqn_net.eval()
        self.dqn_optimizer = dqn_optimizer
        self.dqn_lr = dqn_lr
        self.use_munchausen = use_munchausen





        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.dqn_lr,
        )

        self._calibration_enabled = True
        self.total_it = 0
        self.dqn_decay_lr = torch.optim.lr_scheduler.CosineAnnealingLR(self.dqn_optimizer, T_max=int(1e6))

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.target_dqn_net, self.dqn_net, soft_target_update_rate)


    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled


    def _target_q_values(self, next_adj: torch.Tensor, next_features: torch.Tensor, next_candidates: torch.Tensor,
                         next_action_masks: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor,
                         mb_g_pool: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            batch_size = rewards.shape[0]
            # print(batch_size)
            target_q_values_all = self.target_dqn_net(next_adj, next_features, next_candidates, mb_g_pool,
                                                      next_action_masks)
            masked_q_target = target_q_values_all.mean(dim=2).masked_fill(~next_action_masks, -np.inf)
            action_idx = masked_q_target.argmax(dim=-1)
            # test = action_idx.unsqueeze(-1).expand(batch_size, self.N, 1)

            q_target_values = target_q_values_all.gather(1, action_idx.reshape(batch_size, 1, 1).expand(batch_size, 1, self.N)).transpose(1, 2)
            rewards = rewards.unsqueeze(-1)
            dones = dones.unsqueeze(-1)
            target_q_values = rewards + (1.0 - dones) * self.discount * q_target_values

        return target_q_values


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


        target_q_values = self._target_q_values(
                next_adj,
                next_features,
                next_candidates,
                next_action_masks,
                rewards, dones,
                mb_g_pool
            )

        batch_size = rewards.shape[0]
        q_values = self.dqn_net(adj, features, candidates, mb_g_pool, action_masks)
        q_action = q_values.gather(1, actions.reshape(batch_size, 1, 1).expand(batch_size, 1, self.N))
        td_error = q_action - target_q_values

        huber_loss = calculate_huber_loss(td_error, k=1.0)
        quantil_loss = abs(self.quantile_tau - (td_error.detach() < 0).float()) * huber_loss / 1.0
        qf_loss = quantil_loss.sum(dim=2).mean(dim=1)

        qf_loss = qf_loss.mean()


        q_values = q_values.mean(dim=2)
        q_values = q_values.masked_fill(~action_masks, -np.inf)
        q_action = q_values.gather(1, actions)

        cql_qf_ood = torch.logsumexp(q_values / self.cql_temp, dim=1) * self.cql_temp
        """Subtract the log likelihood of data"""

        cql_qf_diff = torch.clamp(
            cql_qf_ood - q_action,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf_diff - self.cql_target_action_gap)
            )


            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = cql_min_qf_loss
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf_loss = cql_qf_diff * self.cql_alpha
            alpha_prime_loss = actions.new_tensor(0.0)
            alpha_prime = actions.new_tensor(0.0)
        cql_qf_loss = qf_loss + cql_min_qf_loss

        self.dqn_optimizer.zero_grad()
        cql_qf_loss.backward()
        self.dqn_optimizer.step()


        log_dict = dict(
            qf_loss=qf_loss.item(),
            average_qf=q_action.mean().item(),
            std_qf=q_action.std().item(),
            average_target_q=target_q_values.mean().item(),
            std_target_q=target_q_values.std().item(),
            cql_loss=cql_min_qf_loss.item(),
            loss=cql_qf_loss.item(),
            cql_qf_diff=cql_qf_diff.item(),
            alpha_prime_loss=alpha_prime_loss.item(),
            alpha_prime=alpha_prime.item(),
            dqn_lr=self.dqn_optimizer.param_groups[0]["lr"],
        )

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dqn_net": self.dqn_net.state_dict(),
            "target_dqn_net": self.target_dqn_net.state_dict(),
            "dqn_optimizer": self.dqn_optimizer.state_dict(),
            "cql_log_alpha": self.log_alpha_prime,
            "cql_log_alpha_optim": self.alpha_prime_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.dqn_net.load_state_dict(state_dict["dqn_net"])
        self.target_dqn_net.load_state_dict(state_dict["target_dqn_net"])
        self.dqn_optimizer.load_state_dict(state_dict["dqn_optimizer"])
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

            for i in range(epi.actions.shape[0]):
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


@pyrallis.wrap()
def train(config: TrainConfig):
    best_eval_score = -np.inf

    datasets = [config.dataset]
    dataset_min = [minari.load_dataset(d) for d in datasets]
    dataset, adj_shape, features_shape, omega_shape, mask_shape = minari_dataset_to_dict(dataset_min)

    dataset_size = len(dataset["rewards"])
    env = SJSSP(config.n_jobs, config.n_machines)
    eval_env = SJSSP(config.n_jobs, config.n_machines)

    if config.activation_fn == "relu":
        activation_fn = nn.ReLU
    elif config.activation_fn == "tanh":
        activation_fn = nn.Tanh
    else:
        raise ValueError(f"Unknown activation function: {config.activation_fn}")


    if config.normalize_reward:
        dataset["rewards"] = dataset["rewards"] * config.reward_scale + config.reward_bias


    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["features"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1
    dataset["features"] = normalize_states(
        dataset["features"], state_mean, state_std
    )
    dataset["next_features"] = normalize_states(
        dataset["next_features"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std, min_max_normalize=False)
    eval_env = wrap_env(eval_env, state_mean=state_mean, state_std=state_std, min_max_normalize=False)

    offline_buffer = ReplayBuffer(
            adj_shape=adj_shape,
            feature_shape=features_shape,
            omega_shape=omega_shape,
            mask_shape=mask_shape,
            buffer_size=dataset_size,
            device=config.device,
    )
    offline_buffer.load_d4rl_dataset(dataset)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)
    set_env_seed(eval_env, config.eval_seed)


    # feature_extractor = GraphCNN(
    #     num_layers=2,
    #     hidden_dim=64,
    #
    # )

    g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
                             batch_size=torch.Size([1, config.n_jobs * config.n_machines, config.n_jobs * config.n_machines]),
                             n_nodes=config.n_jobs * config.n_machines, device=config.device)

    mb_g_pool = g_pool_cal(config.graph_pool_type,
                           (config.batch_size, config.n_jobs * config.n_machines, config.n_jobs * config.n_machines),
                           config.n_jobs * config.n_machines, config.device)
    dqn_net = QRDQN_net(
        n_j=config.n_jobs,
        n_m=config.n_machines,
        input_dim=2,
        neighbor_pooling_type=config.neighbor_pooling_type,
        hidden_dim=config.hidden_dim_fe,
        num_layers=config.n_dim_fe,
        num_mlp_layers_feature_extract=config.n_dim_fe_mlp,
        learn_eps=False,
        device=config.device,
        gnn_dropout=config.gnn_dropout,
        q_dropout=config.q_dropout,
        orthogonal_init=config.orthogonal_init,
        action_dim=config.n_jobs,
        hidden_dim_q=config.q_hidden_size,
        n_hidden_q=config.q_n_hidden_layers,
        activation_fn=activation_fn,
        dueling=config.dqn_dueling,
    ).to(config.device)
    target_dqn_net = QRDQN_net(
        n_j=config.n_jobs,
        n_m=config.n_machines,
        input_dim=2,
        neighbor_pooling_type=config.neighbor_pooling_type,
        hidden_dim=config.hidden_dim_fe,
        num_layers=config.n_dim_fe,
        num_mlp_layers_feature_extract=config.n_dim_fe_mlp,
        learn_eps=False,
        device=config.device,
        gnn_dropout=config.gnn_dropout,
        q_dropout=config.q_dropout,
        orthogonal_init=config.orthogonal_init,
        action_dim=config.n_jobs,
        hidden_dim_q=config.q_hidden_size,
        n_hidden_q=config.q_n_hidden_layers,
        activation_fn=activation_fn,
        dueling=config.dqn_dueling,
    ).to(config.device)
    target_dqn_net.load_state_dict(dqn_net.state_dict())
    dqn_optimizer = torch.optim.Adam(dqn_net.parameters(), lr=config.dqn_lr, weight_decay=1e-3)


    num_steps_decay = (config.train_epochs * int(np.ceil(dataset_size / config.batch_size)))
    kwargs = {

        "mb_graph_pool": mb_g_pool.clone(),
        "graph_pool_type": config.graph_pool_type,
        "n_tasks": config.n_jobs * config.n_machines,

        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "entropy_tau": config.munchausen_tau,
        "munchausen_alpha": config.munchausen_alpha,

        "dqn_net": dqn_net,
        "target_dqn_net": target_dqn_net,
        "dqn_optimizer": dqn_optimizer,
        "dqn_lr": config.dqn_lr,
        "use_munchausen": config.use_munchausen,
        "target_update_period": config.target_update_period,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
        "max_steps": num_steps_decay,
        "decay_lr_steps": config.decay_lr_steps,
        "decay_gamma": config.decay_gamma,
    }


    print("---------------------------------------")
    print(f"Training CQL-DQN, Env: n_jobs {config.n_jobs}, n_machines {config.n_machines}, Seed: {seed}")
    print("---------------------------------------")

    kwargs["action_dim"] = env.action_space.n
    trainer = DiscreteCQL_DQN(**kwargs)

    wandb_init(asdict(config))

    eval_instance_arr = np.load(config.eval_instances)

    print("Offline pretraining")
    start_time = time.time()
    if config.use_epochs:
        config.offline_iterations = 0
        run_times = np.zeros(config.train_epochs)
        for e in range(config.train_epochs):
            start_time = time.time()
            for batch in offline_buffer.train_epoch(config.batch_size, shuffle=config.shuffle_buffer):
                batch = [b.to(config.device) for b in batch]

                log_dict = trainer.train(batch)
                wandb.log(log_dict, step=trainer.total_it)
                # if config.decay_lr_step:
                #     trainer.step_lr()
            run_times[e] = time.time() - start_time

            # print(f"Time steps: {t + 1}")
            if (e + 1) % config.eval_every_n_epochs == 0:
                eval_scores, success_rate, make_spans, run_times_instances = eval_actor_l2d(
                    env=eval_env,
                    actor=dqn_net,
                    readArrFunc=parseAndMake,
                    eval_instances=eval_instance_arr,
                    device=config.device,
                    g_pool=g_pool_step,
                    deterministic=True,
                )

                eval_score = eval_scores.mean()
                mean_make_span = np.mean(make_spans)


                best_eval_score = max(best_eval_score, eval_score)
                mean_run_time_epoch = np.mean(run_times[:e + 1])
                std_run_time_epoch = np.std(run_times[:e + 1])
                mean_run_time_instance = np.mean(run_times_instances)
                std_run_time_instance = np.std(run_times_instances)
                eval_log = {
                    "eval/score": eval_score,
                    "eval/best_score": best_eval_score,
                    "eval/make_span": mean_make_span,
                    "eval/run_time_instance": mean_run_time_instance,
                }
                # Valid only for envs with goal, e.g. AntMaze, Adroit


                print("---------------------------------------")
                print(
                    f"Epoch {e + 1}\n"
                    f"Evaluation over {config.n_episodes} episodes:"
                    f"\nEval Score: {eval_score:.3f}"
                    f"\nMean Make Span: {mean_make_span:.3f}"
                    f"\nMean Run Time Epoch: {mean_run_time_epoch:.3f} ± {std_run_time_epoch:.3f}"
                    f"\nMean Run Time Instance: {mean_run_time_instance:.3f} ± {std_run_time_instance:.3f}"
                )
                print("---------------------------------------")
                if config.checkpoints_path is not None:
                    results_dict = {
                        "epoch": e + 1,
                        "eval_score": eval_score,
                        "make_span": mean_make_span,
                        "run_time": mean_run_time_instance,
                        "std_run_time": std_run_time_instance,
                        "run_time_epoch": mean_run_time_epoch,
                        "std_run_time_epoch": std_run_time_epoch,
                    }
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_epoch_{e+1}.pt"),
                    )
                    write_results_checkpoint(results_dict, config.checkpoints_path, "results.csv")
                wandb.log(eval_log, step=trainer.total_it)
            # actor_step_lr.step()
    else:
        for t in range(config.offline_iterations):


            offline_batch = offline_buffer.sample_c(config.batch_size)
            batch = [b.to(config.device) for b in offline_batch]
            log_dict = trainer.train(batch)
            wandb.log(log_dict, step=trainer.total_it)

            if (t + 1) % config.eval_freq == 0:
                eval_scores, success_rate, make_spans, run_times_instances = eval_actor_l2d(
                    env=eval_env,
                    actor=dqn_net,
                    readArrFunc=parseAndMake,
                    eval_instances=eval_instance_arr,
                    device=config.device,
                    g_pool=g_pool_step,
                    deterministic=True,
                )

                eval_score = eval_scores.mean()
                mean_make_span = np.mean(make_spans)
                mean_run_time = np.mean(run_times_instances)
                std_run_time = np.std(run_times_instances)

                best_eval_score = max(best_eval_score, eval_score)
                eval_log = {
                    "eval/score": eval_score,
                    "eval/best_score": best_eval_score,
                    "eval/make_span": mean_make_span,
                    "eval/run_time": mean_run_time,
                }
                # Valid only for envs with goal, e.g. AntMaze, Adroit
                curr_run_time = time.time() - start_time
                mean_step_time = curr_run_time / (t + 1)
                expected_time = mean_step_time * (config.offline_iterations - t)

                print("---------------------------------------")
                print(
                    f"Training Step {t + 1}\n"
                    f"Evaluation over {config.n_episodes} episodes:"
                    f"\nEval Score: {eval_score:.3f}"
                    f"\nMean Make Span: {mean_make_span:.3f}"
                    f"\nMean Run Time Instance: {mean_run_time:.3f}"
                    f"\n\nMean Run Time Step: {mean_step_time:.3f}"
                    f"\nExpected Time Remaining: {convert_to_preferred_format(expected_time)}"

                )
                print("---------------------------------------")
                if config.checkpoints_path is not None:
                    results_dict = {
                        "step": trainer.total_it,
                        "eval_score": eval_score,
                        "make_span": mean_make_span,
                        "run_time": mean_run_time,
                        "std_run_time": std_run_time,
                    }

                    torch.save(
                        trainer.state_dict(),
                        os.path.join(config.checkpoints_path, f"checkpoint_{trainer.total_it}.pt"),
                    )
                    write_results_checkpoint(results_dict, config.checkpoints_path, "results.csv")
                wandb.log(eval_log, step=trainer.total_it)



if __name__ == "__main__":
    train()
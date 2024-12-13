import numpy as np
import torch
from pathlib import Path
from .dataset import Dataset, load_dataset, save_dataset
import os
from ..networks.gin_backup import aggr_obs
from typing import List, Callable, Dict
import operator

TensorBatch = List[torch.Tensor]

class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)

class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, 1), dtype=torch.int64, device=device
        )
        self._action_masks = torch.ones(
            (buffer_size, action_dim), dtype=torch.bool, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device)
        self._device = device

    def train_epoch(self, batch_size: int, shuffle: bool = False) -> TensorBatch:
        indices = torch.randperm(self._size) if shuffle else torch.arange(self._size)
        for start in range(0, self._size, batch_size):
            batch_indices = indices[start : start + batch_size]
            yield [self._states[batch_indices], self._actions[batch_indices], \
                   self._rewards[batch_indices], self._next_states[batch_indices], \
                   self._dones[batch_indices], self._action_masks[batch_indices]]

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def add_dataset(self, data: Dataset):

        n_transitions = len(data)
        print("current size: ", self._size)
        if (n_transitions + self._size) > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[self._size:(n_transitions+self._size)] = self._to_tensor(data.states)
        self._actions[self._size:(n_transitions+self._size)] = self._to_tensor(data.actions).unsqueeze(dim=-1)
        self._rewards[self._size:(n_transitions+self._size)] = self._to_tensor(data.rewards).unsqueeze(dim=-1)
        self._next_states[self._size:(n_transitions+self._size)] = self._to_tensor(data.next_states)
        self._dones[self._size:(n_transitions+self._size)] = self._to_tensor(data.dones).unsqueeze(dim=-1)

        self._action_masks[self._size:(n_transitions+self._size)] = self._to_tensor(data.action_masks)
        self._size += n_transitions
        self._pointer = self._size
        self._pointer = self._pointer % self._buffer_size

        print(f"Dataset size: {self._size}")

    def load_datasets_in_buffer(self, datasets: List[Dataset]) -> None:
        for data_set in datasets:
            self.add_dataset(data_set)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        action_masks = self._action_masks[indices]
        return [states, actions, rewards, next_states, dones, action_masks]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._action_masks[self._pointer] = self._to_tensor(action_mask)
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


class ReplayBufferFineTune:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, 1), dtype=torch.int64, device=device
        )
        self._action_masks = torch.ones(
            (buffer_size, action_dim), dtype=torch.bool, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.bool, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def add_dataset(self, data: Dataset):

        n_transitions = len(data)
        print("current size: ", self._size)
        if (n_transitions + self._size) > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[self._size:(n_transitions+self._size)] = self._to_tensor(data.states)
        self._actions[self._size:(n_transitions+self._size)] = self._to_tensor(data.actions).unsqueeze(dim=-1)
        self._rewards[self._size:(n_transitions+self._size)] = self._to_tensor(data.rewards).unsqueeze(dim=-1)
        self._next_states[self._size:(n_transitions+self._size)] = self._to_tensor(data.next_states)
        self._dones[self._size:(n_transitions+self._size)] = self._to_tensor(data.dones).unsqueeze(dim=-1)

        self._action_masks[self._size:(n_transitions+self._size)] = self._to_tensor(data.action_masks)
        self._size += n_transitions
        self._pointer = self._size
        self._pointer = self._pointer % self._buffer_size

        print(f"Dataset size: {self._size}")

    def load_datasets_in_buffer(self, datasets: List[Dataset]) -> None:
        for data_set in datasets:
            self.add_dataset(data_set)

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        action_masks = self._action_masks[indices]
        return [states, actions, rewards, next_states, dones, action_masks]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ):
        # Use this method to add new data into the replay buffer during fine-tuning.
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(reward)
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(done)
        self._action_masks[self._pointer] = self._to_tensor(action_mask)
        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)



class ReplayBufferCQL_L2D:
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
        adj_batch = self._adj[indices]
        adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1]).to(self._device)
        features_batch = self._features[indices]
        features_batch = features_batch.reshape(-1, features_batch.size(-1)).to(self._device)
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices]
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


    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        adj_batch = self._adj[indices].to(torch.float32)
        features_batch = self._features[indices]
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


class PrioritizedReplayBufferCQL_L2D(ReplayBufferCQL_L2D):
    def __init__(
            self,
            adj_shape: np.ndarray,
            feature_shape: np.ndarray,
            omega_shape: np.ndarray,
            mask_shape: np.ndarray,
            buffer_size: int,
            is_discrete: bool = False,
            device: str = "cpu",
            alpha: float = 0.6
    ):
        super(PrioritizedReplayBufferCQL_L2D, self).__init__(
            adj_shape,
            feature_shape,
            omega_shape,
            mask_shape,
            buffer_size,
            is_discrete,
            device)
        self._alpha = alpha
        self._max_priority, self._tree_ptr = 1.0, 0
        tree_capacity = 1
        while tree_capacity < self._buffer_size:
            tree_capacity *= 2
        self._sum_tree = SumSegmentTree(tree_capacity)
        self._min_tree = MinSegmentTree(tree_capacity)

    def sample_c(self, batch_size: int, beta: float = 0.4) -> TensorBatch:
        if self._device == "cpu":
            return self.sample_c_cpu(batch_size, beta)
        else:
            return self.sample_c_gpu(batch_size, beta)

    def sample_c_gpu(self, batch_size: int, beta: float = 0.4) -> TensorBatch:
        indices = self._sample_proportional(batch_size)
        adj_batch = self._adj[indices]
        adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1]).to(self._device)
        features_batch = self._features[indices]
        features_batch = features_batch.reshape(-1, features_batch.size(-1)).to(self._device)
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices]
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
        weights = torch.tensor([self._calculate_weight(idx, beta) for idx in indices], dtype=torch.float32, device=self._device)
        return [adj_batch, features_batch, omega, actions, action_masks, rewards, next_adj_batch, next_features_batch, next_omega, next_action_masks, dones, indices, weights]

    def sample_c_cpu(self, batch_size: int, beta: float=0.4) -> TensorBatch:
        indices = self._sample_proportional(batch_size)
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
        weights = torch.tensor([self._calculate_weight(idx, beta) for idx in indices], dtype=torch.float32, device=self._device)

        return [adj_batch, features_batch, omega, actions, action_masks, rewards, next_adj_batch, next_features_batch, next_omega, next_action_masks, dones, indices, weights]

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        super().load_d4rl_dataset(data)
        for idx in range(self._size):
            self._sum_tree[idx] = self._max_priority ** self._alpha
            self._min_tree[idx] = self._max_priority ** self._alpha

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority.item() > 0
            assert 0 <= idx < self._size

            self._sum_tree[idx] = priority.item() ** self._alpha
            self._min_tree[idx] = priority.item() ** self._alpha

            self._max_priority = max(self._max_priority, priority.item())

    def _sample_proportional(self, batch_size: int) -> torch.Tensor:
        """Sample indices based on proportions."""
        indices = []
        p_total = self._sum_tree.sum(0, self._size - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self._sum_tree.retrieve(upperbound)
            indices.append(idx)

        return torch.tensor(indices, dtype=torch.int64, device=self._device)

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self._min_tree.min() / self._sum_tree.sum()
        max_weight = (p_min * self._size) ** (-beta)

        # calculate weights
        p_sample = self._sum_tree[idx] / self._sum_tree.sum()
        weight = (p_sample * self._size) ** (-beta)
        weight = weight / max_weight

        return weight





class ReplayBufferIQL_L2D:
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

        self._adj = torch.zeros(
            (buffer_size, *adj_shape), dtype=torch.float32
        )
        self._features = torch.zeros(
            (buffer_size, *feature_shape), dtype=torch.float32
        )
        self._omega = torch.zeros(
            (buffer_size, *omega_shape), dtype=torch.int64
        )

        self._actions = torch.zeros(
            (buffer_size, 1), dtype=torch.int64
        )
        self._action_masks = torch.ones(
            (buffer_size, *mask_shape), dtype=torch.bool
        )

        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self._next_adj = torch.zeros(
            (buffer_size, *adj_shape), dtype=torch.float32
        )
        self._next_features = torch.zeros(
            (buffer_size, *feature_shape), dtype=torch.float32
        )
        self._next_omega = torch.zeros(
            (buffer_size, *omega_shape), dtype=torch.int64
        )
        self._next_action_masks = torch.ones(
            (buffer_size, *mask_shape), dtype=torch.bool
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32)

        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32)

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

    def sample_c(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        adj_batch = self._adj[indices]
        adj_batch = aggr_obs(adj_batch.to_sparse(), adj_batch.shape[1])
        features_batch = self._features[indices]
        features_batch = features_batch.reshape(-1, features_batch.size(-1))
        # next_adj_batch = aggr_obs(self._next_adj[indices].to_sparse(), self._next_adj.shape[1])
        next_adj_batch = self._next_adj[indices]
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

    def train_epoch(self, batch_size: int, shuffle: bool = False) -> TensorBatch:
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

# def custom_collate_fn(batch):

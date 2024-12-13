import pickle
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, DefaultDict
from collections import defaultdict
# from ..utils import discounted_cumsum
from tqdm.auto import trange

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum
# def one_hot_encode(arr, n_classes: Optional[int] = None):
#     # Find the number of unique elements (classes)
#     if n_classes is None:
#         n_classes = np.max(arr) + 1
#
#     # Initialize the one-hot encoded array
#     one_hot = np.eye(n_classes)[arr]
#
#     return one_hot

class Dataset(object):
    def __init__(
            self,
            action_space_shape: np.ndarray,
            observation_space_shape: np.ndarray,
            action_mask_shape: np.ndarray,
            size=2000):
        self.action_space_shape = action_space_shape
        self.observation_space_shape = observation_space_shape
        self.action_mask_shape = action_mask_shape
        self.states = np.zeros((size, *observation_space_shape))
        self.actions = np.zeros((size, *action_space_shape))
        self.rewards = np.zeros((size, 1))
        self.next_states = np.zeros((size, *observation_space_shape))
        self.dones = np.zeros((size, 1))
        self.action_masks = np.zeros((size, *action_mask_shape))
        self.next_action_masks = np.zeros((size, *action_mask_shape))
        self.curr_idx = 0
        self.size = size

    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray,
            action_mask: np.ndarray,
            next_action_mask: np.ndarray) -> None:
        self.states[self.curr_idx] = state
        self.actions[self.curr_idx] = action
        self.rewards[self.curr_idx] = reward
        self.next_states[self.curr_idx] = next_state
        self.dones[self.curr_idx] = done
        self.action_masks[self.curr_idx] = action_mask
        self.next_action_masks[self.curr_idx] = next_action_mask
        self.curr_idx = (self.curr_idx + 1) % self.size

    def load_trajectories(self, gamma: float = 1.0, num_actions: Optional[int] = None) \
            -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        data_ = defaultdict(list)

        # self.actions = one_hot_encode(self.actions.astype(int), num_actions)
        for i in trange(len(self), desc="Loading trajectories"):
            data_["states"].append(self.states[i])
            data_["actions"].append(self.actions[i])
            data_["rewards"].append(self.rewards[i])
            data_["action_masks"].append(np.append(self.action_masks[i], 0))

            if self.dones[i]:
                episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                episode_data["returns"] = discounted_cumsum(
                    episode_data["rewards"], gamma=gamma
                )
                traj.append(episode_data)
                traj_len.append(episode_data["actions"].shape[0])
                data_ = defaultdict(list)
        info = {
            "obs_mean": self.states.mean(0, keepdims=True),
            "obs_std": self.states.std(0, keepdims=True) + 1e-6,
            "traj_lens": np.array(traj_len),
            "fill_value": 0,
            "has_mask": True,
        }

        return traj, info

    def compute_mean_std(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.states.mean(0)
        std = self.states.std(0) + eps
        return mean, std

    def normalize_states(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: Tuple[int, List, np.ndarray]):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx],
                self.action_masks[idx], self.next_action_masks[idx])

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


def load_dataset(path: str) -> Dataset:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_dataset(path: str, dataset: Dataset) -> None:
    with open(path, 'wb') as f:
        pickle.dump(dataset, f)


def combine_datasets(dataset_list: List[Dataset]) -> Dataset:
    dataset = Dataset(dataset_list[0].action_space_shape, dataset_list[0].observation_space_shape,
                      dataset_list[0].action_mask_shape,
                      size=sum([len(d) for d in dataset_list]))
    for d in dataset_list:
        for i in range(len(d)):
            dataset.add(d.states[i], d.actions[i], d.rewards[i], d.next_states[i], d.dones[i], d.action_masks[i],
                        d.next_action_masks[i])
        # dataset.add_dataset(d)
    return dataset

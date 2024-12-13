import pickle
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, DefaultDict
from collections import defaultdict
# from ..utils import discounted_cumsum
from tqdm.auto import trange
import minari
import gymnasium as gym

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


class DatasetMinari(object):
    def __init__(
            self,
            datasets: List[minari.MinariDataset],
            action_mask_key: str = "action_mask",
            observation_space_key: str = "real_obs",
    ):
        self.min_datasets = datasets
        self.env = self.min_datasets[0].recover_environment()
        self.total_steps = sum([d.total_steps for d in self.min_datasets])
        # # self.total_steps = self.min_dataset.total_steps
        # self.episode_indices = self.min_dataset.episode_indices
        self.action_space_shape = self.env.action_space.shape
        self.n_actions = self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else None
        self.mask_shape = (self.n_actions,) if self.n_actions is not None else self.action_space_shape
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            print(self.env.observation_space)
            self.observation_space_shape = self.env.observation_space.spaces[observation_space_key].shape
            self.action_mask_shape = self.env.observation_space.spaces[action_mask_key].shape
            self.has_mask = True
        else:
            self.observation_space_shape = self.env.observation_space.shape
            self.action_mask_shape = None
            self.has_mask = False


        self.states = np.zeros((self.total_steps, *self.observation_space_shape))
        self.actions = np.zeros((self.total_steps, *self.action_space_shape))
        self.rewards = np.zeros((self.total_steps))
        self.next_states = np.zeros((self.total_steps, *self.observation_space_shape))
        self.dones = np.zeros((self.total_steps))
        self.action_masks = np.ones((self.total_steps, *self.mask_shape))
        self.next_action_masks = np.ones((self.total_steps, *self.mask_shape))
        self.curr_idx = 0
        def add_episode(episode: minari.EpisodeData):
            start_idx_state = self.curr_idx
            idx_state = 0
            idx_end_state = episode.total_timesteps
            idx_state_next = 1
            idx_end_state_next = episode.total_timesteps + 1
            end_idx_state = start_idx_state + episode.total_timesteps
            # start_idx_state_next = start_idx_state + 1
            # end_idx_state_next = end_idx_state + 1

            if self.has_mask:
                self.states[start_idx_state:end_idx_state] = episode.observations[observation_space_key][idx_state:idx_end_state]
                self.action_masks[start_idx_state:end_idx_state] = episode.observations[action_mask_key][idx_state:idx_end_state]
                self.next_states[start_idx_state:end_idx_state] = episode.observations[observation_space_key][idx_state_next:idx_end_state_next]
                self.next_action_masks[start_idx_state:end_idx_state] = episode.observations[action_mask_key][idx_state_next:idx_end_state_next]
            else:
                self.states[start_idx_state:end_idx_state] = episode.observations[idx_state:idx_end_state]
                self.next_states[start_idx_state:end_idx_state] = episode.observations[idx_state_next:idx_end_state_next]
            self.actions[start_idx_state:end_idx_state] = episode.actions
            self.rewards[start_idx_state:end_idx_state] = episode.rewards
            self.dones[start_idx_state:end_idx_state] = episode.terminations | episode.truncations
            self.curr_idx = end_idx_state
        for min_dataset in self.min_datasets:
            for epi in min_dataset.iterate_episodes(min_dataset.episode_indices):
                add_episode(epi)
        # for epi in self.min_dataset.iterate_episodes(self.episode_indices):
        #     add_episode(epi)

    def get_space(self) -> Tuple[int, int]:
        if self.n_actions is not None:
            return np.prod(self.observation_space_shape), self.n_actions
        else:
            return np.prod(self.observation_space_shape), np.prod(self.action_space_shape)

    # def get_normalized_scores(self, scores: np.ndarray) -> np.ndarray:
    #     # score = minari.get_normalized_scores(self.min_dataset, scores)
    #
    #     min_score = self.min_dataset.ref_min_score
    #     max_score = self.min_dataset.ref_max_score
    #     return (scores - min_score) / (max_score - min_score)

    def get_env(self) -> gym.Env:
        return self.min_datasets[0].recover_environment()

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
        self.curr_idx = (self.curr_idx + 1) % self.total_steps

    def switch_done_mask(self):
        """
        Will go through all the next_actions_masks and dones and switch the mask to all true if the next state is done
        :return:
        """
        for i in range(self.total_steps):
            if self.dones[i]:
                self.next_action_masks[i] = np.ones_like(self.next_action_masks[i])

    def get_dict(self) -> Dict[str, np.ndarray]:
        data = {
            "observations": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_states,
            "terminals": self.dones,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks
        }
        return data

    def load_trajectories(self, gamma: float = 1.0, num_actions: Optional[int] = None) \
            -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        data_ = defaultdict(list)

        for i in trange(len(self), desc="Loading trajectories"):
            data_["states"].append(self.states[i])
            data_["actions"].append(self.actions[i])
            data_["rewards"].append(self.rewards[i])

            if self.has_mask:
                # data_["action_masks"].append(np.append(self.action_masks[i], 0))
                data_["action_masks"].append(self.action_masks[i])

            else:
                data_["action_masks"].append(None)

            if self.dones[i]:
                episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                episode_data["returns"] = discounted_cumsum(
                    episode_data["rewards"], gamma=gamma
                )
                traj.append(episode_data)
                traj_len.append(episode_data["actions"].shape[0])
                data_ = defaultdict(list)
        # print(self.actions.shape)
        # exit()
        info = {
            "obs_mean": self.states.mean(0, keepdims=True),
            "obs_std": self.states.std(0, keepdims=True) + 1e-6,
            "traj_lens": np.array(traj_len),
            "fill_value": self.n_actions,
            "has_mask": self.has_mask
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
        return self.total_steps

    def __getitem__(self, idx: Tuple[int, List, np.ndarray]):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx],
                self.action_masks[idx], self.next_action_masks[idx])

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# def load_dataset(path: str) -> Dataset:
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
#     return data
#
#
# def save_dataset(path: str, dataset: Dataset) -> None:
#     with open(path, 'wb') as f:
#         pickle.dump(dataset, f)
#
#
# def combine_datasets(dataset_list: List[Dataset]) -> Dataset:
#     dataset = Dataset(dataset_list[0].action_space_shape, dataset_list[0].observation_space_shape,
#                       dataset_list[0].action_mask_shape,
#                       size=sum([len(d) for d in dataset_list]))
#     for d in dataset_list:
#         for i in range(len(d)):
#             dataset.add(d.states[i], d.actions[i], d.rewards[i], d.next_states[i], d.dones[i], d.action_masks[i],
#                         d.next_action_masks[i])
#         # dataset.add_dataset(d)
#     return dataset


class DatasetMinariL2D(object):
    def __init__(
            self,
            datasets: List[minari.MinariDataset],
    ):
        self.min_datasets = datasets
        self.total_steps = sum([d.total_steps for d in self.min_datasets])
        # # self.total_steps = self.min_dataset.total_steps
        # self.episode_indices = self.min_dataset.episode_indices
        self.action_space_shape = self.env.action_space.shape
        self.n_actions = self.env.action_space.n if isinstance(self.env.action_space, gym.spaces.Discrete) else None
        self.mask_shape = (self.n_actions,) if self.n_actions is not None else self.action_space_shape
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            print(self.env.observation_space)
            self.observation_space_shape = self.env.observation_space.spaces[observation_space_key].shape
            self.action_mask_shape = self.env.observation_space.spaces[action_mask_key].shape
            self.has_mask = True
        else:
            self.observation_space_shape = self.env.observation_space.shape
            self.action_mask_shape = None
            self.has_mask = False


        self.states = np.zeros((self.total_steps, *self.observation_space_shape))
        self.actions = np.zeros((self.total_steps, *self.action_space_shape))
        self.rewards = np.zeros((self.total_steps))
        self.next_states = np.zeros((self.total_steps, *self.observation_space_shape))
        self.dones = np.zeros((self.total_steps))
        self.action_masks = np.ones((self.total_steps, *self.mask_shape))
        self.next_action_masks = np.ones((self.total_steps, *self.mask_shape))
        self.curr_idx = 0
        def add_episode(episode: minari.EpisodeData):
            start_idx_state = self.curr_idx
            idx_state = 0
            idx_end_state = episode.total_timesteps
            idx_state_next = 1
            idx_end_state_next = episode.total_timesteps + 1
            end_idx_state = start_idx_state + episode.total_timesteps
            # start_idx_state_next = start_idx_state + 1
            # end_idx_state_next = end_idx_state + 1

            if self.has_mask:
                self.states[start_idx_state:end_idx_state] = episode.observations[observation_space_key][idx_state:idx_end_state]
                self.action_masks[start_idx_state:end_idx_state] = episode.observations[action_mask_key][idx_state:idx_end_state]
                self.next_states[start_idx_state:end_idx_state] = episode.observations[observation_space_key][idx_state_next:idx_end_state_next]
                self.next_action_masks[start_idx_state:end_idx_state] = episode.observations[action_mask_key][idx_state_next:idx_end_state_next]
            else:
                self.states[start_idx_state:end_idx_state] = episode.observations[idx_state:idx_end_state]
                self.next_states[start_idx_state:end_idx_state] = episode.observations[idx_state_next:idx_end_state_next]
            self.actions[start_idx_state:end_idx_state] = episode.actions
            self.rewards[start_idx_state:end_idx_state] = episode.rewards
            self.dones[start_idx_state:end_idx_state] = episode.terminations | episode.truncations
            self.curr_idx = end_idx_state
        for min_dataset in self.min_datasets:
            for epi in min_dataset.iterate_episodes(min_dataset.episode_indices):
                add_episode(epi)
        # for epi in self.min_dataset.iterate_episodes(self.episode_indices):
        #     add_episode(epi)

    def get_space(self) -> Tuple[int, int]:
        if self.n_actions is not None:
            return np.prod(self.observation_space_shape), self.n_actions
        else:
            return np.prod(self.observation_space_shape), np.prod(self.action_space_shape)

    # def get_normalized_scores(self, scores: np.ndarray) -> np.ndarray:
    #     # score = minari.get_normalized_scores(self.min_dataset, scores)
    #
    #     min_score = self.min_dataset.ref_min_score
    #     max_score = self.min_dataset.ref_max_score
    #     return (scores - min_score) / (max_score - min_score)

    def get_env(self) -> gym.Env:
        return self.min_datasets[0].recover_environment()

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
        self.curr_idx = (self.curr_idx + 1) % self.total_steps

    def switch_done_mask(self):
        """
        Will go through all the next_actions_masks and dones and switch the mask to all true if the next state is done
        :return:
        """
        for i in range(self.total_steps):
            if self.dones[i]:
                self.next_action_masks[i] = np.ones_like(self.next_action_masks[i])

    def get_dict(self) -> Dict[str, np.ndarray]:
        data = {
            "observations": self.states,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_states,
            "terminals": self.dones,
            "action_masks": self.action_masks,
            "next_action_masks": self.next_action_masks
        }
        return data

    def load_trajectories(self, gamma: float = 1.0, num_actions: Optional[int] = None) \
            -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
        traj, traj_len = [], []
        data_ = defaultdict(list)

        for i in trange(len(self), desc="Loading trajectories"):
            data_["states"].append(self.states[i])
            data_["actions"].append(self.actions[i])
            data_["rewards"].append(self.rewards[i])

            if self.has_mask:
                # data_["action_masks"].append(np.append(self.action_masks[i], 0))
                data_["action_masks"].append(self.action_masks[i])

            else:
                data_["action_masks"].append(None)

            if self.dones[i]:
                episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                episode_data["returns"] = discounted_cumsum(
                    episode_data["rewards"], gamma=gamma
                )
                traj.append(episode_data)
                traj_len.append(episode_data["actions"].shape[0])
                data_ = defaultdict(list)
        # print(self.actions.shape)
        # exit()
        info = {
            "obs_mean": self.states.mean(0, keepdims=True),
            "obs_std": self.states.std(0, keepdims=True) + 1e-6,
            "traj_lens": np.array(traj_len),
            "fill_value": self.n_actions,
            "has_mask": self.has_mask
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
        return self.total_steps

    def __getitem__(self, idx: Tuple[int, List, np.ndarray]):
        return (self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx],
                self.action_masks[idx], self.next_action_masks[idx])

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state
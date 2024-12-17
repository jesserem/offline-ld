
import numpy as np
from typing import Tuple, Union, Dict, Optional, List, Any, Callable
import os
import random
import uuid
import wandb
import torch
from collections import defaultdict
from torch import nn
import gymnasium as gym
import minari
import time

from .buffers.dataset import Dataset
import pandas as pd



def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def compute_mean_std_graph(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:

    mean = np.mean(states, axis=(0,1))
    std = np.std(states, axis=(0, 1)) + eps
    return mean, std


def compute_max_min_graph(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    max_val = np.max(states, axis=(0, 1))
    min_val = np.min(states, axis=(0, 1))
    return min_val, max_val


def scalarize_states(states: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    return (states - min_val) / (max_val - min_val)


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """
    Compute the mean of the tensor along the given dimension, ignoring the masked values.
    :param tensor: The input tensor.
    :param mask: The mask tensor.
    :param dim: The dimension to compute the mean.
    :return: The mean of the tensor along the given dimension, ignoring the masked values.
    """
    masked_tensor = tensor.clone()
    masked_tensor = masked_tensor.masked_fill(~mask.to(torch.bool), 0)
    return masked_tensor.sum(dim=dim, keepdim=keepdim) / (mask.to(torch.bool)).sum(dim=dim, keepdim=keepdim)


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
    has_mask: bool = False,
    min_max_normalize: bool = False,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state_mask(state):
        # return (
        #     state - state_mean
        # ) / state_std  # epsilon should be already added in std.
        return {
            "real_obs": (state["real_obs"] - state_mean) / state_std,
            "action_mask": state["action_mask"],
        }

    def normalize_state(state):
        state["fea"] = (state["fea"] - state_mean) / state_std
        return state

    def min_max_normalize_state(state):
        state["fea"] = (state["fea"] - state_mean) / (state_std - state_mean)
        return state

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward
    if has_mask:
        env = gym.wrappers.TransformObservation(env, normalize_state_mask)
    else:
        if min_max_normalize:
            env = gym.wrappers.TransformObservation(env, min_max_normalize_state)
        else:
            env = gym.wrappers.TransformObservation(env, normalize_state)

    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env

def wrap_env_l2d(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
    min_max_normalize: bool = False,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):

        state["fea"] = (state["fea"] - state_mean) / state_std

        return state

    def min_max_normalize_state(state):
        state["fea"] = (state["fea"] - state_mean) / (state_std - state_mean)
        return state


    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward
    if min_max_normalize:
        env = gym.wrappers.TransformObservation(env, min_max_normalize_state)
    else:
        env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def set_env_seed(env: Optional[gym.Env], seed: int):
    # env.seed(seed)
    env.action_space.seed(seed)


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        set_env_seed(env, seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def is_goal_reached(reward: float, info: Dict) -> bool:
    if "goal_achieved" in info:
        return info["goal_achieved"]
    return reward > 0  # Assuming that reaching target is a positive reward


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int, attributes: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    for _ in range(n_episodes):

        (state, _), done = env.reset(seed=seed), False
        episode_reward = 0.0
        goal_achieved = False
        while not done:
            curr_step += 1
            if hasattr(env, "action_masks"):
                action = actor.act(state["real_obs"], state["action_mask"], device)
            else:
                action = actor.act(state, np.ones(env.action_space.n), device)
                action = action.item()
                # print("The action is: ", action)
            if action == env.action_space.n - 1:
                num_no_ops += 1
            # print("The action is: ", action)
            # print("the action mask is: ", state["action_mask"])
            # print("the current step is: ", curr_step)
            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal

        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward)
        if attributes:
            for att in attributes:

                att_scores_list[att].append(getattr(env, att, None))
        # make_spans.append(env.last_time_step)

    actor.train()
    att_scores = {}
    if attributes:
        for att in attributes:
            att_scores[att] = np.asarray(att_scores_list[att])
    # print("The number of no ops is: ", num_no_ops)
    return np.asarray(episode_rewards), np.mean(successes), att_scores


def write_results_checkpoint(
    results: Dict, checkpoint_dir: str, checkpoint_name: str
) -> None:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        df = pd.DataFrame(results, index=["epoch"])
    else:
        df = pd.read_csv(checkpoint_path)
        df_new = pd.DataFrame(results, index=["epoch"])
        # df = df.append(results, ignore_index=True)
        df = pd.concat([df, df_new], axis=0)
    df.to_csv(checkpoint_path, index=False)


@torch.no_grad()
def eval_actor_l2d_original_dqn(
    env: gym.Env, actor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": eval_instances[ins]}), False

        episode_reward = -env.initQuality
        start_time = time.time()
        goal_achieved = False
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)
            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)
            c_action = actor.act(features=fea, adj=adj, candidate=omega, graph_pool=g_pool, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(int(np.abs(episode_reward - env.posRewards)))


    actor.train()

    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times


@torch.no_grad()
def eval_actor_l2d_original(
    env: gym.Env, actor: nn.Module, feature_extractor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    feature_extractor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": eval_instances[ins]}), False

        episode_reward = -env.initQuality
        start_time = time.time()
        goal_achieved = False
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)
            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)
            x, _ = feature_extractor(features=fea, adj=adj, candidate=omega, graph_pool=g_pool)
            c_action = actor.act(x, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(int(np.abs(episode_reward - env.posRewards)))


    actor.train()
    feature_extractor.train()

    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times

@torch.no_grad()
def eval_actor_l2d(
    env: gym.Env, actor: nn.Module, feature_extractor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    feature_extractor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": readArrFunc(eval_instances[ins])}), False

        episode_reward = -env.initQuality
        start_time = time.time()
        goal_achieved = False
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)

            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)
            x = feature_extractor(features=fea, adj=adj, candidate=omega, graph_pool=g_pool)
            c_action = actor.act(x, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(env.make_span)


    actor.train()
    feature_extractor.train()

    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times


@torch.no_grad()
def eval_actor_l2d_only_actor(
    env: gym.Env, actor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": readArrFunc(eval_instances[ins])}), False

        episode_reward = -env.initQuality
        start_time = time.time()
        goal_achieved = False
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)
            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)

            input_actor = (fea, adj, omega, g_pool)
            # x = feature_extractor(features=fea, adj=adj, candidate=omega, graph_pool=g_pool)
            c_action = actor.act(input_actor, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(env.make_span)


    actor.train()

    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times



@torch.no_grad()
def eval_actor_l2d_other(
        env: gym.Env, actor: nn.Module, feature_extractor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    feature_extractor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": readArrFunc(eval_instances[ins])}), False

        episode_reward = -env.initQuality
        goal_achieved = False
        start_time = time.time()
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)
            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)
            x, _ = feature_extractor(features=fea, adj=adj, candidate=omega, graph_pool=g_pool)
            c_action = actor.act(x, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(env.make_span)


        # make_spans.append(env.last_time_step)

    actor.train()
    feature_extractor.train()

    # print("The number of no ops is: ", num_no_ops)
    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times

@torch.no_grad()
def eval_actor_l2d_other_original(
    env: gym.Env, actor: nn.Module, feature_extractor: nn.Module, g_pool, readArrFunc: Callable, device: str,
        eval_instances: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # env.seed(seed)
    actor.eval()
    feature_extractor.eval()
    episode_rewards = []
    make_spans = []
    successes = []
    num_no_ops = 0
    curr_step = 0
    att_scores_list = defaultdict(list)
    num_instances = eval_instances.shape[0]
    run_times = np.zeros(num_instances)
    for ins in range(num_instances):
        # print("Checking instance: ", ins + 1)

        (state, _), done = env.reset(options={"JSM_env": eval_instances[ins]}), False

        episode_reward = -env.initQuality
        start_time = time.time()
        goal_achieved = False
        while not done:
            curr_step += 1
            adj = state["adj"]
            fea = state["fea"]
            omega = state["omega"]
            mask = ~state["mask"]
            fea = torch.as_tensor(fea, dtype=torch.float, device=device)
            adj = torch.as_tensor(adj, dtype=torch.float, device=device)
            omega = torch.as_tensor(omega, dtype=torch.long, device=device).unsqueeze(0)
            x, _ = feature_extractor(features=fea, adj=adj, candidate=omega, graph_pool=g_pool)
            c_action = actor.act(x, action_mask=mask, deterministic=deterministic, device=device)
            action = state["omega"][c_action]

            state, reward, truncated, terminated, env_infos = env.step(action)
            done = truncated or terminated
            episode_reward += reward
            if not goal_achieved:
                goal_achieved = is_goal_reached(reward, env_infos)
        # Valid only for environments with goal
        run_times[ins] = time.time() - start_time
        successes.append(float(goal_achieved))
        episode_rewards.append(episode_reward - env.posRewards)
        make_spans.append(int(np.abs(episode_reward - env.posRewards)))


    actor.train()
    feature_extractor.train()

    return np.asarray(episode_rewards), np.mean(successes), np.asarray(make_spans), run_times


def return_reward_range(dataset: Dataset, max_episode_steps: int) -> Tuple[float, float]:
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset.rewards, dataset.dones):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset.rewards), "Lengths do not match"
    return min(returns), max(returns)


def modify_reward(dataset: Dataset, max_episode_steps: int = 1000) -> Dict:
    # if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
    min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
    dataset.rewards /= max_ret - min_ret
    dataset.rewards *= max_episode_steps
    return {
        "max_ret": max_ret,
        "min_ret": min_ret,
        "max_episode_steps": max_episode_steps,
    }



def modify_reward_online(reward: float, **kwargs) -> float:

    reward /= kwargs["max_ret"] - kwargs["min_ret"]
    reward *= kwargs["max_episode_steps"]

    return reward


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.action_masks()

def get_env(instance):
    instance_path = f"./JSSEnv/envs/instances/{instance}"
    env = gym.make("jss-v1", env_config={"instance_path": instance_path})
    # env = ActionMasker(env, mask_fn)
    _ = env.reset()
    # check_env(env)
    # env = Monitor(env)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


# def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
#     cumsum = np.zeros_like(x)
#     cumsum[-1] = x[-1]
#     for t in reversed(range(x.shape[0] - 1)):
#         cumsum[t] = x[t] + gamma * cumsum[t + 1]
#     return cumsum


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

TensorBatch = List[torch.Tensor]
import sys
from pathlib import Path

# import gym
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete, Dict, MultiBinary, MultiDiscrete


from ..helper_functions import load_parameters
from .permissibleLS import permissibleLeftShift
from .uniform_instance_gen import override
from .updateAdjMat import getActionNbghs
from .updateEntTimeLB import calEndTimeLB
from .uniform_instance_gen import uni_instance_gen

base_path = Path(__file__).resolve().parents[2]
sys.path.append(str(base_path))

param_file = str(base_path) + "/configs/L2D.toml"
parameters = load_parameters(param_file)
env_parameters = parameters["env_parameters"]
model_parameters = parameters["network_parameters"]
train_parameters = parameters["train_parameters"]
test_parameters = parameters["test_parameters"]


class SJSSP(gym.Env):
    def __init__(self,
                 n_j,
                 n_m):

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        self.action_space = Discrete(self.number_of_tasks)
        self.prev_reward = 0
        self.norm_factor = 1

        # self.observation_space = Dict({
        #     "adj": MultiBinary([self.number_of_tasks, self.number_of_tasks]),
        #     "fea": Box(low=-np.inf, high=np.inf, shape=(self.number_of_tasks, 2), dtype=np.single),
        #     "omega": Box(low=0, high=self.number_of_tasks, shape=(self.number_of_jobs,), dtype=np.int64),
        #     "mask": Box(low=0, high=1, shape=(self.number_of_jobs,), dtype=bool)
        # })

        self.observation_space = Dict({
            "adj": Box(low=0, high=1, shape=(self.number_of_tasks, self.number_of_tasks), dtype=bool),
            "fea": Box(low=-np.inf, high=np.inf, shape=(self.number_of_tasks, 2), dtype=np.single),
            "omega": Box(low=0, high=self.number_of_tasks, shape=(self.number_of_jobs,), dtype=np.int64),
            "mask": Box(low=0, high=1, shape=(self.number_of_jobs,), dtype=bool)
        })

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0


        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1) / env_parameters["et_normalize_coef"],
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = env_parameters["rewardscale"]
            self.posRewards += reward

        self.max_endTime = self.LBs.max()
        adj = np.copy(self.adj).astype(bool)
        mask = np.copy(self.mask)
        omega = np.copy(self.omega)
        reward /= self.norm_factor
        # state = {"adj": self.adj, "fea": fea, "omega": self.omega, "mask": self.mask}
        return {"adj": adj, "fea": fea, "omega": omega, "mask": mask}, reward, self.done(), False, {}


    def reset(self, data=None, options=None, **kwargs):
        super().reset(**kwargs)

        if data is None and options is None:
            data = uni_instance_gen(n_j=self.number_of_jobs, n_m=self.number_of_machines, low=env_parameters['low'],
                                    high=env_parameters["high"])
            raise ValueError("Data is None")

        elif data is None:
            data = options["JSM_env"]

        if options is not None and "norm_factor" in options:
            self.norm_factor = options["norm_factor"]
        else:
            self.norm_factor = 1


        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max()
        self.max_endTime = self.initQuality
        self.initQuality /= self.norm_factor
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1) /env_parameters["et_normalize_coef"],
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -env_parameters["high"] * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        adj = np.copy(self.adj).astype(bool)

        mask = np.copy(self.mask)
        omega = np.copy(self.omega)
        return {"adj": adj, "fea": fea, "omega": omega, "mask": mask}, {}

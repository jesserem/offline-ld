# from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.env_test import NipsJSPEnv_test as SJSSP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.JSSP_Env import SJSSP
import numpy as np
import minari
import json
import os
from collections import defaultdict
import argparse
import gymnasium as gym
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.dispatching_rules.helper_functions import *
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.scheduling_environment.simulationEnv import SimulationEnv
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.solution_methods.L2D.JSSP_Env import SJSSP
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_jsp_fsp import parseArray, parse, parse_to_array, parseAndMake
from gymnasium.spaces import Graph,GraphInstance
from typing import Dict, Union
from Job_Shop_Scheduling_Benchmark_Environments_and_Instances.data_parsers.parser_jsp_fsp import parseAndMake
from minari.serialization import deserialize_space, serialize_space
from minari import StepDataCallback

class RewardModify(gym.Wrapper):
    def __init__(self, env):
        super(RewardModify, self).__init__(env)
        self.reward = 0

    def step(self, action):
        state, reward, termi, trunc, info = self.env.step(action)
        reward = self.reward
        return state, reward, termi, trunc, info

    def set_reward(self, reward):
        self.reward = reward

def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule, noisy_prob=0):
    """use dispatching rules to select the next operation to schedule"""
    operation_priorities = {}
    # print(noisy_prob)
    # exit()

    # Calculate the priority values for the operations based on the dispatching rule and machine assignment rule
    for job in simulationEnv.JobShop.jobs:
        for operation in job.operations:
            if operation not in simulationEnv.processed_operations and operation not in simulationEnv.JobShop.scheduled_operations and machine.machine_id in operation.processing_times:
                if check_precedence_relations(simulationEnv, operation):
                    if dispatching_rule == 'FIFO' and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = operation.job_id

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = get_operations_remaining(simulationEnv, operation)

                    elif dispatching_rule in ['MWR', 'LWR'] and machine_assignment_rule == 'SPT':
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = get_work_remaining(simulationEnv, operation)

                    elif dispatching_rule == 'FIFO' and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = simulationEnv.get_earliest_end_time_machines(operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = operation.job_id

                    elif dispatching_rule in ['MOR', 'LOR'] and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = get_earliest_end_time_machines(simulationEnv, operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = get_operations_remaining(simulationEnv, operation)

                    elif dispatching_rule in ['MWR', 'LWR'] and machine_assignment_rule == 'EET':
                        earliest_end_time_machines = simulationEnv.get_earliest_end_time_machines(operation)
                        if machine.machine_id in earliest_end_time_machines:
                            operation_priorities[operation] = simulationEnv.get_work_remaining(operation)
                    elif dispatching_rule in ['SPT']:
                        min_processing_time = min(operation.processing_times.values())
                        min_keys = [key for key, value in operation.processing_times.items() if
                                    value == min_processing_time]
                        if machine.machine_id in min_keys:
                            operation_priorities[operation] = get_work_remaining(simulationEnv, operation)

    if len(operation_priorities) == 0:
        return None
    else:
        noisy_number = np.random.uniform(0, 1)
        if noisy_number < noisy_prob:
            return np.random.choice(list(operation_priorities.keys()))
        if dispatching_rule == 'FIFO' or dispatching_rule == 'LOR' or dispatching_rule == 'LWR' or dispatching_rule == 'SPT':
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)

def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule, action_list, noisy_prob=0):
    """Schedule operations on the machines based on the priority values"""
    machines_available = [machine for machine in simulationEnv.JobShop.machines if
                          simulationEnv.machine_resources[machine.machine_id].count == 0]
    machines_available.sort(key=lambda m: m.machine_id)

    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule,
                                                 noisy_prob=noisy_prob)
        if operation_to_schedule is not None:
            # print(operation_to_schedule.job_id)
            action_list.append(operation_to_schedule.job_id)
            simulationEnv.JobShop._scheduled_operations.append(operation_to_schedule)
            # Check if all precedence relations are satisfied
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))

def run_simulation(simulationEnv, dispatching_rule, machine_assignment_rule, action_list = [], noisy_prob=0):
    """Schedule simulator and schedule operations with the dispatching rules"""

    if simulationEnv.online_arrivals:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling_environment until all operations are processed
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule, action_list, noisy_prob=noisy_prob)
            yield simulationEnv.simulator.timeout(1)

    else:
        # add machine resources to the environment
        for _ in simulationEnv.JobShop.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling_environment and schedule operations until all operations are processed from the data instance
        while len(simulationEnv.processed_operations) < sum([len(job.operations) for job in simulationEnv.JobShop.jobs]):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule, action_list,
                                noisy_prob=noisy_prob)
            # print("part1", len(simulationEnv.processed_operations))
            # print("part2", sum([len(job.operations) for job in simulationEnv.JobShop.jobs]))
            yield simulationEnv.simulator.timeout(1)


def run_env_instance_random(env, arr, norm_factor=1):
    # options = {"JSM_env": parseAndMake(arr)}
    # print("arr", arr)
    options = {"JSM_env": arr,
               "norm_factor": norm_factor}
    # print(options)
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = -env.initQuality
    # total_reward *= norm_factor
    test = 0
    mask = ~state["mask"]
    done = False
    while not done:
        # print(a)
        # print(mask[a])

        action = np.random.choice(candidate, p=mask/np.sum(mask))
        # print(action)
        # print(action, a, candidate, mask)


        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)
        done = termi or trunc
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        test += reward
        # print(candidate)
    # print("Total reward: ", total_reward)
    # print("test: ", test)
    make_span = total_reward - env.posRewards
    return make_span


def run_env_instance(env, arr, action_list, norm_factor=1):
    # options = {"JSM_env": parseAndMake(arr)}
    # print("arr", arr)
    options = {"JSM_env": arr,
               "norm_factor": norm_factor}
    # print(options)
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = -env.initQuality
    # total_reward *= norm_factor
    test = 0
    mask = ~state["mask"]
    for a in action_list:
        # print(a)
        # print(mask[a])
        if not mask[a]:
            print("ERROR")
            exit()
        action = candidate[a]
        # print(action)
        # print(action, a, candidate, mask)


        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)
        done = termi or trunc
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        test += reward
        # print(candidate)
    # print("Total reward: ", total_reward)
    # print("test: ", test)
    make_span = total_reward - env.posRewards
    return make_span

def run_env_norm(env, arr, action_list, norm_factor=1):
    # options = {"JSM_env": parseAndMake(arr)}
    # print("arr", arr)
    options = {"JSM_env": arr,
               "norm_factor": norm_factor}
    # print(options)
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = 0
    # total_reward *= norm_factor
    test = 0
    mask = ~state["mask"]
    for a in action_list:
        # print(a)
        # print(mask[a])
        if not mask[a]:
            print("ERROR")
            exit()
        action = candidate[a]
        # print(action)
        # print(action, a, candidate, mask)


        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)

        done = termi or trunc
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        test += reward
        # print(candidate)
    # print("Total reward: ", total_reward)
    # print("test: ", test)
    # make_span = total_reward - env.posRewards
    return total_reward


def run_env_instance_noisy_new(env, arr, action_list, norm_factor=1, noisy_prob=0.1):
    # options = {"JSM_env": parseAndMake(arr)}
    # print("arr", arr)
    options = {"JSM_env": arr,
               "norm_factor": norm_factor}
    # print(options)
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = -env.initQuality
    # total_reward *= norm_factor
    test = 0
    mask = ~state["mask"]
    done = False
    i = 0
    # incorrect_n = 0
    while not done:
        a = action_list[i]
        # print(a)
        # print(mask[a])
        random_number = np.random.uniform(0, 1)
        if not mask[a] or random_number < noisy_prob:

            action = np.random.choice(candidate, p=mask/np.sum(mask))
        else:
            action = candidate[a]
        # print(action)
        # print(action, a, candidate, mask)


        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)
        done = termi or trunc
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        test += reward
        i += 1
        # print(candidate)
    # print("Total reward: ", total_reward)
    # print("test: ", test)
    make_span = total_reward - env.posRewards
    print("make_span: ", make_span)
    return make_span

def run_env_instance_noisy(env, arr, action_list, noisy_after, norm_factor=1):
    # options = {"JSM_env": parseAndMake(arr)}
    options = {
        "JSM_env": arr,
        "norm_factor": norm_factor
    }
    # print(options)
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = -env.initQuality
    mask = ~state["mask"]
    i = 0
    done = False
    while i < noisy_after:
        action = candidate[action_list[i]]
        state, reward, trunc, termi, _ = env.step(action)
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        done = termi or trunc
        i += 1
    while not done:
        i += 1
        action = np.random.choice(candidate, p=mask/np.sum(mask))
        # print(action)
        state, reward, trunc, termi, _ = env.step(action)
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
        done = termi or trunc
        # print(done)

    # for a in action_list:
    #     # print(a)
    #     if not mask[a]:
    #         print("ERROR")
    #         exit()
    #     action = candidate[a]
    #     # print(action, a, candidate, mask)
    #
    #
    #     # print("action: ", action)
    #     state, reward, trunc, termi, _ = env.step(action)
    #     total_reward += reward
    #     candidate = state["omega"]
    #     mask = ~state["mask"]
    #     # print(candidate)
    # print(i)

    make_span = total_reward - env.posRewards
    return make_span

def read_json_instance(instance_path):
    with open(instance_path, "r") as f:
        data = json.load(f)
    action_list = data["action_list"]
    make_span = data["objValue"]
    return action_list, make_span

def run_dataset(args):
    dataset = np.load(args.dataset)
    make_span_list = []
    make_span_noisy_list = []
    make_span_env_list = []
    n_instances = dataset.shape[0]
    env_job = dataset[0][0].shape[0]
    env_machine = dataset[0][0].shape[1]
    env = SJSSP(n_j=env_job, n_m=env_machine)
    if args.dataset_name is None:
        args.dataset_name = f"jsspl2d-{env_job}_{env_machine}_CP_SAT_time_limit_3600-v0"
    # env = RewardModify(env)
    env = minari.DataCollector(env, observation_space=env.observation_space, action_space=env.action_space)
    env_norm_factor = SJSSP(n_j=env_job, n_m=env_machine)
    if args.dispatching_rules is not None:
        dispatching_rules = [rule.upper() for rule in args.dispatching_rules]
    dispatch_rules_make_span = defaultdict(list)
    for i in range(0, n_instances):
        path_json = os.path.join(args.folder, "instance_" + str(i), "cp_sat_results.json")
        if not os.path.exists(path_json):
            raise FileNotFoundError("Instance: ", i, " not found")
        action_list, make_span = read_json_instance(path_json)
        print("\nInstance: ", i)
        print("Perfect Make span: ", make_span)
        make_span_list.append(make_span)


        norm_factor = make_span
        if args.norm_env:
            best_make_span = run_env_norm(env_norm_factor, dataset[i], action_list, norm_factor=1)
            norm_factor = abs(best_make_span)
            print(norm_factor)
        elif args.no_norm:
            norm_factor = 1

        # env.reset()
        if not args.no_cp:
            noisy_run_prob = np.random.uniform(0, 1)
            if args.noisy_prob > 0 and noisy_run_prob < 0.5:
                make_span_env = run_env_instance_noisy_new(env, dataset[i], action_list, norm_factor=norm_factor, noisy_prob=args.noisy_prob)
                diff_make_span = np.abs(make_span_env * norm_factor) - make_span
                print(diff_make_span)
                print("Noisy")
            else:
                make_span_env = run_env_instance(env, dataset[i], action_list, norm_factor=norm_factor)
                print("No Noisy")
                print(make_span_env)
            make_span_env_list.append(make_span_env)
        # if np.abs(make_span) != np.abs(int(make_span_env)):
        #     print("Instance: ", i)
        #     print("make span: ", make_span)
        #     print("make span env: ", make_span_env)
        #     exit()
        curr_noisy_runs = []
        for noisy_run in range(args.noisy_runs):
            make_span_env = run_env_instance_noisy_new(env, dataset[i], action_list, norm_factor=norm_factor, noisy_prob=args.noisy_prob)
            # noisy_after = np.random.randint(int(0.25 * len(action_list)), int(0.75 * len(action_list)))
            # # print("Noisy after: ", noisy_after)
            make_span_env = abs(make_span_env * norm_factor)
            print("Noisy run: ", make_span_env)
            # make_span_env = run_env_instance_noisy(env, dataset[i], action_list, noisy_after, norm_factor=norm_factor)
            curr_noisy_runs.append(make_span_env)
            # for i in range(0, len(action_list), 2):
            #     prob = np.random.uniform(0, 1)
            #     if prob < 0.5:
            #         print(action_list[i], action_list[i + 1])
            #         action_list[i], action_list[i + 1] = action_list[i + 1], action_list[i]
            #         print(action_list[i], action_list[i + 1], "\n")

        # exit()
        if args.noisy_runs > 0:
            make_span_noisy_list.append(np.mean(curr_noisy_runs))
        for random_run in range(args.random_runs):
            make_span_random = run_env_instance_random(env, dataset[i], norm_factor=norm_factor)
            print("Random run: ", make_span_random)
        if args.use_dispatch:
            curr_action_list = []

            for dispatching_rule in dispatching_rules:
                make_spans_dispatch = []
                for _ in range(args.n_runs_dispatch):
                    simulationEnv = SimulationEnv(
                        online_arrivals=False,
                    )
                    action_list = []


                    simulationEnv.JobShop = parseArray(simulationEnv.JobShop, dataset[i])


                # exit()

                    simulationEnv.simulator.process(
                        run_simulation(simulationEnv, dispatching_rule, "SPT", action_list, noisy_prob=args.noisy_prob))

                    simulationEnv.simulator.run()
                    already_done = False
                    for prev_action_list in curr_action_list:
                        if prev_action_list == action_list:
                            already_done = True
                            break
                    if already_done:
                        print("Found duplicate")
                        continue

                    make_span_dispatch = simulationEnv.JobShop.makespan


                    make_spans_dispatch.append(make_span_dispatch)
                    print("Make span dispatch: ", make_spans_dispatch)


                    run_env_instance(env, dataset[i], action_list, norm_factor=norm_factor)
                    curr_action_list.append(action_list)
                make_span_dispatch = np.mean(make_spans_dispatch)
                diff_from_cp = np.round((np.abs(make_span_dispatch - make_span) / make_span) * 100, 2)
                print("Dispatching rule:", dispatching_rule, "Make span:", make_span_dispatch, "Diff from CP:",
                      diff_from_cp)
                dispatch_rules_make_span[dispatching_rule].append(make_span_dispatch)
                # print(test_val * make_span)
                # exit()





        # data = read_json_instance(path_json)
        # print(args.folder)
        # exit()




    #
    #
    min_score = np.min(make_span_list)
    max_score = np.max(make_span_list)
    try:
        env.create_dataset(
            dataset_id=args.dataset_name,
            algorithm_name=f"CP-SAT",
            ref_min_score=min_score,
            ref_max_score=max_score,
        )
    except ValueError:
        print("Dataset already exists, updating it")




    print("Make span list: ", np.mean(make_span_list))
    # print("Make span env list: ", np.mean(make_span_env_list))
    if args.noisy_runs > 0:
        make_span_noisy_list = np.array(make_span_noisy_list)
        print("Make span noisy list: ", np.mean(make_span_noisy_list))
    if args.use_dispatch:
        for dispatching_rule in dispatch_rules_make_span:
            mean_make_span_dispatch = np.round(np.mean(dispatch_rules_make_span[dispatching_rule]), 2)
            print("Dispatching rule: ", dispatching_rule, mean_make_span_dispatch)
            # print("Mean diff from CP: ", np.mean(dispatch_rules_make_span[dispatching_rule]))
    print("Min score: ", min_score)
    print("Max score: ", max_score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset file of all the instances")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing the results")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset")
    parser.add_argument("--noisy_runs", type=int, default=0, help="Number of noisy runs")
    parser.add_argument("--random_runs", type=int, default=0, help="Number of random runs")
    parser.add_argument("--use_dispatch", action='store_true', help="Use dispatching rules")
    parser.add_argument("--dispatching_rules", nargs='+', default=["FIFO", "MOR", "LOR", "MWR", "LWR"],
                        help="Dispatching rules")
    parser.add_argument("--no_cp", action='store_true', help="No CP-SAT results")
    parser.add_argument("--no_norm", action='store_true', help="No normalization")
    parser.add_argument("--noisy_prob", type=float, default=0, help="Noisy probability")
    parser.add_argument('--n_runs_dispatch', type=int, default=1, help="Number of runs for dispatching rules")
    parser.add_argument("--norm_reward", action='store_true', help="Normalize reward")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random number generator")
    parser.add_argument("--norm_env", action='store_true', help="Normalize the environment")

    args = parser.parse_args()
    np.random.seed(args.seed)
    run_dataset(args)

if __name__ == "__main__":
    main()

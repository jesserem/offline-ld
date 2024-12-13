import numpy as np
from solution_methods.L2D.env_test import NipsJSPEnv_test as SJSSP
from data_parsers.parser_jsp_fsp import parseArray, parse, parse_to_array, parseAndMake
from solution_methods.cp_sat import JSPmodel
from solution_methods.cp_sat.utils import solve_model
import minari
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE, INFEASIBLE
import argparse
import json
import logging
import os

from plotting.drawer import draw_gantt_chart

logging.basicConfig(level=logging.INFO)
DEFAULT_RESULTS_ROOT = "./results/milp"
PARAM_FILE = "configs/milp.toml"
path_data = "./solution_methods/L2D/generated_data/"

def get_action_list(jobShopEnv):
    action_arr = []
    for job in jobShopEnv.jobs:
        for operation in job.operations:
            action_arr.append((job.job_id, operation.operation_id, operation.scheduled_start_time))

    action_arr.sort(key=lambda x: x[2])
    action_list = [x[0] for x in action_arr]
    return action_list
    # print(action_arr)

def run_method(problem_instance, exp_name, folder, time_limit=30, save_results=False):
    """
    Solve the scheduling problem for the provided input file.
    """

    jobShopEnv = parseAndMake(problem_instance)
    model, vars = JSPmodel.jsp_cp_sat_model(jobShopEnv)

    solver, status, solution_count = solve_model(model, time_limit)


    jobShopEnv, results = JSPmodel.update_env(jobShopEnv, vars, solver, status, solution_count, time_limit)

    # Plot the ganttchart of the solution
    if status == FEASIBLE:
        solution_status = "Feasible"
    elif status == OPTIMAL:
        solution_status = "Optimal"
    else:
        solution_status = "Infeasible"




    action_list = get_action_list(jobShopEnv)
    makeSpan = int(jobShopEnv.makespan)

    # Plot ganttchart
    # if True:
    #     draw_gantt_chart(jobShopEnv)

    # Ensure the directory exists; create if not
    if save_results:
        dir_path = os.path.join(folder, exp_name)
        results_dict = {
            "time_limit": int(time_limit),
            # "status": status,
            "statusString": solution_status,
            "objValue": int(jobShopEnv.makespan),
            "action_list": action_list,
        }
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Specify the full path for the file
        file_path = os.path.join(dir_path, "cp_sat_results.json")
        #
        # Save results to JSON (will create or overwrite the file)
        # res_file =
        with open(file_path, "w") as outfile:
            json.dump(results_dict, outfile, indent=4)
    return action_list, makeSpan


def run_env_instance(env, arr, action_list):
    options = {"JSM_env": parseAndMake(arr)}
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    total_reward = - env.initQuality
    for a in action_list:
        action = candidate[a]

        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)
        total_reward += reward
        candidate = state["omega"]
        # print(candidate)

    make_span = total_reward - env.posRewards
    return make_span

def run_dataset(dataset):
    make_span_list = []
    n_instances = dataset.shape[0]
    env_job = dataset[0][0].shape[0]
    env_machine = dataset[0][0].shape[1]
    env = SJSSP(n_j=env_job, n_m=env_machine)
    env = minari.DataCollector(env, observation_space=env.observation_space, action_space=env.action_space)
    time_span = 60
    save_results_folder = f"./results/cp_sat/problem_{env_job}_{env_machine}/train/time_limit_" + str(time_span) + "/"
    make_span_env_list = []
    for i in range(0, n_instances):
        # print("Instance: ", i)
        # instance_path = "/jsp/taillard/ta01"
        # arr = parse_to_array(instance_path)
        # exit()
        # action_list = []
        name = "instance_" + str(i)
        action_list, make_span = run_method(dataset[i], name, save_results_folder, time_limit=time_span, save_results=True)



        make_span_list.append(make_span)
        make_span_env = run_env_instance(env, dataset[i], action_list)
        make_span_env_list.append(make_span_env)
        if np.abs(make_span) != np.abs(int(make_span_env)):
            print("Instance: ", i)
            exit()
        # print("makespan: ", make_span)
        # print("action list: ", action_list)
        # print("makespan env: ", make_span_env)
        # exit()

    min_score = np.min(make_span_list)
    max_score = np.max(make_span_list)
    # env.create_dataset(
    #     dataset_id=f"jsspl2d-{env_job}_{env_machine}_CAP_ST_TimeLimit_{time_span}-v0",
    #     algorithm_name=f"L2D-MILP",
    #     ref_min_score=min_score,
    #     ref_max_score=max_score,
    # )
    print(make_span_list)
    print(make_span_env_list)
    print("mean make span: ", np.mean(make_span_list))
    print("mean make span env: ", np.mean(make_span_env_list))


    print("mean make span: ", np.mean(make_span_list))
    print("SUCCES!!!")
dataLoaded = np.load(path_data+'generatedData' + str(6) + '_' + str(6) + '_Seed' + str(200) + '.npy')

run_dataset(dataLoaded)

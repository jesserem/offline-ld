import numpy as np
from data_parsers.parser_jsp_fsp import parseAndMake
from solution_methods.cp_sat import JSPmodel
from solution_methods.cp_sat.utils import solve_model
from ortools.sat.python.cp_model import OPTIMAL, FEASIBLE, INFEASIBLE
import argparse
import json
import logging
import time
import os

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
    start_time = time.time()
    solver, status, solution_count = solve_model(model, time_limit)
    runtime = time.time() - start_time


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
            "runtime": runtime,
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


def run_dataset(dataset, instance_id, time_span=600, save_folder="./results/cp_sat", train_folder=False):

    instance = dataset[instance_id]
    env_job = instance[0].shape[0]
    env_machine = instance[0].shape[1]
    if train_folder:
        folder_f = "train"
    else:
        folder_f = "test"


    save_results_folder = f"{save_folder}/folder_f/setup_{env_job}_{env_machine}/time_limit_" + str(time_span) + "/"

        # print("Instance: ", i)
        # instance_path = "/jsp/taillard/ta01"
        # arr = parse_to_array(instance_path)
        # exit()
        # action_list = []
    name = "instance_" + str(instance_id)
    action_list, make_span = run_method(instance, name, save_results_folder, time_limit=time_span, save_results=True)

def main():
    parser = argparse.ArgumentParser(description='Run CP_SAT on a single instance')
    parser.add_argument('--instance_files', type=str, help='Path to the instance file', required=True)
    parser.add_argument('--id', type=int, help='Instance ID', required=True)
    parser.add_argument('--time_limit', type=int, default=600, help='Time limit for the MILP solver')
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument('--save_folder', type=str, default="./results/cp_sat", help='Folder to save the results')

    args = parser.parse_args()
    dataLoaded = np.load(args.instance_files)
    run_dataset(dataLoaded, args.id, args.time_limit, args.save_folder)

def run_all():
    instance_file = "./solution_methods/L2D/large_generated_data/generatedData6_6_Seed200.npy"
    dataLoaded = np.load(instance_file)
    time_limit = 3600
    save_folder = "../results_large_cp_sat"
    for i in range(dataLoaded.shape[0]):
        run_dataset(dataLoaded, i, time_limit, save_folder)
# run_dataset(dataLoaded)
main()
# run_all()
import numpy as np
from solution_methods.dispatching_rules.helper_functions import *
from scheduling_environment.simulationEnv import SimulationEnv
from solution_methods.L2D.env_test import NipsJSPEnv_test as SJSSP
from data_parsers.parser_jsp_fsp import parseArray, parse, parse_to_array, parseAndMake
import minari

def select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule):
    """use dispatching rules to select the next operation to schedule"""
    operation_priorities = {}

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
        if dispatching_rule == 'FIFO' or dispatching_rule == 'LOR' or dispatching_rule == 'LWR' or dispatching_rule == 'SPT':
            return min(operation_priorities, key=operation_priorities.get)
        else:
            return max(operation_priorities, key=operation_priorities.get)

def schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule, action_list):
    """Schedule operations on the machines based on the priority values"""
    machines_available = [machine for machine in simulationEnv.JobShop.machines if
                          simulationEnv.machine_resources[machine.machine_id].count == 0]
    machines_available.sort(key=lambda m: m.machine_id)

    for machine in machines_available:
        operation_to_schedule = select_operation(simulationEnv, machine, dispatching_rule, machine_assignment_rule)
        if operation_to_schedule is not None:
            # print(operation_to_schedule.job_id)
            action_list.append(operation_to_schedule.job_id)
            simulationEnv.JobShop._scheduled_operations.append(operation_to_schedule)
            # Check if all precedence relations are satisfied
            simulationEnv.simulator.process(simulationEnv.perform_operation(operation_to_schedule, machine))

def run_simulation(simulationEnv, dispatching_rule, machine_assignment_rule, action_list = []):
    """Schedule simulator and schedule operations with the dispatching rules"""

    if simulationEnv.online_arrivals:
        # Start the online job generation process
        simulationEnv.simulator.process(simulationEnv.generate_online_job_arrivals())

        # Run the scheduling_environment until all operations are processed
        while True:
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule)
            yield simulationEnv.simulator.timeout(1)

    else:
        # add machine resources to the environment
        for _ in simulationEnv.JobShop.machines:
            simulationEnv.add_machine_resources()

        # Run the scheduling_environment and schedule operations until all operations are processed from the data instance
        while len(simulationEnv.processed_operations) < sum([len(job.operations) for job in simulationEnv.JobShop.jobs]):
            schedule_operations(simulationEnv, dispatching_rule, machine_assignment_rule, action_list)
            # print("part1", len(simulationEnv.processed_operations))
            # print("part2", sum([len(job.operations) for job in simulationEnv.JobShop.jobs]))
            yield simulationEnv.simulator.timeout(1)

path_data = "./solution_methods/L2D/test_generated_data/"



def run_normal_instance():
    instance_path = "/jsp/taillard/ta01"
    arr = parse_to_array(instance_path)
    simulationEnv = SimulationEnv(
        online_arrivals=False,
    )
    simulationEnv.JobShop = parseArray(simulationEnv.JobShop, arr)
    simulationEnv.simulator.process(run_simulation(simulationEnv, "FIFO", "SPT"))
    simulationEnv.simulator.run()
    make_span_copy = simulationEnv.JobShop.makespan
    # print(arr[0])
    # print(arr[1])
    # return
    # print(arr[0])
    # print(arr[1])
    # exit()
    simulationEnv = SimulationEnv(
        online_arrivals=False,
    )
    action_list = []
    simulationEnv.JobShop = parse(simulationEnv.JobShop, instance_path)
    simulationEnv.simulator.process(run_simulation(simulationEnv, "MO", "SPT", action_list))
    simulationEnv.simulator.run()


    make_span_orginal = simulationEnv.JobShop.makespan



    print("make span: ", make_span_orginal)
    print("make span copy: ", make_span_copy)


def run_env_instance(env, arr, action_list):
    options = {"JSM_env": parseAndMake(arr)}
    state,_ = env.reset(options=options)
    # state = env.state
    candidate = state["omega"]
    mask = ~state["mask"]
    total_reward = - env.initQuality
    for a in action_list:
        if not mask[a]:
            print("mask: ", mask)
            print("action: ", a)

        action = candidate[a]

        # print("action: ", action)
        state, reward, trunc, termi, _ = env.step(action)

        # print("state: ", state["fea"])
        # print("reward: ", reward)
        total_reward += reward
        candidate = state["omega"]
        mask = ~state["mask"]
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
    dispatching_rule = "MWR"
    machine_assignment_rule = "SPT"
    for i in range(0, n_instances):

        simulationEnv = SimulationEnv(
            online_arrivals=False,
        )
        # instance_path = "/jsp/taillard/ta01"
        # arr = parse_to_array(instance_path)
        # exit()
        action_list = []
        simulationEnv.JobShop = parseArray(simulationEnv.JobShop, dataset[i])

        # exit()


        simulationEnv.simulator.process(run_simulation(simulationEnv, dispatching_rule, machine_assignment_rule, action_list))


        simulationEnv.simulator.run()


        make_span = simulationEnv.JobShop.makespan
        make_span_list.append(make_span)
        # print("Instance: ", i, "make span: ", make_span)
        make_span_env = run_env_instance(env, dataset[i], action_list)
        # exit()
        # print("make span env: ", make_span_env)
        # print("make span: ", make_span)
    min_score = np.min(make_span_list)
    max_score = np.max(make_span_list)
    env.create_dataset(
        dataset_id=f"jsspl2d-TEST_{env_job}_{env_machine}_{dispatching_rule}_{machine_assignment_rule}-v0",
        algorithm_name=f"L2D-{dispatching_rule}-{machine_assignment_rule}",
        ref_min_score=min_score,
        ref_max_score=max_score,
    )

    print("mean make span: ", np.mean(make_span_list))
dataLoaded = np.load(path_data+'generatedData' + str(6) + '_' + str(6) + '_Seed' + str(300) + '.npy')
# print(dataLoaded[0][0])
# print(dataLoaded[0][1])
# run_normal_instance()

run_dataset(dataLoaded)
# print(dataLoaded.shape)
# print(dataLoaded[0])
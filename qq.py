#!/usr/bin/env python3
import argparse
import csv
import logging

import matplotlib
from qiskit.converters import dag_to_circuit

from qq.constraints import refine_constrain_swaps_added, count_swaps_added
from qq.types import ModelVariables

matplotlib.use('Agg')

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.transpiler.coupling import CouplingMap
from z3 import *

import qq.model
import qq.util

logger = logging.getLogger(__name__)

def parse_arguments():
    argparser = argparse.ArgumentParser()
    # Program input arguments:
    argparser.add_argument('-i', '--input-circuit', required=True, dest='input_circuit', type=str,
                           help='The filename of the input circuit to be optimized. Should be in OpenQASM 2.0 format.')
    argparser.add_argument('-c', '--coupling-graph', required=True, dest='coupling_graph', type=str,
                           help='The filename of the coupling graph for the target device. Should be a CSV file with '
                                'each row an edge in the graph.')

    # Logging arguments:
    levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    argparser.add_argument('-l', '--log-level', dest='log_level', default='INFO', choices=levels, type=str,
                           help='Log level of output to console.')
    argparser.add_argument('-lf', '--log-file', dest='log_file', nargs='?', type=str)

    argparser.add_argument('-o', '--output-circuit', required=False, dest='output_circuit', type=str,
                           help='The filename of the output circuit.')

    argparser.add_argument('-n', '--num-solver-qubits', required=False, default=0, dest='num_solver_qubits', type=int,
                           help='The number of solver qubits to allocate')


    # Parsing arguments.
    args = argparser.parse_args()
    return args


# Configure logging facilities.
# We log all records to a file if specified.
# A log level can be provided for filtered output to the user.
def config_logging(log_level: str, log_file):
    qq_logger = logging.getLogger('qq')
    qq_logger.setLevel(logging.DEBUG)

    user_stream_log_handler = logging.StreamHandler()
    user_stream_log_handler.setLevel(log_level)
    user_stream_log_formatter = logging.Formatter("%(relativeCreated)d:%(levelname)s:%(name)s: %(message)s")
    user_stream_log_handler.setFormatter(user_stream_log_formatter)
    qq_logger.addHandler(user_stream_log_handler)

    if log_file is not None:
        file_log_handler = logging.FileHandler(log_file, mode='w')
        file_log_handler.setLevel(logging.DEBUG)
        file_log_formatter = logging.Formatter("%(asctime)s|%(levelname)s|%(name)s| %(message)s")
        file_log_handler.setFormatter(file_log_formatter)
        qq_logger.addHandler(file_log_handler)
    return qq_logger


def load_input_circuit(circuit_file_path):
    circuit = QuantumCircuit.from_qasm_file(circuit_file_path)
    return circuit


def load_coupling_graph(coupling_graph_file_path):
    coupling_graph = CouplingMap()
    with open(coupling_graph_file_path, newline='') as coupling_graph_file:
        csv_reader = csv.reader(coupling_graph_file)
        for source_qubit, destination_qubit in csv_reader:
            coupling_graph.add_edge(int(source_qubit), int(destination_qubit))
    return coupling_graph


def improve_criteria(criteria_name: str, min_value: int, max_value: int, model_variables: ModelVariables, solver: Solver):
    criteria_var = Int(criteria_name)

    init_check = solver.check()
    if init_check != sat:
        return None

    best_model = solver.model()
    best_criteria_val = max_value

    # Contract range to find a minimum:
    l = min_value
    r = max_value
    while l <= r:
        m = math.floor((l+r)/2)
        solver.push()
        if criteria_name == "swaps_added":
            solver.add(refine_constrain_swaps_added(m, model_variables))
        solver.add(criteria_var <= m)
        is_sat = solver.check()
        print("BS:", criteria_name, l, r, m, is_sat)

        if is_sat == sat:
            best_model = solver.model()
            if criteria_name == "swaps_added":
                best_criteria_val = count_swaps_added(model_variables, best_model)
            else:
                best_criteria_val = best_model[criteria_var].as_long()
            r = best_criteria_val - 1
        elif is_sat == unsat:
            solver.pop()
            l = m + 1
        else:
            logger.error("Solver in unknown state: %s", solver.reason_unknown())
    return best_model, best_criteria_val


def optimize_circuit(input_circuit, coupling_graph, num_solver_qubits, max_circuit_time, max_swaps_addable):
    solver, model_input, model_variables = qq.model.construct_model(input_circuit, coupling_graph, num_solver_qubits,
                                                   max_circuit_time=max_circuit_time,
                                                   max_swaps_addable=max_swaps_addable)

    best_model, best_circuit_end_time = improve_criteria("circuit_end_time", 0, max_circuit_time, model_variables, solver)
    best_model, best_swaps_added = improve_criteria("swaps_added", 0, max_swaps_addable, model_variables, solver)

    print("Best CET:", best_circuit_end_time, best_model[model_variables["circuit_end_time"]])
    print("Best SA:", best_swaps_added, best_model[model_variables["swaps_added"]])

    stats = solver.statistics()
    decisions = stats.get_key_value('sat decisions')
    with open('./experiment_info.dat', 'a') as experiment_info_file:
        experiment_info_file.write("SAT_Decisions: {}\n".format(decisions))

    opt_dag, init_layout = qq.model.construct_circuit_from_model(best_model, model_input)
    opt_circ = dag_to_circuit(opt_dag)
    return opt_circ


# Primary entry point for QQ.
def qq_main():
    global logger

    # Parse arguments
    args = parse_arguments()
    if args.log_level in ("DEBUG"):
        enable_trace('qq')
        # enable_trace('qq_sat_assign')

    logger = config_logging(args.log_level, args.log_file)
    logger.info("QQ: Self-Hosted Quantum Program Compiler.")

    input_circuit = load_input_circuit(args.input_circuit)
    logger.info("Loaded input circuit: {}".format(args.input_circuit))

    coupling_graph = load_coupling_graph(args.coupling_graph)
    logger.info("Loaded coupling graph: {}".format(args.coupling_graph))

    max_circuit_time = len(input_circuit)
    undirected_couplings = [coupling for coupling in qq.util.get_undirected_couplings(coupling_graph)]
    num_undirected_couplings = len(input_circuit)
    max_swaps_addable = len(input_circuit)

    with open('./experiment_info.dat', 'w') as _:
        pass

    best_result_circuit = optimize_circuit(input_circuit, coupling_graph, args.num_solver_qubits, max_circuit_time, max_swaps_addable)

    result_qasm = best_result_circuit.qasm()
    if args.output_circuit is not None:
        with open(args.output_circuit, 'w') as output_circuit_file:
            output_circuit_file.write(result_qasm + "\n")
    else:
        logger.info("Resulting program:\n%s", result_qasm)

    with open('./experiment_info.dat', 'a') as experiment_info_file:
        experiment_info_file.write("Opt_Depth: {}\n".format(best_result_circuit.depth()))
        experiment_info_file.write("Opt_Ops: {}\n".format(len(best_result_circuit)))

    return

    # for circuit_time in range(max_circuit_time, 0, -1):
    #     print("Circuit time:", circuit_time)
    #
    #     if result_circuit is None:
    #         best_circuit_time = max_circuit_time + 1
    #
    #         break
    #
    # for swaps_addable in range(max_swaps_addable, 0, -1):
    #     print("Swaps addable:", swaps_addable)
    #     result_circuit = qq.model.construct_model(input_circuit, coupling_graph, args.num_solver_qubits,
    #                                               max_circuit_time=best_circuit_time, max_swaps_addable=swaps_addable)
    #     if result_circuit is None:
    #         best_swaps_addable = best_swaps_addable + 1
    #         break
    #     best_result_circuit = result_circuit


if __name__ == "__main__":
    qq_main()

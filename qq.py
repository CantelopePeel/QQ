#!/usr/bin/env python3
import argparse
import csv
import logging

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

import qq.model


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


# Primary entry point for QQ.
def qq_main():
    # Parse arguments
    args = parse_arguments()

    logger = config_logging(args.log_level, args.log_file)
    logger.info("QQ: Self-Hosted Quantum Program Compiler.")

    input_circuit = load_input_circuit(args.input_circuit)
    logger.info("Loaded input circuit: {}".format(args.input_circuit))

    coupling_graph = load_coupling_graph(args.coupling_graph)
    logger.info("Loaded coupling graph: {}".format(args.coupling_graph))

    qq.model.construct_model(input_circuit, coupling_graph)


if __name__ == "__main__":
    qq_main()

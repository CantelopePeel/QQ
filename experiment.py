import csv
import math
import subprocess
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

from qiskit import transpile, QuantumCircuit
from qiskit.transpiler import CouplingMap

NUM_CIRCUIT_QUBITS = list(range(2, 11, 3))
NUM_GATES = list(range(1, 11, 3))
PROPORTION_COUPLINGS = list(i / 10 for i in range(1, 11, 3))
NUM_SOLVER_QUBITS = list(range(0, 11, 3))

NUM_TRIALS = 1

DATA_DIR = 'experiment_data'

import random

random.seed(0)


def gen_circuit(num_gates, num_qubits):
    prog_str = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[{}];
""".format(num_qubits)

    for i in range(num_gates):
        if random.randint(0, 1) == 0:
            qubit0 = random.randrange(0, num_qubits)
            prog_str += "x q[{}];\n".format(qubit0)
        else:
            qubit0 = random.randrange(0, num_qubits)
            qubit1 = random.randrange(0, num_qubits)
            while qubit1 == qubit0:
                qubit1 = random.randrange(0, num_qubits)
            prog_str += "cx q[{}],q[{}];\n".format(qubit0, qubit1)
    return prog_str


def gen_coupling_graph(num_qubits, proportion):
    couplings = set()

    for i in range(num_qubits - 1):
        couplings.add((i, i+1))
        couplings.add((i+1, i))

    num_couplings = int(math.ceil(proportion * (num_qubits * (num_qubits - 1)) / 2))
    if len(couplings) < num_couplings:
        for i in range(num_couplings):
            qubit0 = random.randrange(0, num_qubits)
            qubit1 = random.randrange(0, num_qubits)
            while qubit1 == qubit0:
                qubit1 = random.randrange(0, num_qubits)

            while (qubit0, qubit1) in couplings or (qubit1, qubit0) in couplings:
                qubit0 = random.randrange(0, num_qubits)
                qubit1 = random.randrange(0, num_qubits)
                while qubit1 == qubit0:
                    qubit1 = random.randrange(0, num_qubits)

            couplings.add((qubit0, qubit1))
            couplings.add((qubit1, qubit0))

    couplings_str = ""
    for qubit0, qubit1 in couplings:
        couplings_str += "{},{}\n".format(qubit0, qubit1)
    return couplings_str


def gen_experiments():
    pass


def get_experiment_data():
    experiment_count_map = defaultdict(int)
    with open("experiment_info.dat", "r") as expr_file:
        for line in expr_file:
            key, value = line.split(': ')
            experiment_count_map[key] += int(value)
    return experiment_count_map


def load_coupling_graph(coupling_graph_file_path, num_qubits):
    coupling_graph = CouplingMap()
    for i in range(num_qubits):
        coupling_graph.add_physical_qubit(i)

    with open(coupling_graph_file_path, newline='') as coupling_graph_file:
        csv_reader = csv.reader(coupling_graph_file)
        for source_qubit, destination_qubit in csv_reader:
            coupling_graph.add_edge(int(source_qubit), int(destination_qubit))
    return coupling_graph


def experiments():
    performance_csv_file = open("./experiment_performance.csv", "w")
    comparative_csv_file = open("./experiment_comparative.csv", "w")
    performance_csv_file.write("num_circuit_qubits,num_gates,prop_couplings,trial,num_solver_qubits,"
                               "sat_decisions,grover_iterations\n")
    comparative_csv_file.write("num_circuit_qubits,num_gates,prop_couplings,trial,"
                               "input_depth,qq_depth,opt0_depth,opt1_depth,opt2_depth,opt3_depth,"
                               "input_ops,qq_ops,opt0_ops,opt1_ops,opt2_ops,opt3_ops\n")
    num_experiments = len(NUM_GATES) * len(NUM_CIRCUIT_QUBITS) * len(NUM_SOLVER_QUBITS) * len(PROPORTION_COUPLINGS) * NUM_TRIALS
    print("Num experiments:", num_experiments)
    experiment_counter = 0
    for num_circuit_qubits in NUM_CIRCUIT_QUBITS:
        for num_gates in NUM_GATES:
            for proportion_couplings in PROPORTION_COUPLINGS:
                for trial in range(NUM_TRIALS):
                    for num_solver_qubits in NUM_SOLVER_QUBITS:
                        print("Experiment ({}/{}) | NCQ: {} | NG: {} | PC: {} | NSQ: {} | T: {} |".format(
                            experiment_counter, num_experiments,
                            num_circuit_qubits, num_gates, proportion_couplings, num_solver_qubits, trial))
                        experiment_counter += 1
                        with open("./experiment_prog.qprog", "w") as prog_file:
                            prog_file.write(gen_circuit(num_gates, num_circuit_qubits))
                        with open("./experiment_graph.csv", "w") as coupling_file:
                            coupling_file.write(gen_coupling_graph(num_circuit_qubits, proportion_couplings))

                        coupling_graph = load_coupling_graph("./experiment_graph.csv", num_circuit_qubits)
                        input_circ = QuantumCircuit.from_qasm_file("./experiment_prog.qprog")

                        qq_command = "python3 ./qq.py --input-circuit=./experiment_prog.qprog " \
                                     "--coupling-graph=./experiment_graph.csv --log-level=WARNING " \
                                     "--log-file=./qq.log --num-solver-qubits={}".format(num_solver_qubits)
                        subprocess.check_output(qq_command, shell=True).decode('unicode_escape')
                        experiment_data_map = get_experiment_data()

                        performance_csv_file.write("{},{},{},{},{},{},{}\n".format(num_circuit_qubits,
                                                                            num_gates,
                                                                            proportion_couplings,
                                                                            trial,
                                                                            num_solver_qubits,
                                                                            experiment_data_map["SAT_Decisions"],
                                                                            experiment_data_map["Iterations"]))

                        if num_solver_qubits == 0:
                            comparative_csv_file.write(
                                "{},{},{},{},".format(num_circuit_qubits, num_gates, proportion_couplings, trial))
                            opt0_circ = transpile(input_circ, coupling_map=coupling_graph, seed_transpiler=0,
                                                  optimization_level=0)
                            opt1_circ = transpile(input_circ, coupling_map=coupling_graph, seed_transpiler=0,
                                                  optimization_level=1)
                            opt2_circ = transpile(input_circ, coupling_map=coupling_graph, seed_transpiler=0,
                                                  optimization_level=2)
                            opt3_circ = transpile(input_circ, coupling_map=coupling_graph, seed_transpiler=0,
                                                  optimization_level=3)
                            comparative_csv_file.write("{},{},{},{},{},{},".format(input_circ.depth(),
                                                                                   experiment_data_map["Opt_Depth"],
                                                                                   opt0_circ.depth(),
                                                                                   opt1_circ.depth(),
                                                                                   opt2_circ.depth(),
                                                                                   opt3_circ.depth()))
                            comparative_csv_file.write("{},{},{},{},{},{}\n".format(len(input_circ.count_ops()),
                                                                                    experiment_data_map["Opt_Ops"],
                                                                                    len(opt0_circ),
                                                                                    len(opt1_circ),
                                                                                    len(opt2_circ),
                                                                                    len(opt3_circ)))
                            comparative_csv_file.flush()
                        performance_csv_file.flush()


if __name__ == "__main__":
    experiments()

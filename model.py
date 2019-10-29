from functools import reduce
from typing import Dict, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister, Aer
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.circuit import Gate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.extensions import SwapGate
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from z3 import *

from qq.util import get_undirected_couplings, kcnf_split_clauses, strip_unit_clauses


# TODO: If we do a measurement or destroy some state we should revisualize that


def build_circuit():
    # qc = QuantumCircuit(7)
    # qc.h(3)
    # qc.cx(0, 6)
    # qc.cx(6, 0)
    # qc.cx(0, 1)
    # qc.cx(3, 1)
    # qc.cx(3, 0)
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(2, 0)


    # qc = QuantumCircuit(3, 2)
    # qc.h(0)
    # qc.cx(0, 2)
    # qc.measure([0, 1], [0, 1])
    return qc


# Generates the initial remapping of qubits and returns the mapping of physical -> virtual qubits. If
# a value (virtual qubit) is negative, it is not assigned to a realizable virtual qubit.
def generate_input_qubit_mapping(dag: DAGCircuit, coupling_map: CouplingMap, goal: Goal, qubits_used: int):
    num_physical_qubits = len(coupling_map.physical_qubits)
    num_virtual_qubits = qubits_used
    input_qubit_map = IntVector('input_qubit_map', num_physical_qubits)
    for input_qubit in input_qubit_map:
        goal.add(And(input_qubit >= num_virtual_qubits - num_physical_qubits,
                     input_qubit < num_virtual_qubits))

    goal.add(Distinct(input_qubit_map))
    # for i in range(num_physical_qubits):
    #     for j in range(i + 1, num_physical_qubits):
    #         goal.add(input_qubit_map[i] != input_qubit_map[j]) # TODO: Replace with Distinct?
    return input_qubit_map


# TODO: reorder args for 'constrain_*' functions so that solver comes first.
def constrain_gate_starts_before_end_time(node: DAGNode, gate_times: Dict[DAGNode, IntVector], goal: Goal,
                                          max_circuit_time: int):
    for gate_qubit_time in gate_times[node]:
        goal.add(And(gate_qubit_time >= 0,
                     gate_qubit_time < max_circuit_time))

    for qubit_index in range(len(node.qargs) - 1):
        qubit_0 = node.qargs[qubit_index]
        qubit_1 = node.qargs[qubit_index + 1]
        goal.add(gate_times[node][qubit_0.index] == gate_times[node][qubit_1.index])


def constrain_gate_ends_before_end_time(node: DAGNode, dag: DAGCircuit, gate_times: Dict[DAGNode, IntVector],
                                        gate_durations: Dict[DAGNode, IntVector], swap_nodes,
                                        goal: Goal, max_circuit_time: int):
    for i in range(dag.num_qubits()):
        goal.add(And(gate_times[node][i] + gate_durations[node][i] >= 0,
                     gate_times[node][i] + gate_durations[node][i] < max_circuit_time))

    for qubit_index in range(len(node.qargs) - 1):
        qubit_0 = node.qargs[qubit_index]
        qubit_1 = node.qargs[qubit_index + 1]
        goal.add(
            gate_times[node][qubit_0.index] + gate_durations[node][qubit_0.index]
            == gate_times[node][qubit_1.index] + gate_durations[node][qubit_1.index])
        goal.add(
            gate_times[node][qubit_0.index] == gate_times[node][qubit_1.index])

    # TODO change method name now that it has two functions. Or move this into own function
    for layer, coupling_vars in swap_nodes[node].items():
        for coupling, swap_node_exists in coupling_vars.items():
            goal.add(Implies(swap_node_exists,
                             gate_times[node][coupling[0]] == gate_times[node][coupling[1]]))


def calculate_physical_qubit_swap_duration(node_index, physical_qubit: int, node: DAGNode,
                                           swap_nodes: Dict[DAGNode, Dict[int, Dict[Tuple[int, int], Bool]]],
                                           goal: Goal, max_circuit_time: int):
    duration_list = []
    for layer, coupling_vars in swap_nodes[node].items():
        for coupling, swap_node_exists in coupling_vars.items():
            if physical_qubit in coupling:
                duration_list.append(If(swap_node_exists, 1, 0))

    swap_node_qubit_duration = Int("swap_node_duration_%d_qubit_%d" % (node_index, physical_qubit))
    goal.add(And(swap_node_qubit_duration >= 0,
                 swap_node_qubit_duration < max_circuit_time))
    goal.add(swap_node_qubit_duration == Sum(duration_list))
    return swap_node_qubit_duration


def constrain_gate_duration(node_index, node: DAGNode, qubit_mapping: IntVector,
                            gate_durations: Dict[DAGNode, IntVector],
                            gate_times: Dict[DAGNode, IntVector],
                            swap_nodes: Dict[DAGNode, Dict[int, Dict[Tuple[int, int], Bool]]], goal: Goal,
                            max_circuit_time: int):
    for gate_qubit_duration in gate_durations[node]:
        goal.add(And(gate_qubit_duration >= 0,
                     gate_qubit_duration < max_circuit_time))

    # Generate all of the swap node durations and at least try to fix them.
    swap_node_durations = []
    for physical_qubit, virtual_qubit in enumerate(qubit_mapping):
        swap_node_duration = calculate_physical_qubit_swap_duration(node_index, physical_qubit, node, swap_nodes, goal,
                                                                    max_circuit_time)
        swap_node_durations.append(swap_node_duration)
        goal.add(gate_durations[node][physical_qubit] >= swap_node_duration)

    # TODO: move swap durations out of gate_durations into own var.
    if len(node.qargs) == 1:
        for physical_qubit, virtual_qubit in enumerate(qubit_mapping):
            goal.add(Implies(virtual_qubit == node.qargs[0].index,
                             gate_durations[node][physical_qubit] >= 1 + swap_node_durations[physical_qubit]))
    elif len(node.qargs) == 2:
        for physical_qubit0, virtual_qubit0 in enumerate(qubit_mapping):
            for physical_qubit1, virtual_qubit1 in enumerate(qubit_mapping):
                if physical_qubit0 != physical_qubit1:
                    swap_node_qubit_duration0 = swap_node_durations[physical_qubit0]
                    swap_node_qubit_duration1 = swap_node_durations[physical_qubit1]
                    goal.add(Implies(And(virtual_qubit0 == node.qargs[0].index,
                                         virtual_qubit1 == node.qargs[1].index),
                                     And(And(gate_durations[node][physical_qubit0] >= 1 + swap_node_qubit_duration0,
                                             gate_durations[node][physical_qubit1] >= 1 + swap_node_qubit_duration1,
                                             gate_durations[node][physical_qubit0] >= 1 + swap_node_qubit_duration1,
                                             gate_durations[node][physical_qubit1] >= 1 + swap_node_qubit_duration0),
                                         And(gate_times[node][physical_qubit0] + gate_durations[node][physical_qubit0]
                                             == gate_times[node][physical_qubit1]
                                             + gate_durations[node][physical_qubit1]))))
        # TODO fix, should calc distance to correct mapping?
    else:
        # TODO: Support topologies which accept 3+ qubit gate implementations.
        raise TranspilerError('3+ qubit gates are not permitted.')

# Constrains gates to have gates which are physically coupled.
def constrain_gate_input_adjacency(node: DAGNode, qubit_mapping: IntVector, coupling_map: CouplingMap, goal: Goal):
    if len(node.qargs) == 1:
        pass  # TODO: Don't need to stipulate anything, right?
    elif len(node.qargs) == 2:
        adjacency_constraints = []
        for physical_qubit0, virtual_qubit0 in enumerate(qubit_mapping):
            for physical_qubit1, virtual_qubit1 in enumerate(qubit_mapping):
                if physical_qubit0 != physical_qubit1:
                    if coupling_map.distance(physical_qubit0, physical_qubit1) == 1:
                        adjacency_constraints.append(And(virtual_qubit0 == node.qargs[0].index,
                                                         virtual_qubit1 == node.qargs[1].index))
                    else:
                        goal.append(Not(And(virtual_qubit0 == node.qargs[0].index,
                                            virtual_qubit1 == node.qargs[1].index)))
        goal.add(Or(adjacency_constraints))
    else:
        # TODO: Support topologies which accept 3+ qubit gate implementations.
        raise TranspilerError('3+ qubit gates are not permitted.')


# Generates a mapping from node to times for each gate's start time.
def generate_gate_times(dag: DAGCircuit):
    pass


# Generate swaps to remap gates, we will need a certain number of swap layers and ensure distinctness.
# Returns a map of Tuple(swap_layer: int, coupling: Tuple(int, int)) -> Bool.
# TODO: Can simplify this by allowing swaps on disjoint pairs of qubits be allowed in each swap layer.
# TODO: Currently only one allowed.
#
# TODO: Can we just generate possible swaps for paths between source and target? Will this still work and be optimal?
def generate_gate_mapping_swaps(node_index: int, node: DAGNode, dag: DAGCircuit, coupling_map: CouplingMap,
                                prior_mapping: IntVector, goal: Goal, qubits_used: int, max_swaps_addable: int) -> Dict[
    DAGNode, Dict[int, Dict[Tuple[int, int], Bool]]]:
    num_physical_qubits = len(coupling_map.physical_qubits)
    num_virtual_qubits = qubits_used
    node_swap_gates = {}
    swap_gate_exists_list = []
    # TODO: Capture if the previous layer even existed before checking if we should consider the swaps in this layer.
    # TODO: Finish implementing this!
    first_layer_mapping = None
    prior_layer_mapping_exists = None
    undirected_couplings = [coupling for coupling in get_undirected_couplings(coupling_map)]

    num_swap_layers = len(undirected_couplings)
    if num_swap_layers > max_swaps_addable:
        num_swap_layers = max_swaps_addable

    if num_swap_layers <= 0:
        return node_swap_gates, swap_gate_exists_list, prior_mapping

    for swap_layer_index in range(num_swap_layers):
        layer_mapping_exists = Bool('swap_layer_mapping_needed_%d_layer_%d' % (node_index, swap_layer_index))
        layer_mapping = IntVector('swap_layer_mapping_%d_layer_%d' % (node_index, swap_layer_index),
                                  num_physical_qubits)
        for layer_qubit in layer_mapping:
            goal.add(And(layer_qubit >= num_virtual_qubits - num_physical_qubits,
                         layer_qubit < num_virtual_qubits))
        goal.add(Distinct(layer_mapping))

        layer_swap_gates = {}
        for coupling in undirected_couplings:
            swap_gate_exists = Bool(
                'swap_gate_%d_layer_%d_coupling_%d_%d' % (node_index, swap_layer_index, coupling[0], coupling[1]))
            swap_gate_exists_list.append(swap_gate_exists)
            layer_swap_gates[coupling] = swap_gate_exists
            goal.add(Implies(swap_gate_exists,
                             And(prior_mapping[coupling[0]] == layer_mapping[coupling[1]],
                                 prior_mapping[coupling[1]] == layer_mapping[coupling[0]])))
            for other_qubit in range(num_physical_qubits):
                if other_qubit not in coupling:
                    goal.add(Implies(swap_gate_exists,
                                     prior_mapping[other_qubit] == layer_mapping[other_qubit]))

        # Ensure that only one swap per layer.
        for coupling, swap_gate_exists in layer_swap_gates.items():
            for other_coupling, other_swap_gate_exists in layer_swap_gates.items():
                if coupling != other_coupling:
                    goal.add(Not(And(swap_gate_exists, other_swap_gate_exists)))

        # Ensure that if there are no swaps in the layer, that mapping is still constrained.
        goal.add(Implies(Not(Or([swap_gate_exists for swap_gate_exists in layer_swap_gates.values()])),
                         And([prior_mapping[physical_qubit] == layer_mapping[physical_qubit]
                              for physical_qubit in range(num_physical_qubits)])))

        node_swap_gates[swap_layer_index] = layer_swap_gates
        prior_mapping = layer_mapping
        if first_layer_mapping is None:
            first_layer_mapping = layer_mapping
    return node_swap_gates, swap_gate_exists_list, first_layer_mapping


# Generate a mapping of physical -> virtual qubits.
# TODO: Return value is redundant. Pass mapping in.
def generate_gate_mapping(node_index: int, node: DAGNode, dag: DAGCircuit, prior_gate_mapping: IntVector,
                          coupling_map: CouplingMap,
                          goal: Goal, qubits_used: int):
    num_physical_qubits = len(coupling_map.physical_qubits)
    num_virtual_qubits = qubits_used
    gate_mapping = IntVector('gate_mapping_%d' % node_index, num_physical_qubits)

    for i in range(num_physical_qubits):
        goal.add(prior_gate_mapping[i] == gate_mapping[i])

    # TODO can remove most likely because of equality above.
    for gate_qubit in gate_mapping:
        goal.add(And(gate_qubit >= num_virtual_qubits - num_physical_qubits,
                     gate_qubit < num_virtual_qubits))

    goal.add(Distinct(gate_mapping))

    return gate_mapping


# Primary driver to generate constrains on circuit schedule optimization.
def constrain_gate_schedule(dag: DAGCircuit, coupling_map: CouplingMap, goal: Goal, input_qubit_map: IntVector,
                            qubits_used: int, max_circuit_time: int, max_swaps_addable: int):
    print(dag.input_map.values())

    num_physical_qubits = len(coupling_map.physical_qubits)
    num_virtual_qubits = qubits_used
    swap_nodes = {}

    print("NUM_VIRT_QUBITS:", num_virtual_qubits)

    topo_gate_nodes = [node for node in dag.topological_op_nodes() if isinstance(node.op, Gate)]
    prior_mapping = input_qubit_map  # TODO: change
    prior_node = None
    gate_mappings = {node: IntVector(('gate_mapping_%d' % node_index), num_physical_qubits)
                     for node_index, node in enumerate(topo_gate_nodes)}
    gate_times = {node: IntVector(('gate_time_%d' % node_index), num_physical_qubits)
                  for node_index, node in enumerate(topo_gate_nodes)}
    gate_durations = {node: IntVector(('gate_duration_%d' % node_index), num_physical_qubits)
                      for node_index, node in enumerate(topo_gate_nodes)}
    swap_gate_exists_list = []
    for node_index, node in enumerate(topo_gate_nodes):
        print("NODE QR:", node_index, node, node.op, node.qargs, node.data_dict)
        swap_nodes[node], node_swap_gate_exists_list, swap_layer_mapping = generate_gate_mapping_swaps(node_index, node,
                                                                                                       dag,
                                                                                                       coupling_map,
                                                                                                       prior_mapping,
                                                                                                       goal,
                                                                                                       qubits_used,
                                                                                                       max_swaps_addable)
        swap_gate_exists_list.extend(node_swap_gate_exists_list)
        generate_gate_mapping(node_index, node, dag, swap_layer_mapping, coupling_map, goal, qubits_used)

        constrain_gate_starts_before_end_time(node, gate_times, goal, max_circuit_time)
        constrain_gate_ends_before_end_time(node, dag, gate_times, gate_durations, swap_nodes, goal, max_circuit_time)
        constrain_gate_duration(node_index, node, gate_mappings[node], gate_durations, gate_times, swap_nodes, goal,
                                max_circuit_time)
        constrain_gate_input_adjacency(node, gate_mappings[node], coupling_map, goal)

        if prior_node is not None:
            for physical_qubit in range(num_physical_qubits):
                goal.add(gate_times[node][physical_qubit] >=
                         gate_times[prior_node][physical_qubit] + gate_durations[prior_node][physical_qubit])
        # TODO: Delete soon.
        # for successor_node in dag.successors(node):
        #     if successor_node.type == 'op' and isinstance(successor_node.op, Gate):
        #         for node_physical_qubit in range(num_physical_qubits):
        #             for successor_physical_qubit in range(num_physical_qubits):
        #                 # TODO: need to fix this. only should constrain the time for particular qubits
        #                 goal.add(gate_times[successor_node][successor_physical_qubit] >=
        #                            gate_times[node][node_physical_qubit]
        #                            + gate_durations[node][node_physical_qubit])

        prior_node = node
        prior_mapping = gate_mappings[node]

    # TODO: Move into separate function 'contrain_circ_end_time' etc.
    # Circuit end time:
    circuit_end_time = Int('circuit_end_time')
    # goal.add(And(circuit_end_time >= 0,
    #                circuit_end_time < max_circuit_time))
    goal.add(circuit_end_time == max_circuit_time)
    for i in range(num_physical_qubits):
        goal.add(circuit_end_time >= gate_times[prior_node][i] + gate_durations[prior_node][i])
    #
    # Swap added count:
    swaps_added = Int('swaps_added')
    goal.add(And(swaps_added >= 0,
                 swaps_added <= max_swaps_addable))

    swap_gate_exists_pb_list = [(item, 1) for item in swap_gate_exists_list]
    swap_gate_count_constraint = PbLe(swap_gate_exists_pb_list, max_swaps_addable)
    swap_gate_counts = [If(swap_gate_exists, 1, 0) for swap_gate_exists in swap_gate_exists_list]

    goal.add(swap_gate_count_constraint)
    goal.add(swaps_added >= Sum(swap_gate_counts))

    print("NUM_ASSERTS: {}".format(len(goal)))

    return


# Print useful debug info when solver returns an unsat result.
def print_unsat(solver: Solver):
    unsat_solver = Solver()

    assertion_map = {}
    for index, assertion in enumerate(solver.assertions()):
        assert_val = Bool('assert_%d' % index)
        unsat_solver.assert_and_track(assertion, assert_val)
        assertion_map[assert_val] = assertion

    unsat_solver.check()

    print("UNSAT:\n", unsat_solver.unsat_core())
    # print("TRAIL:\n", unsat_solver.trailnode_index())
    unsat_core = unsat_solver.unsat_core()
    for i in range(len(unsat_core)):
        print("CONFLICT: \n", unsat_core[i], ":\n", assertion_map[unsat_core[i]])

    # seen = []
    # var_set = []
    # for e in visitor(And(solver.assertions()), seen):
    #     if is_const(e) and e.decl().kind() == Z3_OP_UNINTERPRETED:
    #         print("Variable:", e)
    #         var_set.append(e)
    # print(solver.check(var_set))

# Checks if the DAG is compatible with the algorithm.
def check_dag_circuit_compatible(dag: DAGCircuit, coupling_map: CouplingMap) -> bool:
    if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
        raise TranspilerError('Model runs on physical circuits only.')

    if len(dag.qubits()) > len(coupling_map.physical_qubits):
        raise TranspilerError('The layout does not match the amount of qubits in the DAG.')
    return True

# Builds the full model.
def build_model(circ: QuantumCircuit, coupling_map: CouplingMap, qubits_used: int, max_circuit_time: int,
                max_swaps_addable: int):
    dag = circuit_to_dag(circ)

    for node_index, node in enumerate(dag.topological_op_nodes()):
        print(node.data_dict)

    if not check_dag_circuit_compatible(dag, coupling_map):
        return

    print(dag.qregs)
    goal = Goal()
    input_qubit_map = generate_input_qubit_mapping(dag, coupling_map, goal, qubits_used)
    constrain_gate_schedule(dag, coupling_map, goal, input_qubit_map, qubits_used, max_circuit_time, max_swaps_addable)

    return goal


# TODO: Take this out and rework once we turn all of this into a transpiler pass.
# Initially remaps qubits so as to minimize qubit usage.
def remap_compact_virtual_qubit_registers(circ: QuantumCircuit, coupling_map: CouplingMap) -> Optional[QuantumCircuit]:
    dag = circuit_to_dag(circ)
    if not check_dag_circuit_compatible(dag, coupling_map):
        return

    qreg = dag.qregs['q']
    num_physical_qubits = len(coupling_map.physical_qubits)
    device_register = QuantumRegister(num_physical_qubits)
    qubit_index_counter = 0
    virtual_qubit_remap_layout = Layout.generate_trivial_layout(device_register)
    idle_wires = [wire for wire in dag.idle_wires()]
    for i in range(len(qreg)):
        qubit = qreg[i]
        if qubit not in idle_wires:
            print(qubit)
            virtual_qubit_remap_layout.add(qubit, qubit_index_counter)
            qubit_index_counter += 1
    for i, wire in enumerate(idle_wires):
        virtual_qubit_remap_layout.add(wire, qubit_index_counter + i)
    print("VIRT LAYOUT:", virtual_qubit_remap_layout)
    remap_pass_manager = PassManager()
    remap_pass_manager.append(SetLayout(virtual_qubit_remap_layout))
    remap_pass_manager.append(ApplyLayout())
    remap_circ = remap_pass_manager.run(circ)
    return remap_circ, qubit_index_counter


# Remaps the circuit using the original layout given the input layout we found to be optimal.
def remap_finalize_layout_virtual_qubit_registers(circ: QuantumCircuit, coupling_map: CouplingMap, layout: Layout) -> \
Optional[QuantumCircuit]:
    dag = circuit_to_dag(circ)
    if not check_dag_circuit_compatible(dag, coupling_map):
        return

    print("FINALIZE LAYOUT:", layout)
    remap_pass_manager = PassManager()
    remap_pass_manager.append(SetLayout(layout))
    remap_pass_manager.append(ApplyLayout())
    remap_circ = remap_pass_manager.run(circ)
    return remap_circ


# Helper form DIMACS stats function.
def combine_counts(counts0: Dict, counts1: Dict):
    for key0 in counts1.keys():
        if key0 in counts0:
            counts0[key0] += counts1[key0]
        else:
            counts0[key0] = counts1[key0]
    return counts0


# Print info about DIMACS CNF representation of the model.
def print_dimacs_stats(cnf_content: str):
    cnf_lines = cnf_content.splitlines()
    comment_lines = list(filter(lambda l: l.startswith('c'), cnf_lines))
    clause_lines = list(filter(lambda l: not (l.startswith('c') or l.startswith('p')), cnf_lines))
    program_lines = list(filter(lambda l: l.startswith('p'), cnf_lines))

    program_info = program_lines[0].split(' ')
    num_variables = int(program_info[2])
    num_clauses = int(program_info[3])
    print("Dimacs info:")
    print("  Num variables: {}".format(num_variables))
    print("  Num clauses: {} ({} lines)".format(num_clauses, len(clause_lines)))

    clause_values = []
    for clause_line in clause_lines:
        clause_value = [int(literal) for literal in clause_line.split(' ')[:-1]]
        clause_values.append(clause_value)

    # Clause length.
    clause_lengths = [len(clause_value) for clause_value in clause_values]
    clause_lengths_agg_map = map(lambda l: {l: 1}, clause_lengths)
    clause_lengths_agg_reduce = reduce(lambda l0, l1: combine_counts(l0, l1), clause_lengths_agg_map)

    avg_clause_length = sum(clause_lengths) / len(clause_lengths)
    min_clause_length = min(clause_lengths)
    max_clause_length = max(clause_lengths)
    print("  Avg. clause len:", avg_clause_length)
    print("  Min. clause len:", min_clause_length)
    print("  Max. clause len:", max_clause_length)

    for i in range(min_clause_length, max_clause_length + 1):
        if i in clause_lengths_agg_reduce.keys():
            num_clauses = clause_lengths_agg_reduce[i]
            print("  Len {} clauses: {}".format(i, num_clauses))

    # Variable occurrence.
    flat_clause_values = [value for clause in clause_values for value in clause]
    variable_values = map(lambda v: abs(v), flat_clause_values)
    variable_occurrences_map = map(lambda v: {v: 1}, variable_values)
    variable_occurrences_reduce = reduce(lambda v0, v1: combine_counts(v0, v1), variable_occurrences_map)

    avg_var_occ = sum(variable_occurrences_reduce.values()) / num_variables
    min_var_occ = min(variable_occurrences_reduce.values())
    max_var_occ = max(variable_occurrences_reduce.values())
    print("  Avg. vars occ:", avg_var_occ)
    print("  Min. vars occ:", min_var_occ)
    print("  Max. vars occ:", max_var_occ)

    for i in range(min_var_occ, max_var_occ + 1):
        if i in variable_occurrences_reduce.keys():
            num_clauses = variable_occurrences_reduce[i]
            print("  Occ {} vars: {}".format(i, num_clauses))


# Constructs the optimized circuit using:
#   1. The model found by the solver.
#   2. The circuit used to generate the model.
# TODO handle passing around of output layout
# TODO see todo above circuit remap function.
# TODO Do something like is done in _copy_circuit_metadata in lookahead_swap.py
def construct_circuit_from_model(model: ModelRef, dag: DAGCircuit, coupling_map: CouplingMap, num_virtual_qubits: int, max_swaps_addable: int):
    model_decls_map = {decl.name(): model[decl] for decl in model.decls()}

    print("NUM VARS:", len(model_decls_map))

    times = {}
    for k, v in model_decls_map.items():
        if 'time' in k:
            times[k] = v.as_long()

    for k, v in sorted(times.items(), key=lambda a: a[0]):
        print("TIMES:", k, v)

    durs = {}
    for k, v in model_decls_map.items():
        if 'duration' in k:
            durs[k] = v.as_long()

    for k, v in sorted(durs.items(), key=lambda a: a[0]):
        print("DURS:", k, v)

    new_dag = DAGCircuit()

    # TODO: Change so that this canonical register actually works with circuits not yet mapped to phys qubits such as in
    # TODO: DenseLayout.run()
    canonical_register = dag.qregs['q']
    trivial_layout = Layout.generate_trivial_layout(canonical_register)
    current_layout = trivial_layout.copy()

    # First fix the initial layout of circuit by setting the layout to the model's inferred input layout.
    for physical_qubit in canonical_register:
        input_qubit_map_decl_name = ("input_qubit_map__%d" % physical_qubit.index)
        input_virtual_qubit = model_decls_map[input_qubit_map_decl_name].as_long()

        # If the virtual qubit is non-negative, it is an actual virtual qubit. Otherwise, the number in
        # input_virtual_qubit is negative and is not used by a virtual qubit. We map this "fake" virtual qubit to some
        # unused physical qubit.
        if input_virtual_qubit >= 0:
            virtual_qubit_index = input_virtual_qubit
        else:
            virtual_qubit_index = num_virtual_qubits + (-input_virtual_qubit) - 1
        print("CST: CAN_Q/IN_VIRT_Q:", virtual_qubit_index, canonical_register[virtual_qubit_index],
              physical_qubit.index)

        current_layout[canonical_register[virtual_qubit_index]] = physical_qubit.index

    init_layout = current_layout.copy()
    print("INIT LAYOUT:", init_layout)

    node_index = 0
    for layer in dag.serial_layers():
        sub_dag = layer['graph']

        # Should be only one node in this layer, may not always be a gate.
        if len(sub_dag.gate_nodes()) == 1:
            # Add swap layers.
            # TODO: If an actual reciprocal undirected edge, can place swap as usual (cost: 1 swap (decomposed 3 cnot),
            # TODO: depth 3). Else if we only have a strictly directed edge need to explicit that cost will be (1 swap
            # TODO: (decomposed 3 cnot) and 4 Hadamard transforms, depth 5).
            undirected_couplings = [coupling for coupling in get_undirected_couplings(coupling_map)]

            num_swap_layers = len(undirected_couplings)

            if num_swap_layers > max_swaps_addable:
                num_swap_layers = max_swaps_addable

            for swap_layer_index in range(num_swap_layers):
                for coupling in undirected_couplings:
                    swap_gate_decl_name = ("swap_gate_%d_layer_%d_coupling_%d_%d" % (node_index, swap_layer_index,
                                                                                     coupling[0], coupling[1]))
                    swap_gate_exists = is_true(model_decls_map[swap_gate_decl_name])

                    # Actually make a swap layer if a swap gate exists.
                    if swap_gate_exists:
                        print("CST: SWP IDX/CPL:", swap_layer_index, coupling)

                        swap_layer = DAGCircuit()
                        swap_layer.add_qreg(canonical_register)

                        qubit_0 = current_layout[coupling[0]]
                        qubit_1 = current_layout[coupling[1]]
                        swap_layer.apply_operation_back(SwapGate(),
                                                        qargs=[qubit_0, qubit_1],
                                                        cargs=[])
                        edge_map = current_layout.combine_into_edge_map(trivial_layout)
                        new_dag.compose_back(swap_layer, edge_map)
                        current_layout.swap(qubit_0, qubit_1)

            # Set the layout as it would be input into gate.
            for qubit in canonical_register:
                gate_qubit_map_decl_name = ("gate_mapping_%d__%d" % (node_index, qubit.index))
                gate_virtual_qubit = model_decls_map[gate_qubit_map_decl_name].as_long()

                # See above comment for why we do this:
                # if gate_virtual_qubit >= 0:
                #     current_layout[qubit] = gate_virtual_qubit
                # else:
                #     current_layout[qubit] = num_virtual_qubits + (-gate_virtual_qubit) - 1
            for name, time in sorted(times.items(), key=lambda a: a[0]):
                if name.startswith('gate_time_{}'.format(node_index)):
                    print("CST TIME:", name, time)

            for name, time in sorted(durs.items(), key=lambda a: a[0]):
                if name.startswith('gate_duration_{}'.format(node_index)):
                    print("CST TIME:", name, time)
            node_index += 1

        edge_map = current_layout.combine_into_edge_map(trivial_layout)
        new_dag.extend_back(sub_dag, edge_map)
        print("CST DAG:", node_index - 1)
        print(dag_to_circuit(new_dag).draw())

    print('CIRC_END_TIME:', model_decls_map['circuit_end_time'])
    print('SWAPS_ADDED:', model_decls_map['swaps_added'])
    print('CIRC_DEPTH:', new_dag.depth())

    return new_dag, init_layout


# Turns a finite domain goal into a k-SAT goal.
def transform_goal_to_ksat(goal: Goal, max_clause_size: int = 3):
    fd_solver = SolverFor('QF_FD')
    simplify_tactic = With('simplify',
                           som=True,
                           pull_cheap_ite=True,
                           push_ite_bv=False,
                           local_ctx=True,
                           local_ctx_limit=10_000_000,
                           flat=True,
                           hoist_mul=False,
                           elim_and=True,
                           blast_distinct=True)
    fd_pre_process_tactic = Then('simplify', 'propagate-values', 'card2bv', simplify_tactic, 'max-bv-sharing',
                                 'bit-blast', simplify_tactic, 'sat-preprocess')
    fd_solver.add(*goal)
    translated_fd_solver = fd_solver.translate(main_ctx())
    translated_fd_goal = Goal()
    translated_fd_goal.add(*translated_fd_solver.assertions())
    preprocessed_fd_goal = fd_pre_process_tactic(translated_fd_goal)[0]
    preprocessed_fd_solver = SolverFor('QF_FD')
    preprocessed_fd_solver.add(*preprocessed_fd_goal)

    print(fd_solver.assertions())
    print(preprocessed_fd_goal)

    print("TRANSFORM:", len(fd_solver.assertions()), len(translated_fd_solver.assertions()), len(preprocessed_fd_solver.assertions()))
    check_solver_sat(fd_solver)
    check_solver_sat(preprocessed_fd_solver)

    with open("output/smt2_file_goal_fd.txt", "w") as smt2_fd_goal_file:
        smt2_fd_goal_file.write(preprocessed_fd_solver.sexpr())
    print(fd_solver.assertions())

    print("SAT ASSERTS:", len(fd_solver.assertions()), len(preprocessed_fd_solver.assertions()))

    print_dimacs_stats(fd_solver.dimacs())
    print(fd_solver.statistics())

    sat_preproc_goal = Goal()
    sat_preproc_goal.append(fd_solver.assertions())
    # preproc_goal = strip_unit_clauses(sat_preproc_goal)
    kcnf_sat_goal = kcnf_split_clauses(sat_preproc_goal, max_clause_length=max_clause_size)

    print("KCNF_STATS:")
    print_dimacs_stats(kcnf_sat_goal.dimacs())
    return kcnf_sat_goal

    # for line in [line for line in fd_solver.dimacs().splitlines() if line.startswith('c')]:
    #     print("C:", line)


def setup_optimizer(goal: Goal):
    optimizer = Optimize()
    optimizer.set(priority='pareto')
    for assertion in goal:
        optimizer.add(assertion)

    optimizer.minimize(Int('circuit_end_time'))
    optimizer.minimize(Int('swaps_added'))
    return optimizer


def check_solver_sat(solver: Solver):
    solver_check = solver.check()
    if solver_check == CheckSatResult(Z3_L_TRUE):  # If sat
        print("SAT CHECK:", sat)
    elif solver == CheckSatResult(Z3_L_FALSE):  # If unsat
        print_unsat(solver)
    else:
        print("REASON:", solver.reason_unknown())


# TODO write test routines to ensure the output of the circuit is the same as for original.

def main():
    # TODO: See above TODOs about undirected edges and cost model.
    # coupling_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
    #                   [1, 0], [2, 1], [3, 2], [4, 3], [5, 4], [6, 5]]
    coupling_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 2],
                      [1, 0], [1, 2], [3, 2], [4, 3], [2, 0]]

    # coupling_edges = [[0, 1], [1, 2], [2, 3]]
    input_coupling_map = CouplingMap(couplinglist=coupling_edges)

    input_circ = build_circuit()

    print(input_circ.draw())
    print(input_circ.qasm())
    remap_circuit, qubits_used = remap_compact_virtual_qubit_registers(input_circ, input_coupling_map)
    print(remap_circuit.draw())
    print(remap_circuit.qasm())

    # dag_drawer(circuit_to_dag(remap_circuit))

    max_swaps_addable = 3
    goal = build_model(remap_circuit, input_coupling_map, qubits_used, max_circuit_time=10,
                       max_swaps_addable=max_swaps_addable)
    # ksat_goal = transform_goal_to_ksat(goal)

    optimizer = setup_optimizer(goal)
    print("OPT CHECK: ", optimizer.check())
    opt_model = optimizer.model()
    print("OPT MODEL DONE")
    opt_dag, init_layout = construct_circuit_from_model(opt_model, circuit_to_dag(remap_circuit), input_coupling_map, qubits_used,
                                                        max_swaps_addable=max_swaps_addable)
    opt_circ = dag_to_circuit(opt_dag)

    print(opt_circ.draw())

    # cnf = """
    # p cnf 2 4
    # 1 2 3 0
    # 1 -2 4 0
    # -3 -4 50
    # """

    # TODO: Separate into method for sim / translation.
    # backend = Aer.get_backend('qasm_simulator')
    # oracle = LogicalExpressionOracle(cnf)  # , optimization=True)
    # algorithm = Grover(oracle, mct_mode='basic')
    #
    # oracle_circuit = oracle.construct_circuit()
    # grover_circuit = algorithm.construct_circuit()
    # print("A:")
    # grover_dag = circuit_to_dag(grover_circuit)
    # oracle_dag = circuit_to_dag(oracle_circuit)
    # # dag_drawer(grover_dag)
    # print("Q:")
    # # dag_drawer(oracle_dag)
    # print("DEPTH:", grover_dag.depth())
    # print("NUM_QUBITS:", grover_dag.num_qubits())
    # result = algorithm.run(backend)
    # print(result["result"])
    # print(result)


if __name__ == "__main__":
    main()
from z3 import *

from qq.util import topological_gate_nodes, get_undirected_couplings
from qq.types import *


def variable_gate_start_time(model_input: ModelInput):
    gate_start_times = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        gate_start_time_for_node = dict()
        for physical_qubit in range(num_physical_qubits):
            gate_start_time_for_node[physical_qubit] = Int('gate_start_time__{}_{}'.format(node_index, physical_qubit))
        gate_start_times[node] = gate_start_time_for_node
    return 'gate_start_time', gate_start_times


def variable_gate_duration(model_input: ModelInput):
    gate_durations = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        gate_durations_for_node = dict()
        for physical_qubit in range(num_physical_qubits):
            gate_durations_for_node[physical_qubit] = Int('gate_duration__{}_{}'.format(node_index, physical_qubit))
        gate_durations[node] = gate_durations_for_node
    return 'gate_duration', gate_durations


def variable_swap_gate_duration(model_input: ModelInput):
    swap_gate_durations = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        swap_gate_durations_for_node = dict()
        for physical_qubit in range(num_physical_qubits):
            swap_gate_durations_for_node[physical_qubit] = Int(
                'swap_gate_duration__{}_{}'.format(node_index, physical_qubit))
        swap_gate_durations[node] = swap_gate_durations_for_node
    return 'swap_gate_duration', swap_gate_durations


def variable_gate_qubit_mapping(model_input: ModelInput):
    gate_qubit_mappings = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        gate_qubit_mappings_for_node = dict()
        for physical_qubit in range(num_physical_qubits):
            gate_qubit_mappings_for_node[physical_qubit] = Int('gate_qubit_mapping__{}_{}'.format(node_index, physical_qubit))
        gate_qubit_mappings[node] = gate_qubit_mappings_for_node
    return 'gate_qubit_mapping', gate_qubit_mappings


def variable_input_qubit_mapping(model_input: ModelInput):
    input_qubit_mappings = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    for physical_qubit in range(num_physical_qubits):
        input_qubit_mappings[physical_qubit] = Int('input_qubit_mapping__{}'.format(physical_qubit))
    return 'input_qubit_mapping', input_qubit_mappings


def variable_swap_qubit_mapping(model_input: ModelInput):
    swap_qubit_mappings = dict()
    num_physical_qubits = model_input.coupling_graph.size()
    undirected_couplings = [coupling for coupling in get_undirected_couplings(model_input.coupling_graph)]
    num_swap_layers_per_node = len(undirected_couplings)
    max_swap_layers_per_node = model_input.parameters.max_swap_layers_per_node

    if max_swap_layers_per_node >= 0:
        num_swap_layers_per_node = max_swap_layers_per_node

    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        swap_qubit_mappings_for_node = dict()
        for layer_index in range(num_swap_layers_per_node):
            swap_qubit_mappings_for_node_for_layer = dict()
            for physical_qubit in range(num_physical_qubits):
                swap_qubit_mappings_for_node_for_layer[physical_qubit] = Int('swap_qubit_mapping__{}_{}_{}'.format(
                    node_index, layer_index, physical_qubit))
            swap_qubit_mappings_for_node[layer_index] = swap_qubit_mappings_for_node_for_layer
        swap_qubit_mappings[node] = swap_qubit_mappings_for_node
    return 'swap_qubit_mapping', swap_qubit_mappings


def variable_swap_gate_insertion(model_input: ModelInput):
    swap_gate_insertions = dict()
    undirected_couplings = [coupling for coupling in get_undirected_couplings(model_input.coupling_graph)]
    num_swap_layers_per_node = len(undirected_couplings)
    max_swap_layers_per_node = model_input.parameters.max_swap_layers_per_node

    if max_swap_layers_per_node >= 0:
        num_swap_layers_per_node = max_swap_layers_per_node

    for node_index, node in enumerate(topological_gate_nodes(model_input.circuit)):
        swap_gate_insertions_for_node = dict()
        for layer_index in range(num_swap_layers_per_node):
            swap_gate_insertions_for_node_for_layer = dict()
            for coupling in undirected_couplings:
                swap_gate_insertions_for_node_for_layer[coupling] = Bool('swap_gate_insertion__{}_{}_{}_{}'.format(
                    node_index, layer_index, coupling[0], coupling[1]))
            swap_gate_insertions_for_node[layer_index] = swap_gate_insertions_for_node_for_layer
        swap_gate_insertions[node] = swap_gate_insertions_for_node
    return 'swap_gate_insertion', swap_gate_insertions


def variable_circuit_end_time(model_input: ModelInput):
    return 'circuit_end_time', Int('circuit_end_time')


def variable_swaps_added(model_input: ModelInput):
    return 'swaps_added', Int('swaps_added')

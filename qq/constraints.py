from qiskit.transpiler import TranspilerError
from z3 import *

from qq.util import topological_gate_nodes, get_undirected_couplings
from qq.types import *


# Constrains the initial remapping of qubits and returns the mapping of physical -> virtual qubits. If
# a value (virtual qubit) is negative, it is not assigned to a realizable virtual qubit.
def constrain_input_qubit_mapping(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    input_qubit_mapping = model_variables['input_qubit_mapping']
    num_physical_qubits = model_input.coupling_graph.size()
    num_virtual_qubits = model_input.parameters.num_virtual_qubits
    for input_qubit in input_qubit_mapping.values():
        goal.add(And(input_qubit >= num_virtual_qubits - num_physical_qubits,
                     input_qubit < num_virtual_qubits))

    goal.add(Distinct(*input_qubit_mapping.values()))
    return goal


def constrain_gate_starts_before_end_time(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    gate_start_times = model_variables['gate_start_time']
    max_circuit_time = model_input.parameters.max_circuit_time
    for node in topological_gate_nodes(model_input.circuit):
        gate_start_time = gate_start_times[node]
        for physical_qubit in gate_start_times[node]:
            goal.add(And(gate_start_time[physical_qubit] >= 0,
                         gate_start_time[physical_qubit] < max_circuit_time))

        for qubit_index in range(len(node.qargs) - 1):
            qubit_0 = node.qargs[qubit_index]
            qubit_1 = node.qargs[qubit_index + 1]
            goal.add(gate_start_time[qubit_0.index] == gate_start_time[qubit_1.index])
    return goal


def constrain_gate_ends_before_end_time(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    gate_start_times = model_variables['gate_start_time']
    gate_durations = model_variables['gate_duration']
    swap_gate_insertions = model_variables['swap_gate_insertion']
    max_circuit_time = model_input.parameters.max_circuit_time
    num_physical_qubits = model_input.coupling_graph.size()
    for node in topological_gate_nodes(model_input.circuit):
        for physical_qubit in range(num_physical_qubits):
            goal.add(And(
                gate_start_times[node][physical_qubit] + gate_durations[node][physical_qubit] >= 0,
                gate_start_times[node][physical_qubit] + gate_durations[node][physical_qubit] < max_circuit_time))

        for qubit_index in range(len(node.qargs) - 1):
            qubit_0 = node.qargs[qubit_index]
            qubit_1 = node.qargs[qubit_index + 1]
            goal.add(
                gate_start_times[node][qubit_0.index] + gate_durations[node][qubit_0.index]
                == gate_start_times[node][qubit_1.index] + gate_durations[node][qubit_1.index])
            goal.add(
                gate_start_times[node][qubit_0.index] == gate_start_times[node][qubit_1.index])

        # TODO change method name now that it has two functions. Or move this into own function
        for layer, coupling_vars in swap_gate_insertions[node].items():
            for coupling, swap_node_exists in coupling_vars.items():
                goal.add(Implies(swap_node_exists,
                                 gate_start_times[node][coupling[0]] == gate_start_times[node][coupling[1]]))
    return goal


# Constrain swaps to remap gates, we will need a certain number of swap layers as well as  ensure distinctness.
# TODO: Can simplify this by allowing swaps on disjoint pairs of qubits be allowed in each swap layer.
# TODO: Currently only one allowed.
# TODO: Can we just generate possible swaps for paths between source and target? Will this still work and be optimal?
def constrain_swap_qubit_mapping(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    num_virtual_qubits = model_input.parameters.num_virtual_qubits
    num_physical_qubits = model_input.coupling_graph.size()

    undirected_couplings = [coupling for coupling in get_undirected_couplings(model_input.coupling_graph)]
    num_swap_layers_per_node = len(undirected_couplings)
    max_swap_layers_per_node = model_input.parameters.max_swap_layers_per_node

    if max_swap_layers_per_node >= 0:
        num_swap_layers_per_node = max_swap_layers_per_node

    prior_mapping = model_variables['input_qubit_mapping']
    # TODO: Capture if the previous layer even existed before checking if we should consider the swaps in this layer.

    swap_gate_insertions = model_variables['swap_gate_insertion']
    swap_qubit_mappings = model_variables['swap_qubit_mapping']
    gate_qubit_mappings = model_variables['gate_qubit_mapping']

    for node in topological_gate_nodes(model_input.circuit):
        swap_gate_insertion = swap_gate_insertions[node]
        swap_qubit_mapping = swap_qubit_mappings[node]

        for swap_layer_index in range(num_swap_layers_per_node):
            swap_layer_qubit_mapping = swap_qubit_mapping[swap_layer_index]
            swap_layer_gate_insertion = swap_gate_insertion[swap_layer_index]

            # Bound the swap layer mapping within a range.
            for swap_layer_qubit in swap_layer_qubit_mapping.values():
                goal.add(And(swap_layer_qubit >= num_virtual_qubits - num_physical_qubits,
                             swap_layer_qubit < num_virtual_qubits))
            goal.add(Distinct(*swap_layer_qubit_mapping.values()))

            # Permute qubit mapping when a swap is to be inserted between a coupling.
            for coupling in swap_layer_gate_insertion:
                goal.add(Implies(swap_layer_gate_insertion[coupling],
                                 And(prior_mapping[coupling[0]] == swap_layer_qubit_mapping[coupling[1]],
                                     prior_mapping[coupling[1]] == swap_layer_qubit_mapping[coupling[0]])))

            # Ensure that only one swap per layer.
            for coupling in swap_layer_gate_insertion:
                for other_coupling in swap_layer_gate_insertion:
                    if coupling != other_coupling:
                        goal.add(Not(And(swap_layer_gate_insertion[coupling],
                                         swap_layer_gate_insertion[other_coupling])))

            # Ensure that if there are no swaps in the layer, that mapping is still constrained.
            goal.add(Implies(Not(Or([swap_gate_exists for swap_gate_exists in swap_layer_gate_insertion.values()])),
                             And([prior_mapping[physical_qubit] == swap_layer_qubit_mapping[physical_qubit]
                                  for physical_qubit in range(num_physical_qubits)])))
            prior_mapping = swap_layer_qubit_mapping
        prior_mapping = gate_qubit_mappings[node]
    return goal


# Constrains gates to have qubits which are physically coupled.
def constrain_gate_input_adjacency(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()

    for node in topological_gate_nodes(model_input.circuit):
        gate_qubit_mapping = model_variables['gate_qubit_mapping'][node]
        if len(node.qargs) == 1:
            pass
        elif len(node.qargs) == 2:
            adjacency_constraints = []
            for physical_qubit0, virtual_qubit0 in gate_qubit_mapping.items():
                for physical_qubit1, virtual_qubit1 in gate_qubit_mapping.items():
                    if physical_qubit0 != physical_qubit1:
                        if model_input.coupling_graph.distance(physical_qubit0, physical_qubit1) == 1:
                            adjacency_constraints.append(And(virtual_qubit0 == node.qargs[0].index,
                                                             virtual_qubit1 == node.qargs[1].index))
                        else:
                            goal.append(Not(And(virtual_qubit0 == node.qargs[0].index,
                                                virtual_qubit1 == node.qargs[1].index)))
            goal.add(Or(adjacency_constraints))
        else:
            # TODO: Support topologies which accept 3+ qubit gate implementations.
            raise TranspilerError('3+ qubit gates are not permitted.')
    return goal


# Constrain the mapping of physical to virtual qubits for a gate.
def constrain_gate_qubit_mapping(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    num_physical_qubits = model_input.coupling_graph.size()
    num_virtual_qubits = model_input.parameters.num_virtual_qubits

    for node in topological_gate_nodes(model_input.circuit):
        gate_qubit_mapping = model_variables['gate_qubit_mapping'][node]
        swap_qubit_mapping = model_variables['swap_qubit_mapping'][node]
        last_swap_layer_index = max(swap_qubit_mapping.keys())
        last_swap_layer_qubit_mapping = swap_qubit_mapping[last_swap_layer_index]

        for physical_qubit in range(num_physical_qubits):
            goal.add(And(gate_qubit_mapping[physical_qubit] >= num_virtual_qubits - num_physical_qubits,
                         gate_qubit_mapping[physical_qubit] < num_virtual_qubits))
            goal.add(last_swap_layer_qubit_mapping[physical_qubit] == gate_qubit_mapping[physical_qubit])

    return goal


def constrain_swap_gate_duration(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    swap_gate_durations = model_variables['swap_gate_duration']
    swap_gate_insertions = model_variables['swap_gate_insertion']
    num_physical_qubits = model_input.coupling_graph.size()
    max_circuit_time = model_input.parameters.max_circuit_time

    for node in swap_gate_durations:
        swap_gate_duration = swap_gate_durations[node]
        swap_gate_insertion = swap_gate_insertions[node]

        for physical_qubit in range(num_physical_qubits):
            swap_gate_qubit_duration = swap_gate_duration[physical_qubit]
            goal.add(And(swap_gate_qubit_duration >= 0,
                         swap_gate_qubit_duration < max_circuit_time))

            duration_list = []
            for swap_layer_index in swap_gate_insertion:
                swap_layer_gate_insertion = swap_gate_insertion[swap_layer_index]
                for coupling in swap_layer_gate_insertion:
                    if physical_qubit in coupling:
                        swap_node_exists = swap_layer_gate_insertion[coupling]
                        duration_list.append(If(swap_node_exists, 1, 0))
            goal.add(swap_gate_qubit_duration == Sum(duration_list))
    return goal


def constrain_gate_duration_ends_before_end_time(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    gate_durations = model_variables['gate_duration']
    max_circuit_time = model_input.parameters.max_circuit_time

    for node in topological_gate_nodes(model_input.circuit):
        gate_duration = gate_durations[node]

        for physical_qubit in gate_duration:
            goal.add(And(gate_duration[physical_qubit] >= 0,
                         gate_duration[physical_qubit] < max_circuit_time))
    return goal


def constrain_gate_duration(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    gate_durations = model_variables['gate_duration']
    swap_gate_durations = model_variables['swap_gate_duration']
    gate_start_times = model_variables['gate_start_time']
    gate_qubit_mappings = model_variables['gate_qubit_mapping']

    num_physical_qubits = model_input.coupling_graph.size()

    for node in topological_gate_nodes(model_input.circuit):
        gate_duration = gate_durations[node]
        swap_gate_duration = swap_gate_durations[node]
        gate_start_time = gate_start_times[node]
        gate_qubit_mapping = gate_qubit_mappings[node]

        # Constrain all of the gate durations to have a threshold at least as high as the swap gate durations.
        for physical_qubit in range(num_physical_qubits):
            goal.add(gate_duration[physical_qubit] >= swap_gate_duration[physical_qubit])

        if len(node.qargs) == 1:
            for physical_qubit, virtual_qubit in gate_qubit_mapping.items():
                goal.add(Implies(virtual_qubit == node.qargs[0].index,
                                 gate_duration[physical_qubit] >= 1 + swap_gate_duration[physical_qubit]))
        elif len(node.qargs) == 2:
            for physical_qubit0, virtual_qubit0 in gate_qubit_mapping.items():
                for physical_qubit1, virtual_qubit1 in gate_qubit_mapping.items():
                    if physical_qubit0 != physical_qubit1:
                        swap_gate_qubit_duration0 = swap_gate_duration[physical_qubit0]
                        swap_gate_qubit_duration1 = swap_gate_duration[physical_qubit1]
                        goal.add(Implies(And(virtual_qubit0 == node.qargs[0].index,
                                             virtual_qubit1 == node.qargs[1].index),
                                         And(And(gate_duration[physical_qubit0] >= 1 + swap_gate_qubit_duration0,
                                                 gate_duration[physical_qubit1] >= 1 + swap_gate_qubit_duration1,
                                                 gate_duration[physical_qubit0] >= 1 + swap_gate_qubit_duration1,
                                                 gate_duration[physical_qubit1] >= 1 + swap_gate_qubit_duration0),
                                             And(gate_start_time[physical_qubit0] + gate_duration[physical_qubit0]
                                                 == gate_start_time[physical_qubit1]
                                                 + gate_duration[physical_qubit1]))))
            # TODO fix, should calc distance to correct mapping?
        else:
            # TODO: Support topologies which accept 3+ qubit gate implementations.
            raise TranspilerError('3+ qubit gates are not permitted.')
    return goal


def constrain_circuit_end_time(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    circuit_end_time = model_variables['circuit_end_time']
    gate_start_times = model_variables['gate_start_time']
    gate_durations = model_variables['gate_duration']
    max_circuit_time = model_input.parameters.max_circuit_time
    num_physical_qubits = model_input.coupling_graph.size()

    goal.add(And(circuit_end_time >= 0,
                 circuit_end_time <= max_circuit_time))

    last_node = list(topological_gate_nodes(model_input.circuit))[-1]

    for physical_qubit in range(num_physical_qubits):
        goal.add(circuit_end_time >=
                 gate_start_times[last_node][physical_qubit] + gate_durations[last_node][physical_qubit])

    return goal


def constrain_swaps_added(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    goal = Goal()
    swaps_added = model_variables['swaps_added']
    swap_gate_insertions = model_variables['swap_gate_insertion']
    max_swaps_addable = model_input.parameters.max_swaps_addable

    goal.add(swaps_added >= 0)
    if max_swaps_addable < 0:
        undirected_couplings = [coupling for coupling in get_undirected_couplings(model_input.coupling_graph)]
        num_swap_layers = len(undirected_couplings)
        max_swaps_addable = (num_swap_layers * len(list(topological_gate_nodes(model_input.circuit))))

    goal.add(swaps_added == max_swaps_addable)

    swap_gate_exists_list = [swap_gate_insertions[node][layer][coupling]
                             for node in swap_gate_insertions
                             for layer in swap_gate_insertions[node]
                             for coupling in swap_gate_insertions[node][layer]]
    # swap_gate_counts = [If(swap_gate_exists, 1, 0) for swap_gate_exists in swap_gate_exists_list]
    # goal.add(swaps_added >= Sum(swap_gate_counts))
    swap_gate_exists_pb_list = [(swap_gate_exists, 1) for swap_gate_exists in swap_gate_exists_list]
    swap_gate_count_constraint = PbLe(swap_gate_exists_pb_list, max_swaps_addable)
    goal.add(swap_gate_count_constraint)

    return goal


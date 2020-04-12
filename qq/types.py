from collections import namedtuple
from typing import NewType, Dict, Any

ModelParameters = namedtuple('ModelParameters', ['max_circuit_time',
                                                 'max_swaps_addable',
                                                 'max_swap_layers_per_node',
                                                 'num_virtual_qubits'])
ModelInput = namedtuple('ModelInput', ['circuit', 'coupling_graph', 'parameters'])
ModelVariables = NewType('ModelVariables', Dict[str, Any])


def Assignment(init_values=None):
    if init_values is None:
        init_values = {}
    assignment_type = NewType('Assignment', Dict[int, bool])
    return assignment_type(init_values)

from collections import namedtuple
from typing import NewType, Dict, Any

ModelParameters = namedtuple('ModelParameters', ['max_circuit_time',
                                                 'max_swaps_addable',
                                                 'max_swap_layers_per_node',
                                                 'num_virtual_qubits'])
ModelInput = namedtuple('ModelInput', ['circuit', 'coupling_graph', 'parameters'])
ModelVariables = NewType('ModelVariables', Dict[str, Any])

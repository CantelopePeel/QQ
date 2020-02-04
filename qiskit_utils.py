import tqdm
from qiskit.dagcircuit import DAGCircuit


def fast_circuit_to_dag(circuit):
    dag = DAGCircuit()
    dag.name = circuit.name

    for qreg in circuit.qregs:
        dag.add_qreg(qreg)
    for creg in circuit.cregs:
        dag.add_creg(creg)

    for instruction, qargs, cargs in tqdm.tqdm(circuit.data, desc="Circuit to DAG"):
        dag.apply_operation_back(instruction.copy(), qargs, cargs,
                                 instruction.condition)
    return dag
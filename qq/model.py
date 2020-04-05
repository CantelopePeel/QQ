import subprocess
from typing import Tuple

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard.swap import SwapGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout

from qq.assignment import Assignment, unit_propagation, clause_subsumption, pure_literal_elimination, \
    clause_subsumption_new, validate_clauses_and_assignment, AssignmentValue
from qq.constraints import *
from qq.sexpr_parser import build_variable_eval_order_list, rename_var, build_eval_dict, build_eval_code_block, \
    run_eval_code_block, parse_sexpr, drop_asserts_sexpr_file
from qq.util import *
from qq.variables import *

logger = logging.getLogger(__name__)


def generate_model_inputs(circuit: DAGCircuit, coupling_graph: CouplingMap, parameters: ModelParameters) -> ModelInput:
    return ModelInput(circuit=circuit,
                      coupling_graph=coupling_graph,
                      parameters=parameters)


def generate_model_parameters(max_circuit_time: int, max_swaps_addable: int, max_swap_layers_per_node: int,
                              num_virtual_qubits: int):
    return ModelParameters(max_circuit_time=max_circuit_time,
                           max_swaps_addable=max_swaps_addable,
                           max_swap_layers_per_node=max_swap_layers_per_node,
                           num_virtual_qubits=num_virtual_qubits)


def generate_model_variables(model_input: ModelInput) -> ModelVariables:
    variable_generator_funcs = [
        variable_gate_start_time,
        variable_gate_duration,
        variable_swap_gate_duration,
        variable_gate_qubit_mapping,
        variable_input_qubit_mapping,
        variable_swap_qubit_mapping,
        variable_swap_gate_insertion,
        variable_circuit_end_time,
        variable_swaps_added,
    ]

    model_variables = ModelVariables(dict())
    for var_func in variable_generator_funcs:
        var_name, var_values = var_func(model_input)
        logger.debug("Variables: %s", var_name)
        model_variables[var_name] = var_values

    # print(model_variables)

    return model_variables


def generate_model_constraints(model_input: ModelInput, model_variables: ModelVariables) -> Goal:
    constraint_generator_funcs = [
        constrain_input_qubit_mapping,
        constrain_gate_starts_before_end_time,
        constrain_gate_ends_before_end_time,
        constrain_swap_qubit_mapping,
        constrain_gate_input_adjacency,
        constrain_gate_qubit_mapping,
        constrain_swap_gate_duration,
        constrain_gate_duration_ends_before_end_time,
        constrain_gate_duration,
        constrain_circuit_end_time,
        constrain_swaps_added,
    ]

    model_constraints = Goal()
    for con_func in constraint_generator_funcs:
        constraint_goal = con_func(model_input, model_variables)
        model_constraints.add(*constraint_goal)
        logger.debug("Generated constraints: %s", con_func.__name__)

    logger.debug("Total constraints generated: {}".format(len(model_constraints)))
    return model_constraints


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
        logger.debug("Circuit node data ({0}): {1} ".format(node_index, node.data_dict))

    if not check_dag_circuit_compatible(dag, coupling_map):
        return

    model_parameters = generate_model_parameters(max_circuit_time=max_circuit_time, max_swaps_addable=max_swaps_addable,
                                                 max_swap_layers_per_node=-1, num_virtual_qubits=qubits_used)
    model_inputs = generate_model_inputs(circuit=dag, coupling_graph=coupling_map, parameters=model_parameters)
    model_variables = generate_model_variables(model_inputs)
    goal = generate_model_constraints(model_inputs, model_variables)

    return goal, model_inputs


# TODO: Take this out and rework once we turn all of this into a transpiler pass.
# Initially remaps qubits so as to minimize qubit usage.
def remap_compact_virtual_qubit_registers(circuit: QuantumCircuit, coupling_map: CouplingMap) -> \
        Optional[Tuple[QuantumCircuit, int]]:
    dag = circuit_to_dag(circuit)
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
            virtual_qubit_remap_layout.add(qubit, qubit_index_counter)
            qubit_index_counter += 1

    for i, wire in enumerate(idle_wires):
        virtual_qubit_remap_layout.add(wire, qubit_index_counter + i)
    logger.debug("Compact circuit layout: {}".format(virtual_qubit_remap_layout))
    remap_pass_manager = PassManager()
    remap_pass_manager.append(SetLayout(virtual_qubit_remap_layout))
    remap_pass_manager.append(ApplyLayout())
    remap_circuit = remap_pass_manager.run(circuit)
    return remap_circuit, qubit_index_counter


# Remaps the circuit using the original layout given the input layout we found to be optimal.
def remap_apply_layout_virtual_qubit_registers(circuit: QuantumCircuit, coupling_map: CouplingMap, layout: Layout) -> \
        Optional[QuantumCircuit]:
    dag = circuit_to_dag(circuit)
    if not check_dag_circuit_compatible(dag, coupling_map):
        return

    print("FINALIZE LAYOUT:", layout)
    remap_pass_manager = PassManager()
    remap_pass_manager.append(SetLayout(layout))
    remap_pass_manager.append(ApplyLayout())
    remap_circ = remap_pass_manager.run(circuit)
    return remap_circ


# Constructs the optimized circuit using:
#   1. The model found by the solver.
#   2. The circuit used to generate the model.
# TODO handle passing around of output layout
# TODO see todo above circuit remap function.
# TODO Do something like is done in _copy_circuit_metadata in lookahead_swap.py
def construct_circuit_from_model(model: ModelRef, model_input: ModelInput):
    num_virtual_qubits = model_input.parameters.num_virtual_qubits
    max_swaps_addable = model_input.parameters.max_swaps_addable
    circuit = model_input.circuit

    model = {decl.name(): model[decl].as_long() for decl in model.decls()}

    # for name, decl in model.items():
    #     print(name, decl)

    logger.debug("NUM VARS: {}".format(len(model)))

    times = {}
    for key, value in model.items():
        if 'time' in key:
            times[key] = value

    for key, value in sorted(times.items(), key=lambda a: a[0]):
        logger.debug("TIMES: {} {}".format(key, value))

    durs = {}
    for key, value in model.items():
        if 'duration' in key:
            durs[key] = value

    for key, value in sorted(durs.items(), key=lambda a: a[0]):
        logger.debug("DURS: {} {}".format(key, value))

    ins = {}
    for key, value in model.items():
        if 'insertion' in key:
            ins[key] = is_true(value)

    for key, value in sorted(ins.items(), key=lambda a: a[0]):
        logger.debug("INS: {} {}".format(key, value))

    new_dag = DAGCircuit()

    # TODO: Change so that this canonical register actually works with circuits not yet mapped to phys qubits such as in
    # TODO: DenseLayout.run()
    canonical_register = circuit.qregs['q']
    trivial_layout = Layout.generate_trivial_layout(canonical_register)
    current_layout = trivial_layout.copy()
    # print(current_layout, ":::", trivial_layout)

    # First fix the initial layout of circuit by setting the layout to the model's inferred input layout.
    for physical_qubit in canonical_register:
        input_qubit_map_decl_name = ("input_qubit_mapping__%d" % physical_qubit.index)
        input_virtual_qubit = model[input_qubit_map_decl_name]

        # If the virtual qubit is non-negative, it is an actual virtual qubit. Otherwise, the number in
        # input_virtual_qubit is negative and is not used by a virtual qubit. We map this "fake" virtual qubit to some
        # unused physical qubit.
        if input_virtual_qubit >= 0:
            virtual_qubit_index = input_virtual_qubit
        else:
            virtual_qubit_index = num_virtual_qubits + (-input_virtual_qubit) - 1
        logger.debug("Current Start Times: Canonical Qubit/Index Virtual Qubit: %s %s %s", virtual_qubit_index,
                     canonical_register[virtual_qubit_index], physical_qubit.index)

        current_layout[canonical_register[virtual_qubit_index]] = physical_qubit.index

    init_layout = current_layout.copy()
    logger.debug("Constructed circuit init layout: %s", init_layout)
    node_index = 0
    for layer in model_input.circuit.serial_layers():
        sub_dag = layer['graph']

        # Should be only one node in this layer, may not always be a gate.
        if len(sub_dag.gate_nodes()) == 1:
            # Add swap layers.
            # TODO: If an actual reciprocal undirected edge, can place swap as usual (cost: 1 swap (decomposed 3 cnot),
            # TODO: depth 3). Else if we only have a strictly directed edge need to explicit that cost will be (1 swap
            # TODO: (decomposed 3 cnot) and 4 Hadamard transforms, depth 5).
            undirected_couplings = [coupling for coupling in get_undirected_couplings(model_input.coupling_graph)]

            num_swap_layers = len(undirected_couplings)

            if num_swap_layers > max_swaps_addable >= 0:
                num_swap_layers = max_swaps_addable

            for swap_layer_index in range(num_swap_layers):
                for coupling in undirected_couplings:
                    swap_gate_decl_name = ("swap_gate_insertion__%d_%d_%d_%d" % (node_index, swap_layer_index,
                                                                                 coupling[0], coupling[1]))
                    swap_gate_exists = model[swap_gate_decl_name]
                    # print(current_layout, ":::", trivial_layout)

                    # Actually make a swap layer if a swap gate exists.
                    if swap_gate_exists != 0:
                        logger.debug("CST: SWP IDX/CPL: {} {}".format(swap_layer_index, coupling))

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
                gate_qubit_map_decl_name = ("gate_qubit_mapping__%d_%d" % (node_index, qubit.index))
                gate_virtual_qubit = model[gate_qubit_map_decl_name]

                # See above comment for why we do this:
                # if gate_virtual_qubit >= 0:
                #     current_layout[qubit] = gate_virtual_qubit
                # else:
                #     current_layout[qubit] = num_virtual_qubits + (-gate_virtual_qubit) - 1
            for name, time in sorted(times.items(), key=lambda a: a[0]):
                if name.startswith('gate_start_time__{}'.format(node_index)):
                    logger.debug("CST TIME:".format(name, time))

            for name, time in sorted(durs.items(), key=lambda a: a[0]):
                if name.startswith('gate_duration__{}'.format(node_index)):
                    logger.debug("CST TIME:".format(name, time))
            node_index += 1

        edge_map = current_layout.combine_into_edge_map(trivial_layout)
        new_dag.extend_back(sub_dag, edge_map)
        logger.debug("CST DAG: {} \n{}".format(node_index - 1, dag_to_circuit(new_dag).draw()))

    logger.debug('CIRC_END_TIME: {}'.format(model['circuit_end_time']))
    logger.debug('SWAPS_ADDED: {}'.format(model['swaps_added']))
    logger.debug('CIRC_DEPTH: {}'.format(new_dag.depth()))

    return new_dag, init_layout


# Turns a finite domain goal into a k-SAT goal.
def transform_goal_to_cnf(goal: Goal, max_clause_size: int = 3):
    fd_pre_process_tactic = Tactic('qffd')
    fd_solver = fd_pre_process_tactic.solver()
    fd_solver.set('max_conflicts', 0)
    fd_solver.add(*goal)
    print_dimacs_stats(fd_solver.dimacs())

    check_solver_sat(fd_solver)

    logger.debug("FD_STATS:")
    print_dimacs_stats(fd_solver.dimacs(), extended_info=True)
    logger.debug(fd_solver.statistics())

    sat_preproc_goal = Goal()
    sat_preproc_goal.append(fd_solver.assertions())

    sat_preproc_clause_list = ClauseList()
    sat_preproc_clause_list.load_from_dimacs(sat_preproc_goal.dimacs(), show_progress=True)
    kcnf_clause_list = split_clauses_in_clause_list(sat_preproc_clause_list, max_clause_length=max_clause_size)

    logger.debug("KCNF_STATS:")
    print_clause_stats(kcnf_clause_list)

    kcnf_sat_sexpr = fd_solver.sexpr()
    return sat_preproc_goal, kcnf_sat_sexpr, kcnf_clause_list


def setup_optimizer(goal: Goal):
    optimizer = Optimize()
    optimizer.set(priority='pareto')
    for assertion in goal:
        optimizer.add(assertion)

    optimizer.minimize(Int('circuit_end_time'))
    optimizer.minimize(Int('swaps_added'))
    return optimizer


def quantum_kcnf_solver(goal: Goal):
    assignment = {}

    quantum_solver_goal = Goal()

    # Preprocess the unit clauses out and into the assignment.
    for assertion in goal:
        if expr_is_literal(assertion):
            if is_not(assertion):
                assignment[assertion.arg(0).decl().name()] = False
            else:
                assignment[assertion.decl().name()] = False
        else:
            quantum_solver_goal.add(assertion)
    logger.debug("Preprocessing assignment length: {}".format(len(assignment)))
    print_dimacs_stats(quantum_solver_goal.dimacs(), extended_info=True)

    return assignment


def check_solver_capable(num_vars, num_clauses) -> bool:
    if num_vars <= 100:
        logger.debug("CCC: V: {} C: {}".format(num_vars, num_clauses))
        return True
    else:
        return False


num_solver_qubits = 0


def run_solver(solver_state: bytes):
    state_str = solver_state.decode('unicode_escape')

    with open('./state_dimacs.txt', 'w') as dimacs_file:
        dimacs_file.write(state_str)

    clause_list = ClauseList()
    clause_list.load_from_dimacs(state_str, make_names=True)
    assignment = Assignment()

    unit_prop_result = unit_propagation(clause_list, assignment, num_levels=1, limit_remaining_items=-1)
    if unit_prop_result is None:
        return b"none"
    clause_list, assignment = unit_prop_result
    # print_clause_stats(clause_list)

    assignment_clauses = []
    print("Vs:", clause_list.num_variables(), clause_list.num_clauses(), clause_list.num_variables() + clause_list.num_clauses() )
    # TODO remove if unneeded.
    # for var in clause_list.variables():
    #     if var in assignment:
    #         print("V:", var)
    #         lit = var if assignment[var] == AssignmentValue.TRUE else -var
    #         assignment_clauses.append(Clause({lit}))
    # for assignment_clause in assignment_clauses:
    #     print("AC:", assignment_clause)
    #     clause_list.add_clause(assignment_clause, make_names=True)

    # # TODO: Optimize this process
    # while clause_list.num_variables() + clause_list.num_clauses() >= 40:
    #     logger.info("A: %s %s %s", str(validate_clauses_and_assignment(clause_list, assignment)), str(clause_list.num_variables()), str(clause_list.num_clauses()))
    #     # elim_clause_list, assignment = pure_literal_elimination(clause_list, assignment)
    #     # logger.info("B: %s %s", str(elim_clause_list.num_variables()), str(elim_clause_list.num_clauses()))
    #     prop_clause_list, assignment = unit_propagation(clause_list, assignment)
    #     logger.info("C: %s %s %s", str(validate_clauses_and_assignment(prop_clause_list, assignment)), str(prop_clause_list.num_variables()), str(prop_clause_list.num_clauses()))
    #     sub_clause_list = clause_subsumption_new(prop_clause_list)
    #     clause_list = sub_clause_list
    #     logger.info("D: %s %s %s", str(validate_clauses_and_assignment(clause_list, assignment)), str(clause_list.num_variables()), str(clause_list.num_clauses()))

    if clause_list.num_variables() == 0:
        return b"none"
    if clause_list.num_variables() + clause_list.num_clauses() > num_solver_qubits - 1:
        return b"none"

    clause_list.fix_non_consecutive_vars()

    print_clause_stats(clause_list)
    sat_cnf = clause_list.to_dimacs()
    print(sat_cnf)
    with open('./dimacs.txt', 'w') as dimacs_file:
        dimacs_file.write(sat_cnf)

    with open('./assignment.txt', 'w') as assignment_file:
        for var, val in assignment.items():
            lit = var if val == AssignmentValue.TRUE else -var
            assignment_file.write("{} ".format(lit))

    num_iterations = 2**(clause_list.num_variables() - 1)
    sim_command = "../qq_sim/build/qq_sim ./dimacs.txt {} {} a a".format(num_iterations, num_solver_qubits)
    sim_command_output = subprocess.check_output(sim_command, shell=True).decode('unicode_escape')
    sim_output, sim_debug_output, _ = sim_command_output.split('\n')
    sim_output = sim_output.strip()

    sim_result = b"undef"
    if sim_output == "none":
        sim_result = b"none"
    else:
        remap_output = ""
        lits = [int(l) for l in sim_output.split(' ')]
        for lit in lits:
            var = int(clause_list.get_variable_name(abs(lit)).split('_')[1])
            sign = 1 if lit > 0 else -1
            remap_output += str(var*sign) + " "
        sim_result = bytes(remap_output, 'unicode_escape')
        with open('./experiment_info.dat', 'a') as experiment_info_file:
            experiment_info_file.write(sim_debug_output+"\n")
    return sim_result


callback_refs = []


def optimize_circ(goal: Goal, model_input):
    optimizer = setup_optimizer(goal)

    logger.debug("OPT CHECK: %s", optimizer.check())

    opt_model = optimizer.model()
    logger.debug(optimizer.statistics())
    logger.debug("OPT MODEL DONE")
    opt_dag, init_layout = construct_circuit_from_model(opt_model, model_input)
    opt_circ = dag_to_circuit(opt_dag)

    logger.debug(opt_circ.draw())


# TODO write test routines to ensure the output of the circuit is the same as for original.
def construct_model(input_circuit: QuantumCircuit, input_coupling_graph: CouplingMap,
                    solver_qubits: int = 0, max_swaps_addable=10, max_circuit_time=10):
    global num_solver_qubits
    num_solver_qubits = solver_qubits

    check_callback_ref = Z3_set_quantum_solver_check_capable_callback(check_solver_capable)
    run_callback_ref = Z3_set_quantum_solver_run_callback(run_solver)

    callback_refs.append(check_callback_ref)
    callback_refs.append(run_callback_ref)

    # For now, input circuit must have same qubits as coupling graph.
    # TODO: Need to change this to accept smaller input circuits.
    if input_circuit.n_qubits != input_coupling_graph.size():
        raise RuntimeError("Input circuit and coupling map should have same number of qubits.")

    logger.debug("Input circuit diagram: \n{}".format(input_circuit.draw()))
    logger.debug("Input circuit QASM: \n{}".format(input_circuit.qasm()))

    remap_circuit, qubits_used = remap_compact_virtual_qubit_registers(input_circuit, input_coupling_graph)
    logger.info("Circuit qubits remapped: Using %d qubits.", qubits_used)

    logger.debug("Remap circuit diagram: \n{}".format(remap_circuit.draw()))
    logger.debug("Remap circuit QASM: \n{}".format(remap_circuit.qasm()))

    logger.info("Generating constraint model.")
    goal, model_input = build_model(remap_circuit, input_coupling_graph, qubits_used,
                                    max_circuit_time=max_circuit_time,
                                    max_swaps_addable=max_swaps_addable)
    logger.info("Constraint model: %d constraints.", goal.size())

    logger.debug("KSAT PROCEDURE")
    # cnf_goal, cnf_sexpr, cnf_clause_list = transform_goal_to_cnf(goal, max_clause_size=3)

    cnf_goal_solver = SolverFor("QF_FD")
    cnf_goal_solver.add(goal)

    if cnf_goal_solver.check() == sat:
        model = cnf_goal_solver.model()

        stats = cnf_goal_solver.statistics()
        decisions = stats.get_key_value('sat decisions')
        with open('./experiment_info.dat', 'a') as experiment_info_file:
            experiment_info_file.write("SAT_Decisions: {}\n".format(decisions))

        opt_dag, init_layout = construct_circuit_from_model(model, model_input)
        opt_circ = dag_to_circuit(opt_dag)
        print(opt_circ)
        logger.info("Model done.")
        return opt_circ
    else:
        return None

# TODO: OLD model stuff.
#
# cnf_solver = SolverFor('QF_FD')
# cnf_solver.from_string(cnf_clause_list.to_dimacs())
# cnf_solver.set('completion', True)
# logger.info("CNF Solver: %s", str(cnf_solver.check()))

# with open("../output/sexpr_solver_sexpr2.txt", "w") as solver_file:
#     solver_file.write(cnf_sexpr)
# model_cnf_sexpr = drop_asserts_sexpr_file(cnf_sexpr.split('\n'))
# with open("../output/sexpr_solver_sexpr3.txt", "w") as solver_file:
#     solver_file.write(model_cnf_sexpr)
#
# sexpr_solver = Solver()
# sexpr_solver.from_string(model_cnf_sexpr)
#
# print(sexpr_solver.check())
# with open("../output/sexpr_solver_sexpr.txt", "w") as solver_file:
#     solver_file.write(sexpr_solver.model().sexpr())
#
# sexpr_result = parse_sexpr(model_cnf_sexpr)
# logger.debug("CNF solver check: %s", cnf_solver.check())

# hybrid_schoning_solver(cnf_clause_list)

# with open("../output/cnf_solver_sexpr.txt", "w") as smt2_fd_goal_file:
#     smt2_fd_goal_file.write(cnf_solver.sexpr())

# cnf_model = cnf_solver.model()
# with open("../output/sexpr_solver_sexpr6.txt", "w") as solver_file:
#     solver_file.write(cnf_model.sexpr())
# cnf_model_var_vals = {rename_var(decl.name()): bool(cnf_model[decl]) for decl in cnf_model.decls()}
# # print("CMVV", cnf_model_var_vals.keys())
# eval_order_list = build_variable_eval_order_list(sexpr_result)
# eval_dict = build_eval_dict(sexpr_result)
# code_block = build_eval_code_block(eval_order_list, eval_dict)
# var_values = run_eval_code_block(code_block, cnf_model_var_vals)
# # print(var_values)

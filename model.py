from typing import Optional, Tuple

from networkx.algorithms import hybrid
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions import SwapGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout

from qq.assignment import Assignment, unit_propagation, clause_subsumption, hybrid_schoning_solver
from qq.constraints import *
from qq.scratch.cube_and_conquer import cube_and_conquer
from qq.sexpr_parser import *
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
        logger.debug(con_func.__name__)
        logger.debug(constraint_goal)

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

    model_decls_map = {decl.name(): model[decl] for decl in model.decls()}

    logger.debug("NUM VARS: {}".format(len(model_decls_map)))

    times = {}
    for key, value in model_decls_map.items():
        if 'time' in key:
            times[key] = value.as_long()

    for key, value in sorted(times.items(), key=lambda a: a[0]):
        logger.debug("TIMES: {} {}".format(key, value))

    durs = {}
    for key, value in model_decls_map.items():
        if 'duration' in key:
            durs[key] = value.as_long()

    for key, value in sorted(durs.items(), key=lambda a: a[0]):
        logger.debug("DURS: {} {}".format(key, value))

    ins = {}
    for key, value in model_decls_map.items():
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

    # First fix the initial layout of circuit by setting the layout to the model's inferred input layout.
    for physical_qubit in canonical_register:
        input_qubit_map_decl_name = ("input_qubit_mapping__%d" % physical_qubit.index)
        input_virtual_qubit = model_decls_map[input_qubit_map_decl_name].as_long()

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
                    swap_gate_exists = is_true(model_decls_map[swap_gate_decl_name])

                    # Actually make a swap layer if a swap gate exists.
                    if swap_gate_exists:
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
                gate_virtual_qubit = model_decls_map[gate_qubit_map_decl_name].as_long()

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

    logger.debug('CIRC_END_TIME: {}'.format(model_decls_map['circuit_end_time']))
    logger.debug('SWAPS_ADDED: {}'.format(model_decls_map['swaps_added']))
    logger.debug('CIRC_DEPTH: {}'.format(new_dag.depth()))

    return new_dag, init_layout


# Turns a finite domain goal into a k-SAT goal.
def transform_goal_to_cnf(goal: Goal, max_clause_size: int = 3):
    # fd_solver = SolverFor('QF_FD')
    # fd_solver.set('max_conflicts', 0)
    # simplify_tactic = With('simplify',
    #                        som=True,
    #                        pull_cheap_ite=True,
    #                        push_ite_bv=False,
    #                        local_ctx=True,
    #                        local_ctx_limit=10_000_000,
    #                        flat=True,
    #                        hoist_mul=False,
    #                        elim_and=True,
    #                        blast_distinct=True)
    # fd_pre_process_tactic = Then('dt2bv', 'eq2bv', 'simplify', 'propagate-values', 'card2bv', simplify_tactic,
    #                              'max-bv-sharing',
    #                              'bit-blast', simplify_tactic, 'sat-preprocess')
    fd2_pre_process_tactic = Tactic('qffd')
    # fd2_pre_process_tactic = Then('propagate-values', 'qffd')

    fd_solver = Solver()
    fd_solver.add(*goal)
    with open("output/fd_solver_smt2.txt", "w") as smt_file:
        smt_file.write(fd_solver.to_smt2())

    fd_solver2 = fd2_pre_process_tactic.solver()
    # fd_solver2 = fd_solver2.translate(goal.ctx)
    fd_solver2.set('max_conflicts', 0)
    fd_solver2.add(*goal)

    # translated_fd_solver = fd_solver.translate(main_ctx())
    # translated_fd_goal = Goal()
    # translated_fd_goal.add(*translated_fd_solver.assertions())
    # preprocessed_fd_goal = fd_pre_process_tactic(translated_fd_goal)[0]
    #
    # logger.debug("PFG: {}".format(len(preprocessed_fd_goal)))
    # print_dimacs_stats(preprocessed_fd_goal.dimacs())
    logger.debug("FD2: {}".format(len(fd_solver2.assertions())))
    print_dimacs_stats(fd_solver2.dimacs())
    #
    # with open("output/preproc_file_goal_ fd_dimacs.txt", "w") as preproc_file_goal_dimacs_file:
    #     preproc_file_goal_dimacs_file.write(preprocessed_fd_goal.dimacs())

    # with open("output/preproc_file_goal_fd.txt", "w") as preproc_file_goal_fd_file:
    #     for assertion in tqdm.tqdm(preprocessed_fd_goal):
    #         preproc_file_goal_fd_file.write(str(assertion) + '\n')
    #         preproc_file_goal_fd_file.flush()

    # preprocessed_fd_solver = SolverFor('QF_FD')
    # # preprocessed_fd_solver.set('max_conflicts', 100000)
    # preprocessed_fd_solver.add(*preprocessed_fd_goal)
    #
    # with open("output/smt2_file_preproc_goal_fd.txt", "w") as smt2_fd_goal_file:
    #     smt2_fd_goal_file.write(fd_solver.to_smt2())
    #
    # logger.debug("TRANSFORM: {} {} {} {}".format(len(fd_solver.assertions()), len(translated_fd_solver.assertions()),
    #                                              len(preprocessed_fd_solver.assertions()),
    #                                              len(fd_solver2.assertions())))

    # check_solver_sat(fd_solver)
    check_solver_sat(fd_solver2)
    # check_solver_sat(preprocessed_fd_solver)

    # with open("output/fd_solver_dimacs.txt", "w") as goal_dimacs_file:
    #     goal_dimacs_file.write(fd_solver.dimacs())
    #
    # with open("output/fd_solver_sexpr.txt", "w") as smt2_fd_goal_file:
    #     smt2_fd_goal_file.write(fd_solver.sexpr())
    #
    # with open("output/fd_solver_smt2.txt", "w") as smt2_fd_goal_file:
    # #     smt2_fd_goal_file.write(fd_solver.to_smt2())
    with open("output/fd_solver2_dimacs.txt", "w") as goal_dimacs_file:
        goal_dimacs_file.write(fd_solver2.dimacs())
    # with open("output/fd_solver2_sexpr.txt", "w") as smt2_fd_goal_file:
    #     smt2_fd_goal_file.write(fd_solver2.sexpr())
    # with open("output/fd_solver2_smt2.txt", "w") as smt2_fd_goal_file:
    #     smt2_fd_goal_file.write(fd_solver2.to_smt2())

    logger.debug("WRT OUT")

    # logger.debug("SAT ASSERTS:".format(len(fd_solver.assertions()), len(preprocessed_fd_solver.assertions()),
    #                                    len(fd_solver2.assertions())))

    logger.debug("FD_STATS2:")
    print_dimacs_stats(fd_solver2.dimacs(), extended_info=True)
    logger.debug(fd_solver2.statistics())

    sat_preproc_goal = Goal()
    # sat_preproc_goal.append(fd_solver.assertions())
    sat_preproc_goal.append(fd_solver2.assertions())
    with open("output/sat_preproc_goal_sepxr.txt", "w") as goal_dimacs_file:
        goal_dimacs_file.write(sat_preproc_goal.sexpr())

    # preproc_goal = strip_unit_clauses(sat_preproc_goal)
    # kcnf_sat_goal = kcnf_split_clauses(sat_preproc_goal, max_clause_length=max_clause_size)
    # kcnf_sat_goal = split_expr_clauses(sat_preproc_goal, max_clause_length=max_clause_size)
    sat_preproc_clause_list = ClauseList()
    sat_preproc_clause_list.load_from_dimacs(sat_preproc_goal.dimacs())
    kcnf_clause_list = split_clauses_in_clause_list(sat_preproc_clause_list, max_clause_length=max_clause_size)

    logger.debug("KCNF_STATS:")
    print_clause_stats(kcnf_clause_list)
    #print_dimacs_stats(kcnf_sat_goal.dimacs(), extended_info=True)
    # print(kcnf_sat_goal)

    kcnf_sat_sexpr = fd_solver2.sexpr()
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


# TODO write test routines to ensure the output of the circuit is the same as for original.
def construct_model(input_circuit: QuantumCircuit, input_coupling_graph: CouplingMap):
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
    max_swaps_addable = 10
    max_circuit_time = 10
    goal, model_input = build_model(remap_circuit, input_coupling_graph, qubits_used,
                                    max_circuit_time=max_circuit_time,
                                    max_swaps_addable=max_swaps_addable)
    logger.info("Constraint model: %d constraints.", goal.size())

    optimizer = setup_optimizer(goal)
    # solver = SolverFor('QF_FD')
    # solver.append(*goal)
    # solver.set('lookahead.cube.depth', 1)
    # cube_and_conquer(solver)

    for i in range(0):
        logger.debug("OPT CHECK: %s", optimizer.check())

        opt_model = optimizer.model()
        logger.debug(optimizer.statistics())
        logger.debug("OPT MODEL DONE")
        opt_dag, init_layout = construct_circuit_from_model(opt_model, model_input)
        opt_circ = dag_to_circuit(opt_dag)

        logger.debug(opt_circ.draw())

    logger.debug("KSAT PROCEDURE")
    cnf_goal, cnf_sexpr, cnf_clause_list = transform_goal_to_cnf(goal, max_clause_size=3)

    cnf_solver = Solver()
    cnf_solver.from_string(cnf_clause_list.to_dimacs())
    # cnf_goal2 = Goal()
    # cnf_goal2.append(*cnf_solver.assertions())
    # print("PROBE1:", Probe('num-bool-consts')(cnf_goal2))
    # partial_solve_tactic = Repeat(
    #     Cond(Probe('num-bool-consts') > 10,
    #          With('qffd', max_conflicts=100),
    #          Tactic('skip')), 1000)
    # res_goal = partial_solve_tactic.apply(cnf_goal2)
    # print("PROBE2:", Probe('num-bool-consts')(res_goal[0]))
    # print("PST", res_goal)

    cube_solver = Solver()
    cube_solver.add(cnf_solver.assertions())
    cube_solver.set("sat.restart.max", 100)
    cube_solver.set("lookahead.cube.depth", 4)

    cube_and_conquer(cube_solver)

    model_cnf_sexpr = drop_asserts_sexpr_file(cnf_sexpr.split('\n'))

    # TODO: Stop. Investigate further. Consider: solver.enforce_model_conversion
    sexpr_solver = Solver()
    sexpr_solver.from_string(model_cnf_sexpr)
    print(sexpr_solver.check())
    print(sexpr_solver.model().decls())
    with open("output/sexpr_solver_sexpr.txt", "w") as solver_file:
        solver_file.write(sexpr_solver.model().sexpr())

    sexpr_result = parse_sexpr(model_cnf_sexpr)
    logger.debug("CNF solver check: %s", cnf_solver.check())

    hybrid_schoning_solver(cnf_clause_list)

    with open("output/cnf_solver_sexpr.txt", "w") as smt2_fd_goal_file:
        smt2_fd_goal_file.write(cnf_solver.sexpr())

    cnf_model = cnf_solver.model()
    cnf_model_var_vals = {rename_var(decl.name()): bool(cnf_model[decl]) for decl in cnf_model.decls()}
    eval_order_list = build_variable_eval_order_list(sexpr_result)
    eval_dict = build_eval_dict(sexpr_result)
    code_block = build_eval_code_block(eval_order_list, eval_dict)
    run_eval_code_block(code_block, cnf_model_var_vals)

    cnf_dimacs = cnf_goal.dimacs()
    cnf_clauses = dimacs_cnf_to_clauses(cnf_dimacs)
    literal_varname_map = dimacs_cnf_to_literal_varname_map(cnf_dimacs)
    logger.debug("CLAUSES 0:")
    print_clause_stats(cnf_clauses)
    assignment = Assignment()
    cnf_clauses = unit_propagation(cnf_clauses, assignment)

    logger.debug("CLAUSES 1:")
    logger.debug("Assn: %d", len(assignment))
    print_clause_stats(cnf_clauses)

    cnf_clauses = clause_subsumption(cnf_clauses)
    logger.debug("CLAUSES 2:")
    logger.debug("Assn: %d", len(assignment))
    print_clause_stats(cnf_clauses, extended_info=True)

    solved_assignment = quantum_kcnf_solver(cnf_goal)

    logger.info("Model done.")


# TODO: Cleanup and refactor to new file.
def construct_grovers_and_oracles(cnf_goal: Goal()):
    # expr_val = dimacs_cnf_to_expression(ksat_dimacs)
    # parse_expr(expr_val, evaluate=True)

    # TODO: Separate into method for sim / translation.
    # backend = Aer.get_backend('qasm_simulator')
    ksat_dimacs = cnf_goal.dimacs()
    cnf_ast = dimacs_cnf_to_cnf_ast(ksat_dimacs)
    num_vars = get_num_variables(ksat_dimacs)
    # oracles = build_oracles(cnf_ast, num_vars)
    oracle = initialize_oracle(cnf_ast, num_vars)
    oracle.construct_circuit()

    oracle_circuit = oracle.circuit
    with open('./output/oracle_instrs.qasm', 'w') as oracle_file:
        oracle_qasm_content = oracle_circuit.qasm()
        oracle_file.write(oracle_qasm_content)
    # dag = fast_circuit_to_dag(oracle.circuit)
    # print("D")
    # dag_drawer(dag, filename='oracle.png')

    # circuit_drawer(oracle.circuit, filename='circ.png', output='latex')

    # parse_expr(cnf_expr, evaluate=False)
    # oracle = LogicalExpressionOracle(cnf_expr)  # , optimization=True)
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
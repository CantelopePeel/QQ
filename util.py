import logging
from collections import deque, Counter
from functools import reduce
from typing import List, Dict

from qiskit.aqua import AquaError
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap

# Generates couplings in the coupling map (which is directed) that are undirected. Useful for determining if we can
# perform a swap on two qubits.
from qq.assignment import Clause
from qq.clause import *
from qq.parallel_cnf_circuit import CNFParallel

logger = logging.getLogger(__name__)


def get_undirected_couplings(coupling_map: CouplingMap):
    couplings = coupling_map.get_edges()
    for coupling in couplings:
        if coupling[0] < coupling[1] and (coupling[1], coupling[0]) in couplings:
            yield coupling


# Split the clause into a number of sections.
def split_clause(clause: ExprRef, sub_clause_size: int):
    if 'auxiliary_var_count' not in split_clause.__dict__:
        split_clause.auxiliary_var_count = 0

    literals = clause.children()

    aux_literal = Bool('aux!{}'.format(split_clause.auxiliary_var_count))
    num_literals = len(literals)
    num_normal_literals = sub_clause_size - 1
    split_clause.auxiliary_var_count += 1

    split_clauses = []
    # Set up the auxiliary literal for first sub clause manually.
    neg_aux_literal = Not(aux_literal)
    sub_clause = Or([neg_aux_literal] + literals[0:num_normal_literals])
    split_clauses.append(sub_clause)

    for i in range(1, num_literals, num_normal_literals):
        split_clauses.append(Or([aux_literal] + literals[i:(i + num_normal_literals)]))
    return split_clauses


def dimacs_cnf_to_expression(dimacs):
    lines = [
        ll for ll in [
            l.strip().lower() for l in dimacs.strip().split('\n')
        ] if len(ll) > 0 and not ll[0] == 'c'
    ]

    if not lines[0][:6] == 'p cnf ':
        raise AquaError('Unrecognized dimacs cnf header {}.'.format(lines[0]))

    def create_var(cnf_tok):
        return ('~v' + cnf_tok[1:]) if cnf_tok[0] == '-' else ('v' + cnf_tok)

    clause_queue = deque()

    for line in tqdm.tqdm(lines[1:]):
        toks = line.split()
        if not toks[-1] == '0':
            raise AquaError('Unrecognized dimacs line {}.'.format(line))
        else:
            clause_queue.append('({})'.format(' | '.join(
                [create_var(t) for t in toks[:-1]]
            )))

    while len(clause_queue) != 1:
        l_clause = clause_queue.popleft()
        r_clause = clause_queue.popleft()
        clause = '({} & {})'.format(l_clause, r_clause)
        clause_queue.append(clause)

    return clause_queue.popleft()


def dimacs_cnf_to_cnf_ast(dimacs):
    lines = [
        ll for ll in [
            l.strip().lower() for l in dimacs.strip().split('\n')
        ] if len(ll) > 0 and not ll[0] == 'c'
    ]

    # for line_partition in partition_lines(lines[1:], partitions):
    clauses = []
    for line in lines[1:]:
        toks = line.split()
        if not toks[-1] == '0':
            raise AquaError('Unrecognized dimacs line {}.'.format(line))
        else:
            clauses.append([int(t) for t in toks[:-1]])

    clause_asts = []
    for clause in clauses:
        ast_lit_nodes = [('lit', literal) for literal in clause]
        ast_clause_node = ('or', *ast_lit_nodes)
        clause_asts.append(ast_clause_node)

    cnf_ast = ('and', *clause_asts)
    return cnf_ast


def dimacs_cnf_to_clauses(dimacs: str) -> List[Clause]:
    lines = [
        ll for ll in [
            l.strip().lower() for l in dimacs.strip().split('\n')
        ] if len(ll) > 0 and not ll[0] == 'c'
    ]

    clauses = []
    for line in lines[1:]:
        toks = line.split()
        if not toks[-1] == '0':
            raise AquaError('Unrecognized dimacs line {}.'.format(line))
        else:
            clauses.append(Clause({int(t) for t in toks[:-1]}))

    return clauses


def dimacs_cnf_to_literal_varname_map(dimacs: str) -> Dict[int, str]:
    lines = [
        ll for ll in [
            l.strip().lower() for l in dimacs.strip().split('\n')
        ] if len(ll) > 0 and ll[0] == 'c'
    ]

    literal_varname_map = dict()
    for line in lines[1:]:
        toks = line.split()
        if not len(toks) == 3:
            raise AquaError('Unrecognized dimacs line {}.'.format(line))
        else:
            literal = int(toks[1])
            varname = Bool(toks[2])
            literal_varname_map[literal] = varname
    return literal_varname_map


def get_num_variables(cnf_content):
    cnf_lines = cnf_content.splitlines()
    program_lines = list(filter(lambda l: l.startswith('p'), cnf_lines))
    program_info = program_lines[0].split(' ')
    num_variables = int(program_info[2])
    return num_variables


def initialize_oracle(ast, num_vars):
    oracle = LogicalExpressionOracle("v0 & v1", mct_mode='basic')
    oracle._circuit = None
    oracle._nf = CNFParallel(ast, num_vars=num_vars)
    return oracle


# Helper form DIMACS stats function.
def combine_counts(counts0: Dict, counts1: Dict):
    for key0 in counts0.keys():
        if key0 in counts1:
            counts1[key0] = counts0[key0] + counts1[key0]
        else:
            counts1[key0] = counts0[key0]
    return counts1


def print_clause_stats(clause_list: ClauseList, extended_info: bool = False):
    stats_content = ""
    all_variables = {abs(literal) for clause in clause_list for literal in clause}
    num_variables = len(all_variables)
    num_clauses = len(clause_list)
    stats_content += "Clauses info:\n"
    stats_content += ("  Num variables: {}\n".format(num_variables))
    stats_content += ("  Num clauses: {}\n".format(num_clauses))

    clause_values = []
    for clause in clause_list:
        clause_value = [int(literal) for literal in clause]
        clause_values.append(clause_value)

    # Clause length.
    clause_lengths = [len(clause_value) for clause_value in clause_values]
    clause_lengths_agg_map = map(lambda l: {l: 1}, clause_lengths)
    clause_lengths_agg_reduce = reduce(lambda l0, l1: combine_counts(l0, l1), clause_lengths_agg_map)

    avg_clause_length = sum(clause_lengths) / len(clause_lengths)
    min_clause_length = min(clause_lengths)
    max_clause_length = max(clause_lengths)
    stats_content += ("  Avg. clause len: {}\n".format(avg_clause_length))
    stats_content += ("  Min. clause len: {}\n".format(min_clause_length))
    stats_content += ("  Max. clause len: {}\n".format(max_clause_length))

    if extended_info:
        for i in range(min_clause_length, max_clause_length + 1):
            if i in clause_lengths_agg_reduce.keys():
                num_clauses = clause_lengths_agg_reduce[i]
                stats_content += ("  Len {} clauses: {}\n".format(i, num_clauses))

    # Variable occurrence.
    flat_clause_values = [abs(value) for clause in clause_values for value in clause]
    variable_values = map(lambda v: abs(v), flat_clause_values)
    variable_occurrences_map = map(lambda v: {v: 1}, variable_values)
    # variable_occurrences_reduce = reduce(lambda v0, v1: combine_counts(v0, v1), variable_occurrences_map, {})
    variable_occurrences_dict = dict(Counter(flat_clause_values))
    occurrence_agg_counts = dict(Counter(variable_occurrences_dict.values()))
    avg_var_occ = sum(variable_occurrences_dict.values()) / num_variables
    min_var_occ = min(variable_occurrences_dict.values())
    max_var_occ = max(variable_occurrences_dict.values())
    stats_content += ("  Avg. vars occ: {}\n".format(avg_var_occ))
    stats_content += ("  Min. vars occ: {}\n".format(min_var_occ))
    stats_content += ("  Max. vars occ: {}\n".format(max_var_occ))

    if extended_info:
        for i in range(min_var_occ, max_var_occ + 1):
            if i in occurrence_agg_counts.keys():
                num_clauses = occurrence_agg_counts[i]
                stats_content += ("  Occ {} vars: {}\n".format(i, num_clauses))
    logger.debug(stats_content[:-1])


# Print info about DIMACS CNF representation of the model.
def print_dimacs_stats(cnf_content: str, extended_info: bool = False):
    stats_content = ""
    cnf_lines = cnf_content.splitlines()
    comment_lines = list(filter(lambda l: l.startswith('c'), cnf_lines))
    clause_lines = list(filter(lambda l: not (l.startswith('c') or l.startswith('p')), cnf_lines))
    program_lines = list(filter(lambda l: l.startswith('p'), cnf_lines))

    program_info = program_lines[0].split(' ')
    num_variables = int(program_info[2])
    num_clauses = int(program_info[3])
    stats_content += ("Dimacs info:\n")
    stats_content += ("  Num variables: {}\n".format(num_variables))
    stats_content += ("  Num clauses: {} ({} lines)\n".format(num_clauses, len(clause_lines)))
    stats_content += ("  Num comments: {}\n".format(len(comment_lines)))

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
    stats_content += ("  Avg. clause len: {}\n".format(avg_clause_length))
    stats_content += ("  Min. clause len: {}\n".format(min_clause_length))
    stats_content += ("  Max. clause len: {}\n".format(max_clause_length))

    if extended_info:
        for i in range(min_clause_length, max_clause_length + 1):
            if i in clause_lengths_agg_reduce.keys():
                num_clauses = clause_lengths_agg_reduce[i]
                stats_content += ("  Len {} clauses: {}\n".format(i, num_clauses))

    # Variable occurrence.
    flat_clause_values = [abs(value) for clause in clause_values for value in clause]
    variable_values = map(lambda v: abs(v), flat_clause_values)
    variable_occurrences_dict = dict(Counter(flat_clause_values))
    occurrence_agg_counts = dict(Counter(variable_occurrences_dict.values()))
    avg_var_occ = sum(variable_occurrences_dict.values()) / num_variables
    min_var_occ = min(variable_occurrences_dict.values())
    max_var_occ = max(variable_occurrences_dict.values())
    stats_content += ("  Avg. vars occ: {}\n".format(avg_var_occ))
    stats_content += ("  Min. vars occ: {}\n".format(min_var_occ))
    stats_content += ("  Max. vars occ: {}\n".format(max_var_occ))

    if extended_info:
        for i in range(min_var_occ, max_var_occ + 1):
            if i in occurrence_agg_counts.keys():
                num_clauses = occurrence_agg_counts[i]
                stats_content += ("  Occ {} vars: {}\n".format(i, num_clauses))
    logger.debug(stats_content[:-1])


def topological_gate_nodes(circuit: DAGCircuit):
    for node in circuit.topological_op_nodes():
        if isinstance(node.op, Gate):
            yield node


def check_solver_sat(solver: Solver):
    solver_check = solver.check()
    if solver_check == CheckSatResult(Z3_L_TRUE):  # If sat
        logger.debug("SAT Check: %s", sat)
    elif solver == CheckSatResult(Z3_L_FALSE):  # If unsat
        logger.debug("SAT Check: %s", sat)
        print_unsat(solver)
    else:
        logger.debug("SAT Check: Unknown Reason: %s", solver.reason_unknown())


# Print useful debug info when solver returns an unsat result.
def print_unsat(solver: Solver):
    unsat_solver = Solver()

    assertion_map = {}
    for index, assertion in enumerate(solver.assertions()):
        assert_val = Bool('assert_%d' % index)
        unsat_solver.assert_and_track(assertion, assert_val)
        assertion_map[assert_val] = assertion

    unsat_solver.check()

    logger.debug("Unsatisfiable core: \n", unsat_solver.unsat_core())
    unsat_core = unsat_solver.unsat_core()
    for i in range(len(unsat_core)):
        logger.debug("Conflit: \n {} :\n {}".format(unsat_core[i], assertion_map[unsat_core[i]]))

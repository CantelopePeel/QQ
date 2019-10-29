import random

import tqdm as tqdm
from qiskit.transpiler import CouplingMap
from z3 import *


# Generates couplings in the coupling map (which is directed) that are undirected. Useful for determining if we can
# perform a swap on two qubits.
def get_undirected_couplings(coupling_map: CouplingMap):
    couplings = coupling_map.get_edges()
    for coupling in couplings:
        if coupling[0] < coupling[1] and (coupling[1], coupling[0]) in couplings:
            yield coupling


def is_literal(literal: ExprRef):
    if is_not(literal):
        return is_const(literal.arg(0))
    elif is_const(literal):
        return True
    else:
        return False


def is_clause(clause: ExprRef):
    if is_or(clause):
        for literal in clause.children():
            if not is_literal(literal):
                return False
        return True
    elif is_literal(clause):
        return True
    else:
        return False


def clause_length(clause: ExprRef):
    # TODO: Should we even check if we do already?
    # if not is_clause(clause):
    #     return -1 # TODO: raise exception.

    if is_or(clause):
        return clause.num_args()
    elif is_literal(clause):
        return 1
    else:
        return -1  # TODO: raise exception. Should never reach here?


# Split the clause into a number of sections.
def split_clause(clause: ExprRef, sub_clause_size: int):
    literals = clause.children()

    aux_literal = FreshBool(prefix='aux')
    num_literals = len(literals)
    num_normal_literals = sub_clause_size - 1

    split_clauses = []
    # Set up the auxiliary literal for first sub clause manually.
    neg_aux_literal = Not(aux_literal)
    sub_clause = Or([neg_aux_literal] + literals[0:num_normal_literals])
    split_clauses.append(sub_clause)

    for i in range(1, num_literals, num_normal_literals):
        split_clauses.append(Or([aux_literal] + literals[i:(i + num_normal_literals)]))
    return split_clauses


# Split a CNF expression until it is k-CNF. This means that every clause contains at most `k' literals.
# TODO: Use a priority queue (heap) to do this faster
# TODO maybe do not split in half, split to the max_length
def kcnf_split_clauses(goal: Goal, max_clause_length: int = 3):
    if max_clause_length < 3:
        return  # TODO: raise exception

    kcnf_goal = Goal()
    mod_count = 0
    num_clauses = len(goal)
    for i in tqdm.tqdm(range(num_clauses)):
        clause = goal.get(i)
        if not is_clause(clause):
            return  # TODO: raise exception

        if clause_length(clause) > max_clause_length:
            mod_count += 1
            split_clauses = split_clause(clause, sub_clause_size=max_clause_length)
            kcnf_goal.append(*split_clauses)
        else:
            kcnf_goal.add(clause)
    print(mod_count)
    return kcnf_goal


def strip_unit_clauses(goal: Goal):
    stripped_goal = Goal()
    for clause in tqdm.tqdm(goal):
        if clause_length(clause) > 1:
            stripped_goal.add(clause)
    return stripped_goal



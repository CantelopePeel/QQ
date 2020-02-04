import logging
import random
from collections import defaultdict, deque, namedtuple
from time import time
from enum import Enum
from typing import Dict, NewType, Optional

import tqdm

from qq.clause import ClauseList, Clause

logger = logging.getLogger(__name__)


class AssignmentValue(Enum):
    FALSE = 0
    TRUE = 1
    UNDEFINED = 2


# def Assignment(init_values=None):
#     if init_values is None:
#         init_values = {}
#     assignment_type = NewType('Assignment', Dict[int, AssignmentValue])
#     return assignment_type(init_values)

def Assignment(init_values=None):
    if init_values is None:
        init_values = {}
    assignment_type = NewType('Assignment', Dict[int, bool])
    return assignment_type(init_values)

Frame = namedtuple('Frame', ['clause_list', 'assignment'])


def clause_subsumption(clause_list: ClauseList):
    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list)}

    # Build the literal-clause lists.
    literal_map_lists = defaultdict(set)
    for clause_idx, clause in clause_dict.items():
        for literal in clause:
            literal_map_lists[literal].add(clause_idx)

    num_subsumed = 0
    subsumed_clauses = set()
    for clause_idx0 in tqdm.trange(len(clause_list), desc="Gathering Subsumed Clauses"):
        if clause_idx0 in subsumed_clauses:
            continue

        clause0 = clause_list[clause_idx0].copy()
        clause_idx_set = literal_map_lists[clause0.pop()]
        clause_idx_set -= subsumed_clauses

        while len(clause0) != 0:
            clause_idx_set &= literal_map_lists[clause0.pop()]
            if len(clause_idx_set) == 0:
                break
        else:
            subsumed_clauses |= clause_idx_set
            num_subsumed += len(clause_idx_set)

    logger.debug("Subsumed clauses: {}".format(num_subsumed))
    for clause_idx in subsumed_clauses:
        clause_dict.pop(clause_idx)

    non_subsumed_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            non_subsumed_clauses.add_clause(clause_dict[idx])

    return non_subsumed_clauses


def unit_propagation(clause_list: ClauseList, assignment: Assignment):
    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list)}

    # Build the occurrence lists.
    unit_clause_stack = deque()
    occurrence_lists = defaultdict(set)
    for clause_idx, clause in clause_dict.items():
        if len(clause) == 1:
            unit_clause_stack.append(clause_idx)
        for literal in clause:
            variable = abs(literal)
            occurrence_lists[clause_idx].add(variable)

    # Perform unit propagation.
    while len(unit_clause_stack) != 0:
        unit_clause_idx = unit_clause_stack.popleft()
        unit_clause = clause_dict[unit_clause_idx]
        unit_literal = unit_clause.pop()
        unit_variable = abs(unit_literal)

        variable_occurrences = occurrence_lists[unit_variable]
        for occurrence_clause_idx in variable_occurrences:
            occurrence_clause = clause_dict[occurrence_clause_idx]
            if unit_literal in occurrence_clause:
                # The literal is the same as the one in this clause. Remove the clause.
                clause_dict.pop(occurrence_clause_idx)
            elif -unit_literal in occurrence_clause:
                # The literal is the negation of the one in this clause. Remove the literal from the clause.
                clause_dict[occurrence_clause_idx].remove(-unit_literal)
                # Clauses is now a unit clause. Add it to the queue of unit clauses to propagate
                if len(clause_dict[occurrence_clause_idx]) == 1:
                    unit_clause_stack.append(occurrence_clause_idx)
            # Set the value of the literal in the assignment.
            if unit_literal > 0:
                assignment[unit_variable] = AssignmentValue.TRUE
            else:
                assignment[unit_variable] = AssignmentValue.FALSE
        # Remove the unit clause.
        clause_dict.pop(unit_clause_idx)

    propagated_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            propagated_clauses.add_clause(clause_dict[idx])

    return propagated_clauses, assignment


def check_clauses_satisfied(clause_list: ClauseList, assignment: Assignment) -> Optional[Clause]:
    for clause in clause_list:

        for literal in clause:
            variable = abs(literal)
            if assignment[variable] and literal > 0:
                break
            elif not assignment[variable] and literal < 0:
                break
        else:  # No satisfied literals found.
            return clause
    return None


def count_clauses_satisfied(clause_list: ClauseList, assignment: Assignment) -> Optional[Clause]:
    sat_clauses = 0
    for clause in clause_list:
        for literal in clause:
            variable = abs(literal)
            if assignment[variable] and literal > 0:
                sat_clauses += 1
                break
            elif not assignment[variable] and literal < 0:
                sat_clauses += 1
                break
    return sat_clauses


def set_pbar(pbar, desc_int, val):
    pbar.n = val
    pbar.last_print_n = val
    pbar.set_description(str(desc_int))
    pbar.update()

def ball_search(clause_list: ClauseList, assignment: Assignment, radius: int):
    if check_clauses_satisfied(clause_list, assignment) is None:
        return True
    if radius <= 0:
        return False


def hybrid_schoning_solver(clause_list: ClauseList) -> Optional[Assignment]:
    random.seed(0)
    unit_clause_list, unit_assignment = unit_propagation(clause_list, Assignment({}))
    unit_clause_list = clause_subsumption(unit_clause_list)
    num_clauses = unit_clause_list.num_clauses()
    print(num_clauses)

    while True:
        assignment = Assignment({var_num: bool(random.getrandbits(1)) for var_num in unit_clause_list.variables()})
        pbar = tqdm.tqdm(total=num_clauses)
        for i in range(3*unit_clause_list.num_variables()):
            if i % 100 == 0:
                count_sat = count_clauses_satisfied(unit_clause_list, assignment)
                set_pbar(pbar, i, count_sat)

            unsat_clause = check_clauses_satisfied(unit_clause_list, assignment)
            if unsat_clause is None:
                return assignment
            unsat_clause_variable = abs(random.sample(unsat_clause, 1)[0])
            assignment[unsat_clause_variable] = not assignment[unsat_clause_variable]


def hybrid_dpll_backtracking(clause_list: ClauseList):
    pass
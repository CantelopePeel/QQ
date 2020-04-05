import copy
import logging
import random
from collections import defaultdict, deque, namedtuple
from time import time
from enum import Enum
from typing import Dict, NewType, Optional, Set

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

def validate_clauses_and_assignment(clause_list: ClauseList, assignment: Assignment):
    for clause_index, clause in enumerate(clause_list.clauses()):
        for lit in clause:
            var = abs(lit)
            if var in assignment:
                logger.warning("Clause list and assignment validation failed: %d %s | %d %s", clause_index, str(clause),
                               var, str(assignment[var]))

                return False
    return True


def clause_subsumption_new(clause_list: ClauseList, show_progress: bool = False):
    def subset_index_lookup(clause):
        lookup_set = set()
        counter = 0
        for literal in clause:
            if counter == 0:
                lookup_set = clause_index[literal]
            else:
                lookup_set.intersection_update(clause_index[literal])
            counter += 1
        return lookup_set

    def superset_index_lookup(clause):
        lookup_set = set()
        counter = 0
        for literal in clause:
            if counter == 0:
                lookup_set = clause_index[literal]
            else:
                lookup_set.update(clause_index[literal])
            counter += 1
        return lookup_set

    result_clause_set = set()
    clause_index = defaultdict(set)
    clause_list.clauses()

    for new_clause in tqdm.tqdm(set(frozenset(clause) for clause in clause_list.clauses())):
        # print(new_clause, len(new_clause))
        was_forward_subsumed = False
        # Forward subsumption.
        for exist_clause in superset_index_lookup(new_clause):
            # if len(exist_clause) < new_clause_len:
            if new_clause.issuperset(exist_clause):
                was_forward_subsumed = True
                break

        # Backward subsumption.
        remove_clause_list = list(subset_index_lookup(new_clause))
        for remove_clause in remove_clause_list:
            result_clause_set.discard(remove_clause)
            for literal in remove_clause:
                clause_index[literal].discard(remove_clause)

        # Add current clause to result and index.
        if not was_forward_subsumed:
            result_clause_set.add(new_clause)
            for literal in new_clause:
                clause_index[literal].add(new_clause)

    non_subsumed_clauses = ClauseList()
    for clause in result_clause_set:
        non_subsumed_clauses.add_clause(clause, skip_literal_loop=True)
    non_subsumed_clauses.bulk_add_variable_names()
    return non_subsumed_clauses


def clause_subsumption(clause_list: ClauseList, show_progress: bool = False):
    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list)}

    # Build the literal->clause lists.
    literal_map_lists = defaultdict(set)
    for clause_idx, clause in clause_dict.items():
        for literal in clause:
            literal_map_lists[literal].add(clause_idx)

    num_subsumed = 0
    subsumed_clauses = set()
    for clause_idx0 in tqdm.trange(len(clause_list), desc="Gathering Subsumed Clauses", disable=not show_progress):
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

    for clause_idx in subsumed_clauses:
        clause_dict[clause_idx] = None

    non_subsumed_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            if clause_dict[idx] is not None:
                non_subsumed_clauses.add_clause(clause_dict[idx], skip_literal_loop=True)
    non_subsumed_clauses.bulk_add_variable_names()

    logger.debug("Subsumed clauses: {} Remaining: {}".format(num_subsumed, non_subsumed_clauses.num_clauses()))
    return non_subsumed_clauses

def special_limited_unit_propagation(clause_list: ClauseList, assignment: Assignment, num_levels: int = -1):
    # No-op if number of propagation levels is zero.
    if num_levels == 0:
        return clause_list, assignment

    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list.clauses())}

    # Build the occurrence lists.
    unit_clause_queue = deque()
    occurrence_lists: Dict[int, Set[int]] = defaultdict(set)
    for clause_idx, clause in clause_dict.items():
        if len(clause) == 1:
            clause_level = 0
            if num_levels < 0 or clause_level < num_levels:
                unit_clause_queue.append((clause_idx, clause_level))
            # print(clause_idx, clause_dict[clause_idx])
        for literal in clause:
            variable = abs(literal)
            occurrence_lists[variable].add(clause_idx)

    num_units_propagated = 0
    num_clauses_elim = 0
    num_neg_lits_elim = 0
    max_prop_level = 0

    num_remaining_vars = clause_list.num_variables()
    num_remaining_clauses = clause_list.num_clauses()

    # Perform unit propagation.
    while len(unit_clause_queue) != 0:
        unit_clause_idx, unit_clause_level = unit_clause_queue.popleft()
        max_prop_level = max(max_prop_level, unit_clause_level)
        if clause_dict[unit_clause_idx] is None:
            continue
        if len(clause_dict[unit_clause_idx]) == 0:
            continue

        unit_literal = clause_dict[unit_clause_idx].pop()
        clause_dict[unit_clause_idx].add(unit_literal)

        unit_variable = abs(unit_literal)

        # Iterate over each clause where the unit variable appears.
        variable_occurrences = occurrence_lists[unit_variable]
        removed_clause_occurences = []
        for occurrence_clause_idx in variable_occurrences:
            occurrence_clause = clause_dict[occurrence_clause_idx]

            if occurrence_clause is None:
                continue
            if unit_literal in occurrence_clause:
                num_clauses_elim += 1
                # The literal is the same as the one in this clause. Remove the clause.
                clause_dict[occurrence_clause_idx] = None
            elif -unit_literal in occurrence_clause:
                num_neg_lits_elim += 1
                # The literal is the negation of the one in this clause. Remove the literal from the clause.
                clause_dict[occurrence_clause_idx].remove(-unit_literal)
                if len(clause_dict[occurrence_clause_idx]) == 0:
                    return None
                # Clauses is now a unit clause. Add it to the queue of unit clauses to propagate
                if len(clause_dict[occurrence_clause_idx]) == 1:
                    occurrence_clause_level = unit_clause_level + 1
                    if num_levels < 0 or occurrence_clause_level < num_levels:
                        unit_clause_queue.append((occurrence_clause_idx, occurrence_clause_level))

        # Set the value of the literal in the assignment.
        if unit_literal > 0:
            assignment[unit_variable] = AssignmentValue.TRUE
        else:
            assignment[unit_variable] = AssignmentValue.FALSE
        num_units_propagated += 1
        # Remove the unit clause.
        clause_dict[unit_clause_idx] = None

    propagated_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            if clause_dict[idx] is not None:
                propagated_clauses.add_clause(clause_dict[idx], make_names=True)
    logger.debug("Propagated units: {} Remaining clauses: {} | Max Prop Level: {} | Elims: C: {} NL: {}".format(
        num_units_propagated, propagated_clauses.num_clauses(), max_prop_level, num_clauses_elim, num_neg_lits_elim))

    return propagated_clauses, assignment


def unit_propagation(clause_list: ClauseList, assignment: Assignment, num_levels: int = -1,
                     limit_remaining_items: int = -1):
    # No-op if number of propagation levels is zero.
    if num_levels == 0:
        return clause_list, assignment

    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list.clauses())}

    # Build the occurrence lists.
    unit_clause_queue = deque()
    occurrence_lists: Dict[int, Set[int]] = defaultdict(set)
    # var_occurrences_counts = defaultdict(int)

    for clause_idx, clause in clause_dict.items():
        if len(clause) == 1:
            clause_level = 0
            if num_levels < 0 or clause_level < num_levels:
                unit_clause_queue.append((clause_idx, clause_level))

        for literal in clause:
            variable = abs(literal)
            occurrence_lists[variable].add(clause_idx)
            # var_occurrences_counts[variable] += 1

    num_units_propagated = 0
    num_clauses_elim = 0
    num_neg_lits_elim = 0
    max_prop_level = 0

    # num_remaining_vars = clause_list.num_variables()
    # num_remaining_clauses = clause_list.num_clauses()

    # Subtract non-occurring variables.
    # for var in clause_list.variables():
    #     if var not in occurrence_lists or len(occurrence_lists[var]) == 0:
    #         num_remaining_vars -= 1

    # Perform unit propagation.
    while len(unit_clause_queue) != 0:
        unit_clause_idx, unit_clause_level = unit_clause_queue.popleft()
        max_prop_level = max(max_prop_level, unit_clause_level)
        if clause_dict[unit_clause_idx] is None:
            continue
        if len(clause_dict[unit_clause_idx]) == 0:
            continue

        unit_literal = clause_dict[unit_clause_idx].pop()
        clause_dict[unit_clause_idx].add(unit_literal)

        unit_variable = abs(unit_literal)

        # Iterate over each clause where the unit variable appears.
        variable_occurrences = occurrence_lists[unit_variable]
        for occurrence_clause_idx in variable_occurrences:
            occurrence_clause = clause_dict[occurrence_clause_idx]

            if occurrence_clause is None:
                continue
            elif unit_literal in occurrence_clause:
                num_clauses_elim += 1
                # The literal is the same as the one in this clause. Remove the clause.
                # for lit in clause_dict[occurrence_clause_idx]:
                #     var = abs(lit)
                #     var_occurrences_counts[var] -= 1
                #     if var_occurrences_counts[var] == 0:
                #         num_remaining_vars -= 1
                clause_dict[occurrence_clause_idx] = None
                # num_remaining_clauses -= 1
            elif -unit_literal in occurrence_clause:
                num_neg_lits_elim += 1
                # The literal is the negation of the one in this clause. Remove the literal from the clause.
                clause_dict[occurrence_clause_idx].remove(-unit_literal)
                # var_occurrences_counts[unit_variable] -= 1
                # if var_occurrences_counts[unit_variable] == 0:
                #     num_remaining_vars -= 1
                if len(clause_dict[occurrence_clause_idx]) == 0:
                    return None
                # Clauses is now a unit clause. Add it to the queue of unit clauses to propagate.
                if len(clause_dict[occurrence_clause_idx]) == 1:
                    occurrence_clause_level = unit_clause_level + 1
                    if num_levels < 0 or occurrence_clause_level < num_levels:
                        unit_clause_queue.append((occurrence_clause_idx, occurrence_clause_level))

        # Set the value of the literal in the assignment.
        if unit_literal > 0:
            assignment[unit_variable] = AssignmentValue.TRUE
        else:
            assignment[unit_variable] = AssignmentValue.FALSE
        num_units_propagated += 1
        # Remove the unit clause.
        clause_dict[unit_clause_idx] = None

        # if num_remaining_clauses + num_remaining_vars <= limit_remaining_items:
        #     print("ILF", num_remaining_clauses, num_remaining_vars)
        #     break

    propagated_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            if clause_dict[idx] is not None:
                propagated_clauses.add_clause(clause_dict[idx], make_names=True)
    logger.debug("Propagated units: {} Remaining clauses: {} | Max Prop Level: {} | Elims: C: {} NL: {}".format(
        num_units_propagated, propagated_clauses.num_clauses(), max_prop_level, num_clauses_elim, num_neg_lits_elim))

    return propagated_clauses, assignment


def pure_literal_elimination(clause_list: ClauseList, assignment: Assignment):
    clause_dict = {idx: clause.copy() for idx, clause in enumerate(clause_list.clauses())}

    # Build the occurrence lists.
    lit_set = set()
    lit_occurrence_lists = defaultdict(set)
    lit_occurrence_counts = defaultdict(int)
    for clause_idx, clause in clause_dict.items():
        for literal in clause:
            lit_set.add(literal)
            lit_set.add(-literal)
            lit_occurrence_lists[literal].add(clause_idx)
            lit_occurrence_counts[literal] += 1

    # Fill queue with pure literals
    pure_lit_queue = deque()
    for lit in lit_set:
        if lit_occurrence_counts[lit] > 0 and lit_occurrence_counts[-lit] == 0:
            pure_lit_queue.append(lit)

    # Perform pure literal elimination.
    while len(pure_lit_queue) != 0:
        # Pop a pure literal.
        pure_lit = pure_lit_queue.popleft()
        if lit_occurrence_counts[pure_lit] > 0 and lit_occurrence_counts[-pure_lit] == 0:
            lit_set.remove(pure_lit)
            lit_set.remove(-pure_lit)

        pure_var = abs(pure_lit)

        # Remove pure literal clauses
        pure_lit_occurrences = lit_occurrence_lists[pure_lit]
        for occurrence_clause_idx in pure_lit_occurrences:
            occurrence_clause = clause_dict[occurrence_clause_idx]
            if occurrence_clause is None:
                continue
            if pure_lit in occurrence_clause:
                # Decrement the literal occurrence counts.
                for lit in occurrence_clause:
                    lit_occurrence_counts[lit] -= 1
                    assert (lit_occurrence_counts[lit] >= 0)
                    if lit_occurrence_counts[-lit] > 0 and lit_occurrence_counts[lit] == 0:
                        pure_lit_queue.append(-lit)
            # The literal is the same as the one in this clause. Remove the clause.
            clause_dict[occurrence_clause_idx] = None

        # Set the value of the literal in the assignment.
        if pure_lit > 0:
            assignment[pure_var] = AssignmentValue.TRUE
        else:
            assignment[pure_var] = AssignmentValue.FALSE

    new_clauses = ClauseList()
    for idx in range(len(clause_list)):
        if idx in clause_dict.keys():
            if clause_dict[idx] is not None:
                new_clauses.add_clause(clause_dict[idx], make_names=True)

    return new_clauses, assignment


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

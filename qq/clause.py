from typing import NewType, Set, List, Dict, Optional

import tqdm
from z3 import *


def Clause(init_literals=None):
    if init_literals is None:
        init_literals = {}
    clause_type = NewType('Clause', Set[int])
    return clause_type(init_literals)


class ClauseList:
    clause_list: List[Clause]
    var_names: Dict[int, str]

    def __init__(self):
        self.clause_list = []
        self.var_names = dict()

    def add_clause(self, clause: Clause):
        self.clause_list.append(clause)
        for literal in clause:
            variable = abs(literal)
            self.var_names.setdefault(variable, None)

    def load_from_dimacs(self, dimacs_content: str):
        dimacs_lines = dimacs_content.splitlines()
        for line in dimacs_lines:
            line_start_char = line[0]
            if line_start_char == 'c':
                comment_info = line.split(' ')
                var_num = int(comment_info[1])
                var_name = comment_info[2]
                self.set_variable_name(var_num, var_name)
            elif line_start_char == 'p':
                pass
            elif line_start_char.isdigit() or line_start_char == '-':
                clause = Clause({int(literal) for literal in line.split(' ')[:-1]})
                self.add_clause(clause)
            else:
                raise RuntimeError("Unexpected character at start of DIMACS line.")

    def to_dimacs(self):
        dimacs_content = ""
        num_variables = len(self.var_names)
        num_clauses = len(self.clause_list)
        dimacs_content += "p cnf {} {}\n".format(num_variables, num_clauses)
        for clause in self.clause_list:
            dimacs_content += "{} 0\n".format(' '.join(str(literal) for literal in clause))

        for var_num in self.var_names:
            dimacs_content += "c {} {}\n".format(var_num, self.get_variable_name(var_num))

        return dimacs_content[:-1]

    def fix_non_consecutive_vars(self):
        ordered_var_nums = sorted(self.variables())
        consec_var_num_map = {var: consec_var for consec_var, var in zip(enumerate(ordered_var_nums, start=1))}

        new_var_names = {consec_var_num_map[var_num]: var_name for var_num, var_name in self.var_names}
        new_clause_list = []
        for clause in self.clause_list:
            new_clause = Clause()
            for literal in clause:
                variable = abs(literal)
                sign = literal // variable
                new_literal = sign * consec_var_num_map[variable]
                new_clause.add(new_literal)
            new_clause_list.append(new_clause)
        self.var_names = new_var_names
        self.clause_list = new_clause_list

    def get_variable_name(self, var_number: int):
        var_name = self.var_names[var_number]
        if var_name is None:
            raise RuntimeError("Variable exists but a name is not set: {}".format(str(var_number)))
        return var_name

    def set_variable_name(self, var_number: int, var_name: str):
        if var_number not in self.var_names:
            raise RuntimeError("Variable does not exist. A name cannot be set.")
        self.var_names[var_number] = var_name

    def num_variables(self):
        return len(self.var_names.keys())

    def num_clauses(self):
        return len(self.clause_list)

    def variables(self):
        return self.var_names.keys()

    def clauses(self):
        return self.clause_list

    def __iter__(self):
        return self.clause_list.__iter__()

    def __len__(self):
        return len(self.clause_list)

    def __getitem__(self, item):
        return self.clause_list[item]


# Get the length of a clause in the Z3 expression format. Caller is responsible
# for checking if the expression is a valid clause prior to calling this.
def expr_clause_length(clause: ExprRef) -> int:
    if is_or(clause):
        return clause.num_args()
    elif expr_is_literal(clause):
        return 1
    else:
        raise RuntimeError("Non-clause expression encountered.")


def expr_is_literal(literal: ExprRef) -> bool:
    if is_not(literal):
        return is_const(literal.arg(0))
    elif is_const(literal):
        return True
    else:
        return False


def expr_is_clause(clause: ExprRef) -> bool:
    if is_or(clause):
        for literal in clause.children():
            if not expr_is_literal(literal):
                return False
        return True
    elif expr_is_literal(clause):
        return True
    else:
        return False


def rename_clause_vars(goal: Goal):
    var_cache = {}

    def rename_var(var: ExprRef):
        name = var.decl().name()
        if not name.startswith('k!'):
            return var

        if name in var_cache:
            return var_cache[name]
        else:
            new_name = name.replace('!', '_')
            new_var = Bool(new_name)
            var_cache[name] = new_var
            return new_var

    renamed_goal = Goal()
    for clause in tqdm.tqdm(goal, desc="Renaming Clause Variables"):
        if is_or(clause):
            literal_list = []
            for literal in clause.children():
                if is_not(literal):
                    new_literal = rename_var(literal.arg(0))
                    literal_list.append(Not(new_literal))
                elif is_const(literal):
                    new_literal = rename_var(literal)
                    literal_list.append(new_literal)
            renamed_goal.add(Or(literal_list))
        elif is_not(clause):
            new_literal = rename_var(clause.arg(0))
            renamed_goal.add(Not(new_literal))
        elif is_const(clause):
            new_literal = rename_var(clause)
            renamed_goal.add(new_literal)
        else:
            raise RuntimeError("An error occurred while renaming variables.")

    return renamed_goal


def split_expr_clauses(goal: Goal, max_clause_length: int = 3):
    if max_clause_length < 3:
        raise RuntimeError("Max clause length must be greater than 2.")

    clauses = [clause for clause in goal]
    split_clauses = []

    for clause in tqdm.tqdm(clauses, desc="Splitting Clauses"):
        clause_split_stack = [clause]
        while len(clause_split_stack) != 0:
            stack_clause = clause_split_stack.pop()
            if expr_clause_length(stack_clause) > max_clause_length:
                split_clause_list = split_expr_clause_half(stack_clause)
                clause_split_stack.extend(split_clause_list)
            else:
                split_clauses.append(stack_clause)

    split_clause_goal = Goal()
    split_clause_goal.add(*split_clauses)
    return split_clause_goal


# Split the clause into two halves.
def split_expr_clause_half(clause: ExprRef):
    if 'auxiliary_var_count' not in split_expr_clause_half.__dict__:
        split_expr_clause_half.auxiliary_var_count = 0

    literals = clause.children()

    aux_literal = Bool('aux!{}'.format(split_expr_clause_half.auxiliary_var_count))
    num_literals = len(literals)
    split_expr_clause_half.auxiliary_var_count += 1

    neg_aux_literal = Not(aux_literal)
    sub_clause0 = Or([neg_aux_literal] + literals[:num_literals//2])
    sub_clause1 = Or([aux_literal] + literals[num_literals//2:])
    split_clauses = [sub_clause0, sub_clause1]

    return split_clauses


def rename_clauses_in_clause_list(clause_list: ClauseList):
    for var_num in clause_list.variables():
        var_name = clause_list.get_variable_name(var_num)
        new_var_name = var_name.replace('!', '_')
        clause_list.set_variable_name(var_num, new_var_name)
    return clause_list


def split_clauses_in_clause_list(clause_list: ClauseList, max_clause_length: int = 3):
    if max_clause_length < 3:
        raise RuntimeError("")

    next_var_num = max(clause_list.variables()) + 1
    aux_var_num = 0
    split_clauses = ClauseList()
    aux_vars = dict()

    for clause in tqdm.tqdm(clause_list.clauses(), desc="Splitting Clauses"):
        clause_split_stack = [clause]

        while len(clause_split_stack) != 0:
            stack_clause = clause_split_stack.pop()
            if len(stack_clause) > max_clause_length:
                split_clause_list = split_expr_clause_half_from_dimacs(stack_clause, next_var_num)
                clause_split_stack.extend(split_clause_list)
                aux_var_name = 'aux_{}'.format(aux_var_num)
                aux_vars[next_var_num] = aux_var_name
                aux_var_num += 1
                next_var_num += 1
            else:
                split_clauses.add_clause(stack_clause)

    for var_num in clause_list.variables():
        var_name = clause_list.get_variable_name(var_num)
        split_clauses.set_variable_name(var_num, var_name)

    for aux_var_num, aux_var_name in aux_vars.items():
        split_clauses.set_variable_name(aux_var_num, aux_var_name)

    return split_clauses


# Split the clause into two halves.
def split_expr_clause_half_from_dimacs(clause: Clause, next_var_num: int):
    literals = list(clause)

    aux_literal = next_var_num
    num_literals = len(literals)

    neg_aux_literal = -aux_literal
    sub_clause0 = set([neg_aux_literal] + literals[:num_literals//2])
    sub_clause1 = set([aux_literal] + literals[num_literals//2:])
    split_clauses = [sub_clause0, sub_clause1]

    return split_clauses

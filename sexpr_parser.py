# sexpParser.py
#
# Demonstration of the pyparsing module, implementing a simple S-expression
# parser.
#
# Updates:
#  November, 2011 - fixed errors in precedence of alternatives in simpleString;
#      fixed exception raised in verifyLen to properly signal the input string
#      and exception location so that markInputline works correctly; fixed
#      definition of decimal to accept a single '0' and optional leading '-'
#      sign; updated tests to improve parser coverage
#
# Copyright 2007-2011, by Paul McGuire
#
"""
BNF reference: http://theory.lcs.mit.edu/~rivest/sexp.txt

<sexp>    	:: <string> | <list>
<string>   	:: <display>? <simple-string> ;
<simple-string>	:: <raw> | <token> | <base-64> | <hexadecimal> | 
		           <quoted-string> ;
<display>  	:: "[" <simple-string> "]" ;
<raw>      	:: <decimal> ":" <bytes> ;
<decimal>  	:: <decimal-digit>+ ;
		-- decimal numbers should have no unnecessary leading zeros
<bytes> 	-- any string of bytes, of the indicated length
<token>    	:: <tokenchar>+ ;
<base-64>  	:: <decimal>? "|" ( <base-64-char> | <whitespace> )* "|" ;
<hexadecimal>   :: "#" ( <hex-digit> | <white-space> )* "#" ;
<quoted-string> :: <decimal>? <quoted-string-body>  
<quoted-string-body> :: "\"" <bytes> "\""
<list>     	:: "(" ( <sexp> | <whitespace> )* ")" ;
<whitespace> 	:: <whitespace-char>* ;
<token-char>  	:: <alpha> | <decimal-digit> | <simple-punc> ;
<alpha>       	:: <upper-case> | <lower-case> | <digit> ;
<lower-case>  	:: "a" | ... | "z" ;
<upper-case>  	:: "A" | ... | "Z" ;
<decimal-digit> :: "0" | ... | "9" ;
<hex-digit>     :: <decimal-digit> | "A" | ... | "F" | "a" | ... | "f" ;
<simple-punc> 	:: "-" | "." | "/" | "_" | ":" | "*" | "+" | "=" ;
<whitespace-char> :: " " | "\t" | "\r" | "\n" ;
<base-64-char> 	:: <alpha> | <decimal-digit> | "+" | "/" | "=" ;
<null>        	:: "" ;
"""
import operator

import tqdm
from pyparsing import Suppress, Literal, Forward, Word, ParseFatalException, ZeroOrMore, Group, alphanums, Keyword

# define punctuation literals
LPAR, RPAR = map(Suppress, "()")

# extended definitions
token = Word(alphanums + "-./_:*+=!<>")

sexp_cmd = Forward()
cmd_names = (Literal('declare-fun') | Literal('model-add'))
# dropped_cmd_names = Suppress(Keyword('assert') | Keyword('model-del'))
# cmd_name = (cmd_names | dropped_cmd_names)
cmd_name = cmd_names

sexp = Forward()
sexp_list = Group(LPAR + ZeroOrMore(sexp) + RPAR)
sexp << (token | sexp_list)

sexp_cmd_list = Group(LPAR + cmd_names + ZeroOrMore(sexp) + RPAR)
# dropped_sexp_cmd_list = Suppress(Group(LPAR + dropped_cmd_names + ZeroOrMore(sexp) + RPAR))

# sexp_cmd << (token | sexp_cmd_list | dropped_sexp_cmd_list)
sexp_cmd << (token | sexp_cmd_list)


sexp_doc = Group(ZeroOrMore(sexp_cmd))


def parse_sexpr(sexpr_str: str):
    sexpr = sexp_doc.parseString(sexpr_str, parseAll=True)
    return sexpr.asList()


def filter_sexpr_lines(sexpr_set_str: str):
    open_paren_count = 0
    output_str = ""

    sexpr_stmt_str = ""
    for c in tqdm.tqdm(sexpr_set_str):
        if open_paren_count == 0 and sexpr_stmt_str != "":
            if not sexpr_stmt_str.startswith('(assert'):
                output_str += sexpr_stmt_str + '\n'
            else:
                sexpr_stmt_str = ""

        if c == '(':
            open_paren_count += 1

        if open_paren_count > 0:
            sexpr_stmt_str += c

        if c == ')':
            open_paren_count -= 1


def drop_asserts_sexpr_file(sexpr_file):
    # Note: This code is really flaky to any change that may be made in Z3's
    # output format. Be careful.
    sexpr_content = ""
    drop_line = False
    for sexpr_line in tqdm.tqdm(sexpr_file, desc="Sifting Model S-expression"):
        if sexpr_line.startswith('(assert'):
            drop_line = True
        elif sexpr_line.startswith('(define-fun'):
            drop_line = False
        elif sexpr_line.startswith('(model-add'):
            drop_line = False
        elif sexpr_line.startswith('(model-del'):
            drop_line = True
        if not drop_line:
            sexpr_content += sexpr_line
    return sexpr_content


def build_variable_value_tables(parse_result):
    var_table = dict()
    for command_list in parse_result[0]:
        if command_list[0] == 'declare-fun':
            var_name = command_list[1]
            var_table[var_name] = None
    return var_table


def build_variable_eval_order_list(parse_result):
    eval_order_list = []
    for command_list in parse_result[0]:
        if command_list[0] == 'declare-fun':
            var_name = rename_var(command_list[1])
            eval_order_list.append(var_name)
    return eval_order_list


def mkbv(*args):
    return [*args]


def bv2int(bit_vec):
    num = 0
    for i, bit_val in enumerate(bit_vec):
        if bit_val:
            num += (2 ** i)
    return num


def build_eval_code(function_def):
    if isinstance(function_def, list):
        code = ''
        op = function_def[0]
        op_map = {
            '+': 'add',
            '-': 'neg',
            'bv2int': 'bv2int',
            'mkbv': 'mkbv',
        }
        op_code = op_map[op]
        code += op_code + '('
        for param in function_def[1:]:
            code += build_eval_code(param) + ','
        code += ')'
        return code
    elif isinstance(function_def, str):
        return rename_var(function_def)
    else:
        raise RuntimeError("Failed to build code. Got an unexpected type.")


def rename_var(var_name: str):
    new_var_name = var_name.replace('!', '_')
    return new_var_name


def build_eval_dict(parse_result):
    eval_dict = dict()
    for command_list in parse_result[0]:
        if command_list[0] == 'model-add':
            var_name = rename_var(command_list[1])
            var_calc_code = build_eval_code(command_list[4])
            var_code = "{0} = {1}".format(var_name, var_calc_code)
            eval_dict[var_name] = var_code

    return eval_dict


def build_eval_code_block(eval_order_list, eval_dict):
    code = ''
    for var_name in eval_order_list:
        if var_name in eval_dict:
            var_code = eval_dict[var_name]
            code += var_code + '\n'
    return code


def run_eval_code_block(code_block, var_values):
    global_funcs = {
        'add': operator.add,
        'neg': operator.neg,
        'bv2int': bv2int,
        'mkbv': mkbv,
    }
    local_vars = var_values
    exec(code_block, global_funcs, local_vars)
    return var_values

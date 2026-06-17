# ═══════════════════════════════════════════════════════════════════════════════
# PINE SCRIPT V6 INTERPRETER — JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# A proper lexer → parser → AST → interpreter pipeline for Pine Script v6.
# Executes strategy logic bar-by-bar on real OHLCV data.
# ═══════════════════════════════════════════════════════════════════════════════

import re
import numpy as np
import pandas as pd
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class TT(Enum):
    # Literals
    INT = auto(); FLOAT = auto(); STRING = auto(); BOOL = auto(); NA = auto(); COLOR = auto()
    # Identifiers & keywords
    IDENT = auto(); VAR = auto(); VARIP = auto()
    IF = auto(); ELSE = auto(); FOR = auto(); TO = auto(); BY = auto()
    WHILE = auto(); SWITCH = auto(); IMPORT = auto(); EXPORT = auto()
    TRUE = auto(); FALSE = auto(); NOT = auto(); AND = auto(); OR = auto()
    TYPE = auto(); METHOD = auto(); SERIES = auto(); SIMPLE = auto()
    # Operators
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    ASSIGN = auto(); PLUS_ASSIGN = auto(); MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto(); SLASH_ASSIGN = auto()
    COLON_ASSIGN = auto()  # :=
    ARROW = auto()  # =>
    QUESTION = auto(); COLON = auto()
    # Delimiters
    LPAREN = auto(); RPAREN = auto(); LBRACKET = auto(); RBRACKET = auto()
    COMMA = auto(); DOT = auto(); NEWLINE = auto()
    # Special
    EOF = auto(); INDENT = auto(); DEDENT = auto()


@dataclass
class Token:
    type: TT
    value: Any
    line: int = 0
    col: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LEXER
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORDS = {
    'var': TT.VAR, 'varip': TT.VARIP, 'if': TT.IF, 'else': TT.ELSE,
    'for': TT.FOR, 'to': TT.TO, 'by': TT.BY, 'while': TT.WHILE,
    'switch': TT.SWITCH, 'true': TT.TRUE, 'false': TT.FALSE,
    'not': TT.NOT, 'and': TT.AND, 'or': TT.OR, 'na': TT.NA,
    'import': TT.IMPORT, 'export': TT.EXPORT, 'type': TT.TYPE,
    'method': TT.METHOD, 'series': TT.SERIES, 'simple': TT.SIMPLE,
}


def lexer(source: str) -> List[Token]:
    """Tokenize Pine Script v6 source code."""
    tokens = []
    lines = source.split('\n')
    indent_stack = [0]
    _in_continuation = False  # True when previous line ended with operator
    _paren_depth = 0  # Track open parens/brackets — suppress NEWLINE inside

    for line_num, raw_line in enumerate(lines, 1):
        # Remove comments
        line = re.sub(r'//.*$', '', raw_line)

        # Skip empty lines
        stripped = line.strip()
        if not stripped:
            continue

        # Handle indentation — suppress during line continuations AND inside parens
        indent = len(line) - len(line.lstrip())
        if not _in_continuation and _paren_depth == 0:
            if indent > indent_stack[-1]:
                indent_stack.append(indent)
                tokens.append(Token(TT.INDENT, indent, line_num))
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.DEDENT, indent, line_num))

        # Tokenize the line content
        i = 0
        s = stripped
        while i < len(s):
            c = s[i]

            # Skip whitespace
            if c in ' \t':
                i += 1
                continue

            # Multi-char operators
            if i + 1 < len(s):
                two = s[i:i+2]
                if two == ':=':
                    tokens.append(Token(TT.COLON_ASSIGN, ':=', line_num)); i += 2; continue
                if two == '=>':
                    tokens.append(Token(TT.ARROW, '=>', line_num)); i += 2; continue
                if two == '==':
                    tokens.append(Token(TT.EQ, '==', line_num)); i += 2; continue
                if two == '!=':
                    tokens.append(Token(TT.NEQ, '!=', line_num)); i += 2; continue
                if two == '<=':
                    tokens.append(Token(TT.LTE, '<=', line_num)); i += 2; continue
                if two == '>=':
                    tokens.append(Token(TT.GTE, '>=', line_num)); i += 2; continue
                if two == '+=':
                    tokens.append(Token(TT.PLUS_ASSIGN, '+=', line_num)); i += 2; continue
                if two == '-=':
                    tokens.append(Token(TT.MINUS_ASSIGN, '-=', line_num)); i += 2; continue
                if two == '*=':
                    tokens.append(Token(TT.STAR_ASSIGN, '*=', line_num)); i += 2; continue
                if two == '/=':
                    tokens.append(Token(TT.SLASH_ASSIGN, '/=', line_num)); i += 2; continue

            # Single-char operators
            single_map = {
                '+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH,
                '%': TT.PERCENT, '<': TT.LT, '>': TT.GT, '=': TT.ASSIGN,
                '(': TT.LPAREN, ')': TT.RPAREN, '[': TT.LBRACKET, ']': TT.RBRACKET,
                ',': TT.COMMA, '.': TT.DOT, '?': TT.QUESTION, ':': TT.COLON,
            }
            if c in single_map:
                tokens.append(Token(single_map[c], c, line_num))
                if c in '([': _paren_depth += 1
                elif c in '])': _paren_depth = max(0, _paren_depth - 1)
                i += 1; continue

            # Strings
            if c in '"\'':
                quote = c
                i += 1
                start = i
                while i < len(s) and s[i] != quote:
                    if s[i] == '\\': i += 1  # skip escaped
                    i += 1
                tokens.append(Token(TT.STRING, s[start:i], line_num))
                i += 1  # skip closing quote
                continue

            # Numbers
            if c.isdigit() or (c == '.' and i + 1 < len(s) and s[i+1].isdigit()):
                start = i
                has_dot = False
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    if s[i] == '.': has_dot = True
                    i += 1
                val = s[start:i]
                if has_dot:
                    tokens.append(Token(TT.FLOAT, float(val), line_num))
                else:
                    tokens.append(Token(TT.INT, int(val), line_num))
                continue

            # Color literals (#rrggbb or #rrggbbaa)
            if c == '#':
                start = i
                i += 1
                while i < len(s) and s[i] in '0123456789abcdefABCDEF':
                    i += 1
                tokens.append(Token(TT.COLOR, s[start:i], line_num))
                continue

            # Identifiers / keywords
            if c.isalpha() or c == '_':
                start = i
                while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                    i += 1
                word = s[start:i]
                if word in KEYWORDS:
                    tokens.append(Token(KEYWORDS[word], word, line_num))
                else:
                    tokens.append(Token(TT.IDENT, word, line_num))
                continue

            # Unknown char — skip
            i += 1

        # Line continuation: suppress NEWLINE if line ends with an operator or inside parens
        _continuation_types = {TT.COLON, TT.QUESTION, TT.COMMA,
                               TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH,
                               TT.EQ, TT.NEQ, TT.GT, TT.LT, TT.GTE, TT.LTE,
                               TT.AND, TT.OR, TT.NOT, TT.ASSIGN, TT.COLON_ASSIGN}
        if _paren_depth > 0:
            _in_continuation = True  # inside parens — suppress newlines
        elif tokens and tokens[-1].type in _continuation_types:
            _in_continuation = True  # next line continues this expression
        else:
            _in_continuation = False
            tokens.append(Token(TT.NEWLINE, '\n', line_num))

    # Close remaining indents
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.DEDENT, 0, line_num))

    tokens.append(Token(TT.EOF, None, line_num))
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# AST NODES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    pass

@dataclass
class NumberLiteral(ASTNode):
    value: float

@dataclass
class StringLiteral(ASTNode):
    value: str

@dataclass
class BoolLiteral(ASTNode):
    value: bool

@dataclass
class NALiteral(ASTNode):
    pass

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class TernaryOp(ASTNode):
    condition: ASTNode
    true_val: ASTNode
    false_val: ASTNode

@dataclass
class FunctionCall(ASTNode):
    name: str
    args: List[ASTNode] = field(default_factory=list)
    kwargs: Dict[str, ASTNode] = field(default_factory=dict)

@dataclass
class MethodCall(ASTNode):
    obj: ASTNode
    method: str
    args: List[ASTNode] = field(default_factory=list)
    kwargs: Dict[str, ASTNode] = field(default_factory=dict)

@dataclass
class DotAccess(ASTNode):
    obj: ASTNode
    attr: str

@dataclass
class IndexAccess(ASTNode):
    obj: ASTNode
    index: ASTNode

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode
    is_var: bool = False
    is_varip: bool = False
    op: str = '='  # =, :=, +=, -=, *=, /=

@dataclass
class TupleUnpack(ASTNode):
    targets: List[str]
    value: ASTNode

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    elif_clauses: List[Tuple[ASTNode, List[ASTNode]]] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None

@dataclass
class ForLoop(ASTNode):
    var: str
    start: ASTNode
    end: ASTNode
    step: Optional[ASTNode] = None
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class WhileLoop(ASTNode):
    condition: ASTNode
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class FunctionDef(ASTNode):
    name: str
    params: List[Tuple[str, Optional[ASTNode]]]  # (name, default)
    body: List[ASTNode] = field(default_factory=list)
    is_method: bool = False

@dataclass
class SwitchStatement(ASTNode):
    expr: Optional[ASTNode]
    cases: List[Tuple[Optional[ASTNode], List[ASTNode]]]  # (condition/None for default, body)

@dataclass
class ArrayLiteral(ASTNode):
    elements: List[ASTNode]

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Recursive descent parser for Pine Script v6."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors = []

    def peek(self, offset=0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token(TT.EOF, None)

    def current(self) -> Token:
        return self.peek()

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def expect(self, tt: TT) -> Token:
        tok = self.current()
        if tok.type != tt:
            self.errors.append(f"Line {tok.line}: expected {tt.name}, got {tok.type.name} ({tok.value!r})")
            return tok
        return self.advance()

    def match(self, *types) -> Optional[Token]:
        if self.current().type in types:
            return self.advance()
        return None

    def skip_newlines(self):
        while self.current().type == TT.NEWLINE:
            self.advance()

    def parse(self) -> Program:
        stmts = []
        self.skip_newlines()
        while self.current().type != TT.EOF:
            try:
                stmt = self.parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
            except Exception as e:
                self.errors.append(f"Parse error at line {self.current().line}: {e}")
                # Skip to next line
                while self.current().type not in (TT.NEWLINE, TT.EOF):
                    self.advance()
            self.skip_newlines()
        return Program(stmts)

    def parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        tok = self.current()

        # Skip annotations like //@version=6
        if tok.type == TT.IDENT and tok.value in ('indicator', 'library'):
            # Skip the whole function call line
            while self.current().type not in (TT.NEWLINE, TT.EOF):
                self.advance()
            return None

        # var / varip declaration
        if tok.type in (TT.VAR, TT.VARIP):
            return self.parse_var_decl()

        # if statement
        if tok.type == TT.IF:
            return self.parse_if()

        # for loop
        if tok.type == TT.FOR:
            return self.parse_for()

        # while loop
        if tok.type == TT.WHILE:
            return self.parse_while()

        # switch
        if tok.type == TT.SWITCH:
            return self.parse_switch()

        # function/method definition: name(params) =>
        if tok.type in (TT.IDENT, TT.METHOD):
            return self.parse_assignment_or_expr()

        # Tuple unpacking: [a, b, c] = ...
        if tok.type == TT.LBRACKET:
            return self.parse_tuple_unpack()

        # type keyword — skip type definitions
        if tok.type == TT.TYPE:
            while self.current().type not in (TT.NEWLINE, TT.EOF, TT.DEDENT):
                self.advance()
            return None

        # import/export — skip
        if tok.type in (TT.IMPORT, TT.EXPORT):
            while self.current().type not in (TT.NEWLINE, TT.EOF):
                self.advance()
            return None

        # series/simple type annotations — skip the keyword, parse the rest
        if tok.type in (TT.SERIES, TT.SIMPLE):
            self.advance()
            return self.parse_statement()

        # Expression statement
        expr = self.parse_expression()
        return expr

    def parse_var_decl(self) -> ASTNode:
        is_var = self.current().type == TT.VAR
        is_varip = self.current().type == TT.VARIP
        self.advance()  # skip var/varip

        # Skip optional type annotation (int, float, bool, string, etc.)
        if self.current().type == TT.IDENT and self.peek(1).type == TT.IDENT:
            type_tok = self.current()
            if type_tok.value in ('int', 'float', 'bool', 'string', 'color', 'line',
                                   'label', 'box', 'table', 'array', 'matrix', 'map'):
                self.advance()

        name = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        value = self.parse_expression()
        return Assignment(name, value, is_var=is_var, is_varip=is_varip)

    def parse_assignment_or_expr(self) -> ASTNode:
        # Could be: identifier = expr, identifier := expr, func def, or just expression
        if self.current().type == TT.METHOD:
            return self.parse_func_def(is_method=True)

        # Look ahead for assignment
        if self.current().type == TT.IDENT:
            # Check for function definition: name(params) =>
            if self.peek(1).type == TT.LPAREN:
                # Could be func def or func call assigned
                # Scan for => after matching parens
                save = self.pos
                name = self.advance().value
                self.advance()  # (
                depth = 1
                while depth > 0 and self.current().type != TT.EOF:
                    if self.current().type == TT.LPAREN: depth += 1
                    if self.current().type == TT.RPAREN: depth -= 1
                    self.advance()
                if self.current().type == TT.ARROW:
                    self.pos = save
                    return self.parse_func_def()
                self.pos = save

            # Check for: IDENT = expr / IDENT := expr / IDENT op= expr
            if self.peek(1).type in (TT.ASSIGN, TT.COLON_ASSIGN, TT.PLUS_ASSIGN,
                                      TT.MINUS_ASSIGN, TT.STAR_ASSIGN, TT.SLASH_ASSIGN):
                name = self.advance().value
                op_tok = self.advance()
                op = op_tok.value

                # Skip optional type annotations after =
                # e.g., myVar = float(na)
                value = self.parse_expression()
                return Assignment(name, value, op=op)

            # Type-annotated assignment: float myVar = expr
            if (self.current().value in ('int', 'float', 'bool', 'string', 'color')
                    and self.peek(1).type == TT.IDENT
                    and self.peek(2).type in (TT.ASSIGN, TT.COLON_ASSIGN)):
                self.advance()  # skip type
                name = self.advance().value
                op = self.advance().value
                value = self.parse_expression()
                return Assignment(name, value, op=op)

        return self.parse_expression()

    def parse_tuple_unpack(self) -> ASTNode:
        self.expect(TT.LBRACKET)
        names = []
        while True:
            names.append(self.expect(TT.IDENT).value)
            if not self.match(TT.COMMA):
                break
        self.expect(TT.RBRACKET)
        self.expect(TT.ASSIGN)
        value = self.parse_expression()
        return TupleUnpack(names, value)

    def parse_if(self) -> ASTNode:
        self.expect(TT.IF)
        condition = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()

        elif_clauses = []
        else_body = None

        while True:
            self.skip_newlines()
            if self.current().type == TT.ELSE:
                self.advance()
                if self.current().type == TT.IF:
                    self.advance()
                    elif_cond = self.parse_expression()
                    self.skip_newlines()
                    elif_body = self.parse_block()
                    elif_clauses.append((elif_cond, elif_body))
                else:
                    self.skip_newlines()
                    else_body = self.parse_block()
                    break
            else:
                break

        return IfStatement(condition, body, elif_clauses, else_body)

    def parse_for(self) -> ASTNode:
        self.expect(TT.FOR)
        var_name = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        start = self.parse_expression()
        self.expect(TT.TO)
        end = self.parse_expression()
        step = None
        if self.match(TT.BY):
            step = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return ForLoop(var_name, start, end, step, body)

    def parse_while(self) -> ASTNode:
        self.expect(TT.WHILE)
        condition = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return WhileLoop(condition, body)

    def parse_switch(self) -> ASTNode:
        self.expect(TT.SWITCH)
        expr = None
        if self.current().type not in (TT.NEWLINE, TT.INDENT):
            expr = self.parse_expression()
        self.skip_newlines()
        cases = []
        if self.match(TT.INDENT):
            while self.current().type != TT.DEDENT and self.current().type != TT.EOF:
                self.skip_newlines()
                if self.current().type == TT.DEDENT:
                    break
                if self.current().type == TT.ARROW:
                    # default case
                    self.advance()
                    self.skip_newlines()
                    body = self.parse_block() if self.current().type == TT.INDENT else [self.parse_expression()]
                    cases.append((None, body))
                else:
                    cond = self.parse_expression()
                    self.skip_newlines()
                    if self.match(TT.ARROW):
                        self.skip_newlines()
                        body = self.parse_block() if self.current().type == TT.INDENT else [self.parse_expression()]
                    else:
                        body = []
                    cases.append((cond, body))
                self.skip_newlines()
            self.match(TT.DEDENT)
        return SwitchStatement(expr, cases)

    def parse_func_def(self, is_method=False) -> ASTNode:
        if is_method:
            self.advance()  # skip 'method'
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        while self.current().type != TT.RPAREN and self.current().type != TT.EOF:
            # Skip type annotations
            if (self.current().type == TT.IDENT and self.peek(1).type == TT.IDENT
                    and self.current().value in ('int', 'float', 'bool', 'string',
                                                  'series', 'simple', 'color', 'array')):
                self.advance()
            param_name = self.expect(TT.IDENT).value
            default = None
            if self.match(TT.ASSIGN):
                default = self.parse_expression()
            params.append((param_name, default))
            self.match(TT.COMMA)
        self.expect(TT.RPAREN)
        self.expect(TT.ARROW)
        self.skip_newlines()
        if self.current().type == TT.INDENT:
            body = self.parse_block()
        else:
            body = [self.parse_expression()]
        return FunctionDef(name, params, body, is_method)

    def parse_block(self) -> List[ASTNode]:
        stmts = []
        if self.match(TT.INDENT):
            while self.current().type != TT.DEDENT and self.current().type != TT.EOF:
                self.skip_newlines()
                if self.current().type == TT.DEDENT:
                    break
                stmt = self.parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
                self.skip_newlines()
            self.match(TT.DEDENT)
        else:
            # Single-line block
            stmt = self.parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    # ── Expression parsing (precedence climbing) ──────────────────────────

    def parse_expression(self) -> ASTNode:
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        expr = self.parse_or()
        if self.match(TT.QUESTION):
            true_val = self.parse_expression()
            self.expect(TT.COLON)
            false_val = self.parse_expression()
            return TernaryOp(expr, true_val, false_val)
        return expr

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TT.OR):
            right = self.parse_and()
            left = BinaryOp('or', left, right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.match(TT.AND):
            right = self.parse_not()
            left = BinaryOp('and', left, right)
        return left

    def parse_not(self) -> ASTNode:
        if self.match(TT.NOT):
            return UnaryOp('not', self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_addition()
        ops = {TT.EQ: '==', TT.NEQ: '!=', TT.LT: '<', TT.GT: '>',
               TT.LTE: '<=', TT.GTE: '>='}
        while self.current().type in ops:
            op = ops[self.advance().type]
            right = self.parse_addition()
            left = BinaryOp(op, left, right)
        return left

    def parse_addition(self) -> ASTNode:
        left = self.parse_multiplication()
        while self.current().type in (TT.PLUS, TT.MINUS):
            op = '+' if self.advance().type == TT.PLUS else '-'
            right = self.parse_multiplication()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiplication(self) -> ASTNode:
        left = self.parse_unary()
        while self.current().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            tok = self.advance()
            op = {TT.STAR: '*', TT.SLASH: '/', TT.PERCENT: '%'}[tok.type]
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.current().type == TT.MINUS:
            self.advance()
            return UnaryOp('-', self.parse_unary())
        if self.current().type == TT.PLUS:
            self.advance()
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        node = self.parse_primary()
        while True:
            if self.current().type == TT.DOT:
                self.advance()
                attr = self.expect(TT.IDENT).value
                # Check for method call: obj.method(args)
                if self.current().type == TT.LPAREN:
                    self.advance()
                    args, kwargs = self.parse_arg_list()
                    self.expect(TT.RPAREN)
                    node = MethodCall(node, attr, args, kwargs)
                else:
                    node = DotAccess(node, attr)
            elif self.current().type == TT.LBRACKET:
                self.advance()
                index = self.parse_expression()
                self.expect(TT.RBRACKET)
                node = IndexAccess(node, index)
            elif self.current().type == TT.LPAREN and isinstance(node, Identifier):
                self.advance()
                args, kwargs = self.parse_arg_list()
                self.expect(TT.RPAREN)
                node = FunctionCall(node.name, args, kwargs)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        tok = self.current()

        if tok.type == TT.INT:
            self.advance(); return NumberLiteral(float(tok.value))
        if tok.type == TT.FLOAT:
            self.advance(); return NumberLiteral(tok.value)
        if tok.type == TT.STRING:
            self.advance(); return StringLiteral(tok.value)
        if tok.type in (TT.TRUE, TT.FALSE):
            self.advance(); return BoolLiteral(tok.type == TT.TRUE)
        if tok.type == TT.NA:
            # If followed by '(', treat as function call na(x), not literal
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TT.LPAREN:
                self.advance()
                return Identifier('na')
            self.advance(); return NALiteral()
        if tok.type == TT.COLOR:
            self.advance(); return StringLiteral(tok.value)  # treat color as string

        if tok.type == TT.IDENT:
            self.advance()
            return Identifier(tok.value)

        if tok.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TT.RPAREN)
            return expr

        # Array literal: [expr, expr, ...]
        if tok.type == TT.LBRACKET:
            self.advance()
            elements = []
            while self.current().type != TT.RBRACKET and self.current().type != TT.EOF:
                elements.append(self.parse_expression())
                self.match(TT.COMMA)
            self.expect(TT.RBRACKET)
            return ArrayLiteral(elements)

        # Type cast expressions like int(x), float(x)
        if tok.type in (TT.SERIES, TT.SIMPLE):
            self.advance()
            return self.parse_postfix()

        self.advance()  # skip unknown
        return NALiteral()

    def parse_arg_list(self) -> Tuple[List[ASTNode], Dict[str, ASTNode]]:
        args = []
        kwargs = {}
        while self.current().type != TT.RPAREN and self.current().type != TT.EOF:
            # Check for kwarg: name = expr
            if (self.current().type == TT.IDENT and self.peek(1).type == TT.ASSIGN):
                name = self.advance().value
                self.advance()  # skip =
                val = self.parse_expression()
                kwargs[name] = val
            else:
                args.append(self.parse_expression())
            self.match(TT.COMMA)
        return args, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETER (bar-by-bar execution)
# ═══════════════════════════════════════════════════════════════════════════════

class PineInterpreter:
    """
    Executes a parsed Pine Script AST bar-by-bar on OHLCV data.
    Maintains proper variable scoping, var/varip persistence,
    and history operator ([N]) support.
    """

    def __init__(self, ast: Program, df: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission_pct: float = 0.0,
                 default_qty: int = 1,
                 pyramiding: int = 1,
                 slippage: int = 0,
                 mintick: float = 0.01,
                 margin_long: int = 0,
                 margin_short: int = 0,
                 input_overrides: dict = None,
                 point_value: float = 1.0,
                 exchange_timezone: str = None,
                 # ── Institutional features ──
                 security_data: dict = None,
                 margin_per_contract: float = 0.0,
                 volume_participation_limit: float = 0.0,
                 market_impact_bps: float = 0.0):
        self.ast = ast
        self.df = df.copy()
        self.n_bars = len(df)
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.default_qty = default_qty

        # State
        self.bar_index = 0
        self.globals = {}       # variable name -> current value
        self.var_inited = {}    # var declarations — True if already initialized
        self.history = {}       # variable name -> list of past values
        self.functions = {}     # user-defined functions
        self.trades = []
        self.warnings = []
        self.strategy_name = "Pine Strategy"
        self.strategy_params = {}

        # ── Input system ──────────────────────────────────────────────────
        self.input_defs = []       # list of input definitions for UI generation
        self.input_overrides = input_overrides or {}  # user-set values from UI

        # ── Position management ───────────────────────────────────────────
        self.position = 0          # 1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.entry_qty = default_qty
        self.entry_bar = 0
        self.open_qty = 0          # remaining qty (for partial exits)
        self.positions = []        # for pyramiding: list of {dir, price, qty, bar}

        # ── Strategy settings ─────────────────────────────────────────────
        self.pyramiding = pyramiding        # max entries in same direction
        self.slippage = slippage            # ticks of slippage per trade
        self.calc_on_every_tick = False
        self.process_orders_on_close = False
        self.commission_type = 'percent'    # 'percent', 'cash_per_contract', 'cash_per_order'
        self.commission_value = commission_pct  # raw value from strategy() declaration
        self.point_value = point_value       # dollar value per point (MNQ=2, NQ=20, ES=50)

        # ── TP/SL from strategy.exit ──────────────────────────────────────
        self.exit_rules = {}       # from_entry -> {tp, sl, tp_pct, sl_pct, trail, qty_pct}

        # ── Pending orders from strategy.order() with limit/stop ─────────
        self.pending_orders = {}   # name -> {dir, qty, limit, stop, name}

        # ── Queued market orders — fill at NEXT bar's open (TradingView default) ──
        self._queued_market_orders = []

        # ── request.security fallback values ─────────────────────────────
        self.security_defaults = {
            'VIX': 16.0, 'VIX1!': 16.0, 'CBOE:VIX': 16.0, 'TVC:VIX': 16.0,
            'VXX': 20.0, 'MOVE': 100.0, 'DXY': 104.0, 'US10Y': 4.3, 'TNX': 43.0,
        }

        # ── syminfo ───────────────────────────────────────────────────────
        self.mintick = mintick
        self.margin_long = margin_long
        self.margin_short = margin_short
        self.exchange_timezone = exchange_timezone

        # ── Multi-timeframe security data ─────────────────────────────
        # Dict of ticker -> DataFrame (with DatetimeIndex + OHLCV columns)
        # Used by request.security() to return real data instead of fallbacks
        self.security_data = security_data or {}
        self._security_cache = {}  # ticker -> {field -> precomputed array}

        # ── Portfolio / Margin tracking ───────────────────────────────
        self.margin_per_contract = margin_per_contract  # e.g. $2,100 for MNQ
        self.buying_power = initial_capital
        self.margin_used = 0.0
        self.margin_calls = 0  # count of blocked entries due to margin

        # ── Execution modeling ────────────────────────────────────────
        # Volume participation: max % of bar volume we can fill (0 = unlimited)
        self.volume_participation_limit = volume_participation_limit  # e.g. 0.10 = 10%
        self.partial_fills = 0  # count of qty-capped fills
        # Market impact: basis points of price impact per contract (0 = none)
        self.market_impact_bps = market_impact_bps  # e.g. 0.5 = 0.5 bps per contract


        self._call_counter = {}
        self._bar_call_reset = -1
        self._setup_builtins()


    def _call_key(self, prefix):
        """Generate a stable key for ta.* history buffers.
        Same call site produces same key across bars."""
        if self._bar_call_reset != self.bar_index:
            self._call_counter = {}
            self._bar_call_reset = self.bar_index
        count = self._call_counter.get(prefix, 0)
        self._call_counter[prefix] = count + 1
        return f"_{prefix}_{count}"

    def _setup_builtins(self):
        """Register all built-in Pine Script functions and variables."""
        self.builtins = {}

        # ── math.* ──
        self.builtins['math.abs'] = lambda args, kw: abs(self._num(args[0]))
        self.builtins['math.max'] = lambda args, kw: max(self._num(args[0]), self._num(args[1]))
        self.builtins['math.min'] = lambda args, kw: min(self._num(args[0]), self._num(args[1]))
        self.builtins['math.sqrt'] = lambda args, kw: np.sqrt(self._num(args[0]))
        self.builtins['math.pow'] = lambda args, kw: self._num(args[0]) ** self._num(args[1])
        self.builtins['math.log'] = lambda args, kw: np.log(self._num(args[0]))
        self.builtins['math.log10'] = lambda args, kw: np.log10(self._num(args[0]))
        self.builtins['math.ceil'] = lambda args, kw: int(np.ceil(self._num(args[0])))
        self.builtins['math.floor'] = lambda args, kw: int(np.floor(self._num(args[0])))
        self.builtins['math.round'] = lambda args, kw: round(self._num(args[0]), int(args[1]) if len(args) > 1 else 0)
        self.builtins['math.sign'] = lambda args, kw: np.sign(self._num(args[0]))
        self.builtins['math.avg'] = lambda args, kw: np.mean([self._num(a) for a in args])
        self.builtins['math.sum'] = lambda args, kw: self._ta_sum(args, kw)

        # ── nz, na, fixnan ──
        self.builtins['nz'] = lambda args, kw: args[0] if args[0] is not None and not self._is_na(args[0]) else (args[1] if len(args) > 1 else 0)
        self.builtins['na'] = lambda args, kw: self._is_na(args[0]) if args else None
        self.builtins['fixnan'] = lambda args, kw: args[0] if not self._is_na(args[0]) else self._prev(str(id(args)), args[0])

        # ── Type casts ──
        self.builtins['int'] = lambda args, kw: int(self._num(args[0])) if not self._is_na(args[0]) else None
        self.builtins['float'] = lambda args, kw: float(self._num(args[0])) if not self._is_na(args[0]) else None
        self.builtins['bool'] = lambda args, kw: bool(args[0])
        self.builtins['str.tostring'] = lambda args, kw: str(args[0])
        self.builtins['string'] = lambda args, kw: str(args[0]) if args else ""

        # ── ta.* indicators ──
        self.builtins['ta.sma'] = lambda args, kw: self._ta_sma(args, kw)
        self.builtins['ta.ema'] = lambda args, kw: self._ta_ema(args, kw)
        self.builtins['ta.wma'] = lambda args, kw: self._ta_wma(args, kw)
        self.builtins['ta.vwma'] = lambda args, kw: self._ta_vwma(args, kw)
        self.builtins['ta.hma'] = lambda args, kw: self._ta_hma(args, kw)
        self.builtins['ta.rma'] = lambda args, kw: self._ta_rma(args, kw)
        self.builtins['ta.rsi'] = lambda args, kw: self._ta_rsi(args, kw)
        self.builtins['ta.macd'] = lambda args, kw: self._ta_macd(args, kw)
        self.builtins['ta.bb'] = lambda args, kw: self._ta_bb(args, kw)
        self.builtins['ta.bbw'] = lambda args, kw: self._ta_bbw(args, kw)
        self.builtins['ta.atr'] = lambda args, kw: self._ta_atr(args, kw)
        self.builtins['ta.tr'] = lambda args, kw: self._ta_tr(args, kw)
        self.builtins['ta.stoch'] = lambda args, kw: self._ta_stoch(args, kw)
        self.builtins['ta.cci'] = lambda args, kw: self._ta_cci(args, kw)
        self.builtins['ta.mfi'] = lambda args, kw: self._ta_mfi(args, kw)
        self.builtins['ta.adx'] = lambda args, kw: self._ta_adx(args, kw)
        self.builtins['ta.dmi'] = lambda args, kw: self._ta_adx(args, kw)
        self.builtins['ta.supertrend'] = lambda args, kw: self._ta_supertrend(args, kw)
        self.builtins['ta.crossover'] = lambda args, kw: self._ta_crossover(args, kw)
        self.builtins['ta.crossunder'] = lambda args, kw: self._ta_crossunder(args, kw)
        self.builtins['ta.cross'] = lambda args, kw: self._ta_cross(args, kw)
        self.builtins['ta.highest'] = lambda args, kw: self._ta_highest(args, kw)
        self.builtins['ta.lowest'] = lambda args, kw: self._ta_lowest(args, kw)
        self.builtins['ta.highestbars'] = lambda args, kw: self._ta_highestbars(args, kw)
        self.builtins['ta.lowestbars'] = lambda args, kw: self._ta_lowestbars(args, kw)
        self.builtins['ta.change'] = lambda args, kw: self._ta_change(args, kw)
        self.builtins['ta.mom'] = lambda args, kw: self._ta_change(args, kw)
        self.builtins['ta.roc'] = lambda args, kw: self._ta_roc(args, kw)
        self.builtins['ta.valuewhen'] = lambda args, kw: self._ta_valuewhen(args, kw)
        self.builtins['ta.barssince'] = lambda args, kw: self._ta_barssince(args, kw)
        self.builtins['ta.cum'] = lambda args, kw: self._ta_cum(args, kw)
        self.builtins['ta.rising'] = lambda args, kw: self._ta_rising(args, kw)
        self.builtins['ta.falling'] = lambda args, kw: self._ta_falling(args, kw)
        self.builtins['ta.pivothigh'] = lambda args, kw: self._ta_pivothigh(args, kw)
        self.builtins['ta.pivotlow'] = lambda args, kw: self._ta_pivotlow(args, kw)
        self.builtins['ta.percentrank'] = lambda args, kw: self._ta_percentrank(args, kw)
        self.builtins['ta.percentile_nearest_rank'] = lambda args, kw: self._ta_percentile(args, kw)
        self.builtins['ta.correlation'] = lambda args, kw: self._ta_correlation(args, kw)
        self.builtins['ta.dev'] = lambda args, kw: self._ta_dev(args, kw)
        self.builtins['ta.stdev'] = lambda args, kw: self._ta_dev(args, kw)
        self.builtins['ta.variance'] = lambda args, kw: self._ta_variance(args, kw)
        self.builtins['ta.median'] = lambda args, kw: self._ta_median(args, kw)
        self.builtins['ta.mode'] = lambda args, kw: self._ta_mode(args, kw)
        self.builtins['ta.linreg'] = lambda args, kw: self._ta_linreg(args, kw)
        self.builtins['ta.swma'] = lambda args, kw: self._ta_swma(args, kw)
        self.builtins['ta.alma'] = lambda args, kw: self._ta_alma(args, kw)
        self.builtins['ta.vwap'] = lambda args, kw: self._ta_vwap(args, kw)
        self.builtins['ta.obv'] = lambda args, kw: self._ta_obv(args, kw)

        # ── strategy.* ──
        self.builtins['strategy'] = lambda args, kw: self._strategy_decl(args, kw)
        self.builtins['strategy.entry'] = lambda args, kw: self._strategy_entry(args, kw)
        self.builtins['strategy.close'] = lambda args, kw: self._strategy_close(args, kw)
        self.builtins['strategy.close_all'] = lambda args, kw: self._strategy_close(["all"], kw)
        self.builtins['strategy.exit'] = lambda args, kw: self._strategy_exit(args, kw)
        self.builtins['strategy.order'] = lambda args, kw: self._strategy_order(args, kw)
        self.builtins['strategy.cancel'] = lambda args, kw: self._strategy_cancel(args, kw)
        self.builtins['strategy.cancel_all'] = lambda args, kw: self._strategy_cancel_all()
        self.builtins['strategy.closedtrades.profit'] = lambda args, kw: self._closedtrades_profit(args)
        self.builtins['strategy.closedtrades.size'] = lambda args, kw: self._closedtrades_size(args)
        self.builtins['strategy.closedtrades.entry_price'] = lambda args, kw: self._closedtrades_field(args, 'Price')
        self.builtins['strategy.closedtrades.exit_price'] = lambda args, kw: self._closedtrades_field(args, 'Exit Price')

        # ── input.* ──
        self.builtins['input'] = lambda args, kw: self._input(args, kw)
        self.builtins['input.int'] = lambda args, kw: self._input(args, kw, cast=int)
        self.builtins['input.float'] = lambda args, kw: self._input(args, kw, cast=float)
        self.builtins['input.bool'] = lambda args, kw: self._input(args, kw, cast=bool)
        self.builtins['input.string'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.source'] = lambda args, kw: self._input_source(args, kw)
        self.builtins['input.timeframe'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.symbol'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.color'] = lambda args, kw: self._input(args, kw, cast=str)

        # ── v3/v4/v5 compatibility — bare function names without ta. prefix ──
        for _fn in ('sma', 'ema', 'wma', 'vwma', 'hma', 'rma', 'rsi', 'macd', 'bb', 'bbw',
                     'atr', 'tr', 'stoch', 'cci', 'mfi', 'adx', 'dmi', 'supertrend',
                     'crossover', 'crossunder', 'cross',
                     'highest', 'lowest', 'highestbars', 'lowestbars',
                     'change', 'mom', 'roc', 'valuewhen', 'barssince', 'cum',
                     'rising', 'falling', 'pivothigh', 'pivotlow',
                     'percentrank', 'percentile_nearest_rank', 'correlation',
                     'dev', 'stdev', 'variance', 'median', 'mode', 'linreg',
                     'swma', 'alma', 'vwap', 'obv'):
            _ta_key = f'ta.{_fn}'
            if _ta_key in self.builtins:
                self.builtins[_fn] = self.builtins[_ta_key]
#JUST TRADES CONFIDENTIAL PROPERTY
        # ── color.* / plot / drawing — no-ops ──
        for fn in ('plot', 'plotshape', 'plotchar', 'plotarrow', 'plotcandle',
                    'plotbar', 'bgcolor', 'barcolor', 'fill', 'hline',
                    'label.new', 'label.delete', 'label.set_xy',
                    'line.new', 'line.delete', 'line.set_xy1', 'line.set_xy2',
                    'box.new', 'box.delete', 'box.set_lefttop', 'box.set_rightbottom',
                    'table.new', 'table.cell', 'table.delete',
                    'alert', 'alertcondition', 'runtime.error',
                    'log.info', 'log.warning', 'log.error',
                    'color.new', 'color.rgb', 'color.from_gradient',
                    'request.financial', 'request.quandl',
                    'request.dividends', 'request.splits', 'request.earnings'):
            self.builtins[fn] = lambda args, kw, _fn=fn: None

        # ── request.security — return fallback values ──
        self.builtins['request.security'] = lambda args, kw: self._request_security(args, kw)

        # ── array.* ──
        self.builtins['array.new_float'] = lambda args, kw: [self._num(args[1]) if len(args) > 1 else 0.0] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_int'] = lambda args, kw: [int(self._num(args[1])) if len(args) > 1 else 0] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_bool'] = lambda args, kw: [bool(args[1]) if len(args) > 1 else False] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_string'] = lambda args, kw: [str(args[1]) if len(args) > 1 else ""] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new'] = self.builtins['array.new_float']
        self.builtins['array.size'] = lambda args, kw: len(args[0]) if isinstance(args[0], list) else 0
        self.builtins['array.get'] = lambda args, kw: args[0][int(self._num(args[1]))] if isinstance(args[0], list) and int(self._num(args[1])) < len(args[0]) else None
        self.builtins['array.set'] = lambda args, kw: self._array_set(args)
        self.builtins['array.push'] = lambda args, kw: args[0].append(args[1]) if isinstance(args[0], list) else None
        self.builtins['array.pop'] = lambda args, kw: args[0].pop() if isinstance(args[0], list) and args[0] else None
        self.builtins['array.remove'] = lambda args, kw: args[0].pop(int(self._num(args[1]))) if isinstance(args[0], list) else None
        self.builtins['array.insert'] = lambda args, kw: args[0].insert(int(self._num(args[1])), args[2]) if isinstance(args[0], list) else None
        self.builtins['array.clear'] = lambda args, kw: args[0].clear() if isinstance(args[0], list) else None
        self.builtins['array.sort'] = lambda args, kw: args[0].sort() if isinstance(args[0], list) else None
        self.builtins['array.reverse'] = lambda args, kw: args[0].reverse() if isinstance(args[0], list) else None
        self.builtins['array.slice'] = lambda args, kw: args[0][int(self._num(args[1])):int(self._num(args[2]))] if isinstance(args[0], list) else []
        self.builtins['array.includes'] = lambda args, kw: args[1] in args[0] if isinstance(args[0], list) else False
        self.builtins['array.indexof'] = lambda args, kw: args[0].index(args[1]) if isinstance(args[0], list) and args[1] in args[0] else -1
        self.builtins['array.max'] = lambda args, kw: max(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.min'] = lambda args, kw: min(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.avg'] = lambda args, kw: np.mean(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.sum'] = lambda args, kw: sum(args[0]) if isinstance(args[0], list) else 0
        self.builtins['array.stdev'] = lambda args, kw: np.std(args[0]) if isinstance(args[0], list) and len(args[0]) > 1 else 0
        self.builtins['array.median'] = lambda args, kw: np.median(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.copy'] = lambda args, kw: list(args[0]) if isinstance(args[0], list) else []
        self.builtins['array.concat'] = lambda args, kw: (args[0].extend(args[1]) if isinstance(args[0], list) and isinstance(args[1], list) else None) or args[0]
        self.builtins['array.from'] = lambda args, kw: list(args)
        self.builtins['array.fill'] = lambda args, kw: self._array_fill(args)

        # str.* basics
        self.builtins['str.contains'] = lambda args, kw: str(args[1]) in str(args[0]) if len(args) >= 2 else False
        self.builtins['str.length'] = lambda args, kw: len(str(args[0])) if args else 0
        self.builtins['str.tonumber'] = lambda args, kw: float(args[0]) if args else None
        self.builtins['str.format'] = lambda args, kw: str(args[0]) if args else ""
        self.builtins['str.tostring'] = lambda args, kw: str(args[0]) if args else ""
        self.builtins['str.substring'] = lambda args, kw: str(args[0])[int(self._num(args[1])):int(self._num(args[2]))] if len(args) >= 3 else str(args[0])
        self.builtins['str.replace'] = lambda args, kw: str(args[0]).replace(str(args[1]), str(args[2])) if len(args) >= 3 else str(args[0])
        self.builtins['str.lower'] = lambda args, kw: str(args[0]).lower() if args else ""
        self.builtins['str.upper'] = lambda args, kw: str(args[0]).upper() if args else ""
        self.builtins['str.startswith'] = lambda args, kw: str(args[0]).startswith(str(args[1])) if len(args) >= 2 else False
        self.builtins['str.endswith'] = lambda args, kw: str(args[0]).endswith(str(args[1])) if len(args) >= 2 else False
        self.builtins['str.trim'] = lambda args, kw: str(args[0]).strip() if args else ""
        self.builtins['str.split'] = lambda args, kw: str(args[0]).split(str(args[1])) if len(args) >= 2 else [str(args[0])]
        self.builtins['str.pos'] = lambda args, kw: str(args[0]).find(str(args[1])) if len(args) >= 2 else -1

        # time / session functions
        self.builtins['time'] = lambda args, kw: self._time_func(args, kw)
        self.builtins['time_close'] = lambda args, kw: self._time_func(args, kw)
        self.builtins['hour'] = lambda args, kw: self._hour_func(args, kw)
        self.builtins['minute'] = lambda args, kw: self._minute_func(args, kw)
        self.builtins['second'] = lambda args, kw: 0
        self.builtins['year'] = lambda args, kw: self._year_func(args, kw)
        self.builtins['month'] = lambda args, kw: self._month_func(args, kw)
        self.builtins['dayofmonth'] = lambda args, kw: self._dom_func(args, kw)
        self.builtins['timestamp'] = lambda args, kw: self._timestamp_func(args, kw)

    def _array_set(self, args):
        if isinstance(args[0], list) and int(self._num(args[1])) < len(args[0]):
            args[0][int(self._num(args[1]))] = args[2]

    def _array_fill(self, args):
        if isinstance(args[0], list):
            val = args[1] if len(args) > 1 else 0
            for i in range(len(args[0])):
                args[0][i] = val

    # ── Helper methods ────────────────────────────────────────────────────

    def _num(self, v):
        if v is None: return 0.0
        if isinstance(v, bool): return 1.0 if v else 0.0
        try: return float(v)
        except (TypeError, ValueError): return 0.0

    def _is_na(self, v):
        if v is None: return True
        if isinstance(v, float) and np.isnan(v): return True
        return False

    def _get_history(self, name, offset=0):
        hist = self.history.get(name, [])
        idx = len(hist) - 1 - offset
        if idx < 0 or idx >= len(hist):
            return None
        return hist[idx]

    def _get_ohlcv(self, name):
        """Get OHLCV value for current bar — reads from pre-set globals (fast)."""
        if name in self.globals:
            return self.globals[name]
        if name == 'hlcc4':
            return (self.globals.get('high', 0) + self.globals.get('low', 0) + self.globals.get('close', 0) * 2) / 4
        return None

    def _get_source_history(self, source_name, length):
        """Get last `length` values of a source from history."""
        hist = self.history.get(source_name, [])
        if len(hist) < length:
            return None
        return hist[-length:]

    # ── ta.* implementations (bar-by-bar with history) ────────────────────

    def _vec_get(self, key, source_arr, length, compute_fn):
        """Get precomputed vectorized indicator value for current bar."""
        if key not in self._vec_cache:
            self._vec_cache[key] = compute_fn(source_arr, length)
        arr = self._vec_cache[key]
        i = self.bar_index
        if i < len(arr):
            v = arr[i]
            return None if np.isnan(v) else float(v)
        return None

    def _vec_source(self, source_val):
        """Map a source value to the correct precomputed array."""
        if isinstance(source_val, str):
            m = {'close': self._vec_close, 'open': self._vec_open,
                 'high': self._vec_high, 'low': self._vec_low, 'volume': self._vec_volume}
            return m.get(source_val, self._vec_close)
        # If it's a number, it's the current bar's value — need to use history-based approach
        return None

    def _ta_sma(self, args, kw):
        src_val = args[0]; length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        # Try vectorized path
        if hasattr(self, '_vec_cache'):
            key = self._call_key("sma")
            if key not in self._vec_cache:
                # Check if source is a simple OHLCV reference
                src_key = key + "_src"
                if src_key not in self.history: self.history[src_key] = []
                self.history[src_key].append(self._num(src_val))
                vals = self.history[src_key]
                if len(vals) < length: return None
                return float(np.mean(vals[-length:]))
            return self._vec_get(key, None, length, lambda s, l: None)
        # Fallback
        src_key = self._call_key("src")
        if src_key not in self.history: self.history[src_key] = []
        self.history[src_key].append(self._num(src_val))
        vals = self.history[src_key]
        if len(vals) < length: return None
        return float(np.mean(vals[-length:]))

    def _ta_ema(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        key = self._call_key("ema")
        prev = self.globals.get(key)
        if prev is None or self._is_na(prev):
            # Initialize with SMA
            buf_key = self._call_key("ema_buf")
            if buf_key not in self.history: self.history[buf_key] = []
            self.history[buf_key].append(source)
            if len(self.history[buf_key]) >= length:
                result = np.mean(self.history[buf_key][-length:])
                self.globals[key] = result
                return result
            return None
        alpha = 2.0 / (length + 1)
        result = alpha * source + (1 - alpha) * prev
        self.globals[key] = result
        return result

    def _ta_rma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        key = self._call_key("rma")
        prev = self.globals.get(key)
        if prev is None or self._is_na(prev):
            buf_key = self._call_key("rma_buf")
            if buf_key not in self.history: self.history[buf_key] = []
            self.history[buf_key].append(source)
            if len(self.history[buf_key]) >= length:
                result = np.mean(self.history[buf_key][-length:])
                self.globals[key] = result
                return result
            return None
        alpha = 1.0 / length
        result = alpha * source + (1 - alpha) * prev
        self.globals[key] = result
        return result

    def _ta_wma(self, args, kw):
        source = args[0]; length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("wma")
        if key not in self.history: self.history[key] = []
        self.history[key].append(self._num(source))
        vals = self.history[key]
        if len(vals) < length: return None
        w = np.arange(1, length+1, dtype=float)
        return np.dot(vals[-length:], w) / w.sum()

    def _ta_vwma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        vol = self._get_ohlcv('volume') or 1.0
        key = self._call_key("vwma_s"); vkey = self._call_key("vwma_v")
        if key not in self.history: self.history[key] = []; self.history[vkey] = []
        self.history[key].append(source * vol); self.history[vkey].append(vol)
        if len(self.history[key]) < length: return None
        return sum(self.history[key][-length:]) / sum(self.history[vkey][-length:])

    def _ta_hma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("hma")
        if key not in self.history: self.history[key] = []
        self.history[key].append(source)
        vals = self.history[key]
        n = length
        if len(vals) < n: return None
        s = pd.Series(vals[-max(n, int(np.sqrt(n))+n):])
        half = s.rolling(n//2).apply(lambda x: np.dot(x, np.arange(1,n//2+1))/np.arange(1,n//2+1).sum(), raw=True)
        full = s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
        diff = 2*half - full
        sq = int(np.sqrt(n))
        hma = diff.rolling(sq).apply(lambda x: np.dot(x, np.arange(1,sq+1))/np.arange(1,sq+1).sum(), raw=True)
        return hma.iloc[-1] if not np.isnan(hma.iloc[-1]) else None

    def _ta_rsi(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("rsi")
        buf_key = self._call_key("rsi_buf")
        if buf_key not in self.history: self.history[buf_key] = []
        self.history[buf_key].append(source)
        vals = self.history[buf_key]
        if len(vals) < length + 1: return None
        deltas = np.diff(vals[-(length+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        prev = self.globals.get(key)
        if prev is None:
            avg_gain = np.mean(gains); avg_loss = np.mean(losses)
        else:
            prev_ag, prev_al = prev
            avg_gain = (prev_ag * (length-1) + gains[-1]) / length
            avg_loss = (prev_al * (length-1) + losses[-1]) / length
        self.globals[key] = (avg_gain, avg_loss)
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _ta_macd(self, args, kw):
        source = self._num(args[0])
        fast_len = int(self._num(args[1] if len(args) > 1 else kw.get('fastlen', 12)))
        slow_len = int(self._num(args[2] if len(args) > 2 else kw.get('slowlen', 26)))
        sig_len = int(self._num(args[3] if len(args) > 3 else kw.get('siglen', 9)))
        # Need to compute EMA of source at two lengths, then EMA of diff
        buf = self._call_key("macd_buf")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        vals = self.history[buf]
        if len(vals) < slow_len: return (None, None, None)
        s = pd.Series(vals)
        fast_ema = s.ewm(span=fast_len, adjust=False).mean().iloc[-1]
        slow_ema = s.ewm(span=slow_len, adjust=False).mean().iloc[-1]
        macd_line = fast_ema - slow_ema
        # Signal line needs macd history
        macd_buf = self._call_key("macd_line")
        if macd_buf not in self.history: self.history[macd_buf] = []
        self.history[macd_buf].append(macd_line)
        if len(self.history[macd_buf]) >= sig_len:
            signal = pd.Series(self.history[macd_buf]).ewm(span=sig_len, adjust=False).mean().iloc[-1]
        else:
            signal = None
        hist = (macd_line - signal) if signal is not None else None
        return (macd_line, signal, hist)

    def _ta_bb(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 20))
        mult = self._num(args[2] if len(args) > 2 else kw.get('mult', 2.0))
        buf = self._call_key("bb")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return (None, None, None)
        vals = self.history[buf][-length:]
        basis = np.mean(vals); dev = np.std(vals)
        return (basis + mult*dev, basis, basis - mult*dev)

    def _ta_bbw(self, args, kw):
        result = self._ta_bb(args, kw)
        if result[0] is None: return None
        upper, basis, lower = result
        return (upper - lower) / basis if basis != 0 else 0

    def _ta_atr(self, args, kw):
        length = int(self._num(args[0] if args else kw.get('length', 14)))
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low'); c_prev = self._get_history('close', 1)
        if c_prev is None: c_prev = self._get_ohlcv('close')
        tr = max(h-l, abs(h-c_prev), abs(l-c_prev))
        buf = self._call_key("atr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(tr)
        prev = self.globals.get(buf)
        if prev is None:
            if len(self.history[buf]) < length: return None
            result = np.mean(self.history[buf][-length:])
        else:
            result = (prev * (length-1) + tr) / length
        self.globals[buf] = result
        return result

    def _ta_tr(self, args, kw):
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low')
        c_prev = self._get_history('close', 1)
        if c_prev is None: c_prev = self._get_ohlcv('close')
        return max(h-l, abs(h-c_prev), abs(l-c_prev))

    def _ta_stoch(self, args, kw):
        high = self._num(args[0]); low = self._num(args[1]); close = self._num(args[2])
        k_len = int(self._num(args[3] if len(args) > 3 else 14))
        hbuf = self._call_key("stoch_h"); lbuf = self._call_key("stoch_l")
        if hbuf not in self.history: self.history[hbuf] = []; self.history[lbuf] = []
        self.history[hbuf].append(high); self.history[lbuf].append(low)
        if len(self.history[hbuf]) < k_len: return None
        hh = max(self.history[hbuf][-k_len:]); ll = min(self.history[lbuf][-k_len:])
        if hh == ll: return 50.0
        return 100.0 * (close - ll) / (hh - ll)

    def _ta_cci(self, args, kw):
        length = int(self._num(args[0] if args else 20))
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        buf = self._call_key("cci")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(tp)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        mean = np.mean(vals); mad = np.mean(np.abs(np.array(vals) - mean))
        if mad == 0: return 0
        return (tp - mean) / (0.015 * mad)

    def _ta_mfi(self, args, kw):
        length = int(self._num(args[1] if len(args) > 1 else args[0] if args else 14))
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        vol = self._get_ohlcv('volume') or 1.0
        mf = tp * vol
        buf = self._call_key("mfi"); tp_buf = self._call_key("mfi_tp")
        if buf not in self.history: self.history[buf] = []; self.history[tp_buf] = []
        self.history[buf].append(mf); self.history[tp_buf].append(tp)
        if len(self.history[buf]) < length + 1: return None
        pos = sum(self.history[buf][-(length):][i] for i in range(length) if self.history[tp_buf][-(length):][i] > self.history[tp_buf][-(length+1):-1][i])
        neg = sum(self.history[buf][-(length):][i] for i in range(length) if self.history[tp_buf][-(length):][i] <= self.history[tp_buf][-(length+1):-1][i])
        if neg == 0: return 100.0
        return 100.0 - 100.0 / (1.0 + pos/neg)

    def _ta_adx(self, args, kw):
        """Returns (plus_di, minus_di, adx) tuple — matches ta.dmi() in Pine Script."""
        di_len = int(self._num(args[0] if args else 14))
        adx_smooth = int(self._num(args[1] if len(args) > 1 else di_len))
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low')
        prev_h = self._get_history('high', 1) or h; prev_l = self._get_history('low', 1) or l
        up_move = h - prev_h; down_move = prev_l - l
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        # Use RMA for smoothing (like TradingView)
        buf_p = self._call_key("dmi_p"); buf_m = self._call_key("dmi_m")
        buf_atr = self._call_key("dmi_atr"); buf_dx = self._call_key("dmi_dx")
        for b in (buf_p, buf_m, buf_atr, buf_dx):
            if b not in self.history: self.history[b] = []

        # True Range for ATR
        c_prev = self._get_history('close', 1) or self._get_ohlcv('close')
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))

        self.history[buf_p].append(plus_dm)
        self.history[buf_m].append(minus_dm)
        self.history[buf_atr].append(tr)

        if len(self.history[buf_p]) < di_len:
            return (None, None, None)

        # RMA smoothing
        prev_state = self.globals.get(buf_p + '_rma')
        if prev_state is None:
            sm_plus = np.mean(self.history[buf_p][-di_len:])
            sm_minus = np.mean(self.history[buf_m][-di_len:])
            sm_atr = np.mean(self.history[buf_atr][-di_len:])
        else:
            sm_plus, sm_minus, sm_atr = prev_state
            alpha = 1.0 / di_len
            sm_plus = alpha * plus_dm + (1 - alpha) * sm_plus
            sm_minus = alpha * minus_dm + (1 - alpha) * sm_minus
            sm_atr = alpha * tr + (1 - alpha) * sm_atr

        self.globals[buf_p + '_rma'] = (sm_plus, sm_minus, sm_atr)

        plus_di = 100 * sm_plus / sm_atr if sm_atr > 0 else 0
        minus_di = 100 * sm_minus / sm_atr if sm_atr > 0 else 0
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

        self.history[buf_dx].append(dx)

        # Smooth ADX
        prev_adx = self.globals.get(buf_dx + '_rma')
        if prev_adx is None:
            if len(self.history[buf_dx]) < adx_smooth:
                return (plus_di, minus_di, None)
            adx = np.mean(self.history[buf_dx][-adx_smooth:])
        else:
            alpha = 1.0 / adx_smooth
            adx = alpha * dx + (1 - alpha) * prev_adx
        self.globals[buf_dx + '_rma'] = adx

        return (plus_di, minus_di, adx)

    def _ta_supertrend(self, args, kw):
        factor = self._num(args[0] if args else 3.0); length = int(self._num(args[1] if len(args) > 1 else 10))
        atr_val = self._ta_atr([length], kw) or 0
        hl2 = (self._get_ohlcv('high') + self._get_ohlcv('low')) / 2
        up = hl2 + factor * atr_val; dn = hl2 - factor * atr_val
        close = self._get_ohlcv('close')
        key = self._call_key("st")
        prev = self.globals.get(key, {'dir': 1, 'up': up, 'dn': dn})
        if prev['dir'] == 1:
            dn = max(dn, prev['dn']) if close > prev['dn'] else dn
            direction = 1 if close >= dn else -1
        else:
            up = min(up, prev['up']) if close < prev['up'] else up
            direction = -1 if close <= up else 1
        self.globals[key] = {'dir': direction, 'up': up, 'dn': dn}
        st_val = dn if direction == 1 else up
        return (st_val, direction)

    def _ta_crossover(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("co_a"); key_b = self._call_key("co_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        # Read previous values BEFORE appending current
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return a > b and a_prev <= b_prev

    def _ta_crossunder(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("cu_a"); key_b = self._call_key("cu_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        # Read previous values BEFORE appending current
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return a < b and a_prev >= b_prev

    def _ta_cross(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("cx_a"); key_b = self._call_key("cx_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return (a > b and a_prev <= b_prev) or (a < b and a_prev >= b_prev)

    def _ta_highest(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("highest")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return max(self.history[buf][-length:])

    def _ta_lowest(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("lowest")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return min(self.history[buf][-length:])

    def _ta_highestbars(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("hbars")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return -(length - 1 - np.argmax(vals))

    def _ta_lowestbars(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("lbars")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return -(length - 1 - np.argmin(vals))

    def _ta_change(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("change")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return None
        return source - self.history[buf][-(length+1)]

    def _ta_roc(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 9))
        buf = self._call_key("roc")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return None
        prev = self.history[buf][-(length+1)]
        return 100 * (source - prev) / prev if prev != 0 else 0

    def _ta_sum(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("sum")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return sum(self.history[buf][-length:])

    def _ta_valuewhen(self, args, kw):
        cond = bool(args[0]); source = args[1]; occur = int(self._num(args[2])) if len(args) > 2 else 0
        buf = self._call_key("vw")
        if buf not in self.history: self.history[buf] = []
        if cond: self.history[buf].append(source)
        idx = -(occur + 1)
        return self.history[buf][idx] if abs(idx) <= len(self.history[buf]) else None

    def _ta_barssince(self, args, kw):
        cond = bool(args[0])
        buf = self._call_key("bs")
        if cond: self.globals[buf] = 0
        elif buf in self.globals: self.globals[buf] += 1
        return self.globals.get(buf)

    def _ta_cum(self, args, kw):
        source = self._num(args[0])
        buf = self._call_key("cum")
        self.globals[buf] = self.globals.get(buf, 0) + source
        return self.globals[buf]

    def _ta_rising(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("rising")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return False
        return all(self.history[buf][-(i+1)] > self.history[buf][-(i+2)] for i in range(length))

    def _ta_falling(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("falling")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return False
        return all(self.history[buf][-(i+1)] < self.history[buf][-(i+2)] for i in range(length))

    def _ta_pivothigh(self, args, kw):
        source = self._num(args[0]) if len(args) > 2 else self._get_ohlcv('high')
        lb = int(self._num(args[-2] if len(args) >= 2 else 5))
        rb = int(self._num(args[-1] if len(args) >= 1 else 5))
        buf = self._call_key("ph")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < lb + rb + 1: return None
        idx = -(rb + 1)
        pivot = self.history[buf][idx]
        left = self.history[buf][idx-lb:idx]
        right = self.history[buf][idx+1:] if rb > 0 else []
        if all(pivot >= v for v in left) and all(pivot >= v for v in right): return pivot
        return None

    def _ta_pivotlow(self, args, kw):
        source = self._num(args[0]) if len(args) > 2 else self._get_ohlcv('low')
        lb = int(self._num(args[-2] if len(args) >= 2 else 5))
        rb = int(self._num(args[-1] if len(args) >= 1 else 5))
        buf = self._call_key("pl")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < lb + rb + 1: return None
        idx = -(rb + 1)
        pivot = self.history[buf][idx]
        left = self.history[buf][idx-lb:idx]
        right = self.history[buf][idx+1:] if rb > 0 else []
        if all(pivot <= v for v in left) and all(pivot <= v for v in right): return pivot
        return None

    def _ta_percentrank(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("pr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return 100 * sum(1 for v in vals if v <= source) / length

    def _ta_percentile(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        pct = self._num(args[2] if len(args) > 2 else 50)
        buf = self._call_key("pctl")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.percentile(self.history[buf][-length:], pct)

    def _ta_correlation(self, args, kw):
        s1 = self._num(args[0]); s2 = self._num(args[1])
        length = int(self._num(args[2] if len(args) > 2 else 14))
        b1 = self._call_key("corr1"); b2 = self._call_key("corr2")
        if b1 not in self.history: self.history[b1] = []; self.history[b2] = []
        self.history[b1].append(s1); self.history[b2].append(s2)
        if len(self.history[b1]) < length: return None
        return np.corrcoef(self.history[b1][-length:], self.history[b2][-length:])[0,1]

    def _ta_dev(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("dev")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.std(self.history[buf][-length:])

    def _ta_variance(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("var")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.var(self.history[buf][-length:])

    def _ta_median(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("med")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.median(self.history[buf][-length:])

    def _ta_mode(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("mode")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(round(source, 4))
        if len(self.history[buf]) < length: return None
        from collections import Counter
        c = Counter(self.history[buf][-length:])
        return c.most_common(1)[0][0]

    def _ta_linreg(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        offset = int(self._num(args[2] if len(args) > 2 else 0))
        buf = self._call_key("lr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        y = np.array(self.history[buf][-length:]); x = np.arange(length)
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (length - 1 - offset)

    def _ta_swma(self, args, kw):
        source = self._num(args[0])
        buf = self._call_key("swma")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < 4: return None
        v = self.history[buf][-4:]
        return (v[0]*1 + v[1]*2 + v[2]*2 + v[3]*1) / 6

    def _ta_alma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 9))
        offset = self._num(args[2] if len(args) > 2 else 0.85)
        sigma = self._num(args[3] if len(args) > 3 else 6)
        buf = self._call_key("alma")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = np.array(self.history[buf][-length:])
        m = offset * (length - 1); s = length / sigma
        w = np.exp(-((np.arange(length) - m)**2) / (2*s*s))
        return np.dot(vals, w) / w.sum()

    def _ta_vwap(self, args, kw):
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        vol = self._get_ohlcv('volume') or 1.0
        buf_tv = self._call_key("vwap_tv"); buf_v = self._call_key("vwap_v")
        self.globals[buf_tv] = self.globals.get(buf_tv, 0) + tp * vol
        self.globals[buf_v] = self.globals.get(buf_v, 0) + vol
        return self.globals[buf_tv] / self.globals[buf_v] if self.globals[buf_v] > 0 else tp

    def _ta_obv(self, args, kw):
        close = self._get_ohlcv('close'); vol = self._get_ohlcv('volume') or 0
        prev_close = self._get_history('close', 1)
        buf = self._call_key("obv")
        prev_obv = self.globals.get(buf, 0)
        if prev_close is not None:
            if close > prev_close: prev_obv += vol
            elif close < prev_close: prev_obv -= vol
        self.globals[buf] = prev_obv
        return prev_obv

    # ── input.* ───────────────────────────────────────────────────────────
# CREATED BY JUST TRADES
    # ── Time / Session functions ────────────────────────────────────────

    def _get_bar_timestamp(self):
        """Get current bar's timestamp."""
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index[self.bar_index]
        return None

    def _time_func(self, args, kw):
        """time(timeframe, session, timezone) — returns timestamp if bar is in session, na otherwise."""
        ts = self._get_bar_timestamp()
        if ts is None:
            return self.globals.get('time', 0)

        # If called with session string, check if current bar is within session
        if len(args) >= 2:
            session = str(args[1]) if args[1] else None
            if session and '-' in session:
                try:
                    parts = session.split('-')
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    sh = int(start_str[:2]); sm = int(start_str[2:4]) if len(start_str) >= 4 else 0
                    eh = int(end_str[:2]); em = int(end_str[2:4]) if len(end_str) >= 4 else 0
                    bar_minutes = ts.hour * 60 + ts.minute
                    start_minutes = sh * 60 + sm
                    end_minutes = eh * 60 + em
                    if start_minutes <= bar_minutes < end_minutes:
                        return int(ts.timestamp() * 1000)
                    return None
                except (ValueError, IndexError):
                    pass

        return int(ts.timestamp() * 1000)

    def _hour_func(self, args, kw):
        """hour(time, timezone) — extract hour from timestamp."""
        if args:
            ts = self._get_bar_timestamp()
            if ts is None:
                return 0
            # If timezone argument provided, convert
            tz_str = str(args[1]) if len(args) > 1 else None
            if tz_str:
                try:
                    import pytz
                    tz = pytz.timezone(tz_str)
                    if ts.tzinfo is None:
                        ts = pytz.utc.localize(ts)
                    return ts.astimezone(tz).hour
                except Exception:
                    pass
            return ts.hour
        return self.globals.get('hour', 0)

    def _minute_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            if ts is None:
                return 0
            tz_str = str(args[1]) if len(args) > 1 else None
            if tz_str:
                try:
                    import pytz
                    tz = pytz.timezone(tz_str)
                    if ts.tzinfo is None:
                        ts = pytz.utc.localize(ts)
                    return ts.astimezone(tz).minute
                except Exception:
                    pass
            return ts.minute
        return self.globals.get('minute', 0)

    def _year_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.year if ts else 0
        return self.globals.get('year', 0)

    def _month_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.month if ts else 0
        return self.globals.get('month', 0)

    def _dom_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.day if ts else 0
        return self.globals.get('dayofmonth', 0)

    def _timestamp_func(self, args, kw):
        """timestamp(year, month, day, hour, minute) or timestamp(datestring).
        Pine's timestamp() creates a time in the exchange timezone, returns UTC ms."""
        if len(args) >= 5:
            try:
                from datetime import datetime
                dt = datetime(int(self._num(args[0])), int(self._num(args[1])),
                             int(self._num(args[2])), int(self._num(args[3])),
                             int(self._num(args[4])))
                if self.exchange_timezone:
                    import pytz
                    tz = pytz.timezone(self.exchange_timezone)
                    dt = tz.localize(dt)
                return int(dt.timestamp() * 1000)
            except (ValueError, TypeError):
                return 0
        return 0

    def _input(self, args, kw, cast=None):
        """Handle input.*() calls — capture definition + return value (or override)."""
        defval = kw.get('defval', args[0] if args else 0)
        title = kw.get('title', args[1] if len(args) > 1 and isinstance(args[1], str) else str(args[0]) if args and isinstance(args[0], str) and cast is None else "")
        if isinstance(title, str) and title.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            title = ""  # numeric first arg isn't a title

        group = kw.get('group', '')
        tooltip = kw.get('tooltip', '')
        options = kw.get('options')
        minval = kw.get('minval')
        maxval = kw.get('maxval')
        step = kw.get('step')

        # Build input definition for UI
        input_def = {
            'title': str(title) if title else f"Input {len(self.input_defs)+1}",
            'defval': defval,
            'type': 'int' if cast == int else 'float' if cast == float else 'bool' if cast == bool else 'string',
            'group': str(group) if group else '',
            'tooltip': str(tooltip) if tooltip else '',
            'options': options if isinstance(options, list) else None,
            'minval': self._num(minval) if minval is not None else None,
            'maxval': self._num(maxval) if maxval is not None else None,
            'step': self._num(step) if step is not None else None,
        }

        # Assign a stable key based on definition count
        input_key = input_def['title']
        input_def['key'] = input_key

        # Only register on first bar to avoid duplicates
        if self.bar_index == 0:
            self.input_defs.append(input_def)

        # Check for user override
        if input_key in self.input_overrides:
            override_val = self.input_overrides[input_key]
            if cast:
                try:
                    return cast(self._num(override_val) if cast in (int, float) else override_val)
                except (ValueError, TypeError):
                    return override_val
            return override_val

        # Return default
        if cast:
            try:
                return cast(self._num(defval) if cast in (int, float) else defval)
            except (ValueError, TypeError):
                return defval
        return defval

    def _init_security_cache(self):
        """Pre-align security DataFrames to the main data's timestamps."""
        for ticker, sec_df in self.security_data.items():
            clean = ticker.upper().split(":")[-1] if ":" in ticker else ticker.upper()
            if clean in self._security_cache:
                continue
            # Normalize columns
            col_map = {c.lower(): c for c in sec_df.columns}
            cache = {}
            for field in ('open', 'high', 'low', 'close', 'volume'):
                col = col_map.get(field)
                if col:
                    # Reindex to main DataFrame timestamps using forward-fill
                    # (last known value at each main bar timestamp)
                    aligned = sec_df[col].reindex(self.df.index, method='ffill')
                    cache[field] = aligned.values.astype(float)
            # Derived fields
            if 'high' in cache and 'low' in cache:
                cache['hl2'] = (cache['high'] + cache['low']) / 2.0
            if 'high' in cache and 'low' in cache and 'close' in cache:
                cache['hlc3'] = (cache['high'] + cache['low'] + cache['close']) / 3.0
            if 'open' in cache and 'high' in cache and 'low' in cache and 'close' in cache:
                cache['ohlc4'] = (cache['open'] + cache['high'] + cache['low'] + cache['close']) / 4.0
            self._security_cache[clean] = cache
            # Also cache common aliases
            for alias in (ticker.upper(), clean):
                self._security_cache[alias] = cache

    def _request_security(self, args, kw):
        """Handle request.security() — multi-timeframe data with real DataFrames.

        Supports:
        1. Real security data passed via security_data={ticker: DataFrame}
        2. User overrides via input_overrides
        3. Static fallback defaults for common tickers (VIX, DXY, etc.)
        4. Current symbol OHLCV as last resort
        """
        ticker = str(args[0]).upper().strip() if args else ""
        clean_ticker = ticker.split(":")[-1] if ":" in ticker else ticker

        # Determine which field to return (3rd arg is the expression)
        field = 'close'  # default
        if len(args) >= 3:
            expr = args[2]
            if isinstance(expr, str) and expr.lower() in ('close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4'):
                field = expr.lower()

        # 1. Real security data (multi-timeframe DataFrames)
        for key in (clean_ticker, ticker):
            if key in self._security_cache:
                cache = self._security_cache[key]
                if field in cache and self.bar_index < len(cache[field]):
                    val = cache[field][self.bar_index]
                    if not np.isnan(val):
                        return val

        # 2. User override
        override_key = f"_security_{clean_ticker}"
        if override_key in self.input_overrides:
            return self._num(self.input_overrides[override_key])

        # 3. Static fallback defaults
        for key, val in self.security_defaults.items():
            if clean_ticker == key.upper().split(":")[-1]:
                if self.bar_index == 0:
                    self.input_defs.append({
                        'title': f"_security_{clean_ticker}",
                        'defval': val, 'type': 'float',
                        'group': 'External Data Fallbacks',
                        'tooltip': f'Fallback for request.security("{ticker}", ...)',
                        'key': f"_security_{clean_ticker}",
                        'minval': None, 'maxval': None, 'step': None, 'options': None,
                    })
                return val

        # 4. Current symbol OHLCV fallback
        if len(args) >= 3:
            expr = args[2]
            if isinstance(expr, str) and expr.lower() in ('close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4'):
                return self._get_ohlcv(expr.lower())
            if isinstance(expr, (int, float)):
                return expr

        return 0.0

    def _input_source(self, args, kw):
        defval = kw.get('defval', args[0] if args else 'close')
        title = kw.get('title', 'Source')
        group = kw.get('group', '')
        tooltip = kw.get('tooltip', '')

        if self.bar_index == 0:
            self.input_defs.append({
                'title': str(title), 'defval': str(defval) if isinstance(defval, str) else 'close',
                'type': 'source', 'group': str(group), 'tooltip': str(tooltip),
                'options': ['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'],
                'key': str(title),
            })

        key = str(title)
        if key in self.input_overrides:
            return str(self.input_overrides[key])
        if isinstance(defval, str):
            return defval
        return 'close'

    # ── strategy.* ────────────────────────────────────────────────────────

    def _strategy_decl(self, args, kw):
        if args:
            self.strategy_name = str(args[0])
        self.strategy_params = kw
        # Parse strategy() params
        if 'pyramiding' in kw:
            p = int(self._num(kw['pyramiding']))
            self.pyramiding = max(1, p)  # Pine: pyramiding=0 means no pyramiding = 1 entry max
        if 'slippage' in kw:
            self.slippage = int(self._num(kw['slippage']))
        if 'default_qty_value' in kw:
            self.default_qty = int(self._num(kw['default_qty_value']))
            self.entry_qty = self.default_qty
        if 'calc_on_every_tick' in kw:
            self.calc_on_every_tick = bool(kw['calc_on_every_tick'])
        if 'process_orders_on_close' in kw:
            self.process_orders_on_close = bool(kw['process_orders_on_close'])
        if 'commission_type' in kw:
            ct = kw['commission_type']
            if isinstance(ct, str):
                if 'cash_per_contract' in ct:
                    self.commission_type = 'cash_per_contract'
                elif 'cash_per_order' in ct:
                    self.commission_type = 'cash_per_order'
                else:
                    self.commission_type = 'percent'
        if 'commission_value' in kw:
            cv = self._num(kw['commission_value'])
            self.commission_value = cv
            # Only set commission_pct for percentage mode (backward compat)
            if self.commission_type == 'percent':
                self.commission_pct = cv / 100.0
            else:
                self.commission_pct = 0.0  # don't use pct mode
        if 'initial_capital' in kw:
            self.initial_capital = self._num(kw['initial_capital'])
        if 'margin_long' in kw:
            self.margin_long = int(self._num(kw['margin_long']))
        if 'margin_short' in kw:
            self.margin_short = int(self._num(kw['margin_short']))

    def _strategy_entry(self, args, kw):
        if len(args) < 2: return
        name = str(args[0])
        direction_val = args[1]
        if isinstance(direction_val, str):
            new_dir = 1 if 'long' in direction_val.lower() else -1
        else:
            new_dir = 1 if self._num(direction_val) >= 0 else -1

        qty = int(self._num(kw.get('qty', args[2] if len(args) > 2 else self.default_qty)))
        if qty <= 0: qty = self.default_qty

        # When condition
        when = kw.get('when', True)
        if isinstance(when, bool) and not when: return
        if self._is_na(when): return

        # Queue for next bar's open (TradingView default behavior)
        self._queued_market_orders.append({
            'type': 'entry', 'name': name, 'dir': new_dir, 'qty': qty,
        })

    def _fill_entry(self, name, new_dir, qty, fill_price):
        """Actually fill an entry order at the given price."""
        # Apply slippage
        if self.slippage > 0:
            slip = self.slippage * self.mintick
            fill_price = fill_price + slip if new_dir == 1 else fill_price - slip

        # ── Volume participation limit (execution modeling) ──
        if self.volume_participation_limit > 0:
            bar_volume = self.globals.get('volume', 0)
            if bar_volume > 0:
                max_qty = max(1, int(bar_volume * self.volume_participation_limit))
                if qty > max_qty:
                    self.partial_fills += 1
                    qty = max_qty

        # ── Market impact (execution modeling) ──
        if self.market_impact_bps > 0 and qty > 0:
            impact = fill_price * (self.market_impact_bps / 10000.0) * qty
            fill_price = fill_price + impact if new_dir == 1 else fill_price - impact

        # Pyramiding check: how many open positions in same direction
        same_dir_count = sum(1 for p in self.positions if p['dir'] == new_dir)

        # Close opposite positions first
        opposite = [p for p in self.positions if p['dir'] != new_dir]
        if opposite:
            for p in opposite:
                self._close_single_position(p, fill_price, 'Reverse')
            self.positions = [p for p in self.positions if p['dir'] == new_dir]

        # Check pyramiding limit
        if same_dir_count >= self.pyramiding:
            return

        # ── Margin check (portfolio-level) ──
        if self.margin_per_contract > 0:
            required_margin = self.margin_per_contract * qty
            if self.margin_used + required_margin > self.buying_power:
                self.margin_calls += 1
                return

        # Open new position
        pos = {'dir': new_dir, 'price': fill_price, 'qty': qty, 'bar': self.bar_index,
               'name': name, 'initial_qty': qty, 'be_activated': False, 'sl': None}

        # Apply exit rules if already registered for this entry name
        if name in self.exit_rules:
            rules = self.exit_rules[name]
            if rules.get('sl_ticks'):
                if new_dir == 1:
                    pos['sl'] = fill_price - rules['sl_ticks'] * self.mintick
                else:
                    pos['sl'] = fill_price + rules['sl_ticks'] * self.mintick

        self.positions.append(pos)
        self._update_position_state()

    def _strategy_close(self, args, kw):
        if not self.positions: return
        when = kw.get('when', True)
        if isinstance(when, bool) and not when: return
        if self._is_na(when): return

        name = str(args[0]) if args else "all"

        # Queue for next bar's open (TradingView default behavior)
        self._queued_market_orders.append({
            'type': 'close', 'name': name,
        })

    def _fill_close(self, name, fill_price):
        """Actually fill a close order at the given price."""
        if not self.positions: return

        slip = self.slippage * self.mintick if self.slippage > 0 else 0.0

        if name == "all":
            targets = list(self.positions)
        else:
            targets = [p for p in self.positions if p.get('name') == name]

        for p in targets:
            exit_price = fill_price
            if slip > 0:
                exit_price = fill_price - slip if p['dir'] == 1 else fill_price + slip
            self._close_single_position(p, exit_price, 'Signal Exit')

        self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
        self._update_position_state()

    def _strategy_order(self, args, kw):
        """Place a pending order (limit or stop). If no limit/stop, treat as market."""
        if len(args) < 2: return
        name = str(args[0])
        direction_val = args[1]
        if isinstance(direction_val, str):
            new_dir = 1 if 'long' in direction_val.lower() else -1
        else:
            new_dir = 1 if self._num(direction_val) >= 0 else -1

        qty = int(self._num(kw.get('qty', args[2] if len(args) > 2 else self.default_qty)))
        if qty <= 0: qty = self.default_qty

        limit_price = kw.get('limit')
        stop_price = kw.get('stop')

        # If no limit or stop price, execute as market order
        if limit_price is None and stop_price is None:
            return self._strategy_entry(args, kw)

        # Register as pending order (replaces any existing order with same name)
        self.pending_orders[name] = {
            'name': name,
            'dir': new_dir,
            'qty': qty,
            'limit': self._num(limit_price) if limit_price is not None else None,
            'stop': self._num(stop_price) if stop_price is not None else None,
        }

    def _strategy_cancel(self, args, kw):
        """Cancel a pending order by name."""
        if not args: return
        name = str(args[0])
        self.pending_orders.pop(name, None)

    def _strategy_cancel_all(self):
        """Cancel all pending orders."""
        self.pending_orders.clear()

    def _closedtrades_profit(self, args):
        """strategy.closedtrades.profit(index) — return profit of trade at index."""
        if not args: return 0.0
        idx = int(self._num(args[0]))
        if 0 <= idx < len(self.trades):
            return self.trades[idx].get('Profit', 0.0)
        return 0.0

    def _closedtrades_size(self, args):
        """strategy.closedtrades.size(index) — return qty of trade at index."""
        if not args: return 0
        idx = int(self._num(args[0]))
        if 0 <= idx < len(self.trades):
            return self.trades[idx].get('Contracts', 1)
        return 0

    def _closedtrades_field(self, args, field):
        """Generic accessor for closed trade fields."""
        if not args: return 0.0
        idx = int(self._num(args[0]))
        if 0 <= idx < len(self.trades):
            return self.trades[idx].get(field, 0.0)
        return 0.0

    def _fill_queued_market_orders(self):
        """Fill market orders queued from previous bar at current bar's OPEN."""
        if not self._queued_market_orders:
            return
        fill_price = self.globals['open']
        for order in self._queued_market_orders:
            if order['type'] == 'entry':
                self._fill_entry(order['name'], order['dir'], order['qty'], fill_price)
            elif order['type'] == 'close':
                self._fill_close(order['name'], fill_price)
        self._queued_market_orders.clear()

    def _check_pending_orders(self):
        """Check if any pending limit/stop orders should fill this bar.

        TradingView intrabar price path estimation:
        - If open closer to high: path = Open → High → Low → Close
        - If open closer to low:  path = Open → Low → High → Close

        Gap handling: if price gaps through a limit/stop level, fill at
        the bar's OPEN (not the limit/stop price).

        TradingView behavior: strategy.order() can both CLOSE and OPEN positions.
        If there are matching opposite-side positions, it closes them (FIFO).
        If remaining qty after closing, or if no opposite positions exist,
        it OPENS a new position in the order's direction.
        """
        if not self.pending_orders:
            return

        o = self.globals['open']
        h = self.globals['high']
        l = self.globals['low']
        filled = []

        for name, order in list(self.pending_orders.items()):
            limit_price = order.get('limit')
            stop_price = order.get('stop')
            fill_price = None

            # Limit orders: sell limit fills when high >= limit, buy limit fills when low <= limit
            # TV default (backtest_fill_limits_assumption=0): always fills at LIMIT PRICE,
            # even when open gaps through. Gap improvement only when assumption=1.
            if limit_price is not None:
                if order['dir'] == -1 and h >= limit_price:  # sell limit
                    fill_price = limit_price
                elif order['dir'] == 1 and l <= limit_price:  # buy limit
                    fill_price = limit_price

            # Stop orders: sell stop fills when low <= stop, buy stop fills when high >= stop
            # Stop orders get slippage (unlike limit orders)
            if fill_price is None and stop_price is not None:
                slip = self.slippage * self.mintick if self.slippage > 0 else 0.0
                if order['dir'] == -1 and l <= stop_price:  # sell stop
                    base = o if o <= stop_price else stop_price
                    fill_price = base - slip  # sell stop fills worse (lower)
                elif order['dir'] == 1 and h >= stop_price:  # buy stop
                    base = o if o >= stop_price else stop_price
                    fill_price = base + slip  # buy stop fills worse (higher)

            if fill_price is not None:
                qty_remaining = order['qty']
                # Close opposite-side positions first (FIFO)
                target_dir = -order['dir']
                matching = sorted(
                    [p for p in self.positions if p['dir'] == target_dir],
                    key=lambda p: p.get('bar', 0)
                )
                for p in matching:
                    if qty_remaining <= 0:
                        break
                    close_qty = min(p['qty'], qty_remaining)
                    if close_qty == p['qty']:
                        self._close_single_position(p, fill_price, f'Order Fill ({name})')
                    else:
                        partial = dict(p)
                        partial['qty'] = close_qty
                        self._close_single_position(partial, fill_price, f'Order Fill ({name})')
                        p['qty'] -= close_qty
                    qty_remaining -= close_qty

                # If remaining qty, open new position in order's direction
                if qty_remaining > 0:
                    self._fill_entry(name, order['dir'], qty_remaining, fill_price)

                self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
                self._update_position_state()
                filled.append(name)

        for name in filled:
            self.pending_orders.pop(name, None)

    def _strategy_exit(self, args, kw):
        """Register exit rules (TP/SL/trailing) for a named entry."""
        exit_name = str(args[0]) if args else ""
        from_entry = str(args[1]) if len(args) > 1 else kw.get('from_entry', '')

        rules = self.exit_rules.get(from_entry, {})

        if 'profit' in kw:
            rules['tp_ticks'] = self._num(kw['profit'])
        if 'loss' in kw:
            rules['sl_ticks'] = self._num(kw['loss'])
        if 'limit' in kw:
            rules['tp_price'] = self._num(kw['limit'])
        if 'stop' in kw:
            rules['sl_price'] = self._num(kw['stop'])
        if 'trail_points' in kw:
            rules['trail_points'] = self._num(kw['trail_points'])
        if 'trail_offset' in kw:
            rules['trail_offset'] = self._num(kw['trail_offset'])
        if 'qty' in kw:
            rules['exit_qty'] = int(self._num(kw['qty']))
        if 'qty_percent' in kw:
            rules['exit_qty_pct'] = self._num(kw['qty_percent'])

        self.exit_rules[from_entry] = rules

        # Also register without from_entry as fallback
        if from_entry:
            self.exit_rules[from_entry] = rules
        if not from_entry:
            self.exit_rules['_default'] = rules

    def _close_single_position(self, pos, exit_price, signal='Signal Exit'):
        """Close a single position and record the trade."""
        qty = pos.get('qty', pos.get('initial_qty', self.default_qty))
        if qty <= 0: return

        entry_price = pos['price']
        entry_bar = pos['bar']
        direction = pos['dir']

        pv = self.point_value
        if direction == 1:
            pnl = (exit_price - entry_price) * qty * pv
        else:
            pnl = (entry_price - exit_price) * qty * pv

        # Commission calculation based on type
        # TradingView charges commission on BOTH entry and exit orders (×2)
        if self.commission_type == 'cash_per_contract':
            commission = self.commission_value * qty * 2  # entry + exit
        elif self.commission_type == 'cash_per_order':
            commission = self.commission_value * 2  # entry + exit
        else:
            commission = (entry_price + exit_price) * qty * self.commission_pct

        # Compute run-up / drawdown
        if entry_bar < self.bar_index:
            bars = self.df.iloc[entry_bar:self.bar_index+1]
            _col_map = {c.lower(): c for c in bars.columns}
            _h_col = _col_map.get('high', 'High')
            _l_col = _col_map.get('low', 'Low')
            if direction == 1:
                runup = (bars[_h_col].max() - entry_price) * qty * pv
                dd = (entry_price - bars[_l_col].min()) * qty * pv
            else:
                runup = (entry_price - bars[_l_col].min()) * qty * pv
                dd = (bars[_h_col].max() - entry_price) * qty * pv
        else:
            runup = max(abs(pnl), 0)
            dd = max(abs(pnl), 0) if pnl < 0 else 0

        idx = self.df.index
        entry_date = str(idx[entry_bar]) if isinstance(idx, pd.DatetimeIndex) else str(entry_bar)
        exit_date = str(idx[self.bar_index]) if isinstance(idx, pd.DatetimeIndex) else str(self.bar_index)

        self.trades.append({
            'Date/Time': entry_date,
            'Exit Date/Time': exit_date,
            'Type': 'Long' if direction == 1 else 'Short',
            'Signal': signal,
            'Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Contracts': qty,
            'Profit': round(pnl - commission, 2),
            'Run-up': round(max(runup, 0), 2),
            'Drawdown': round(max(dd, 0), 2),
            'Commission': round(commission, 2),
        })

        pos['qty'] = 0  # mark as closed

    def _partial_close(self, pos, exit_price, qty_to_close, signal='Partial TP'):
        """Close part of a position."""
        if qty_to_close <= 0 or qty_to_close > pos['qty']:
            qty_to_close = pos['qty']

        entry_price = pos['price']
        entry_bar = pos['bar']
        direction = pos['dir']

        pv = self.point_value
        if direction == 1:
            pnl = (exit_price - entry_price) * qty_to_close * pv
        else:
            pnl = (entry_price - exit_price) * qty_to_close * pv

        # Commission — TradingView charges on BOTH entry and exit (×2)
        if self.commission_type == 'cash_per_contract':
            commission = self.commission_value * qty_to_close * 2
        elif self.commission_type == 'cash_per_order':
            commission = self.commission_value * 2
        else:
            commission = (entry_price + exit_price) * qty_to_close * self.commission_pct

        # Run-up/drawdown
        if entry_bar < self.bar_index:
            bars = self.df.iloc[entry_bar:self.bar_index+1]
            _col_map = {c.lower(): c for c in bars.columns}
            _h_col = _col_map.get('high', 'High')
            _l_col = _col_map.get('low', 'Low')
            if direction == 1:
                runup = (bars[_h_col].max() - entry_price) * qty_to_close * pv
                dd = (entry_price - bars[_l_col].min()) * qty_to_close * pv
            else:
                runup = (entry_price - bars[_l_col].min()) * qty_to_close * pv
                dd = (bars[_h_col].max() - entry_price) * qty_to_close * pv
        else:
            runup = max(abs(pnl), 0)
            dd = max(abs(pnl), 0) if pnl < 0 else 0

        idx = self.df.index
        entry_date = str(idx[entry_bar]) if isinstance(idx, pd.DatetimeIndex) else str(entry_bar)
        exit_date = str(idx[self.bar_index]) if isinstance(idx, pd.DatetimeIndex) else str(self.bar_index)

        self.trades.append({
            'Date/Time': entry_date,
            'Exit Date/Time': exit_date,
            'Type': 'Long' if direction == 1 else 'Short',
            'Signal': signal,
            'Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Contracts': qty_to_close,
            'Profit': round(pnl - commission, 2),
            'Run-up': round(max(runup, 0), 2),
            'Drawdown': round(max(dd, 0), 2),
            'Commission': round(commission, 2),
        })

        pos['qty'] -= qty_to_close

    def _update_position_state(self):
        """Update aggregate position state from individual positions."""
        self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
        if not self.positions:
            self.position = 0
            self.entry_price = 0.0
            self.entry_qty = self.default_qty
            self.open_qty = 0
        else:
            self.position = self.positions[0]['dir']
            total_qty = sum(p['qty'] for p in self.positions)
            self.entry_price = sum(p['price'] * p['qty'] for p in self.positions) / total_qty if total_qty > 0 else 0
            self.entry_qty = total_qty
            self.open_qty = total_qty
            self.entry_bar = self.positions[0]['bar']

        # ── Update margin tracking ──
        if self.margin_per_contract > 0:
            total_qty = sum(p['qty'] for p in self.positions)
            self.margin_used = self.margin_per_contract * total_qty

    # ── AST Execution ─────────────────────────────────────────────────────

    def execute(self):
        """Run the strategy bar-by-bar across all OHLCV data."""
        # ── Normalize column names to title-case ──
        _col_map = {c.lower(): c for c in self.df.columns}
        _get = lambda name: self.df[_col_map.get(name.lower(), name)]

        # ── Pre-compute numpy arrays for fast access (avoid df.iloc per bar) ──
        _open = _get('Open').values.astype(float)
        _high = _get('High').values.astype(float)
        _low = _get('Low').values.astype(float)
        _close = _get('Close').values.astype(float)
        _vol_key = _col_map.get('volume')
        _volume = self.df[_vol_key].values.astype(float) if _vol_key else np.zeros(self.n_bars)

        # ── Precompute vectorized indicator cache ─────────────────────
        # This runs common ta.* functions on the FULL series once using
        # pandas vectorized ops (milliseconds), then bar-by-bar just looks up values.
        self._vec_cache = {}
        self._vec_open = _open
        self._vec_high = _high
        self._vec_low = _low
        self._vec_close = _close

        # ── Pre-align security data to main timestamps ──
        if self.security_data:
            self._init_security_cache()
        self._vec_volume = _volume

        _has_dt_index = isinstance(self.df.index, pd.DatetimeIndex)
        if _has_dt_index:
            _timestamps = self.df.index
            _ts_ms = (_timestamps.astype(np.int64) // 10**6).tolist()
            # Convert to exchange timezone for dayofweek/hour/minute (TradingView uses exchange tz)
            if self.exchange_timezone:
                try:
                    import pytz
                    _etz = pytz.timezone(self.exchange_timezone)
                    _local = _timestamps.tz_localize('UTC').tz_convert(_etz) if _timestamps.tz is None else _timestamps.tz_convert(_etz)
                    _hours = _local.hour.tolist()
                    _minutes = _local.minute.tolist()
                    _dows = ((_local.weekday + 1) % 7 + 1).tolist()
                    _months = _local.month.tolist()
                    _years = _local.year.tolist()
                except Exception:
                    _hours = _timestamps.hour.tolist()
                    _minutes = _timestamps.minute.tolist()
                    _dows = ((_timestamps.weekday + 1) % 7 + 1).tolist()
                    _months = _timestamps.month.tolist()
                    _years = _timestamps.year.tolist()
            else:
                _hours = _timestamps.hour.tolist()
                _minutes = _timestamps.minute.tolist()
                _dows = ((_timestamps.weekday + 1) % 7 + 1).tolist()  # Pine: Sun=1, Mon=2, ..., Sat=7
                _months = _timestamps.month.tolist()
                _years = _timestamps.year.tolist()

        # ── Set static globals once (not every bar) ──
        g = self.globals
        g['strategy.long'] = 1
        g['strategy.short'] = -1
        # Strategy enum constants
        g['strategy.commission.percent'] = 'strategy.commission.percent'
        g['strategy.commission.cash_per_contract'] = 'strategy.commission.cash_per_contract'
        g['strategy.commission.cash_per_order'] = 'strategy.commission.cash_per_order'
        g['strategy.fixed'] = 'strategy.fixed'
        g['strategy.percent_of_equity'] = 'strategy.percent_of_equity'
        g['strategy.cash'] = 'strategy.cash'
        # Pine dayofweek constants: Sunday=1, Monday=2, ..., Saturday=7
        g['dayofweek.sunday'] = 1
        g['dayofweek.monday'] = 2
        g['dayofweek.tuesday'] = 3
        g['dayofweek.wednesday'] = 4
        g['dayofweek.thursday'] = 5
        g['dayofweek.friday'] = 6
        g['dayofweek.saturday'] = 7
        g['strategy.initial_capital'] = self.initial_capital
        g['last_bar_index'] = self.n_bars - 1
        g['barstate.isconfirmed'] = True
        g['syminfo.mintick'] = self.mintick
        g['syminfo.pointvalue'] = 1.0
        g['syminfo.tickerid'] = ''
        g['syminfo.ticker'] = ''
        g['syminfo.type'] = 'stock'
        g['syminfo.timezone'] = 'UTC'
        g['timeframe.period'] = '1D'
        g['timeframe.multiplier'] = 1
        g['timeframe.isweekly'] = False
        g['timeframe.isdaily'] = True
        g['timeframe.isintraday'] = False
#JUST TRADES CONFIDENTIAL PROPERTY
        # ── Track which user variables need history (lazy — add on first [N] access) ──
        self._tracked_vars = getattr(self, '_tracked_vars', set())
        # Pre-scan AST for [N] history references so tracking starts from bar 0
        self._prescan_history_refs(self.ast)
        _tracked_vars = self._tracked_vars
        _statements = self.ast.statements
        _running_pnl = 0.0

        for i in range(self.n_bars):
            self.bar_index = i

            # ── Fast OHLCV set (numpy arrays, no iloc) ──
            o = _open[i]; h = _high[i]; l = _low[i]; c = _close[i]; v = _volume[i]
            g['open'] = o; g['high'] = h; g['low'] = l; g['close'] = c; g['volume'] = v
            g['hl2'] = (h + l) * 0.5
            g['hlc3'] = (h + l + c) / 3.0
            g['ohlc4'] = (o + h + l + c) * 0.25
            g['bar_index'] = i
            g['barstate.isfirst'] = (i == 0)
            g['barstate.islast'] = (i == self.n_bars - 1)

            # ── Time (pre-computed lists, no .iloc) ──
            if _has_dt_index:
                g['time'] = _ts_ms[i]
                g['hour'] = _hours[i]
                g['minute'] = _minutes[i]
                g['dayofweek'] = _dows[i]
                g['month'] = _months[i]
                g['year'] = _years[i]

            # ── Record ALL history BEFORE script execution ──
            # This ensures _get_history(name, 1) at bar N returns bar N-1.
            # OHLCV values are known from the data; user variables use the
            # previous bar's computed value as a placeholder (updated after script).
            for var_name in ('open', 'high', 'low', 'close', 'volume'):
                if var_name not in self.history:
                    self.history[var_name] = []
                self.history[var_name].append(g[var_name])
            # User variable placeholders — g[var] holds previous bar's value
            for var_name in _tracked_vars:
                if var_name not in self.history:
                    self.history[var_name] = []
                self.history[var_name].append(g.get(var_name))

            # Fill market orders queued from previous bar at this bar's OPEN
            if self._queued_market_orders:
                self._fill_queued_market_orders()

            # Check TP/SL on open positions
            if self.positions:
                self._check_tp_sl()

            # Check pending limit/stop orders from strategy.order()
            if self.pending_orders:
                self._check_pending_orders()

            # ── Position state — AFTER all fills so script sees current reality ──
            _pos_size = self.open_qty * (self.position if self.positions else 0)
            g['strategy.position_size'] = _pos_size
            g['strategy.position_avg_price'] = self.entry_price if self.positions else 0
            g['strategy.opentrades'] = len(self.positions)
            g['strategy.closedtrades'] = len(self.trades)
            # Compute open profit using current bar's close
            if self.positions:
                _open_pnl = 0.0
                _pv = self.point_value
                for _p in self.positions:
                    if _p['dir'] == 1:
                        _open_pnl += (c - _p['price']) * _p['qty'] * _pv
                    else:
                        _open_pnl += (_p['price'] - c) * _p['qty'] * _pv
                g['strategy.openprofit'] = _open_pnl
            else:
                g['strategy.openprofit'] = 0.0
            g['strategy.equity'] = self.initial_capital + _running_pnl
            g['strategy.netprofit'] = _running_pnl
            # ── Portfolio-level globals ──
            g['strategy.margin_used'] = self.margin_used
            g['strategy.buying_power'] = self.buying_power

            # Execute all statements
            _prev_trade_count = len(self.trades)
            for stmt in _statements:
                try:
                    self._exec(stmt)
                except Exception as e:
                    if i <= 1 or len(self.warnings) < 50:
                        self.warnings.append(f"Bar {i}: {type(e).__name__}: {e}")

            # After bar 0, strategy() may have updated initial_capital
            if i == 0:
                g['strategy.initial_capital'] = self.initial_capital
                _running_pnl = 0.0

            # Update running PnL if new trades were closed
            if len(self.trades) > _prev_trade_count:
                _running_pnl = sum(t['Profit'] for t in self.trades)

            # ── Update buying power (portfolio tracking) ──
            if self.margin_per_contract > 0:
                self.buying_power = self.initial_capital + _running_pnl

            # ── Update user variable history with actual computed values ──
            # (Placeholders were appended before script; now overwrite with real values)
            for var_name in _tracked_vars:
                if var_name in self.history and self.history[var_name]:
                    self.history[var_name][-1] = g.get(var_name)

        # Close any open positions at end
        if self.positions:
            close = _close[-1]
            for pos in list(self.positions):
                if pos.get('qty', 0) > 0:
                    self._close_single_position(pos, close, 'End of Data')
            self.positions = []
            self._update_position_state()

        # ── Attach execution quality report ──
        self.execution_report = {
            'margin_calls': self.margin_calls,
            'partial_fills': self.partial_fills,
            'margin_per_contract': self.margin_per_contract,
            'market_impact_bps': self.market_impact_bps,
            'volume_participation_limit': self.volume_participation_limit,
            'security_data_tickers': list(self._security_cache.keys()),
        }

        return self.trades

    def _prescan_history_refs(self, node):
        """Walk AST to find all variable[N] references and register them for tracking."""
        if isinstance(node, IndexAccess):
            if isinstance(node.obj, Identifier):
                self._tracked_vars.add(node.obj.name)
            elif isinstance(node.obj, DotAccess):
                # Handle dotted names like strategy.position_size[1]
                self._tracked_vars.add(self._dot_name(node.obj))
        if isinstance(node, Program):
            for s in node.statements: self._prescan_history_refs(s)
        for attr in ('left', 'right', 'operand', 'condition', 'true_val', 'false_val',
                      'value', 'obj', 'index', 'expr', 'start', 'end', 'step'):
            child = getattr(node, attr, None)
            if child and isinstance(child, ASTNode): self._prescan_history_refs(child)
        for attr in ('body', 'else_body', 'statements', 'args', 'elements', 'targets'):
            children = getattr(node, attr, None)
            if isinstance(children, list):
                for c in children:
                    if isinstance(c, ASTNode): self._prescan_history_refs(c)
        for attr in ('elif_clauses', 'cases'):
            items = getattr(node, attr, None)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, tuple):
                        for elem in item:
                            if isinstance(elem, ASTNode): self._prescan_history_refs(elem)
                            elif isinstance(elem, list):
                                for e in elem:
                                    if isinstance(e, ASTNode): self._prescan_history_refs(e)
        kw = getattr(node, 'kwargs', None)
        if isinstance(kw, dict):
            for v in kw.values():
                if isinstance(v, ASTNode): self._prescan_history_refs(v)

    def _check_tp_sl(self):
        """Check TP/SL/trailing/breakeven for all open positions."""
        # Use globals instead of df.iloc
        closed_any = False

        for pos in list(self.positions):
            if pos.get('qty', 0) <= 0:
                continue

            entry_name = pos.get('name', '')
            rules = self.exit_rules.get(entry_name, self.exit_rules.get('_default', {}))
            if not rules:
                continue

            direction = pos['dir']
            entry_price = pos['price']
            qty = pos['qty']

            # ── Stop Loss ─────────────────────────────────────────────────
            sl_price = pos.get('sl')  # dynamic SL (breakeven, trailing)
            if sl_price is None and rules.get('sl_ticks'):
                if direction == 1:
                    sl_price = entry_price - rules['sl_ticks'] * self.mintick
                else:
                    sl_price = entry_price + rules['sl_ticks'] * self.mintick
            if sl_price is None and rules.get('sl_price'):
                sl_price = rules['sl_price']

            if sl_price is not None:
                hit = (direction == 1 and self.globals['low'] <= sl_price) or \
                      (direction == -1 and self.globals['high'] >= sl_price)
                if hit:
                    self._close_single_position(pos, sl_price, 'SL Hit')
                    closed_any = True
                    continue

            # ── Take Profit (full or partial) ─────────────────────────────
            tp_ticks = rules.get('tp_ticks')
            tp_price = rules.get('tp_price')

            if tp_ticks:
                if direction == 1:
                    tp_target = entry_price + tp_ticks * self.mintick
                else:
                    tp_target = entry_price - tp_ticks * self.mintick
            elif tp_price:
                tp_target = tp_price
            else:
                tp_target = None

            if tp_target is not None:
                hit = (direction == 1 and self.globals['high'] >= tp_target) or \
                      (direction == -1 and self.globals['low'] <= tp_target)
                if hit:
                    exit_qty_pct = rules.get('exit_qty_pct')
                    exit_qty = rules.get('exit_qty')
                    if exit_qty_pct and exit_qty_pct < 100:
                        qty_close = max(1, int(pos['initial_qty'] * exit_qty_pct / 100))
                        qty_close = min(qty_close, pos['qty'])
                        self._partial_close(pos, tp_target, qty_close, 'Partial TP')
                        # Move stop to breakeven after partial TP
                        pos['sl'] = entry_price + (self.slippage * self.mintick if direction == 1 else -self.slippage * self.mintick)
                        pos['be_activated'] = True
                    elif exit_qty and exit_qty < pos['qty']:
                        self._partial_close(pos, tp_target, exit_qty, 'Partial TP')
                        pos['sl'] = entry_price
                        pos['be_activated'] = True
                    else:
                        self._close_single_position(pos, tp_target, 'TP Hit')
                    closed_any = True
                    continue

            # ── Trailing Stop ─────────────────────────────────────────────
            trail_points = rules.get('trail_points')
            trail_offset = rules.get('trail_offset', 0)
            if trail_points:
                trail_key = f"_trail_{id(pos)}"
                if direction == 1:
                    highest = self.globals.get(trail_key, entry_price)
                    if self.globals['high'] > highest:
                        highest = self.globals['high']
                    self.globals[trail_key] = highest
                    activation = entry_price + trail_points * self.mintick
                    if highest >= activation:
                        trail_sl = highest - trail_offset * self.mintick
                        if self.globals['low'] <= trail_sl:
                            self._close_single_position(pos, trail_sl, 'Trail Stop')
                            closed_any = True
                            continue
                else:
                    lowest = self.globals.get(trail_key, entry_price)
                    if self.globals['low'] < lowest:
                        lowest = self.globals['low']
                    self.globals[trail_key] = lowest
                    activation = entry_price - trail_points * self.mintick
                    if lowest <= activation:
                        trail_sl = lowest + trail_offset * self.mintick
                        if self.globals['high'] >= trail_sl:
                            self._close_single_position(pos, trail_sl, 'Trail Stop')
                            closed_any = True
                            continue

        if closed_any:
            self._update_position_state()

    def _exec(self, node) -> Any:
        if node is None: return None

        if isinstance(node, NumberLiteral): return node.value
        if isinstance(node, StringLiteral): return node.value
        if isinstance(node, BoolLiteral): return node.value
        if isinstance(node, NALiteral): return None
        if isinstance(node, ArrayLiteral): return [self._exec(e) for e in node.elements]

        if isinstance(node, Identifier):
            name = node.name
            # Check globals
            if name in self.globals: return self.globals[name]
            # Check OHLCV
            v = self._get_ohlcv(name)
            if v is not None: return v
            return None

        if isinstance(node, BinaryOp):
            # Short-circuit for and/or (Pine Script semantics)
            if node.op == 'or':
                left = self._exec(node.left)
                if not self._is_na(left) and left:
                    return True
                right = self._exec(node.right)
                return self._binary_op('or', left, right)
            if node.op == 'and':
                left = self._exec(node.left)
                if self._is_na(left) or not left:
                    return False
                right = self._exec(node.right)
                return self._binary_op('and', left, right)
            left = self._exec(node.left)
            right = self._exec(node.right)
            return self._binary_op(node.op, left, right)

        if isinstance(node, UnaryOp):
            val = self._exec(node.operand)
            if node.op == '-': return -self._num(val)
            if node.op == 'not': return not val
            return val

        if isinstance(node, TernaryOp):
            cond = self._exec(node.condition)
            return self._exec(node.true_val) if cond else self._exec(node.false_val)

        if isinstance(node, Assignment):
            val = self._exec(node.value)
            if node.is_var:
                if node.target not in self.var_inited:
                    self.globals[node.target] = val
                    self.var_inited[node.target] = True
                return self.globals.get(node.target)
            elif node.is_varip:
                if node.target not in self.var_inited:
                    self.globals[node.target] = val
                    self.var_inited[node.target] = True
                return self.globals.get(node.target)
            elif node.op == ':=':
                self.globals[node.target] = val
            elif node.op == '+=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) + self._num(val)
            elif node.op == '-=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) - self._num(val)
            elif node.op == '*=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) * self._num(val)
            elif node.op == '/=':
                d = self._num(val)
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) / d if d != 0 else None
            else:
                self.globals[node.target] = val
            return val

        if isinstance(node, TupleUnpack):
            result = self._exec(node.value)
            if isinstance(result, tuple):
                for i, name in enumerate(node.targets):
                    self.globals[name] = result[i] if i < len(result) else None
            return result

        if isinstance(node, IfStatement):
            cond = self._exec(node.condition)
            if cond:
                return self._exec_block(node.body)
            for elif_cond, elif_body in node.elif_clauses:
                if self._exec(elif_cond):
                    return self._exec_block(elif_body)
            if node.else_body:
                return self._exec_block(node.else_body)
            return None

        if isinstance(node, ForLoop):
            start = int(self._num(self._exec(node.start)))
            end = int(self._num(self._exec(node.end)))
            step = int(self._num(self._exec(node.step))) if node.step else 1
            if step == 0: step = 1
            result = None
            count = 0
            for val in range(start, end + (1 if step > 0 else -1), step):
                self.globals[node.var] = val
                result = self._exec_block(node.body)
                count += 1
                if count > 10000: break  # safety
            return result

        if isinstance(node, WhileLoop):
            result = None
            count = 0
            while self._exec(node.condition):
                result = self._exec_block(node.body)
                count += 1
                if count > 10000: break
            return result

        if isinstance(node, SwitchStatement):
            expr_val = self._exec(node.expr) if node.expr else None
            for case_cond, case_body in node.cases:
                if case_cond is None:  # default
                    return self._exec_block(case_body)
                case_val = self._exec(case_cond)
                if expr_val is not None:
                    if case_val == expr_val:
                        return self._exec_block(case_body)
                else:
                    if case_val:
                        return self._exec_block(case_body)
            return None

        if isinstance(node, FunctionDef):
            self.functions[node.name] = node
            return None

        if isinstance(node, FunctionCall):
            return self._call_function(node.name, node.args, node.kwargs)

        if isinstance(node, MethodCall):
            obj = self._exec(node.obj)
            # Try as dotted builtin: obj.method
            if isinstance(node.obj, Identifier):
                full_name = f"{node.obj.name}.{node.method}"
                if full_name in self.builtins:
                    args = [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
                # Two-level dot: ta.ema etc is handled as DotAccess -> MethodCall
            if isinstance(node.obj, DotAccess):
                full_name = f"{self._dot_name(node.obj)}.{node.method}"
                if full_name in self.builtins:
                    args = [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
            # Array method calls
            if isinstance(obj, list):
                full_name = f"array.{node.method}"
                if full_name in self.builtins:
                    args = [obj] + [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
            return None

        if isinstance(node, DotAccess):
            name = self._dot_name_full(node)
            if name in self.globals: return self.globals[name]
            if name in self.builtins:
                # It's a namespace, not a call — return the name for later resolution
                return name
            obj = self._exec(node.obj)
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return self.globals.get(name)

        if isinstance(node, IndexAccess):
            # Resolve name for history lookback — handles both simple and dotted names
            if isinstance(node.obj, Identifier):
                obj_name = node.obj.name
            elif isinstance(node.obj, DotAccess):
                obj_name = self._dot_name(node.obj)
            else:
                obj_name = None
            idx = int(self._num(self._exec(node.index)))
            # History operator: variable[N] means N bars ago
            if obj_name:
                # Register for history tracking
                if hasattr(self, '_tracked_vars'):
                    self._tracked_vars.add(obj_name)
                obj_val = self._exec(node.obj)
                if isinstance(obj_val, list):
                    return obj_val[idx] if 0 <= idx < len(obj_val) else None
                # Bar lookback
                return self._get_history(obj_name, idx)
            obj = self._exec(node.obj)
            if isinstance(obj, list):
                return obj[idx] if 0 <= idx < len(obj) else None
            return None

        return None

    def _dot_name(self, node):
        if isinstance(node, DotAccess):
            return f"{self._dot_name(node.obj)}.{node.attr}"
        if isinstance(node, Identifier):
            return node.name
        return str(node)

    def _dot_name_full(self, node):
        return self._dot_name(node)

    def _exec_block(self, stmts):
        result = None
        for s in stmts:
            result = self._exec(s)
        return result

    def _call_function(self, name, arg_nodes, kwarg_nodes):
        # Resolve dotted name for builtins
        if name in self.builtins:
            args = [self._exec(a) for a in arg_nodes]
            kwargs = {k: self._exec(v) for k, v in kwarg_nodes.items()}
            return self.builtins[name](args, kwargs)

        # User-defined functions
        if name in self.functions:
            func = self.functions[name]
            # Save globals, set params
            saved = {}
            args = [self._exec(a) for a in arg_nodes]
            kwargs = {k: self._exec(v) for k, v in kwarg_nodes.items()}
            for i, (param_name, default) in enumerate(func.params):
                saved[param_name] = self.globals.get(param_name)
                if param_name in kwargs:
                    self.globals[param_name] = kwargs[param_name]
                elif i < len(args):
                    self.globals[param_name] = args[i]
                elif default is not None:
                    self.globals[param_name] = self._exec(default)
            result = self._exec_block(func.body)
            # Restore
            for param_name, old_val in saved.items():
                if old_val is None:
                    self.globals.pop(param_name, None)
                else:
                    self.globals[param_name] = old_val
            return result

        # Type cast functions
        if name in ('int', 'float', 'bool', 'string', 'color'):
            if name in self.builtins:
                args = [self._exec(a) for a in arg_nodes]
                return self.builtins[name](args, {})

        return None

    def _binary_op(self, op, left, right):
        if op == 'and':
            if self._is_na(left) or self._is_na(right): return False
            return bool(left) and bool(right)
        if op == 'or':
            if self._is_na(left) and self._is_na(right): return False
            return bool(left if not self._is_na(left) else False) or bool(right if not self._is_na(right) else False)

        # Comparisons with None/na always return False (like Pine Script)
        if op in ('>', '<', '>=', '<='):
            if self._is_na(left) or self._is_na(right): return False

        l = self._num(left); r = self._num(right)

        if op == '+':
            if isinstance(left, str) or isinstance(right, str):
                return str(left if left is not None else '') + str(right if right is not None else '')
            if self._is_na(left) or self._is_na(right): return None
            return l + r
        if op == '-':
            if self._is_na(left) or self._is_na(right): return None
            return l - r
        if op == '*':
            if self._is_na(left) or self._is_na(right): return None
            return l * r
        if op == '/': return l / r if r != 0 and not self._is_na(left) else None
        if op == '%': return l % r if r != 0 and not self._is_na(left) else None
        if op == '>': return l > r
        if op == '<': return l < r
        if op == '>=': return l >= r
        if op == '<=': return l <= r
        if op == '==':
            if self._is_na(left) and self._is_na(right): return True
            if self._is_na(left) or self._is_na(right): return False
            # String comparison — compare directly, don't convert to numbers
            if isinstance(left, str) or isinstance(right, str):
                return str(left) == str(right)
            return l == r
        if op == '!=':
            if self._is_na(left) and self._is_na(right): return False
            if self._is_na(left) or self._is_na(right): return True
            # String comparison — compare directly, don't convert to numbers
            if isinstance(left, str) or isinstance(right, str):
                return str(left) != str(right)
            return l != r
        return None

    def to_dataframe(self):
        if not self.trades: return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        df.insert(0, 'Trade #', range(1, len(df)+1))
        df['Cum. Profit'] = df['Profit'].cumsum().round(2)
        return df
from __future__ import annotations
import sys
import math
import ast
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# RRL : Basic Language Logic

# ========== Import allow-list & safe builtins (security) ==========
ALLOWED_MODULES = {
    'math', 'random', 'time', 'datetime', 'itertools', 'functools',
    'operator', 'statistics', 'json', 'os', 'sys', 're', 'string',

}

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "int": int, "float": float, "str": str, "bool": bool, "range": range,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "enumerate": enumerate, "zip": zip, "sum": sum, "any": any, "all": all,
    "sorted": sorted, "reversed": reversed,
    "object": object, "__name__": "__main__",
}
SAFE_GLOBALS = {"__builtins__": SAFE_BUILTINS, "math": math}

# small helper: safe import used by interpreter
def _is_module_allowed(modname: str) -> bool:
    top = modname.split('.')[0]
    return top in ALLOWED_MODULES

def _safe_import_interpreter(name: str, fromlist: Optional[List[str]] = None):
    if not _is_module_allowed(name):
        raise ImportError(f"Import of module '{name}' not allowed by ALLOWED_MODULES")
    return __import__(name, fromlist=fromlist or ['*'])

# safe import wrapper to be inserted into transpiled exec globals
def _make_safe_import_for_exec():
    real_import = __import__
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        top = name.split('.')[0]
        if top not in ALLOWED_MODULES:
            raise ImportError(f"Import of module '{name}' not allowed by ALLOWED_MODULES")
        return real_import(name, globals, locals, fromlist, level)
    return _safe_import

# ========== Expression evaluation ==========
# Prefer ast.literal_eval for pure literals (safe), fall back to controlled eval.

def eval_expr(expr: str, env: Dict[str, Any], line: Optional[int] = None) -> Any:
    s = expr.strip()
    # try literal first
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    # fallback to safe eval (uses SAFE_GLOBALS)
    try:
        return eval(s, SAFE_GLOBALS, env)
    except Exception as e:
        if line:
            raise RRLRuntimeError(f"[line {line}] error evaluating expression: {e}")
        raise


def strip_comment(line: str) -> str:
    s = line
    out = []
    in_single = False
    in_double = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        if c == "#" and not in_single and not in_double:
            break
        out.append(c)
        i += 1
    return "".join(out).rstrip()

# ========== AST nodes ==========
@dataclass
class Node:
    line: int

@dataclass
class Expr(Node):
    expr: str

@dataclass
class Program(Node):
    body: List[Node]

@dataclass
class Assign(Node):
    name: str
    expr: str

@dataclass
class Display(Node):
    args_expr: str

@dataclass
class IfBlock(Node):
    branches: List[Tuple[str, List[Node]]]
    else_block: Optional[List[Node]]

@dataclass
class Case:
    pattern: str
    body: List[Node]

@dataclass
class switchBlock(Node):
    expr: Node
    cases: List[Case]
    default_block: Optional[List[Node]]

@dataclass
class RepeatBlock(Node):
    count_expr: str
    body: List[Node]

@dataclass
class WhileBlock(Node):
    cond_expr: str
    body: List[Node]

@dataclass
class DoWhileBlock(Node):
    cond_expr: str
    body: List[Node]

@dataclass
class ForBlock(Node):
    var_name: str
    iterable_expr: str
    body: List[Node]

@dataclass
class FunctionDef(Node):
    name: str
    params: List[str]
    body: List[Node]

@dataclass
class ReturnNode(Node):
    expr: Optional[str]

@dataclass
class ClassDef(Node):
    name: str
    bases: List[str]
    body: List[Node]

# New AST nodes for imports
@dataclass
class ImportNode(Node):
    modules: List[Tuple[str, Optional[str]]]

@dataclass
class FromImportNode(Node):
    module: str
    names: List[Tuple[str, Optional[str]]]

# ========== Parser ==========
class ParserError(Exception):
    pass

class Parser:
    def __init__(self, lines: List[str]):
        self.lines = lines
        self.i = 0

    def parse(self) -> Program:
        body = self.parse_block(stop_tokens=[])
        return Program(line=1, body=body)

    def current(self):
        if self.i >= len(self.lines):
            return (len(self.lines)+1, "")
        return (self.i+1, self.lines[self.i])

    def advance(self):
        self.i += 1

    def parse_block(self, stop_tokens: List[str]) -> List[Node]:
        nodes: List[Node] = []
        while self.i < len(self.lines):
            line_no, raw = self.current()
            line = strip_comment(raw).strip()
            if line == "":
                self.advance()
                continue

            lower = line.lower()
            if any(lower == t or lower.startswith(t + " ") for t in stop_tokens):
                return nodes

            # IMPORT support
            if lower.startswith("import "):
                rest = line[len("import "):].strip()
                modules = []
                for part in [p.strip() for p in rest.split(",") if p.strip()]:
                    if " as " in part.lower():
                        idx = part.lower().rfind(" as ")
                        mod = part[:idx].strip()
                        alias = part[idx+4:].strip()
                        modules.append((mod, alias))
                    else:
                        modules.append((part, None))
                nodes.append(ImportNode(line=line_no, modules=modules))
                self.advance()
                continue

            if lower.startswith("from "):
                idx = lower.find(" import ")
                if idx == -1:
                    raise ParserError(f"[line {line_no}] invalid from-import syntax")
                module_part = line[5:idx].strip()
                names_part = line[idx+8:].strip()
                if not names_part:
                    raise ParserError(f"[line {line_no}] from-import needs names")
                names = []
                for part in [p.strip() for p in names_part.split(",") if p.strip()]:
                    if " as " in part.lower():
                        idx2 = part.lower().rfind(" as ")
                        name = part[:idx2].strip()
                        alias = part[idx2+4:].strip()
                        names.append((name, alias))
                    else:
                        names.append((part, None))
                nodes.append(FromImportNode(line=line_no, module=module_part, names=names))
                self.advance()
                continue

            if lower.startswith("display(") and line.endswith(")"):
                inside = line[len("display("):-1].strip()
                nodes.append(Display(line=line_no, args_expr=inside))
                self.advance()
                continue

            if lower.startswith("class "):
                nodes.append(self.parse_class())
                continue

            if lower.startswith("if "):
                nodes.append(self.parse_if())
                continue

            if lower.startswith("match "):
                nodes.append(self.parse_match())
                continue

            if lower.startswith("not "):
                nodes.append(Expr(line=line_no,expr=line))
                self.advance()
                continue

            if lower.startswith("or "):
                nodes.append(self.parse_or())
                continue

            if lower.startswith("and "):
                nodes.append(self.parse_and())
                continue

            if lower.startswith("not "):
                nodes.append(self.parse_not())
                continue

            if lower.startswith("repeat "):
                nodes.append(self.parse_repeat())
                continue

            if lower.startswith("do while "):
                nodes.append(self.parse_do_while())
                continue

            if lower.startswith("while "):
                nodes.append(self.parse_while())
                continue

            if lower.startswith("for "):
                nodes.append(self.parse_for())
                continue

            if lower.startswith("def "):
                nodes.append(self.parse_def())
                continue

            if lower.startswith("return"):
                expr = line[len("return"):].strip()
                expr = expr if expr != "" else None
                nodes.append(ReturnNode(line=line_no, expr=expr))
                self.advance()
                continue

            if "=" in line and not lower.startswith(("elif ", "else", "endif", "endrepeat", "endwhile", "endfor", "enddef", "endclass")):
                left, right = line.split("=", 1)
                name = left.strip()
                parts = name.split(".")
                if not all(p.isidentifier() for p in parts):
                    raise ParserError(f"[line {line_no}] Invalid assignment target: {name}")
                expr = right.strip()
                nodes.append(Assign(line=line_no, name=name, expr=expr))
                self.advance()
                continue

            nodes.append(Expr(line=line_no, expr=line))
            self.advance()
            continue

        if stop_tokens:
            exp = " or ".join(stop_tokens)
            raise ParserError(f"Unexpected end of file: expected {exp}")
        return nodes

    def parse_if(self) -> IfBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        cond = header[len("if "):].strip()
        self.advance()
        branches = []
        body = self.parse_block(stop_tokens=["elif", "else", "endif"])
        branches.append((cond, body))
        else_block = None
        while self.i < len(self.lines):
            line_no, raw = self.current()
            line = strip_comment(raw).strip()
            lower = line.lower()
            if lower.startswith("elif "):
                cond = line[5:].strip()
                self.advance()
                body = self.parse_block(stop_tokens=["elif", "else", "endif"])
                branches.append((cond, body))
            elif lower == "else":
                self.advance()
                else_block = self.parse_block(stop_tokens=["endif"])
            elif lower == "endif":
                self.advance()
                return IfBlock(line=start_line, branches=branches, else_block=else_block)
            else:
                raise ParserError(f"[line {line_no}] Expected elif/else/endif, got: {line}")
        raise ParserError(f"[line {start_line}] if-block not closed")

    def parse_match(self) -> switchBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        expr = header[len("match "):].strip()
        self.advance()

        cases: List[Case] = []
        default_block: Optional[List[Node]] = None

        while self.i < len(self.lines):
            line_no, raw = self.current()
            line = strip_comment(raw).strip()
            lower = line.lower()

            if lower.startswith("case "):
                pattern_str = line[len("case "):].strip()
                self.advance()
                body = self.parse_block(stop_tokens=["case", "default", "endmatch"])
                cases.append(Case(pattern=Expr(line=line_no, expr=pattern_str), body=body))

            elif lower == "default":
                self.advance()
                default_block = self.parse_block(stop_tokens=["endmatch"])

            elif lower == "endmatch":
                self.advance()
                return switchBlock(
                    line=start_line,
                    expr=Expr(line=start_line, expr=expr),
                    cases=cases,
                    default_block=default_block,
                )

            else:
                raise ParserError(
                    f"[line {line_no}] Unexpected token in match block: '{line}' "
                    f"(expected case/default/endmatch)"
                )

        raise ParserError(f"[line {start_line}] match-block not closed")


    def parse_or(self) -> Expr:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        expr = header
        self.advance()
        return Expr(line=start_line, expr=expr)

    def parse_and(self) -> Expr:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        expr = header
        self.advance()
        return Expr(line=start_line, expr=expr)    

    def parse_not(self) -> Expr:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        expr = header
        self.advance()
        return Expr(line=start_line, expr=expr)

    def parse_repeat(self) -> RepeatBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        count_expr = header[len("repeat "):].strip()
        self.advance()
        body = self.parse_block(stop_tokens=["endrepeat"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "endrepeat":
            self.advance()
            return RepeatBlock(line=start_line, count_expr=count_expr, body=body)
        raise ParserError(f"[line {start_line}] repeat-block not closed")

    def parse_while(self) -> WhileBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        cond_expr = header[len("while "):].strip()
        self.advance()
        body = self.parse_block(stop_tokens=["endwhile"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "endwhile":
            self.advance()
            return WhileBlock(line=start_line, cond_expr=cond_expr, body=body)
        raise ParserError(f"[line {start_line}] while-block not closed")

    def parse_do_while(self) -> DoWhileBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        cond_expr = header[len("do while "):].strip()
        if cond_expr == "":
            raise ParserError(f"[line {start_line}] do while requires a condition")
        self.advance()
        body = self.parse_block(stop_tokens=["endwhile"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "endwhile":
            self.advance()
            return DoWhileBlock(line=start_line, cond_expr=cond_expr, body=body)
        raise ParserError(f"[line {start_line}] do while-block not closed")

    def parse_for(self) -> ForBlock:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        rest = header[len("for "):].strip()
        lower_rest = rest.lower()
        idx = lower_rest.find(" in ")
        if idx == -1:
            raise ParserError(f"[line {start_line}] for-loop requires 'in' keyword")
        var_name = rest[:idx].strip()
        iterable_expr = rest[idx+4:].strip()
        if not var_name.isidentifier():
            raise ParserError(f"[line {start_line}] invalid for-loop variable: {var_name}")
        if iterable_expr == "":
            raise ParserError(f"[line {start_line}] for-loop requires an iterable expression")
        self.advance()
        body = self.parse_block(stop_tokens=["endfor"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "endfor":
            self.advance()
            return ForBlock(line=start_line, var_name=var_name, iterable_expr=iterable_expr, body=body)
        raise ParserError(f"[line {start_line}] for-loop not closed")

    def parse_class(self) -> ClassDef:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        rest = header[len("class "):].strip()
        name = rest
        bases: List[str] = []
        if "(" in rest and rest.endswith(")"):
            name = rest.split("(", 1)[0].strip()
            bases_str = rest[rest.find("(")+1:-1].strip()
            bases = [b.strip() for b in bases_str.split(",") if b.strip()]
        if not name.isidentifier():
            raise ParserError(f"[line {start_line}] invalid class name: {name}")
        self.advance()
        body = self.parse_block(stop_tokens=["endclass"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "endclass":
            self.advance()
            return ClassDef(line=start_line, name=name, bases=bases, body=body)
        raise ParserError(f"[line {start_line}] class-block not closed with endclass")
        
    def parse_def(self) -> FunctionDef:
        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        rest = header[len("def "):].strip()
        header = rest
        if "(" in header and header.endswith(")"):
            name = header.split("(", 1)[0].strip()
            params_raw = header[len(name):].strip()
            if not (params_raw.startswith("(") and params_raw.endswith(")")):
                raise ParserError(f"[line {start_line}] invalid def header: {header}")
            params_str = params_raw[1:-1].strip()
            params = [p.strip() for p in params_str.split(",") if p.strip()] if params_str else []
        else:
            name = header
            params = []
        if not name.isidentifier():
            raise ParserError(f"[line {start_line}] invalid function name: {name}")
        self.advance()
        body = self.parse_block(stop_tokens=["enddef"])
        if self.i < len(self.lines) and strip_comment(self.lines[self.i]).strip().lower() == "enddef":
            self.advance()
            return FunctionDef(line=start_line, name=name, params=params, body=body)
        raise ParserError(f"[line {start_line}] def-block not closed with enddef")

# ========== Runtime (interpreter) ==========
class RRLRuntimeError(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

#RRL : Rototics domain
class RobotSim:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.battery = 100.0
        self.status = "idle"

    @property
    def position(self):
        return (self.x, self.y)

    def move(self, meters: float):
        rad = math.radians(self.heading)
        dx = meters * math.cos(rad)
        dy = meters * math.sin(rad)
        self.x += dx
        self.y += dy
        self.battery = max(0.0, self.battery - abs(meters) * 0.5)
        self.status = "moving"

    def rotate(self, deg: float):
        self.heading = (self.heading + deg) % 360.0
        self.battery = max(0.0, self.battery - abs(deg) * 0.01)
        self.status = "rotating"

    def stop(self):
        self.status = "idle"

    def __repr__(self):
        return f"RobotSim(pos=({self.x:.2f},{self.y:.2f}), h={self.heading:.1f}, bat={self.battery:.1f}, status={self.status})"

# helper constructors for easy collection creation
def _list_of(*items):
    return list(items)

def _tuple_of(*items):
    return tuple(items)

def _set_of(*items):
    return set(items)

def _dict_of(*args, **kwargs):
    # allow either dict_of(k1, v1, k2, v2, ...)
    # or dict_of({'a':1, ...}) or dict_of(k1=v1, ...)
    if len(args) == 1 and isinstance(args[0], dict):
        return dict(args[0])
    if args and len(args) % 2 == 0:
        it = iter(args)
        return {k: v for k, v in zip(it, it)}
    if kwargs:
        d = dict(kwargs)
        # if args present and odd, raise
        if args:
            raise TypeError("dict_of: invalid arguments")
        return d
    return {}

class Interpreter:
    def __init__(self, output: Optional[List[str]] = None):
        self.env: Dict[str, Any] = {}
        self.output = output
        self.env['robot'] = RobotSim()
        # helpful collection constructors available in RRL
        self.env['list_of'] = _list_of
        self.env['tuple_of'] = _tuple_of
        self.env['set_of'] = _set_of
        self.env['dict_of'] = _dict_of

    def run_program(self, prog: Program):
        try:
            self.exec_block(prog.body)
        except RRLRuntimeError as e:
            self._emit(f"RuntimeError: {e}")
        except ReturnSignal as rs:
            self._emit(f"Return outside function: {rs.value}")
        except Exception:
            traceback.print_exc()

    def _emit(self, *args):
        text = " ".join(str(x) for x in args)
        if self.output is not None:
            self.output.append(text)
        else:
            print(text)

    def exec_block(self, nodes: List[Node]):
        for node in nodes:
            self.exec(node)

    def _set_dotted(self, name: str, value: Any):
        parts = name.split(".")
        obj_name = parts[0]
        if obj_name not in self.env:
            raise RRLRuntimeError(f"Unknown name: {obj_name}")
        obj = self.env[obj_name]
        for attr in parts[1:-1]:
            obj = getattr(obj, attr)
        setattr(obj, parts[-1], value)

    def exec(self, node: Node):
        if isinstance(node, Assign):
            value = eval_expr(node.expr, self.env, line=node.line)
            if "." in node.name:
                try:
                    self._set_dotted(node.name, value)
                except Exception as e:
                    raise RRLRuntimeError(f"[line {node.line}] attribute assignment failed: {e}")
            else:
                self.env[node.name] = value

        elif isinstance(node, ImportNode):
            for mod, alias in node.modules:
                try:
                    imported = _safe_import_interpreter(mod, fromlist=['*'])
                except Exception as e:
                    raise RRLRuntimeError(f"[line {node.line}] import failed: {e}")
                name = alias if alias else mod.split('.')[0]
                self.env[name] = imported

        elif isinstance(node, FromImportNode):
            for name, alias in node.names:
                try:
                    mod_obj = _safe_import_interpreter(node.module, fromlist=[name])
                    val = getattr(mod_obj, name)
                except Exception as e:
                    raise RRLRuntimeError(f"[line {node.line}] from-import failed: {e}")
                dest = alias if alias else name
                self.env[dest] = val

        elif isinstance(node, switchBlock):
            switch_value = eval_expr(node.expr.expr, self.env, line=node.line)
            executed = False
            for case in node.cases:
                case_value = eval_expr(case.pattern.expr, self.env, line=case.pattern.line)
                if switch_value == case_value:
                    self.exec_block(case.body)
                    executed = True
                    break
            if not executed and node.default_block is not None:
                self.exec_block(node.default_block)

        elif isinstance(node, Display):
            text = node.args_expr.strip()
            if text == "":
                self._emit("")
                return
            try:
                tuple_expr = f"({text},)" if "," not in text else f"({text})"
                args = eval_expr(tuple_expr, self.env, line=node.line)
                if not isinstance(args, tuple):
                    args = (args,)
                self._emit(*args)
            except RRLRuntimeError:
                raise
            except Exception:
                # fallback to raw text if eval fails
                self._emit(text)

        elif isinstance(node, Expr):
            try:
                eval_expr(node.expr, self.env, line=node.line)
            except Exception as e:
                raise RRLRuntimeError(f"[line {node.line}] error evaluating expression: {e}")

        elif isinstance(node, IfBlock):
            executed = False
            for cond_expr, body in node.branches:
                if eval_expr(cond_expr, self.env, line=node.line):
                    self.exec_block(body)
                    executed = True
                    break
            if not executed and node.else_block is not None:
                self.exec_block(node.else_block)

        elif isinstance(node, RepeatBlock):
            n = eval_expr(node.count_expr, self.env, line=node.line)
            try:
                count = int(n)
            except Exception:
                raise RRLRuntimeError(f"[line {node.line}] repeat expects integer, got: {n}")
            if count < 0:
                return
            for _ in range(count):
                self.exec_block(node.body)

        elif isinstance(node, WhileBlock):
            iterations = 0
            MAX_ITER = 1_000_000
            while True:
                cond = eval_expr(node.cond_expr, self.env, line=node.line)
                if not cond:
                    break
                self.exec_block(node.body)
                iterations += 1
                if iterations > MAX_ITER:
                    raise RRLRuntimeError(f"[line {node.line}] while-loop exceeded {MAX_ITER} iterations")

        elif isinstance(node, DoWhileBlock):
            iterations = 0
            MAX_ITER = 1_000_000
            while True:
                self.exec_block(node.body)
                iterations += 1
                if iterations > MAX_ITER:
                    raise RRLRuntimeError(f"[line {node.line}] do while-loop exceeded {MAX_ITER} iterations")
                cond = eval_expr(node.cond_expr, self.env, line=node.line)
                if not cond:
                    break

        elif isinstance(node, ForBlock):
            iterations = 0
            MAX_ITER = 1_000_000
            iterable = eval_expr(node.iterable_expr, self.env, line=node.line)
            try:
                iterator = iter(iterable)
            except Exception as e:
                raise RRLRuntimeError(f"[line {node.line}] for-loop expects iterable, got: {iterable}") from e
            for value in iterator:
                self.env[node.var_name] = value
                self.exec_block(node.body)
                iterations += 1
                if iterations > MAX_ITER:
                    raise RRLRuntimeError(f"[line {node.line}] for-loop exceeded {MAX_ITER} iterations")

        elif isinstance(node, FunctionDef):
            def make_func(name, params, body, def_line):
                def fn(*args):
                    if len(args) != len(params):
                        raise TypeError(f"{name}() expected {len(params)} args, got {len(args)}")
                    # lexical scope: new local environment chained to current env
                    old_env = self.env
                    local_env = dict(old_env)
                    for p, a in zip(params, args):
                        local_env[p] = a
                    self.env = local_env
                    try:
                        try:
                            self.exec_block(body)
                            return None
                        except ReturnSignal as rs:
                            return rs.value
                    finally:
                        self.env = old_env
                return fn
            self.env[node.name] = make_func(node.name, node.params, node.body, node.line)

        elif isinstance(node, ClassDef):
            methods = {}
            for m in node.body:
                if isinstance(m, FunctionDef):
                    def make_method(fn_name, params, fn_body):
                        def method(*call_args):
                            if len(call_args) != (1 + len(params)):
                                raise TypeError(f"{fn_name}() expected {1+len(params)} args, got {len(call_args)}")
                            inst = call_args[0]
                            args = call_args[1:]
                            old_env = self.env
                            local_env = dict(old_env)
                            local_env['self'] = inst
                            for p, a in zip(params, args):
                                local_env[p] = a
                            self.env = local_env
                            try:
                                try:
                                    self.exec_block(fn_body)
                                    return None
                                except ReturnSignal as rs:
                                    return rs.value
                            finally:
                                self.env = old_env
                        return method
                    methods[m.name] = make_method(m.name, m.params, m.body)
                elif isinstance(m, Assign):
                    val = eval_expr(m.expr, self.env, line=m.line)
                    methods[m.name] = val
            bases_objs = []
            for bname in node.bases:
                if bname in self.env and isinstance(self.env[bname], type):
                    bases_objs.append(self.env[bname])
            if not bases_objs:
                bases_objs = (object,)
            else:
                bases_objs = tuple(bases_objs)
            klass = type(node.name, bases_objs, methods)
            self.env[node.name] = klass

        elif isinstance(node, ReturnNode):
            val = None
            if node.expr is not None:
                val = eval_expr(node.expr, self.env, line=node.line)
            raise ReturnSignal(val)

        else:
            raise RRLRuntimeError(f"Unknown node type at line {node.line}: {type(node).__name__}")

# ========== Transpiler ==========
def _indent(level: int) -> str:
    return "    " * level

def transpile_node(node: Node, level: int = 0, in_class: bool = False) -> List[str]:
    ind = _indent(level)
    lines: List[str] = []

    if isinstance(node, Assign):
        lines.append(f"{ind}{node.name} = {node.expr}")

    elif isinstance(node, Display):
        args = node.args_expr.strip()
        if args == "":
            lines.append(f'{ind}rrl_print("")')
        else:
            lines.append(f"{ind}rrl_print({args})")

    elif isinstance(node, Expr):
        lines.append(f"{ind}{node.expr}")

    elif isinstance(node, ImportNode):
        parts = []
        for mod, alias in node.modules:
            if alias:
                parts.append(f"{mod} as {alias}")
            else:
                parts.append(mod)
        lines.append(f"{ind}import {', '.join(parts)}")

    elif isinstance(node, FromImportNode):
        parts = []
        for name, alias in node.names:
            if alias:
                parts.append(f"{name} as {alias}")
            else:
                parts.append(name)
        lines.append(f"{ind}from {node.module} import {', '.join(parts)}")

    elif isinstance(node, switchBlock):
        expr_code = node.expr.expr
        lines.append(f"{ind}match {expr_code}:")
        for case in node.cases:
            pattern_code = case.pattern.expr
            lines.append(f"{ind}    case {pattern_code}:")
            for stmt in case.body:
                lines.extend(transpile_node(stmt, level+2, in_class=in_class))
        if node.default_block is not None:
            lines.append(f"{ind}    case _:")
            for stmt in node.default_block:
                lines.extend(transpile_node(stmt, level+2, in_class=in_class))

    elif isinstance(node, IfBlock):
        first = True
        for cond, body in node.branches:
            if first:
                lines.append(f"{ind}if {cond}:")
                first = False
            else:
                lines.append(f"{ind}elif {cond}:")
            for n in body:
                lines.extend(transpile_node(n, level+1, in_class=in_class))
        if node.else_block is not None:
            lines.append(f"{ind}else:")
            for n in node.else_block:
                lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, RepeatBlock):
        lines.append(f"{ind}for _rrl_i in range(int({node.count_expr})): ")
        for n in node.body:
            lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, WhileBlock):
        lines.append(f"{ind}while {node.cond_expr}:")
        for n in node.body:
            lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, DoWhileBlock):
        lines.append(f"{ind}while True:")
        for n in node.body:
            lines.extend(transpile_node(n, level+1, in_class=in_class))
        inner_ind = _indent(level + 1)
        lines.append(f"{inner_ind}if not ({node.cond_expr}):")
        lines.append(f"{inner_ind}    break")

    elif isinstance(node, ForBlock):
        lines.append(f"{ind}for {node.var_name} in {node.iterable_expr}:")
        for n in node.body:
            lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, FunctionDef):
        params = ", ".join(node.params)
        if in_class:
            if not params or params.split(",")[0].strip() != 'self':
                params = ("self, " + params) if params else "self"
        lines.append(f"{ind}def {node.name}({params}):")
        if not node.body:
            lines.append(f"{ind}    pass")
        else:
            for n in node.body:
                lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, ClassDef):
        bases = ", ".join(node.bases) if node.bases else "object"
        lines.append(f"{ind}class {node.name}({bases}):")
        if not node.body:
            lines.append(f"{ind}    pass")
        else:
            for n in node.body:
                lines.extend(transpile_node(n, level+1, in_class=True))

    elif isinstance(node, ReturnNode):
        if node.expr is None:
            lines.append(f"{ind}return")
        else:
            lines.append(f"{ind}return {node.expr}")

    else:
        raise RuntimeError(f"transpile: unknown node type {type(node)}")

    return lines

def transpile_program(prog: Program) -> str:
    out: List[str] = []
    out.append("# Transpiled RRL -> Python code")
    for node in prog.body:
        out.extend(transpile_node(node, level=0))
    return "\n".join(out) + "\n"

# Centralized set of helper names for filtering from result_env
RRL_HELPER_NAMES = {'__builtins__', 'math', 'rrl_print', 'RobotSim'}

# ========== Transpiled code executor ==========
def exec_transpiled(source: str, capture_output: Optional[List[str]] = None) -> Dict[str, Any]:
    import builtins as _builtins

    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__build_class__"] = _builtins.__build_class__
    safe_builtins["__import__"] = _make_safe_import_for_exec()

    exec_globals = {"__builtins__": safe_builtins, "math": math}

    def rrl_print(*args):
        text = " ".join(str(x) for x in args)
        if capture_output is not None:
            capture_output.append(text)
        else:
            print(text)

    exec_globals['rrl_print'] = rrl_print
    exec_globals['RobotSim'] = RobotSim
    # expose collection helpers to transpiled code
    exec_globals['list_of'] = _list_of
    exec_globals['tuple_of'] = _tuple_of
    exec_globals['set_of'] = _set_of
    exec_globals['dict_of'] = _dict_of

    ns: Dict[str, Any] = dict(exec_globals)
    if 'robot' not in ns:
        ns['robot'] = RobotSim()

    try:
        compiled = compile(source, '<rrl-transpiled>', 'exec')
        exec(compiled, ns)
    except Exception:
        traceback.print_exc()
        raise

    result_env = {k: v for k, v in ns.items() if k not in RRL_HELPER_NAMES}
    return {"output": capture_output if capture_output is not None else None, "env": result_env}

# ========== Runner helpers ==========
def run_rrl_code(code: str, capture_output: Optional[List[str]] = None, transpile: bool = True) -> Dict[str, Any]:
    lines = code.splitlines()
    parser = Parser(lines)
    prog = parser.parse()
    if transpile:
        src = transpile_program(prog)
        try:
            return exec_transpiled(src, capture_output=capture_output)
        except Exception:
            interp = Interpreter(output=capture_output)
            interp.run_program(prog)
            return {"output": capture_output if capture_output is not None else None, "env": interp.env}
    else:
        interp = Interpreter(output=capture_output)
        interp.run_program(prog)
        return {"output": capture_output if capture_output is not None else None, "env": interp.env}

def run_rrl_file(filename: str, capture_output: Optional[List[str]] = None, transpile: bool = True) -> Dict[str, Any]:
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    return run_rrl_code(code, capture_output=capture_output, transpile=transpile)

# ========== REPL ==========
BANNER = """RRL v2.0.0 — (transpile mode)
Blocks: if/elif/else/endif, repeat/endrepeat, while/endwhile, do while/endwhile, for/endfor, def/enddef, class/endclass, match/case/default/endmatch blocks.
Assignments: x = 10, obj.attr = value
Expressions: arithmetic, function calls, method calls, object attributes
Control flow: if, match (like switch), loops (repeat, while, do while, for)
Functions: def name(params) ... enddef
Classes: class Name(bases) ... endclass
RobotSim API: robot.move(meters), robot.rotate(degrees), robot.stop(), robot.position, robot.battery, robot.status
Import support: import math, import os as myos, from math import sqrt, pi (subject to ALLOWED_MODULES)
Collection helpers: list_of(...), tuple_of(...), set_of(...), dict_of(...)
Use display(...) for output. Type :help for help, :env to see variables, :quit to exit.
"""

HELP = """RRL quick help:
  x = 10
  display("Value:", x)

Collections (easy helpers):
  a = list_of(1, 2, 3)
  b = tuple_of(1, 2, 3)
  s = set_of(1, 2, 3)
  d = dict_of('a', 1, 'b', 2)
  # or dict_of({'a':1, 'b':2}) or dict_of(a=1, b=2)

  display(a)
  a.append(4)
  display(a)

  from math import pi
  display(pi)

  def add_all(arr)
    total = 0
    repeat len(arr)
      # example using Python-style indexing
      # note: eval_expr allows Python list indexing for created lists
      total = total + arr[_rrl_i]
    endrepeat
    return total
  enddef

Commands: :help  :env  :quit  :transpile on|off
"""

def repl():
    print(BANNER)
    buffer: List[str] = []
    open_blocks = 0
    transpile_mode = True

    def needs_more(text_line: str) -> int:
        t = strip_comment(text_line).strip().lower()
        if t.startswith("if "): return 1
        if t.startswith("while "): return 1
        if t.startswith("do while "): return 1
        if t.startswith("repeat "): return 1
        if t.startswith("for "): return 1
        if t.startswith("def "): return 1
        if t.startswith("class "): return 1
        if t == "else": return 0
        if t.startswith("elif "): return 0
        if t == "endif": return -1
        if t == "endwhile": return -1
        if t == "endrepeat": return -1
        if t == "endfor": return -1
        if t == "enddef": return -1
        if t == "endclass": return -1
        return 0

    interp = Interpreter()

    while True:
        try:
            prompt = "... " if open_blocks > 0 else "rrl> "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if open_blocks == 0 and line.startswith(":" ):
            cmd = line.strip().lower()
            if cmd == ":quit":
                break
            elif cmd == ":help":
                print(HELP)
                continue
            elif cmd == ":env":
                print(interp.env)
                continue
            elif cmd.startswith(":transpile"):
                parts = cmd.split()
                if len(parts) == 2 and parts[1] in ("on", "off"):
                    transpile_mode = (parts[1] == "on")
                    print(f"transpile = {transpile_mode}")
                else:
                    print("Usage: :transpile on|off")
                continue
            else:
                print("Unknown command. Try :help")
                continue

        buffer.append(line)
        open_blocks += needs_more(line)

        if open_blocks < 0:
            print("Syntax error: unexpected block end")
            buffer.clear()
            open_blocks = 0
            continue

        if open_blocks == 0:
            code = "\n".join(buffer)   
            try:
                res = run_rrl_code(code, capture_output=None, transpile=transpile_mode)
                if res.get('output'):
                    for o in res['output']:
                        print(o)
                if res.get('env'):
                    interp.env.update(res['env'])
            except (ParserError, RRLRuntimeError) as e:
                print(f"Error: {e}")
            except Exception:
                traceback.print_exc()
            buffer.clear()

# ========== CLI ==========
def main():
    if len(sys.argv) == 1:
        repl()
    elif len(sys.argv) in (2, 3):
        transpile_mode = True
        if len(sys.argv) == 3 and sys.argv[2] == "--no-transpile":
            transpile_mode = False
        res = run_rrl_file(sys.argv[1], capture_output=[], transpile=transpile_mode)
        outs = res.get("output") or []
        for o in outs:
            print(o)
    else:
        print("Usage: python rrl.py            # start REPL")
        print("       python rrl.py file.rrl  # run RRL file")

if __name__ == "__main__":
    main()

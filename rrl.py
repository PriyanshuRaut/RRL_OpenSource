#!/usr/bin/env python3
# rrl_v0.3_transpile_fixed.py — RRL
from __future__ import annotations
import sys
import math
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# ========== Safe eval env ==========
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "int": int, "float": float, "str": str, "bool": bool, "range": range,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
    "enumerate": enumerate, "zip": zip, "sum": sum, "any": any, "all": all,
    "sorted": sorted, "reversed": reversed,
    "setattr": setattr, "getattr": getattr, "hasattr": hasattr,
}
SAFE_GLOBALS = {"__builtins__": SAFE_BUILTINS, "math": math}

def safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    return eval(expr, SAFE_GLOBALS, env)

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
class RepeatBlock(Node):
    count_expr: str
    body: List[Node]

@dataclass
class WhileBlock(Node):
    cond_expr: str
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
            # If the line equals a stop token or starts with token + space => stop
            if any(lower == t or lower.startswith(t + " ") for t in stop_tokens):
                return nodes

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

            if lower.startswith("repeat "):
                nodes.append(self.parse_repeat())
                continue

            if lower.startswith("while "):
                nodes.append(self.parse_while())
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

            if "=" in line and not lower.startswith(("elif ", "else", "endif", "endrepeat", "endwhile", "enddef", "endclass")):
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

        start_line, raw = self.current()
        header = strip_comment(raw).strip()
        name = header
        bases: List[str] = []
        if "(" in header and header.endswith(")"):
            name = header.split("(", 1)[0].strip()
            bases_str = header[header.find("(")+1:-1].strip()
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

# ========== Runtime (interpreter preserved) ==========
class RRLRuntimeError(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value

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

class Interpreter:
    def __init__(self, output: Optional[List[str]] = None):
        self.env: Dict[str, Any] = {}
        self.output = output
        self.env['robot'] = RobotSim()

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
            value = safe_eval(node.expr, self.env)
            if "." in node.name:
                try:
                    self._set_dotted(node.name, value)
                except Exception as e:
                    raise RRLRuntimeError(f"[line {node.line}] attribute assignment failed: {e}")
            else:
                self.env[node.name] = value

        elif isinstance(node, Display):
            text = node.args_expr.strip()
            if text == "":
                self._emit("")
                return
            try:
                tuple_expr = f"({text},)" if "," not in text else f"({text})"
                args = safe_eval(tuple_expr, self.env)
                if not isinstance(args, tuple):
                    args = (args,)
                self._emit(*args)
            except Exception:
                self._emit(text)

        elif isinstance(node, Expr):
            try:
                safe_eval(node.expr, self.env)
            except Exception as e:
                raise RRLRuntimeError(f"[line {node.line}] error evaluating expression: {e}")

        elif isinstance(node, IfBlock):
            executed = False
            for cond_expr, body in node.branches:
                if safe_eval(cond_expr, self.env):
                    self.exec_block(body)
                    executed = True
                    break
            if not executed and node.else_block is not None:
                self.exec_block(node.else_block)

        elif isinstance(node, RepeatBlock):
            n = safe_eval(node.count_expr, self.env)
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
                cond = safe_eval(node.cond_expr, self.env)
                if not cond:
                    break
                self.exec_block(node.body)
                iterations += 1
                if iterations > MAX_ITER:
                    raise RRLRuntimeError(f"[line {node.line}] while-loop exceeded {MAX_ITER} iterations")

        elif isinstance(node, FunctionDef):
            def make_func(name, params, body):
                def fn(*args):
                    if len(args) != len(params):
                        raise TypeError(f"{name}() expected {len(params)} args, got {len(args)}")
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
            self.env[node.name] = make_func(node.name, node.params, node.body)

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
                    val = safe_eval(m.expr, self.env)
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
                val = safe_eval(node.expr, self.env)
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
            # if multiple args are provided separated by commas keep them
            lines.append(f"{ind}rrl_print({args})")

    elif isinstance(node, Expr):
        lines.append(f"{ind}{node.expr}")

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
        lines.append(f"{ind}for _rrl_i in range(int({node.count_expr})):")
        for n in node.body:
            lines.extend(transpile_node(n, level+1, in_class=in_class))

    elif isinstance(node, WhileBlock):
        lines.append(f"{ind}while {node.cond_expr}:")
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

def exec_transpiled(source: str, capture_output: Optional[List[str]] = None) -> Dict[str, Any]:
    import builtins as _builtins

    safe_builtins = dict(SAFE_BUILTINS)
    safe_builtins["__build_class__"] = _builtins.__build_class__

    exec_globals = {"__builtins__": safe_builtins, "math": math}

    def rrl_print(*args):
        text = " ".join(str(x) for x in args)
        if capture_output is not None:
            capture_output.append(text)
        else:
            print(text)

    exec_globals['rrl_print'] = rrl_print
    exec_globals['RobotSim'] = RobotSim

    ns: Dict[str, Any] = dict(exec_globals)
    ns['robot'] = RobotSim()

    try:
        compiled = compile(source, '<rrl-transpiled>', 'exec')
        exec(compiled, ns)
    except Exception:
        traceback.print_exc()
        raise

    # prepare env snapshot excluding internal helpers
    result_env = {k: v for k, v in ns.items() if k not in ('__builtins__', 'math', 'rrl_print', 'RobotSim')}
    return {"output": capture_output if capture_output is not None else None, "env": result_env}

    exec_globals = {"__builtins__": SAFE_BUILTINS, "math": math}
    def rrl_print(*args):
        text = " ".join(str(x) for x in args)
        if capture_output is not None:
            capture_output.append(text)
        else:
            print(text)
    exec_globals['rrl_print'] = rrl_print
    exec_globals['RobotSim'] = RobotSim

    # single shared namespace for exec (globals + locals together)
    ns: Dict[str, Any] = dict(exec_globals)
    ns['robot'] = RobotSim()

    try:
        compiled = compile(source, '<rrl-transpiled>', 'exec')
        exec(compiled, ns)
    except Exception:
        traceback.print_exc()
        raise

    # prepare env snapshot excluding internal helpers
    result_env = {k: v for k, v in ns.items() if k not in ('__builtins__', 'math', 'rrl_print', 'RobotSim')}
    return {"output": capture_output if capture_output is not None else None, "env": result_env}

# ========== Runner helpers ==========
def run_rrl_code(code: str, capture_output: Optional[List[str]] = None, transpile: bool = True) -> Dict[str, Any]:
    lines = code.splitlines()
    parser = Parser(lines)
    prog = parser.parse()
    if transpile:
        src = transpile_program(prog)
        # debug: show the generated Python source so you can inspect it
        try:
            return exec_transpiled(src, capture_output=capture_output)
        except Exception:
            # fallback to interpreter if exec_transpiled fails
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
BANNER = """RRL v0.3 — RRL (transpile mode)
Blocks: if/elif/else/endif, repeat/endrepeat, while/endwhile, def/enddef, class/endclass
Use display(...) for output. Type :help for help, :env to see variables, :quit to exit.
"""

HELP = """RRL quick help:
  x = 10
  display("Value:", x)
  def add(a, b)
    return a + b
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
        if t.startswith("repeat "): return 1
        if t.startswith("def "): return 1
        if t.startswith("class "): return 1
        if t == "else": return 0
        if t.startswith("elif "): return 0
        if t == "endif": return -1
        if t == "endwhile": return -1
        if t == "endrepeat": return -1
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

        if open_blocks == 0 and line.startswith(":"):
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
            code = "\n".join(buffer)        # preserve line breaks
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
        print("Usage: python rrl_v0.3_transpile_fixed.py            # start REPL")
        print("       python rrl_v0.3_transpile_fixed.py file.rrl  # run RRL file")

if __name__ == "__main__":
    main()

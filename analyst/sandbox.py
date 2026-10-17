"""Restricted Python executor for model-written post-processing code.

Defence in depth (for real deployments also run the whole app in Docker with `--network none`):
  1. static AST check: allow-listed imports only, no dunder access, no open/exec/eval/...
  2. separate interpreter process in isolated mode (-I) with CPU + memory rlimits
  3. hard wall-clock timeout
The code receives a DataFrame `df` and must assign its answer to `result`.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ALLOWED_IMPORTS = {"pandas", "numpy", "math", "statistics", "datetime", "json"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint"}


class SandboxError(RuntimeError):
    pass


def check_code(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise SandboxError(f"import not allowed: {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise SandboxError(f"import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SandboxError(f"name not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError(f"dunder attribute not allowed: {node.attr}")


_RUNNER = r'''
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
except Exception:
    pass
import pandas as pd, numpy as np
df = pd.read_pickle(sys.argv[1])
ns = {"pd": pd, "np": np, "df": df}
exec(open(sys.argv[2]).read(), ns)
res = ns.get("result")
if isinstance(res, pd.DataFrame):
    out = {"kind": "frame", "data": json.loads(res.to_json(orient="split", date_format="iso"))}
elif isinstance(res, pd.Series):
    out = {"kind": "frame", "data": json.loads(res.reset_index().to_json(orient="split", date_format="iso"))}
else:
    out = {"kind": "scalar", "data": res if isinstance(res, (int, float, str, bool, type(None))) else str(res)}
print("@@RESULT@@" + json.dumps(out))
'''


def run_python(code: str, df: pd.DataFrame, timeout: float = 10.0):
    """Run `code` against `df`; returns a DataFrame or a scalar."""
    check_code(code)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        df.to_pickle(tmp / "df.pkl")
        (tmp / "user.py").write_text(code)
        (tmp / "runner.py").write_text(_RUNNER)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(tmp / "runner.py"), str(tmp / "df.pkl"), str(tmp / "user.py")],
                capture_output=True, text=True, timeout=timeout, cwd=tmp,
            )
        except subprocess.TimeoutExpired as e:
            raise SandboxError(f"timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise SandboxError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "execution failed")
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@RESULT@@")]
    if not marker:
        raise SandboxError("code did not produce a result")
    out = json.loads(marker[-1][len("@@RESULT@@"):])
    if out["kind"] == "frame":
        d = out["data"]
        return pd.DataFrame(d["data"], columns=d["columns"])
    return out["data"]

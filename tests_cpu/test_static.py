"""Static checks that run without a GPU, torch or triton."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solution"
KERNELS = {
    "fused_head.py": {"head_ce_rows_kernel"},
    "rmsnorm.py": {"rmsnorm_fwd_kernel", "rmsnorm_bwd_kernel"},
    "nf4.py": {"nf4_dequant_kernel"},
    "swiglu8.py": {"swiglu8_fwd_kernel", "swiglu8_bwd_kernel"},
    "lora_dropout.py": {"lora_dropout_fwd_kernel", "lora_dropout_bwd_kernel"},
}


def _jit_functions(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if ast.unparse(target) == "triton.jit":
                    names.add(node.name)
    return names


def test_all_python_files_parse():
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or "site-packages" in path.parts:
            continue
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_our_eight_triton_kernels_exist():
    found = {name: _jit_functions(SOLUTION / name) for name in KERNELS}
    assert found == KERNELS
    assert sum(len(v) for v in found.values()) == 8


def test_apply_installs_every_patch():
    tree = ast.parse((SOLUTION / "__init__.py").read_text(encoding="utf-8"))
    apply = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    called = {ast.unparse(n.func) for n in ast.walk(apply) if isinstance(n, ast.Call)}
    imported = {a.asname or a.name for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imported <= called, f"imported but never applied: {imported - called}"


def test_kernel_tests_skip_without_cuda():
    for path in SOLUTION.glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        assert "torch.cuda.is_available()" in text, f"{path.name} would fail on a CPU-only machine"


def test_notebook_is_valid_and_points_at_this_repo():
    nb = json.loads((ROOT / "templates" / "kaggle_notebook.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "github.com/Goslinuly/decentrathon-triton-kernels" in code
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")


def test_task_config_is_the_fixed_one():
    text = (ROOT / "task" / "task.yaml").read_text(encoding="utf-8")
    assert re.search(r"^base: Qwen/Qwen2\.5-1\.5B-Instruct$", text, re.M)
    assert re.search(r"^\s+r: 16$", text, re.M)
    assert re.search(r"^\s+quantization: 4bit$", text, re.M)

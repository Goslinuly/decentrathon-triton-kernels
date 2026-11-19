"""Root pytest config: collect only the test suites whose dependencies are installed.

- `solution/` kernel tests need `torch` and `triton` (and skip themselves without a CUDA GPU);
- `tests/` are Soup's regression tests and need `soup-cli==0.77.0`;
- `tests_cpu/` are static checks that run anywhere.
"""

import importlib.util


def _has(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


collect_ignore_glob = []
if not (_has("torch") and _has("triton")):
    collect_ignore_glob.append("solution/*")
if not _has("soup_cli"):
    collect_ignore_glob.append("tests/*")

"""Where the CPU time of a solution run goes: one `soup train` under cProfile.

Run after `python harness/run_task.py --only solution --out runs_try` (it reuses that config):
    python tools/cpu_profile.py runs_try/solution_1/task.yaml
Prints the functions with the most own time and the most cumulative time. For analysis only.
"""

import cProfile
import os
import pstats
import runpy
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    config = sys.argv[1]
    sys.path.insert(0, str(REPO / "harness"))
    from common import OFFLINE_ENV

    os.environ.update(OFFLINE_ENV)
    os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton_")
    out = tempfile.mkdtemp(prefix="cpuprof_")
    sys.argv = [str(REPO / "harness" / "_train.py"), "--config", config,
                "--metrics", os.path.join(out, "metrics.json"), "--solution"]
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    except SystemExit:
        pass
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime").print_stats(35)
    stats.sort_stats("cumulative").print_stats(45)


if __name__ == "__main__":
    main()

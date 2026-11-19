"""Where the time and the memory of each run go, from a run_task.py result.json.

python tools/diag.py runs/result.json
"""

import json
import sys


def main():
    result = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "runs/result.json"))
    for side, runs in result["runs"].items():
        for run in runs:
            m = dict(run.get("metrics") or {})
            # run_task drops the per-step times from result.json; the run's metrics.json has them
            import os
            run_dir = os.path.dirname(run.get("adapter") or "")
            try:
                m.setdefault("step_s", json.load(open(os.path.join(run_dir, "metrics.json"))).get("step_s"))
            except (OSError, ValueError):
                pass
            wall, loop = run.get("wall_s"), m.get("train_loop_s")
            own = sum((m.get("own_kernel_ms") or {}).values())
            steps = m.get("step_s") or []
            print(f"{run.get('name', side)}:")
            print(f"  time   wall {wall} s = setup+save {round(wall - loop, 1) if wall and loop else '?'} s"
                  f" + train loop {loop} s; apply {m.get('apply_s')} s;"
                  f" steps {len(steps)}, median {m.get('median_step_s')} s,"
                  f" first two {steps[:2]}, slowest {max(steps) if steps else None}")
            print(f"  memory nvidia-smi peak {run.get('gpu_peak_mib')} MiB (start {run.get('gpu_start_mib')});"
                  f" torch allocated {m.get('torch_max_allocated_mib')} MiB,"
                  f" reserved {m.get('torch_max_reserved_mib')} MiB")
            if steps:
                print(f"  steps  first ten {steps[:10]}; sum of steps 1-3 minus 3 medians:"
                      f" {round(sum(steps[:3]) - 3 * sorted(steps)[len(steps) // 2], 2)} s")
            log = os.path.join(run_dir, "train.log")
            if os.path.exists(log):
                for line in open(log, errors="replace"):
                    if "[solution]" in line:
                        print("  log   ", line.rstrip()[line.find("[solution]"):][:160])
            if own:
                share = own / (1000 * m["timed_loop_s"]) if m.get("timed_loop_s") else 0
                print(f"  own kernels {own:.0f} ms over timed steps = {100 * share:.1f}%:"
                      f" {json.dumps({k: round(v) for k, v in m['own_kernel_ms'].items()})}")


if __name__ == "__main__":
    main()

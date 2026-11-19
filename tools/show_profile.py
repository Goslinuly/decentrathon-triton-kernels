"""Top CUDA kernels of a `run_task.py --profile` run, with their share of the GPU time.

python tools/show_profile.py runs_try/solution_profile/profile.json [top]
"""

import json
import sys


def rows_of(data):
    if isinstance(data, list):
        return data
    for value in data.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "time_us" in value[0]:
            return value
        if isinstance(value, dict):
            found = rows_of(value)
            if found:
                return found
    return []


def main():
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    data = json.load(open(path))
    rows = rows_of(data)
    total = sum(r["time_us"] for r in rows) or 1.0
    print(f"total GPU time {total / 1e3:.0f} ms over {len(rows)} kernels")
    for r in rows[:top]:
        print(f"{100 * r['time_us'] / total:5.1f}%  {r['time_us'] / 1e3:8.1f} ms  x{r['count']:<6d} "
              f"{r['name'][:110]}")
    if isinstance(data, dict):
        print({k: v for k, v in data.items() if not isinstance(v, (list, dict))})


if __name__ == "__main__":
    main()

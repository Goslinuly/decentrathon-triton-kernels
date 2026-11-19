"""Which imports of a solution run cost the most: `python -X importtime` over `solution.apply()`.

python tools/import_time.py   (prints the 30 slowest top-level packages, cumulative seconds)
"""

import collections
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
code = ("import sys; sys.path.insert(0, %r); import torch, transformers; import solution; "
        "solution.apply(); import soup_cli.cli, soup_cli.trainer.sft" % str(REPO))
err = subprocess.run([sys.executable, "-X", "importtime", "-c", code], capture_output=True,
                     text=True, cwd=REPO).stderr
by_top = collections.Counter()
rows = []
for line in err.splitlines():
    if not line.startswith("import time:") or "|" not in line:
        continue
    _, self_us, cum_us, name = [x.strip() for x in line.split("|")]
    if not cum_us.isdigit():
        continue
    depth = len(name) - len(name.lstrip())
    rows.append((int(cum_us), name.strip(), depth))
    by_top[name.strip().split(".")[0]] += int(self_us)
total = sum(by_top.values())
print(f"total self import time {total / 1e6:.2f} s")
for pkg, us in by_top.most_common(30):
    print(f"{us / 1e6:7.2f} s  {pkg}")
print("\nslowest single imports (cumulative):")
for cum, name, depth in sorted(rows, reverse=True)[:15]:
    print(f"{cum / 1e6:7.2f} s  {name}")

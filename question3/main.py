"""Convenient entry point for the official practice runner.

Run from the project root with ``python -m question3.main``.  The complete
implementation lives beside the pushed strategy in ``global_policy``.
"""

import sys
from pathlib import Path

if __package__:
    from .global_policy.tour_optimization.official_main import main
else:
    # 支持 ``python question3/main.py``；直接执行时没有相对导入上下文。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from question3.global_policy.tour_optimization.official_main import main


if __name__ == "__main__":
    raise SystemExit(main())

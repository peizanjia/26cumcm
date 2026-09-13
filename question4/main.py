"""Q4 practice/formal entry: python question4/main.py or python -m question4.main."""
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from question4.official_main import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Build an offline, paired replay of complete Q4 action histories.

The input is evaluation data, never input to a policy.  Every displayed action,
observation, cost and belief polygon comes from recorded frames.  The renderer
does not reconstruct observations from hidden truth or silently repair a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render(data: dict) -> str:
    if not isinstance(data.get("replays"), dict) or not data["replays"]:
        raise ValueError("Expected nonempty replays[seed][strategy] action histories")
    template = Path(__file__).with_name("replay_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace("__REPLAY_DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("outputs") / "data.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("outputs") / "report.html")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    print(json.dumps({"report": str(args.output.resolve()), "bytes": args.output.stat().st_size,
                      "scenarios": len(data["replays"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

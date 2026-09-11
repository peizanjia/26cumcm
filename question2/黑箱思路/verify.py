"""Run regression tests and persist actual validation/environment evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    repo = root.parent.parent
    started = time.perf_counter()
    command = [sys.executable, "-m", "pytest", "question1/tests", "question2/黑箱思路/tests", "-q"]
    result = subprocess.run(command, cwd=repo, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    sources = [repo/"B题.pdf", repo/"附件1.docx", repo/"附件2.docx", root/"checkpoints"/"best.pt"]
    if (root/"checkpoints"/"symmetry"/"best.pt").exists():
        sources.append(root/"checkpoints"/"symmetry"/"best.pt")
    report = dict(command=command, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
                  elapsed_s=time.perf_counter()-started, python=sys.version, executable=sys.executable,
                  sha256={str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (root/"results"/"verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

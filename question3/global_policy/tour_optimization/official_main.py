r"""Run the pushed tour policy against the official practice HTTP simulator.

The simulator UI still owns the test-start button.  This process waits for
``/enter`` to become accepted, runs one complete policy, saves all public
commands and responses, then waits for the next manual test start.  It never
uses hidden simulator state and never clicks the UI.

Typical use from the project root::

    .\.venv\Scripts\python.exe -m question3.global_policy.tour_optimization.official_main \
        --robot-id YOUR_LOGIN_ID

Use ``--once`` for one practice run.  Press Ctrl+C while waiting or between
runs to stop; pressing it during an active run attempts ``/exit`` first.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .parameters import Parameters
from .planner import Planner


DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_PARAMS = Path(__file__).with_name("best_parameters.json")
DEFAULT_OUTPUT = Path(__file__).with_name("outputs") / "official_practice"


class OfficialHTTPClient:
    """Small protocol client with a retry-safe, indefinite /enter poll."""

    def __init__(self, robot_id: str, base_url: str, *, retries: int = 3,
                 request_timeout: float = 15.0, enter_poll_s: float = 1.0,
                 wait_enter_s: float = 0.0, verbose: bool = True):
        if not 1 <= len(robot_id.encode("utf-8")) <= 64:
            raise ValueError("robot_id must contain 1..64 UTF-8 bytes")
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"invalid base URL: {base_url!r}")
        if retries < 1 or request_timeout <= 0 or enter_poll_s <= 0:
            raise ValueError("retries, request timeout and poll interval must be positive")
        if wait_enter_s < 0:
            raise ValueError("wait_enter_s cannot be negative")
        self.robot_id = robot_id
        self.base_url = base_url.rstrip("/")
        self.retries = int(retries)
        self.request_timeout = float(request_timeout)
        self.enter_poll_s = float(enter_poll_s)
        self.wait_enter_s = float(wait_enter_s)
        self.verbose = verbose
        self.prefix = uuid.uuid4().hex
        self.counter = 0
        self.enter_polls = 0
        self.last_enter_status = None
        self.history = []

    def _next_request_id(self, tag: str) -> str:
        self.counter += 1
        return f"{self.prefix}-{tag}-{self.counter}"

    def _payload(self, request_id: str, position=None, channel=None) -> dict:
        payload = {
            "arena_id": "default",
            "robot_id": self.robot_id,
            "request_id": request_id,
        }
        if position is not None:
            payload["position"] = {"x": float(position[0]), "y": float(position[1])}
            payload["channel"] = int(channel)
        return payload

    def _post_once(self, path: str, payload: dict) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.request_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise RuntimeError(f"{path} returned a non-object JSON response")
        self.history.append({"path": path, "request": payload, "response": body})
        return body

    def _post(self, path: str, payload: dict) -> dict:
        """Retry only transport failures, retaining the same request ID."""
        last = None
        for attempt in range(self.retries):
            try:
                return self._post_once(path, payload)
            except HTTPError as exc:
                try:
                    body = json.loads(exc.read().decode("utf-8", errors="replace"))
                except Exception:
                    body = None
                detail = body if body is not None else f"HTTP {exc.code}"
                raise RuntimeError(f"{path} rejected with {detail}") from exc
            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(.25 * (2 ** attempt), 2.0))
        raise RuntimeError(f"{path} transport failed after {self.retries} attempts: {last}")

    def _wait_for_enter(self) -> dict:
        request_id = self._next_request_id("enter")
        payload = self._payload(request_id)
        started = time.monotonic()
        last_report = 0.0
        while self.wait_enter_s == 0 or time.monotonic() - started < self.wait_enter_s:
            self.enter_polls += 1
            try:
                response = self._post("/enter", payload)
                self.last_enter_status = response
                if response.get("accepted") is True:
                    remaining = response.get("remaining_real_duration_s")
                    print(f"[enter] 測試已開始，剩餘實時預算：{remaining} 秒", flush=True)
                    return response
                now = time.monotonic()
                if self.verbose and now - last_report >= 10.0:
                    print("[enter] 尚未開始；請在模擬器中點擊一次開始測試，腳本會自動接續。",
                          flush=True)
                    last_report = now
            except RuntimeError as exc:
                now = time.monotonic()
                if self.verbose and now - last_report >= 10.0:
                    print(f"[enter] 接口尚未就緒（{exc}），持續等待。", flush=True)
                    last_report = now
            time.sleep(self.enter_poll_s)
        raise TimeoutError("等待 /enter 超時；請確認模擬器已啟動、robot_id正確且已點擊開始測試")

    def command(self, path: str, position=None, channel=None) -> dict:
        if path == "/enter":
            return self._wait_for_enter()
        request_id = self._next_request_id(path.strip("/") or "request")
        payload = self._payload(request_id, position, channel) if position is not None else self._payload(request_id)
        return self._post(path, payload)


def load_parameters(path: Path) -> Parameters:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Parameters(**data).validate()


def launch_simulator(executable: Path) -> subprocess.Popen:
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"simulator executable not found: {executable}")
    print(f"[simulator] 啟動 {executable}", flush=True)
    return subprocess.Popen([str(executable)], cwd=str(executable.parent))


def run_one(args, run_number: int, params: Parameters) -> dict:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output).expanduser() / f"run_{stamp}_{run_number:03d}"
    output.mkdir(parents=True, exist_ok=False)
    client = OfficialHTTPClient(
        args.robot_id, args.base_url, retries=args.retries,
        request_timeout=args.request_timeout, enter_poll_s=args.enter_poll_s,
        wait_enter_s=args.wait_enter_s, verbose=not args.quiet,
    )
    planner = Planner(client, params)
    print(f"\n=== 等待第 {run_number} 局：{args.base_url} ===", flush=True)
    try:
        summary = planner.run()
    except KeyboardInterrupt:
        print("\n[stop] 收到 Ctrl+C，嘗試通知模擬器退出。", flush=True)
        if planner.entered:
            try:
                client.command("/exit")
            except Exception as exc:
                print(f"[stop] /exit 未完成：{exc}", flush=True)
        raise
    summary.update(
        evaluation="official_practice",
        run_number=run_number,
        robot_id=args.robot_id,
        base_url=args.base_url,
        enter_polls=client.enter_polls,
        request_prefix=client.prefix,
    )
    planner.save(output, summary)
    (output / "http_requests.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in client.history) + "\n",
        encoding="utf-8",
    )
    (output / "run_configuration.json").write_text(
        json.dumps({"parameters": params.__dict__, "robot_id": args.robot_id,
                    "base_url": args.base_url, "run_number": run_number},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary.get(k) for k in (
        "complete", "cleared", "virtual_time_s", "average_time_s", "distance_m",
        "measures", "clears", "misses", "failure", "enter_polls")},
        ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {output}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-id", default=os.environ.get("JAMMER_ROBOT_ID"),
                        help="目前登入模擬器的隊伍/機器人標識，也可用 JAMMER_ROBOT_ID")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--once", action="store_true", help="完成一局後退出，不等待下一局")
    parser.add_argument("--wait-enter-s", type=float, default=0.0,
                        help="等待手動開始的秒數；0表示無限等待")
    parser.add_argument("--enter-poll-s", type=float, default=1.0)
    parser.add_argument("--request-timeout", type=float, default=15.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--launch-simulator", action="store_true",
                        help="先啟動已解壓的模擬器可執行檔，開始測試仍由UI手動點擊")
    parser.add_argument("--simulator-exe", type=Path,
                        help="--launch-simulator使用的jammers-simulator.exe路徑")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.robot_id:
        try:
            args.robot_id = input("robot_id: ").strip()
        except (EOFError, KeyboardInterrupt):
            parser.error("robot_id is required")
        if not args.robot_id:
            parser.error("robot_id cannot be empty")
    if args.launch_simulator and not args.simulator_exe:
        parser.error("--launch-simulator 需要 --simulator-exe")
    if urlsplit(args.base_url).port == 2026:
        print("[practice] 使用官方/練習接口；請確認模擬器當前登入與測試模式。", flush=True)
    params = load_parameters(args.params)
    args.output.mkdir(parents=True, exist_ok=True)
    simulator_process = launch_simulator(args.simulator_exe) if args.launch_simulator else None
    try:
        run_number = 1
        while True:
            summary = run_one(args, run_number, params)
            if not summary.get("complete"):
                print("[warning] 本局未取得完整公開完成證書；仍返回等待下一局。", flush=True)
            if args.once:
                return 0 if summary.get("complete") else 1
            run_number += 1
            print("[ready] 本局已結束。請在模擬器中再次點擊開始測試，腳本會自動執行下一局。",
                  flush=True)
    except KeyboardInterrupt:
        print("\n[stop] 已停止常駐測試腳本。", flush=True)
        return 130
    finally:
        if simulator_process is not None and simulator_process.poll() is None:
            simulator_process.terminate()
            try:
                simulator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                simulator_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

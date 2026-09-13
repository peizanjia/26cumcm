"""Run the frozen Q4 tuned_joint policy against the four public HTTP endpoints.

Select Q4 in the simulator and start each test manually. By default this
process runs one mission, saves its public history, then waits for another
manual start. --once exits after one mission. No UI or hidden state is read.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import subprocess
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener
import uuid

import numpy as np

from question4.tuned_joint.planner import Parameters, Planner


DEFAULT_BASE_URL = "http://127.0.0.1:2026"
DEFAULT_PARAMS = Path(__file__).with_name("tuned_joint") / "best_config.json"
DEFAULT_OUTPUT = Path(__file__).with_name("official_runs")


def json_value(value):
    """JSON-safe diagnostics; never used to transform planner state or feedback."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_value(value), ensure_ascii=False,
                                    allow_nan=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, value):
    # Close each append so an interrupted run retains all preceding observations.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(json_value(value), ensure_ascii=False, allow_nan=False) + "\n")


def load_parameters(path: Path) -> Parameters:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Parameter file must be a JSON object")
    if "parameters" in data:
        if data.get("policy") != "tuned_joint":
            raise ValueError("This entry requires policy=tuned_joint")
        data = data["parameters"]
    defaults = asdict(Parameters())
    if not isinstance(data, dict) or set(data) != set(defaults):
        raise ValueError("Supply all 45 tuned_joint parameters; partial/default fallback is disabled")
    if data["strategy"] != "tuned_joint":
        raise ValueError("strategy must be tuned_joint")
    for name, default in defaults.items():
        value = data[name]
        if isinstance(default, bool):
            valid = type(value) is bool
        elif isinstance(default, int):
            valid = type(value) is int and value >= 0
        elif isinstance(default, float):
            valid = type(value) in (float, int) and math.isfinite(value) and value >= 0
        else:
            valid = isinstance(value, str)
        if not valid:
            raise ValueError(f"Invalid parameter {name}={value!r}")
    for name in ("max_commands", "max_dynamic_stops", "candidate_limit", "max_radio_steps",
                 "coverage_spacing", "coverage_directions", "information_samples",
                 "integration_min", "integration_max", "service_candidates_per_source"):
        if data[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if data["integration_min"] > data["integration_max"]:
        raise ValueError("integration_min cannot exceed integration_max")
    return Parameters(**data)


class ProtocolError(RuntimeError):
    pass


class TransportError(RuntimeError):
    pass


class OfficialHTTPClient:
    """Serial HTTP transport with idempotent retries and public time limits."""

    def __init__(self, robot_id, base_url=DEFAULT_BASE_URL, *, retries=3,
                 request_timeout=15., enter_poll_s=1., wait_enter_s=0.,
                 verbose=True, journal=None):
        if (not isinstance(robot_id, str) or not 1 <= len(robot_id.encode("utf-8")) <= 64
                or any(unicodedata.category(c).startswith("C") for c in robot_id)):
            raise ValueError("robot_id must contain 1..64 UTF-8 bytes without control/format characters")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or parsed.username or parsed.password):
            raise ValueError("Use a local simulator URL such as http://127.0.0.1:2026")
        _ = parsed.port  # Validate an explicitly supplied port before contacting the simulator.
        if (type(retries) is not int or retries < 1 or
                any(not math.isfinite(v) or v <= 0 for v in (request_timeout, enter_poll_s))
                or not math.isfinite(wait_enter_s) or wait_enter_s < 0):
            raise ValueError("Retries/timeouts/poll interval must be positive; wait_enter_s may be zero")
        self.robot_id, self.base_url = robot_id, base_url.rstrip("/")
        self.retries, self.request_timeout = retries, request_timeout
        self.enter_poll_s, self.wait_enter_s, self.verbose = enter_poll_s, wait_enter_s, verbose
        self.journal = Path(journal) if journal else None
        # The simulator is local; system web proxies must not redirect robot requests.
        self.opener = build_opener(ProxyHandler({}))
        self.prefix, self.counter, self.enter_polls = uuid.uuid4().hex, 0, 0
        self.entered = self.exited = self.session_unavailable = False
        self.deadline = math.inf
        self.max_virtual_duration_s = 360000.
        self.last_virtual_time_s = 0.
        self.accepted_actions = self.cleared = 0
        self.detected_channels = set()
        self.distance_m = 0.
        self.position = (0., 0.)
        self.enter_response = None

    def _log(self, event, path, payload, **extra):
        if self.journal:
            append_jsonl(self.journal, dict(event=event, path=path, request=payload,
                                          logged_at=datetime.now().isoformat(), **extra))

    def _payload(self, path, position=None, channel=None):
        if path not in ("/enter", "/measure", "/clear", "/exit"):
            raise ValueError(f"Unsupported action: {path}")
        self.counter += 1
        payload = dict(arena_id="default", robot_id=self.robot_id,
                       request_id=f"{self.prefix}-{path[1:]}-{self.counter}")
        if path in ("/measure", "/clear"):
            q = np.asarray(position, dtype=float)
            if q.shape != (2,) or not np.all(np.isfinite(q)) or np.any(np.abs(q) > 2e6):
                raise ValueError("Position must contain two finite coordinates within +/-2000000")
            if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, (int, np.integer)) or not 1 <= channel <= 20:
                raise ValueError("Channel must be an integer in 1..20")
            payload.update(position=dict(x=float(q[0]), y=float(q[1])), channel=int(channel))
        elif position is not None or channel is not None:
            raise ValueError("enter/exit do not accept position or channel")
        return payload

    def _post(self, path, payload, deadline=math.inf):
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        for attempt in range(1, self.retries + 1):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Public real-time budget exhausted")
            self._log("request", path, payload, attempt=attempt)
            request = Request(self.base_url + path, data=encoded,
                              headers={"Content-Type": "application/json"}, method="POST")
            try:
                with self.opener.open(request, timeout=min(self.request_timeout, remaining)) as response:
                    raw = response.read().decode("utf-8")
                    status = response.status
            except HTTPError as exc:
                self._log("http_error", path, payload, attempt=attempt, status=exc.code,
                          body=exc.read().decode("utf-8", errors="replace"))
                self.session_unavailable = True
                raise ProtocolError(f"{path}: HTTP {exc.code}; action not retried with a new ID") from exc
            except (URLError, TimeoutError, ConnectionError, OSError, HTTPException) as exc:
                self._log("transport_error", path, payload, attempt=attempt, error=str(exc))
                if attempt == self.retries:
                    if self.entered:
                        self.session_unavailable = True
                    raise TransportError(f"{path}: transport failed after {attempt} attempts: {exc}") from exc
                time.sleep(max(0., min(.25 * 2 ** (attempt - 1), 2., deadline-time.monotonic())))
                continue
            try:
                body = json.loads(raw)
            except (ValueError, TypeError) as exc:
                self._log("invalid_response", path, payload, status=status, body=raw)
                self.session_unavailable = True
                raise ProtocolError(f"{path}: invalid JSON response") from exc
            self._log("response", path, payload, attempt=attempt, status=status, response=body)
            if status != 200 or not isinstance(body, dict) or type(body.get("accepted")) is not bool:
                self.session_unavailable = True
                raise ProtocolError(f"{path}: missing boolean accepted")
            return body
        raise AssertionError("Unreachable retry state")

    @staticmethod
    def _number(response, name):
        value = response.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ProtocolError(f"Invalid or missing {name}")
        return float(value)

    def enter(self):
        payload = self._payload("/enter")
        started = time.monotonic()
        wait_deadline = started + self.wait_enter_s if self.wait_enter_s else math.inf
        last_report = -math.inf
        while time.monotonic() < wait_deadline:
            self.enter_polls += 1
            sent_at = time.monotonic()
            detail = "尚未開始；請確認 robot_id 並在模擬器點擊開始測試"
            try:
                response = self._post("/enter", payload, wait_deadline)
                if response["accepted"]:
                    self.entered = True
                    self.enter_response = response
                    self.last_virtual_time_s = self._number(response, "virtual_time_s")
                    remaining = self._number(response, "remaining_real_duration_s")
                    self.deadline = sent_at + remaining
                    self.max_virtual_duration_s = self._number(response, "max_virtual_duration_s")
                    if self.last_virtual_time_s != 0:
                        raise ProtocolError("Q4 planner requires a fresh mission at virtual time zero")
                    print(f"[enter] 第四問已進入，可用實時預算 {remaining:g} 秒。", flush=True)
                    return response
            except TransportError as exc:
                detail = f"接口尚未就緒（{exc}），繼續等待"
            if self.verbose and time.monotonic() - last_report >= 10.:
                print(f"[enter] {detail}。", flush=True)
                last_report = time.monotonic()
            time.sleep(max(0., min(self.enter_poll_s, wait_deadline-time.monotonic())))
        raise TimeoutError("等待 /enter 超時")

    def command(self, kind, position=None, channel=None):
        path = "/" + kind.removeprefix("/")
        if path == "/enter":
            if self.entered:
                raise ProtocolError("This client already entered a mission")
            return self.enter()
        if not self.entered or self.exited or self.session_unavailable:
            raise ProtocolError("No active public session")
        if self.last_virtual_time_s >= self.max_virtual_duration_s:
            self.session_unavailable = True
            raise TimeoutError("Public virtual-time limit reached")
        # Leave a short exit margin, including after an expensive planning step.
        deadline = self.deadline if path == "/exit" else self.deadline - 2.
        payload = self._payload(path, position, channel)
        response = self._post(path, payload, deadline)
        if response["accepted"] is not True:
            self.session_unavailable = True
            raise ProtocolError(f"{path}: accepted=false; no position/time/belief update")
        virtual = self._number(response, "virtual_time_s")
        if virtual < self.last_virtual_time_s:
            raise ProtocolError("Accepted response moved the public clock backwards")
        self.last_virtual_time_s = virtual
        if path == "/exit":
            self.exited = True
            if response.get("exit_reason") != "user_exit":
                raise ProtocolError("Unexpected exit response")
        else:
            # Count accepted physical actions even if a malformed observation
            # prevents the planner from incorporating this last response.
            self.accepted_actions += 1
            q = (float(position[0]), float(position[1]))
            self.distance_m += math.dist(self.position, q)
            self.position = q
            result = response.get("measure_result" if path == "/measure" else "clear_result")
            allowed = ("no_signal", "near", "direction") if path == "/measure" else ("success", "no_target_in_range")
            if result not in allowed:
                raise ProtocolError(f"{path}: invalid result {result!r}")
            if result == "direction" and not 0 <= self._number(response, "svd_deg") < 360:
                raise ProtocolError("svd_deg must lie in [0,360)")
            if result in ("near", "direction", "success"):
                self.detected_channels.add(int(channel))
            self.cleared += int(path == "/clear" and result == "success")
        return response


class RecordingPlanner(Planner):
    """Keep the frozen algorithm intact and stream completed decisions to disk."""

    def __init__(self, command, params, output):
        super().__init__(command, params, record=True)
        self.output = output

    def execute(self, action, phase, decision=None):
        before = len(self.commands)
        try:
            return super().execute(action, phase, decision)
        finally:
            for row in self.commands[before:]:
                append_jsonl(self.output / "commands.jsonl", row)
            for frame in self.frames[before:]:
                append_jsonl(self.output / "frames.jsonl", frame)
            if decision is not None and len(self.commands) > before:
                append_jsonl(self.output / "decisions.jsonl", dict(command_index=before, **decision))


def run_one(args, run_number, params):
    output = Path(args.output) / args.test_type / (
        f"run_{datetime.now():%Y%m%d_%H%M%S_%f}_{run_number:03d}")
    output.mkdir(parents=True, exist_ok=False)
    client = OfficialHTTPClient(args.robot_id, args.base_url, retries=args.retries,
        request_timeout=args.request_timeout, enter_poll_s=args.enter_poll_s,
        wait_enter_s=args.wait_enter_s, verbose=not args.quiet,
        journal=output / "http_requests.jsonl")
    parameter_hash = hashlib.sha256(json.dumps(asdict(params), sort_keys=True).encode()).hexdigest()
    evaluation = f"official_{args.test_type}" if args.http_context == "official" else "local_synthetic_http"
    write_json(output / "run_configuration.json", dict(problem=4, policy="tuned_joint", evaluation=evaluation,
        parameters=asdict(params), parameter_sha256=parameter_hash,
        parameter_file=str(args.params.resolve()), robot_id=args.robot_id,
        base_url=args.base_url, test_type=args.test_type, run_number=run_number))
    # Initialize geometry before /enter starts the official real-time clock.
    planner = RecordingPlanner(client.command, params, output)
    failure = exit_failure = None
    interrupted = False
    result = {}
    active_start = None
    print(f"\n=== 第四問 {args.test_type}，等待第 {run_number} 局：{args.base_url} ===", flush=True)
    try:
        client.command("/enter")
        active_start = time.monotonic()
        result = planner.run()
    except KeyboardInterrupt:
        interrupted, failure = True, "KeyboardInterrupt"
        print("\n[stop] 正在保存本局公開歷史。", flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"[incomplete] {failure}", flush=True)
    finally:
        if client.entered and not client.exited and not client.session_unavailable and time.monotonic() < client.deadline:
            try:
                client.command("/exit")
            except (Exception, KeyboardInterrupt) as exc:
                exit_failure = f"{type(exc).__name__}: {exc}"
        complete = bool(result.get("completed") and failure is None and client.exited and exit_failure is None)
        summary = dict(result, evaluation=evaluation, problem=4,
            policy="tuned_joint", test_type=args.test_type, run_number=run_number,
            complete=complete, completed=complete, coverage_complete=bool(result.get("completed")),
            detected=len(client.detected_channels), detected_channels=sorted(client.detected_channels),
            inferred_total_sources=len(client.detected_channels) if complete else None,
            cleared=client.cleared, remaining_detected=len(client.detected_channels)-client.cleared,
            omnidirectional_count=None, directional_count=None, type_counts_available=False,
            type_counts_reason="public_protocol_does_not_expose_emitter_type",
            virtual_time_s=client.last_virtual_time_s,
            time_s=client.last_virtual_time_s,
            average_time_s=client.last_virtual_time_s/client.cleared if client.cleared else None,
            distance_m=client.distance_m, accepted_action_count=client.accepted_actions,
            command_count=len(planner.commands), costs=planner.costs, counters=planner.counters,
            parameters=asdict(params), parameter_sha256=parameter_hash,
            source_times=planner.target_times, entered=client.entered, exit_acknowledged=client.exited,
            failure=failure, exit_failure=exit_failure, interrupted=interrupted,
            enter_polls=client.enter_polls, request_prefix=client.prefix,
            enter_response=client.enter_response, output=str(output),
            active_wall_s=time.monotonic()-active_start if active_start is not None else None)
        write_json(output / "summary.json", summary)
        (output / "detected_count.txt").write_text(f"{len(client.detected_channels)}\n", encoding="utf-8")
        write_json(output / "map.json", dict(position=planner.position.tolist(), channel=planner.channel,
            targets=planner.snapshots(), unknown_domains=[planner.unknown_map.snapshot(c) for c in range(1, 21)]))
        print(json.dumps({k: summary[k] for k in ("complete", "detected", "cleared", "remaining_detected", "virtual_time_s",
            "average_time_s", "distance_m", "failure", "exit_failure")}, ensure_ascii=False, indent=2), flush=True)
        print(f"[saved] {output}", flush=True)
        print("[types] 全向／定向實際數量：HTTP 接口未提供，不能由示向度直接判定。", flush=True)
        if not complete:
            print("[incomplete] 本局未完整完成，檢測數僅為已發現數，不能視為總數。", flush=True)
        print(f"本局檢測到的干擾源個數：{len(client.detected_channels)}", flush=True)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-id", default=os.environ.get("JAMMER_ROBOT_ID"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--http-context", choices=("official", "local"), default="official",
                        help="local 仅用于临时HTTP模拟器联调，日志不计为官方成绩")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS,
                        help="完整最佳配置；默认 tuned_joint/best_config.json（031）")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--once", action="store_true", help="完成一局后退出")
    parser.add_argument("--test-type", choices=("practice", "formal"), default="practice")
    parser.add_argument("--formal-confirmed", action="store_true", help="已在模拟器选择问题4正式测试")
    parser.add_argument("--wait-enter-s", type=float, default=0., help="等待开始秒数；0为无限等待")
    parser.add_argument("--enter-poll-s", type=float, default=1.)
    parser.add_argument("--request-timeout", type=float, default=15.)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--quiet", action="store_true", help="隐藏重复等待提示")
    parser.add_argument("--launch-simulator", action="store_true")
    parser.add_argument("--simulator-exe", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test_type == "formal" and not args.formal_confirmed:
        parser.error("formal requires --formal-confirmed after selecting Q4 formal in the simulator")
    if not args.robot_id:
        try:
            args.robot_id = input("robot_id: ").strip()
        except (EOFError, KeyboardInterrupt):
            parser.error("robot_id is required")
    try:
        params = load_parameters(args.params)
        OfficialHTTPClient(args.robot_id, args.base_url, retries=args.retries,
            request_timeout=args.request_timeout, enter_poll_s=args.enter_poll_s,
            wait_enter_s=args.wait_enter_s)
        if args.http_context == "local" and urlsplit(args.base_url).port == 2026:
            raise ValueError("Local synthetic checks must use a different port from official 2026")
        if args.launch_simulator and (not args.simulator_exe or not args.simulator_exe.is_file()):
            raise ValueError("--launch-simulator requires an existing --simulator-exe")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    mode = "正式" if args.test_type == "formal" else "演練"
    if args.http_context == "local":
        mode = "本地 HTTP 聯調（非官方成績）"
    print(f"[{args.test_type}] 第四問{mode}；策略 tuned_joint，參數 {args.params.resolve()}。", flush=True)
    print("[ready] 請在模擬器選擇問題4對應模式並手動開始；腳本不會切換題號或點擊開始。", flush=True)
    if args.launch_simulator:
        executable = args.simulator_exe.resolve()
        # The user requested the interactive simulator UI; leave it running for
        # result inspection and its own log uploads when this script exits.
        subprocess.Popen([str(executable)], cwd=str(executable.parent))
    try:
        run_number = 1
        while True:
            summary = run_one(args, run_number, params)
            if summary["interrupted"]:
                return 130
            if args.once or not summary["entered"]:
                return 0 if summary["complete"] else 1
            run_number += 1
            print("[ready] 本局已保存。等待您手動開始下一局第四問測試。", flush=True)
    except KeyboardInterrupt:
        print("\n[stop] 已停止常駐腳本。", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

"""Local-only HTTP protocol, lifecycle and frozen-policy integration checks."""
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from question4 import official_main as runner
from question4.full_mission.simulator import Simulator, Source


@contextmanager
def synthetic_http(*, seed=20296075, sources=None, remaining=1200.,
                   drop_path=None, reject_action=None):
    """Emulate only the documented public protocol on an ephemeral local port.

    Truth stays inside the test server. A response can be deliberately lost
    after execution; duplicate requests must return the cached first response.
    Each /exit schedules another synthetic manual-start cycle for loop tests.
    """
    state = SimpleNamespace(simulator=None, requests=[], polls=0, active=False,
                            cache={}, dropped=False, rounds=0, virtual_us=0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, payload))
            assert self.headers["Content-Type"] == "application/json"
            allowed = {"arena_id", "robot_id", "request_id"}
            if self.path in ("/measure", "/clear"):
                allowed |= {"position", "channel"}
                assert set(payload["position"]) == {"x", "y"}
            assert set(payload) == allowed
            assert payload["arena_id"] == "default"
            assert payload["robot_id"] == "local-protocol-test"
            rid = payload["request_id"]
            if rid in state.cache:
                old_path, old_payload, body = state.cache[rid]
                assert (self.path, payload) == (old_path, old_payload)
            elif self.path == "/enter":
                state.polls += 1
                if state.active or state.polls % 2:
                    body = dict(accepted=False, virtual_time_s=0, real_timestamp_ms=0)
                else:
                    state.simulator = Simulator(seed, sources=sources)
                    state.virtual_us = 0
                    state.active = True
                    state.rounds += 1
                    body = dict(accepted=True, virtual_time_s=0, real_timestamp_ms=0,
                        max_virtual_duration_s=360000, max_real_duration_s=1200,
                        remaining_real_duration_s=remaining)
            elif not state.active:
                body = dict(accepted=False, virtual_time_s=0, real_timestamp_ms=0)
            elif self.path == "/exit":
                state.active = False
                body = dict(accepted=True, virtual_time_s=state.virtual_us/1e6,
                            real_timestamp_ms=0, exit_reason="user_exit")
            elif reject_action == len(state.simulator.history) + 1:
                body = dict(accepted=False, virtual_time_s=0, real_timestamp_ms=0)
            else:
                old_time = state.simulator.time_s
                body = state.simulator.command(self.path, payload["position"], payload["channel"])
                # Official protocol accumulates time in microseconds.
                state.virtual_us += round((state.simulator.time_s-old_time)*1e6)
                body.update(virtual_time_s=state.virtual_us/1e6, real_timestamp_ms=0)
            if body["accepted"]:
                state.cache[rid] = (self.path, payload, body)
            if self.path == drop_path and not state.dropped:
                state.dropped = True
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def client_for(state, **kw):
    return runner.OfficialHTTPClient("local-protocol-test", state.url,
        request_timeout=2, enter_poll_s=.001, wait_enter_s=3, verbose=False, **kw)


def args_for(state, output):
    return runner.build_parser().parse_args([
        "--robot-id", "local-protocol-test", "--base-url", state.url,
        "--http-context", "local", "--output", str(output), "--enter-poll-s", ".001",
        "--wait-enter-s", "3", "--request-timeout", "2", "--quiet"])


def near_sources():
    return [Source(c, 2., 0., 1000., c % 2 == 0, 3.141592653589793) for c in range(1, 17)]


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_is_full_frozen_selection_not_training_winner():
    config = json.loads(runner.DEFAULT_PARAMS.read_text(encoding="utf-8"))
    params = runner.load_parameters(runner.DEFAULT_PARAMS)
    assert asdict(params) == config["parameters"]
    assert len(asdict(params)) == 45
    assert params.forecast_weight == 0.9791095109100708
    assert params.probe_fraction == 0.721142380055286
    assert params.future_scan_weight == 1.0
    assert runner.load_parameters(runner.DEFAULT_PARAMS.with_name("best_parameters.json")) == params


@pytest.mark.parametrize("change", [
    lambda d: d.pop("probe_fraction"),
    lambda d: d.update(strategy="free_joint"),
    lambda d: d.update(bundle_scans="false"),
    lambda d: d.update(candidate_limit=0),
    lambda d: d.update(max_commands=2.5),
    lambda d: d.update(coverage_spacing=float("nan")),
    lambda d: d.update(integration_min=4096, integration_max=128),
])
def test_bad_parameters_fail_before_enter(tmp_path, change):
    data = asdict(runner.load_parameters(runner.DEFAULT_PARAMS))
    change(data)
    path = tmp_path / "params.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_parameters(path)


def test_public_physics_and_idempotent_retry(tmp_path):
    sources = [Source(3, 300., 0., 1000., True, 0.)]
    with synthetic_http(sources=sources, drop_path="/clear") as state:
        client = client_for(state, journal=tmp_path / "http.jsonl")
        client.command("/enter")
        assert client.enter_polls == 2
        assert state.requests[0] == state.requests[1]
        # A negative directional response stays negative. Clearing it from its
        # back side still succeeds and must not change receiver channel 2.
        a = client.command("measure", (300., 400.), 1)
        b = client.command("measure", (300., 400.), 2)
        c = client.command("clear", (290., 0.), 3)
        d = client.command("measure", (290., 0.), 2)
        assert a["virtual_time_s"] == 105
        assert b["virtual_time_s"] == 111
        assert c["clear_result"] == "success"
        assert d["measure_result"] == "no_signal"
        assert d["virtual_time_s"] - c["virtual_time_s"] == 5
        clears = [p for path, p in state.requests if path == "/clear"]
        assert len(clears) == 2 and clears[0] == clears[1]
        assert len(state.simulator.history) == 4
        assert client.cleared == 1
        assert client.accepted_actions == 4
        client.command("exit")
        assert client.exited
    log = read_lines(tmp_path / "http.jsonl")
    assert any(r["event"] == "transport_error" for r in log)
    assert sum(r["event"] == "response" and r["path"] == "/clear" for r in log) == 1


def test_rejected_action_does_not_reset_clock_or_position():
    with synthetic_http(reject_action=2) as state:
        client = client_for(state)
        client.command("enter")
        client.command("measure", (300., 400.), 1)
        with pytest.raises(runner.ProtocolError, match="accepted=false"):
            client.command("measure", (500., 500.), 2)
        assert client.last_virtual_time_s == 105
        assert client.position == (300., 400.)
        assert client.accepted_actions == 1
        assert len(state.simulator.history) == 1


def test_detected_count_deduplicates_channels_and_survives_clear(capsys):
    with synthetic_http(sources=[Source(1, 2., 0., 1000., False, 0.)]) as state:
        client = client_for(state)
        client.command("enter")
        client.command("measure", (0., 0.), 1)
        client.command("measure", (0., 1.), 1)
        client.command("measure", (0., 1.), 20)
        client.command("clear", (0., 1.), 1)
        assert client.detected_channels == {1} and client.cleared == 1
        log = capsys.readouterr().out
        assert "[found]" not in log and "[clear]" not in log
        assert "本局檢測到" not in log  # Count output belongs to the end-of-run summary.


def test_frozen_policy_completes_and_streams_public_history(tmp_path, capsys):
    with synthetic_http(sources=near_sources()) as state:
        summary = runner.run_one(args_for(state, tmp_path), 1, runner.load_parameters(runner.DEFAULT_PARAMS))
        assert summary["complete"] and summary["cleared"] == 16
        assert summary["detected"] == 16 and summary["remaining_detected"] == 0
        assert summary["inferred_total_sources"] == 16
        assert summary["detected_channels"] == list(range(1, 17))
        assert summary["omnidirectional_count"] is None
        assert summary["directional_count"] is None
        assert not summary["type_counts_available"]
        assert summary["evaluation"] == "local_synthetic_http"
        assert all(s.cleared for s in state.simulator.sources)
        assert state.requests[-1][0] == "/exit"
        output = Path(summary["output"])
        commands = read_lines(output / "commands.jsonl")
        assert len(commands) == len(state.simulator.history) == summary["command_count"]
        assert [(r["kind"], r["channel"], r["position"]) for r in commands[:20]] == [
            ("measure", c, [0., 0.]) for c in range(1, 21)]
        assert len(read_lines(output / "frames.jsonl")) == len(commands)
        assert (output / "decisions.jsonl").is_file()
        assert "truth" not in json.loads((output / "map.json").read_text(encoding="utf-8"))
        saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        assert (output / "detected_count.txt").read_text(encoding="utf-8") == "16\n"
        assert saved["average_time_s"] == state.virtual_us/1e6/16
        assert summary["exit_acknowledged"]
        assert capsys.readouterr().out.count("本局檢測到的干擾源個數：16") == 1


@pytest.mark.parametrize("remaining", [0., 2.])
def test_public_remaining_time_overrides_default_budget(tmp_path, remaining):
    with synthetic_http(sources=near_sources(), remaining=remaining) as state:
        summary = runner.run_one(args_for(state, tmp_path), 1, runner.load_parameters(runner.DEFAULT_PARAMS))
        assert not summary["complete"]
        assert "budget exhausted" in summary["failure"]
        assert summary["accepted_action_count"] == 0
        assert not state.simulator.history
        assert summary["exit_acknowledged"] is (remaining > 0)


def test_failure_saves_partial_run_without_another_action(tmp_path):
    with synthetic_http(sources=near_sources(), reject_action=2) as state:
        summary = runner.run_one(args_for(state, tmp_path), 1, runner.load_parameters(runner.DEFAULT_PARAMS))
        assert not summary["complete"] and not summary["exit_acknowledged"]
        assert summary["detected"] == 1 and summary["inferred_total_sources"] is None
        assert summary["virtual_time_s"] == 5
        output = Path(summary["output"])
        assert len(read_lines(output / "commands.jsonl")) == 1
        assert state.requests[-1][0] == "/measure"
        assert (output / "summary.json").is_file()
        assert (output / "map.json").is_file()


def test_ctrl_c_during_mission_saves_and_exits(tmp_path, monkeypatch):
    original = runner.OfficialHTTPClient.command

    def interrupt(self, kind, position=None, channel=None):
        if kind == "measure" and self.accepted_actions:
            raise KeyboardInterrupt
        return original(self, kind, position, channel)

    monkeypatch.setattr(runner.OfficialHTTPClient, "command", interrupt)
    with synthetic_http(sources=near_sources()) as state:
        summary = runner.run_one(args_for(state, tmp_path), 1, runner.load_parameters(runner.DEFAULT_PARAMS))
        assert summary["interrupted"] and summary["exit_acknowledged"]
        assert not summary["complete"]
        assert summary["virtual_time_s"] == 5
        assert state.requests[-1][0] == "/exit"
        assert len(read_lines(Path(summary["output"]) / "commands.jsonl")) == 1


def test_formal_requires_same_explicit_flag_as_q3(capsys):
    with pytest.raises(SystemExit) as exc:
        runner.main(["--test-type", "formal", "--robot-id", "local-protocol-test"])
    assert exc.value.code == 2
    assert "Q4 formal" in capsys.readouterr().err


def test_formal_cli_marks_local_test_separately(tmp_path, capsys):
    with synthetic_http(sources=near_sources()) as state:
        code = runner.main(["--robot-id", "local-protocol-test", "--base-url", state.url,
            "--http-context", "local", "--test-type", "formal", "--formal-confirmed", "--once",
            "--output", str(tmp_path), "--enter-poll-s", ".001"])
    assert code == 0
    assert "[formal] 第四問" in capsys.readouterr().out
    files = list(tmp_path.glob("formal/run_*/summary.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["evaluation"] == "local_synthetic_http" and data["test_type"] == "formal"


def test_persistent_mode_starts_fresh_after_next_manual_enter(tmp_path, monkeypatch):
    original = runner.run_one
    summaries = []

    def two_rounds(*args):
        result = original(*args)
        summaries.append(dict(result))
        if len(summaries) == 2:
            result["interrupted"] = True  # End the otherwise persistent CLI loop.
        return result

    monkeypatch.setattr(runner, "run_one", two_rounds)
    with synthetic_http(sources=near_sources()) as state:
        code = runner.main(["--robot-id", "local-protocol-test", "--base-url", state.url,
            "--http-context", "local", "--output", str(tmp_path), "--enter-poll-s", ".001"])
        assert state.rounds == 2
    assert code == 130
    assert all(s["complete"] and s["cleared"] == 16 for s in summaries)
    assert summaries[0]["virtual_time_s"] == summaries[1]["virtual_time_s"]
    assert summaries[0]["request_prefix"] != summaries[1]["request_prefix"]
    assert summaries[0]["output"] != summaries[1]["output"]


def test_direct_file_and_module_entrypoints(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for command, cwd in [([sys.executable, "-X", "utf8", str(root / "question4/main.py"), "--help"], tmp_path),
                         ([sys.executable, "-X", "utf8", "-m", "question4.main", "--help"], root)]:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 0, result.stderr
        assert "--formal-confirmed" in result.stdout

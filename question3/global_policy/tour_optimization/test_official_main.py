"""Protocol-only tests for the manual-start official practice runner."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import unittest

from .official_main import OfficialHTTPClient


class _Handler(BaseHTTPRequestHandler):
    polls = 0
    requests = []

    def log_message(self, *_args):
        return

    def do_POST(self):  # noqa: N802 - stdlib protocol name
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        self.__class__.requests.append((self.path, payload))
        if self.path == "/enter":
            self.__class__.polls += 1
            body = ({"accepted": False} if self.__class__.polls == 1 else
                    {"accepted": True, "remaining_real_duration_s": 1200.0,
                     "virtual_time_s": 0.0})
        else:
            body = {"accepted": True, "virtual_time_s": 5.0}
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class OfficialMainTests(unittest.TestCase):
    def setUp(self):
        _Handler.polls = 0
        _Handler.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.thread.join(timeout=2)

    def test_enter_reuses_request_id_until_manual_start(self):
        client = OfficialHTTPClient(
            "practice-bot", f"http://127.0.0.1:{self.server.server_port}",
            enter_poll_s=0.001, request_timeout=2, verbose=False,
        )
        response = client.command("/enter")
        self.assertTrue(response["accepted"])
        self.assertEqual(client.enter_polls, 2)
        self.assertEqual(_Handler.requests[0][0], "/enter")
        self.assertEqual(_Handler.requests[0][1]["request_id"],
                         _Handler.requests[1][1]["request_id"])

    def test_action_uses_new_request_id_and_public_payload(self):
        client = OfficialHTTPClient(
            "practice-bot", f"http://127.0.0.1:{self.server.server_port}",
            request_timeout=2, verbose=False,
        )
        client.command("/measure", (12.5, -3), 7)
        path, payload = _Handler.requests[-1]
        self.assertEqual(path, "/measure")
        self.assertEqual(payload["position"], {"x": 12.5, "y": -3.0})
        self.assertEqual(payload["channel"], 7)
        self.assertEqual(payload["arena_id"], "default")
        self.assertEqual(client.history[-1]["request"], payload)
        self.assertEqual(client.history[-1]["response"]["accepted"], True)


if __name__ == "__main__":
    unittest.main()

"""Synthetic Q3 physics and documented strategy protocol (not official server)."""
import argparse
import hashlib
import json
import math
import random
import time
import unicodedata
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


@dataclass
class Source:
    channel: int
    x: float
    y: float
    radius: float
    cleared: bool = False


def random_scene(seed):
    rng = random.Random(seed)
    sources = []
    for c in rng.sample(range(1, 21), rng.randint(10, 16)):
        r, a = 1800 * math.sqrt(rng.random()), rng.uniform(0, 2 * math.pi)
        sources.append(Source(c, r * math.cos(a), r * math.sin(a), rng.uniform(1000, 1500)))
    return sources


class Simulator:
    def __init__(self, seed=0, robot_id="local", sources=None):
        self.seed, self.robot_id = seed, robot_id
        self.sources = random_scene(seed) if sources is None else sources
        self.position, self.channel, self.time_us = (0., 0.), 1, 0
        self.entered = self.closed = False
        self.created = time.monotonic()
        self.deadline = self.created + 1500
        self.cache, self.history = {}, []

    @property
    def virtual_time(self):
        return self.time_us / 1e6

    def response(self, accepted=False, **extra):
        return dict(accepted=accepted, real_timestamp_ms=int(time.time()*1000),
                    virtual_time_s=self.virtual_time if accepted else 0, **extra)

    def error(self, channel, pos):
        # Stable local field: revisiting exactly the same coordinate gives the same error.
        key = f"{self.seed}:{channel}:{float(pos[0]).hex()}:{float(pos[1]).hex()}"
        u = int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'big') / 2**64
        return 2*u-1

    def handle(self, path, p):
        if path not in ('/enter', '/measure', '/clear', '/exit'):
            return 404, self.response()
        action = path in ('/measure', '/clear')
        base = {'arena_id', 'robot_id', 'request_id'}
        required = base | ({'position', 'channel'} if action else set())
        if not isinstance(p, dict) or not required <= p.keys():
            return 400, self.response()
        for k, limit in [('arena_id', 64), ('robot_id', 64), ('request_id', 128)]:
            s = p[k]
            if not isinstance(s, str) or not 1 <= len(s.encode('utf8')) <= limit or any(unicodedata.category(c).startswith('C') for c in s):
                return 400, self.response()
        if action:
            xy, c = p['position'], p['channel']
            if not isinstance(xy, dict) or not {'x', 'y'} <= xy.keys():
                return 400, self.response()
            if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or c != int(c) or not 1 <= c <= 20:
                return 400, self.response()
            if any(isinstance(xy[k], bool) or not isinstance(xy[k], (int, float)) or not math.isfinite(xy[k]) or abs(xy[k]) > 2e6 for k in ('x','y')):
                return 400, self.response()
            if xy.keys() - {'x','y'}:
                return 200, self.response()
        if p.keys() - required or p['arena_id'] != 'default' or p['robot_id'] != self.robot_id:
            return 200, self.response()
        canonical = json.dumps([path, p], sort_keys=True)
        rid = p['request_id']
        if rid in self.cache:
            previous, result = self.cache[rid]
            return (200, result) if previous == canonical else (409, self.response())
        if self.closed or time.monotonic() >= self.deadline:
            self.closed = True
            return 200, self.response()
        if path == '/enter':
            if self.entered:
                return 200, self.response()
            self.entered = True
            self.deadline = min(self.deadline, time.monotonic()+1200)
            r = self.response(True, max_virtual_duration_s=360000, max_real_duration_s=1200,
                              remaining_real_duration_s=max(0, round(self.deadline-time.monotonic())))
        elif not self.entered:
            return 200, self.response()
        elif path == '/exit':
            self.closed = True
            r = self.response(True, exit_reason='user_exit')
        else:
            pos = (float(xy['x']) or 0., float(xy['y']) or 0.)
            dtime = math.dist(self.position, pos)/5
            source = next((s for s in self.sources if s.channel == c and not s.cleared), None)
            distance = math.dist(pos, (source.x, source.y)) if source else math.inf
            if path == '/measure':
                dtime += 5 + (c != self.channel)
                self.channel = int(c)
                if distance > (source.radius if source else 0):
                    fields = dict(measure_result='no_signal')
                elif distance <= 5:
                    fields = dict(measure_result='near')
                else:
                    bearing = math.degrees(math.atan2(source.y-pos[1], source.x-pos[0]))
                    fields = dict(measure_result='direction', svd_deg=round((bearing+self.error(c,pos)) % 360, 2) % 360)
            else:
                success = distance <= 20
                dtime += 5 if success else 3
                if success:
                    source.cleared = True
                fields = dict(clear_result='success' if success else 'no_target_in_range')
            self.position = pos
            self.time_us += round(dtime*1e6)
            r = self.response(True, **fields)
            if self.virtual_time >= 360000:
                self.closed = True
        self.cache[rid] = canonical, r.copy()
        self.history.append(dict(path=path, req=p, resp=r.copy()))
        return 200, r

    def dump(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Evaluator-only outputs: never returned by the strategy API.
        (directory/'truth.json').write_text(json.dumps(dict(seed=self.seed, sources=[asdict(s) for s in self.sources]), indent=2), encoding='utf8')
        (directory/'commands.jsonl').write_text('\n'.join(json.dumps(x) for x in self.history), encoding='utf8')


class Client:
    """Identical payloads via in-process transport or HTTP; strategy sees responses only."""
    def __init__(self, robot_id='local', simulator=None, base_url='http://127.0.0.1:2027'):
        self.robot_id, self.simulator, self.base_url = robot_id, simulator, base_url
        self.counter = 0

    def command(self, path, position=None, channel=None):
        self.counter += 1
        p = dict(arena_id='default', robot_id=self.robot_id, request_id=f'action-{self.counter}')
        if position is not None:
            p.update(position=dict(x=float(position[0]), y=float(position[1])), channel=int(channel))
        if self.simulator is not None:
            code, response = self.simulator.handle(path, p)
        else:
            from urllib.request import Request, urlopen
            req = Request(self.base_url+path, json.dumps(p).encode(), {'Content-Type':'application/json'})
            with urlopen(req, timeout=10) as r:
                code, response = r.status, json.load(r)
        if code != 200 or not response['accepted']:
            raise RuntimeError((code, response))
        return response


def serve(sim, port, output):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            def unique(pairs):
                d = {}
                for k,v in pairs:
                    if k in d: raise ValueError('duplicate key')
                    d[k] = v
                return d
            try:
                n = int(self.headers.get('Content-Length', '0'))
                ct = self.headers.get('Content-Type', '').lower().replace(' ', '')
                if ct not in ('application/json', 'application/json;charset=utf-8') or self.headers.get('Content-Encoding','identity') != 'identity':
                    code, result = 415, sim.response()
                elif n > 65536:
                    code, result = 413, sim.response()
                else:
                    payload = json.loads(self.rfile.read(n).decode('utf8'), object_pairs_hook=unique,
                                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                    code, result = sim.handle(self.path, payload)
            except (ValueError, UnicodeError, RecursionError):
                code, result = 400, sim.response()
            data = json.dumps(result).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if sim.closed: sim.dump(output)

        def do_GET(self):
            self.send_response(405 if self.path in ('/enter','/exit','/measure','/clear') else 404)
            self.end_headers()

        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Local synthetic Q3 server http://127.0.0.1:{port}; robot_id={sim.robot_id}', flush=True)
    try:
        server.serve_forever()
    finally:
        sim.dump(output)
        server.server_close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=20260911)
    p.add_argument('--port', type=int, default=2027)
    p.add_argument('--robot-id', default='local')
    p.add_argument('--output', default='question3/local_sim/outputs/server')
    a = p.parse_args()
    serve(Simulator(a.seed, a.robot_id), a.port, a.output)

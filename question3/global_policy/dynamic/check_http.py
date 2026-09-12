"""Exercise the dynamic planner against a fresh private local HTTP simulator."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from .runner import Planner,HTTPClient


def main():
    out=Path('question3/global_policy/dynamic/outputs/http_smoke');out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    proc=subprocess.Popen([sys.executable,'-m','question3.local_sim.simulator','--seed','20261209',
                           '--port',str(port),'--output',str(out/'evaluator')],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                          creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        if not proc.stdout.readline():raise RuntimeError(proc.stderr.read())
        planner=Planner(HTTPClient('local',f'http://127.0.0.1:{port}'));result=planner.run()
        for _ in range(100):
            try:truth=json.loads((out/'evaluator/truth.json').read_text(encoding='utf8'));break
            except (FileNotFoundError,json.JSONDecodeError):time.sleep(.01)
        else:raise RuntimeError('Evaluator output missing')
        result.update(true_total=len(truth['sources']),true_cleared=sum(s['cleared'] for s in truth['sources']),
                      evaluation='local_synthetic_http',seed=20261209)
        planner.save(out,result)
        print(json.dumps({k:result[k] for k in ('complete','true_total','true_cleared','average_time_s','failure')},indent=2))
        assert result['complete'] and result['true_total']==result['true_cleared'],result
    finally:proc.terminate();proc.wait(timeout=10)


if __name__=='__main__':main()

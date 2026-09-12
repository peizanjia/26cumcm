"""Run the existing robot unchanged against a subprocess local HTTP server."""
import contextlib
import io
import json
import socket
import subprocess
import sys
from pathlib import Path
from question3.robot import Robot


def main():
    out=Path('question3/local_sim/outputs/http_smoke'); out.mkdir(parents=True,exist_ok=True)
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    proc=subprocess.Popen([sys.executable,'-m','question3.local_sim.simulator','--seed','20260911',
                           '--port',str(port),'--output',str(out)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        line=proc.stdout.readline()
        if not line: raise RuntimeError(proc.stderr.read())
        robot=Robot('local',base_url=f'http://127.0.0.1:{port}',log_path=str(out/'robot.jsonl'))
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            result=robot.run()
            robot.dump_log()
        (out/'console.txt').write_text(captured.getvalue(),encoding='utf8')
        # /exit response precedes evaluator dump; wait for file, not virtual action delays.
        import time
        for _ in range(100):
            try:
                truth=json.loads((out/'truth.json').read_text(encoding='utf8'))
                break
            except (FileNotFoundError,json.JSONDecodeError): time.sleep(.01)
        else: raise RuntimeError('Evaluator dump missing')
        summary=dict(seed=20260911,total=len(truth['sources']),cleared=sum(s['cleared'] for s in truth['sources']),
                     virtual_time_s=result['virtual_time_s'],commands=len(robot.log_lines),
                     note='Existing robot unchanged, local synthetic HTTP test, not official score')
        assert summary['cleared']==summary['total'],summary
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
        print(json.dumps(summary,indent=2))
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__=='__main__': main()

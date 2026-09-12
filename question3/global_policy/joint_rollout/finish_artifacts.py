"""Finish read-only reports once a local batch has fully written its summary."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    root=Path(__file__).parent/'outputs/validation';summary=root/'summary.json'
    while not summary.exists():
        if not a.wait:raise RuntimeError('Validation has not completed')
        time.sleep(5)
    results=json.loads(summary.read_text());manifest=json.loads((root/'manifest.json').read_text())
    if any(r['count']!=manifest['count'] for r in results.values()):raise RuntimeError('Incomplete batch')
    if any(r['failures'] for r in results.values()):raise RuntimeError('Failures require separate review before paired timing conclusions')
    for module in ('analyze_results','build_report'):
        subprocess.run([sys.executable,'-m','question3.global_policy.joint_rollout.'+module],check=True)
    subprocess.run(['node','question3/global_policy/joint_rollout/check_report.cjs'],check=True)
    print('Final comparison, adverse-case replays, factorial analysis and browser checks completed.',flush=True)


if __name__=='__main__':main()

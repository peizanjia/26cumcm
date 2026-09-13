"""Copy the selected actual case histories into a small portable review bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def package(output):
    output = Path(output).resolve()
    validation = json.loads((output/'validate.json').read_text(encoding='utf-8'))
    selection = json.loads((output/'figures/case_selection.json').read_text(encoding='utf-8'))
    seeds = {int(c['seed']) for c in selection['cases']}
    entries = []
    for row in validation['rows']:
        if row['seed'] not in seeds:
            continue
        source = Path(row['history_file']).resolve()
        if not source.is_relative_to(output):
            raise ValueError(f'Case history must be within this experiment: {source}')
        if row['strategy'] not in validation['configs'] or not row['completed']:
            raise ValueError('Publish only indexed complete case histories')
        destination = output/'case_histories'/row['strategy']/f"{row['seed']}.json.gz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != hashlib.sha256(destination.read_bytes()).hexdigest():
            raise AssertionError('Portable history differs from the original evidence')
        entries.append(dict(seed=row['seed'], strategy=row['strategy'],
            file=destination.relative_to(output).as_posix(), bytes=destination.stat().st_size,
            sha256=digest, per_source_s=row['per_source_s']))
    if len(entries) != len(seeds)*len(validation['configs']):
        raise AssertionError('Portable case matrix is incomplete')
    manifest = dict(scope='local synthetic actual histories; identical bytes to evaluated runs',
                    seeds=sorted(seeds), entries=entries, total_bytes=sum(r['bytes'] for r in entries))
    (output/'case_histories/index.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'outputs')
    args = parser.parse_args()
    print(json.dumps(package(args.output), ensure_ascii=False, indent=2))

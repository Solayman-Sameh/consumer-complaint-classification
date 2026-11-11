"""Synchronize embedded release sources without rerunning historical training."""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'notebooks/consumer_complaint_training.ipynb'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    notebook = json.loads(PATH.read_text(encoding='utf-8'))
    changed = []
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if not source.startswith('# Embedded file: '):
            continue
        relative = source.splitlines()[0].split(': ', 1)[1]
        actual = (ROOT / relative).read_text(encoding='utf-8')
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'embedded_source' for t in node.targets))
        if ast.literal_eval(assignment.value) != actual:
            changed.append(relative)
            lines = source.splitlines(keepends=True)
            lines[assignment.lineno-1:assignment.end_lineno] = ['embedded_source = ' + repr(actual) + '\n']
            cell['source'] = ''.join(lines).splitlines(keepends=True)
    if args.check:
        if changed:
            raise SystemExit('Embedded files differ: ' + ', '.join(changed))
        print('All embedded sources and configurations match.')
        return
    notebook['metadata']['release_sources_synchronized'] = '2026-11-10'
    PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print('Synchronized:', changed)

if __name__ == '__main__': main()

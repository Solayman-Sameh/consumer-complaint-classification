"""Publish the minimal app bundle to the user's free CPU Hugging Face Space.

Credentials come only from the existing HF login/environment. Nothing secret is
embedded or logged. Explicit file mappings exclude datasets, reports and envs.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='Solayman-Sameh/consumer-complaint-classifier')
    parser.add_argument('--prepare-only', action='store_true', help='Build a local deployment ZIP and manifest without publishing.')
    parser.add_argument('--direct-network', action='store_true', help='Ignore a broken local proxy for this process only.')
    args = parser.parse_args()
    if args.direct_network:
        for variable in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:
            os.environ.pop(variable, None)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    from huggingface_hub import HfApi, CommitOperationAdd
    verification = json.loads((ROOT/'reports/application_verification.json').read_text())
    if verification['status'] != 'passed' or not verification.get('source_sha256'):
        raise RuntimeError('Verify the current app before preparing or publishing.')
    for name, expected in verification['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_text(encoding='utf-8').encode('utf-8')).hexdigest() != expected:
            raise RuntimeError('Application sources changed since verification; rerun scripts/verify_app.py.')
    files = {'app.py': ROOT/'app.py', 'requirements.txt': ROOT/'requirements-space.txt',
             'README.md': ROOT/'docs/EXPORTED_APP_README.md', 'LICENSE': ROOT/'LICENSE'}
    for name in ['__init__.py','text.py','models.py','inference.py']:
        files['src/complaints/'+name] = ROOT/'src/complaints'/name
    for path in sorted((ROOT/'deployment_model').rglob('*')):
        if path.is_file(): files[path.relative_to(ROOT).as_posix()] = path
    for relative, expected in json.loads((ROOT/'deployment_model/bundle.json').read_text())['file_sha256'].items():
        with (ROOT/'deployment_model'/relative).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == expected
    if args.prepare_only:
        import zipfile
        target = ROOT / 'dist'
        target.mkdir(exist_ok=True)
        package = target / 'consumer-complaint-classifier-release.zip'
        hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
        with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in files.items():
                archive.write(path, name)
        manifest = {'status': 'prepared_not_published', 'repo_id': args.repo, 'files': hashes}
        (target / 'release_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print('Prepared local deployment package:', package)
        return
    api = HfApi()
    account = api.whoami()['name']
    if args.repo.split('/')[0] != account:
        raise ValueError('This script only publishes under the authenticated personal account.')
    api.create_repo(repo_id=args.repo,repo_type='space',space_sdk='gradio',private=False,exist_ok=True)
    print('Publishing',len(files),'files to',args.repo,flush=True)
    commit = api.create_commit(repo_id=args.repo,repo_type='space',
        operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in files.items()],
        commit_message='Deploy verified consumer complaint classifier',num_threads=2)
    report = {'status':'uploaded_build_pending','repo_id':args.repo,
        'url':'https://huggingface.co/spaces/'+args.repo,'commit':commit.oid,
        'uploaded_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'hardware':'default free CPU; no paid hardware requested',
        'files':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}}
    (ROOT/'reports/deployment.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','url','commit']}),flush=True)


if __name__ == '__main__':main()

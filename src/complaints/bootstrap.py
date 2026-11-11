"""Download the pinned source and transactionally prepare/audit Kaggle inputs."""
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

SOURCE = 'shashwatwork/consume-complaints-dataset-fo-nlp/versions/1'
CSV_SHA = 'a120ac8450cb3839bf0085bbae68f7c0f3b5351302cafd0bc85ae27a251010b4'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_csv(path):
    if digest(path)!=CSV_SHA:
        raise RuntimeError('Downloaded CSV does not match the audited version 1 SHA-256. No alternative source will be used.')
    return Path(path)


def reference_check(root, reference):
    root=Path(root)
    differences={}
    actual=read(root/'audit.json')
    keys=['retained_rows','split_counts','class_counts_by_split','labels','exclusion_flow',
          'source_rows','source_columns','duplicate_overlap_between_splits','split_group_count',
          'encoded_collision_grouping_passes']
    for key in keys:
        if actual[key]!=reference['audit'][key]:
            differences[key]={'actual':actual[key],'expected':reference['audit'][key]}
    with gzip.open(root/'records.csv.gz','rb') as f:
        records_hash=hashlib.file_digest(f,'sha256').hexdigest()
    if records_hash!=reference['records_csv_sha256']:
        differences['records_and_split_assignments']='Decompressed canonical CSV hash differs'
    if canonical(read(root/'vocabulary.json'))!=reference['vocabulary_sha256']:
        differences['training_vocabulary']='Canonical vocabulary hash differs'
    with zipfile.ZipFile(root/'encoded_inputs.npz') as z:
        if set(z.namelist())!=set(reference['npy_sha256']):
            differences['encoded_array_names']='Array members differ'
        for name,expected in reference['npy_sha256'].items():
            if name not in z.namelist() or hashlib.sha256(z.read(name)).hexdigest()!=expected:
                differences[name]='Encoded array shape/dtype/content differs'
    report={'status':'failed' if differences else 'passed','differences':differences,
            'method':'Canonical record CSV, vocabulary, NPY array bytes and audited counts',
            'reference_source_csv_sha256':CSV_SHA}
    (root/'reference_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if differences:
        raise RuntimeError(f'Preparation differs from supplied reference. Inspect {root / "reference_comparison.json"}; do not train until investigated.')
    return report


def preparation_identity(project_dir):
    project_dir=Path(project_dir)
    versions={name:importlib.metadata.version(name) for name in
              ['numpy','pandas','scikit-learn','transformers','tokenizers']}
    return {'source_csv_sha256':CSV_SHA,
            'config_sha256':canonical(read(project_dir/'configs/experiment.json')),
            'reference_sha256':digest(project_dir/'configs/reference.json'),
            'code':{name:digest(project_dir/'src/complaints'/name) for name in
                    ['preparation.py','bootstrap.py','text.py']},'versions':versions}


def prepare_all(project_dir, prepared_dir):
    from .data import PreparedData, write_json
    project_dir,prepared_dir=Path(project_dir),Path(prepared_dir)
    reference=read(project_dir/'configs/reference.json')
    identity=preparation_identity(project_dir)
    marker=prepared_dir/'preparation_complete.json'
    if marker.exists():
        if read(marker)['identity']!=identity:
            raise RuntimeError(f'Incompatible preparation cache: {prepared_dir}. Choose a new preparation directory or restore compatible code/dependencies.')
        data=PreparedData(prepared_dir,verify=True)
        reference_check(prepared_dir,reference)
        print('Reusing reference-verified complete preparation',flush=True)
        return data.root
    if prepared_dir.exists():
        raise RuntimeError(f'Incomplete preparation exists at {prepared_dir}. Preserve it for diagnosis and use a new path; partial files are never accepted.')
    import kagglehub
    print('Acquiring source version 1 directly inside Kaggle...',flush=True)
    try:
        source_dir=Path(kagglehub.dataset_download(SOURCE))
    except Exception as exc:
        raise RuntimeError('Cannot acquire the public version-1 source. Enable Internet, check Kaggle account/consent access, and rerun. No credentials or alternate dataset are embedded. '+str(exc)) from exc
    found=list(source_dir.rglob('complaints_processed.csv'))
    if len(found)!=1:
        raise RuntimeError('Expected exactly one complaints_processed.csv in source version 1.')
    csv_path=verify_csv(found[0])
    write_json(project_dir/'reports/source_download.json',{
        'handle':SOURCE,'source_csv':str(csv_path),'sha256':CSV_SHA,
        'authentication':'No explicit login/API key supplied by this notebook; Kaggle session or public access used',
        'kagglehub_version':importlib.metadata.version('kagglehub')})
    staging=prepared_dir.with_name(prepared_dir.name+'.building')
    staging.mkdir(parents=True,exist_ok=True)
    previous=staging/'build_identity.json'
    if previous.exists() and read(previous)!=identity:
        raise RuntimeError(f'Incompatible partial preparation at {staging}; use a different preparation path.')
    for stale in ['prepared_manifest.json', 'reference_comparison.json', 'preparation_complete.json']:
        (staging / stale).unlink(missing_ok=True)
    write_json(previous,identity)
    # Rebuild a matching interrupted preparation from scratch; no partial reuse.
    # Its directory is retained if preparation fails, with diagnostic reports.
    subprocess.check_call([sys.executable,'-u','-m','complaints.preparation',
                           '--csv',str(csv_path),'--output',str(staging)],
                          env={**__import__('os').environ,'PYTHONPATH':str(project_dir/'src')})
    data=PreparedData(staging,verify=True)
    comparison=reference_check(staging,reference)
    # Independently reconstruct train-only vocabulary and re-tokenize sampled rows.
    from .verify_preparation import verify
    verification=verify(staging)
    del data
    write_json(staging/'preparation_complete.json',{'identity':identity,
        'reference':comparison,'independent_verification':verification})
    staging.rename(prepared_dir)
    print('Preparation and independent reference checks passed.',flush=True)
    return prepared_dir

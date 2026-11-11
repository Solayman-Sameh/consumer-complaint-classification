"""Independent-model GPU scheduling using fresh, isolated Python processes.

This module deliberately imports neither torch nor transformers. GPU visibility
is set before the worker interpreter starts, avoiding CUDA-after-fork problems.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def schedule_jobs(jobs, gpu_ids, log_dir, poll_seconds=1, heartbeat_seconds=30):
    """Run at most one job per GPU, returning assignments after every success.

    Jobs are dictionaries containing unique 'name', list 'command', and optional
    environment overrides. A failure stops the remaining workers and raises;
    it never permits downstream selection/export to run on partial results.
    """
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Provide at least one unique GPU identifier')
    if len({job['name'] for job in jobs}) != len(jobs):
        raise ValueError('Job names must be unique')
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending, active, assignments = list(jobs), {}, []
    handles = []
    last_heartbeat = time.monotonic()
    start = time.monotonic()
    try:
        while pending or active:
            for gpu in gpu_ids:
                if gpu in active or not pending:
                    continue
                job = pending.pop(0)
                name = job['name']
                if not name or Path(name).name != name or name in {'.', '..'}:
                    raise ValueError('Job name must be a simple filename')
                log_path = log_dir / (name + '.log')
                handle = log_path.open('w', encoding='utf-8')
                handles.append(handle)
                environment = dict(os.environ)
                environment.update(job.get('env', {}))
                environment.update(CUDA_VISIBLE_DEVICES=str(gpu),
                                   COMPLAINTS_ASSIGNED_GPU=str(gpu),
                                   PYTHONUNBUFFERED='1')
                options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
                process = subprocess.Popen(job['command'], env=environment,
                                           stdout=handle, stderr=subprocess.STDOUT, **options)
                active[gpu] = {'process': process, 'job': job, 'log': log_path,
                               'offset': 0, 'started': time.monotonic()}
                assignments.append({'model': name, 'assigned_gpu': str(gpu),
                                    'started_seconds': time.monotonic() - start,
                                    'log': str(log_path)})
                print(f'Started {name} on GPU {gpu}; log: {log_path}', flush=True)
            finished = []
            for gpu, item in active.items():
                # Files rather than pipes avoid deadlock from full stdout buffers.
                with item['log'].open(encoding='utf-8', errors='replace') as stream:
                    stream.seek(item['offset'])
                    chunk = stream.read(65536)
                    item['offset'] = stream.tell()
                if chunk:
                    print(f"[{item['job']['name']} / GPU {gpu}]\n{chunk}", end='', flush=True)
                status = item['process'].poll()
                if status is None:
                    continue
                # Drain any last messages written since the previous read.
                with item['log'].open(encoding='utf-8', errors='replace') as stream:
                    stream.seek(item['offset'])
                    tail = stream.read()
                if tail:
                    print(tail, end='', flush=True)
                if status != 0:
                    raise RuntimeError(f"{item['job']['name']} failed on GPU {gpu} (exit {status}). "
                                       f"Inspect {item['log']}; completed models are retained.")
                elapsed = time.monotonic() - item['started']
                print(f"Finished {item['job']['name']} on GPU {gpu} in {elapsed / 60:.1f} min", flush=True)
                for assignment in assignments:
                    if assignment['model'] == item['job']['name']:
                        assignment['elapsed_seconds'] = elapsed
                        assignment['status'] = 'complete'
                finished.append(gpu)
            for gpu in finished:
                del active[gpu]
            if active and time.monotonic() - last_heartbeat >= heartbeat_seconds:
                print('Running: ' + ', '.join(f"{v['job']['name']} on GPU {g}" for g, v in active.items())
                      + f'; waiting jobs: {len(pending)}', flush=True)
                last_heartbeat = time.monotonic()
            if pending or active:
                time.sleep(poll_seconds)
    finally:
        # Includes keyboard interrupts and notebook cell cancellation.
        for item in active.values():
            _stop(item['process'])
        for handle in handles:
            handle.close()
    return assignments


def train_models_parallel(data, config, output_root, models, gpu_ids, project_dir):
    from .data import read_json, write_json
    output_root, project_dir = Path(output_root).resolve(), Path(project_dir).resolve()
    if not models or len(set(models)) != len(models):
        raise ValueError('Select at least one model, without duplicate names')
    allowed = ['distilbert', 'rnn', 'lstm', 'gru']
    if not set(models).issubset(allowed):
        raise ValueError('Unknown model')
    config_path = output_root / 'parallel_config.json'
    write_json(config_path, config)
    # Start the generally longer transformer first, alongside the first RNN.
    order = [kind for kind in allowed if kind in models]
    inherited = os.environ.get('PYTHONPATH', '')
    jobs = [{'name': kind, 'command': [sys.executable, '-u', '-m', 'complaints.worker',
             '--model', kind, '--data', str(data.root.resolve()), '--config', str(config_path),
             '--output', str(output_root)],
             'env': {'PYTHONPATH': str(project_dir / 'src') + (os.pathsep + inherited if inherited else ''),
                     'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                     'USE_TF': '0', 'USE_FLAX': '0'}} for kind in order]
    assignments = schedule_jobs(jobs, gpu_ids, output_root / 'worker_logs')
    write_json(output_root / 'parallel_execution.json', {
        'strategy': 'independent models; one process per assigned GPU',
        'gpu_ids': [str(gpu) for gpu in gpu_ids], 'assignments': assignments,
        'effective_transformer_batch_size': config['transformer_batch_size'] * config['gradient_accumulation_steps'],
        'note': 'No model is replicated across GPUs; batch sizes and learning rates retain their single-model meaning.'})
    return {kind: read_json(output_root / 'models' / kind / 'result.json') for kind in models}

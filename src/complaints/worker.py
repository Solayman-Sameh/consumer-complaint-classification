"""A fresh worker sees exactly one assigned GPU before importing PyTorch."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['rnn', 'lstm', 'gru', 'distilbert'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from .data import PreparedData, read_json, write_json
    from .training import run_model, environment_record
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Worker requires exactly one CUDA-visible GPU. Check accelerator and scheduler setup.')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    print(f"Worker {args.model}: assigned GPU {os.environ.get('COMPLAINTS_ASSIGNED_GPU')}; "
          f"visible device cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
    data = PreparedData(args.data, verify=True)
    config = read_json(args.config)
    result = run_model(args.model, data, config, args.output, torch.device('cuda:0'))
    write_json(Path(args.output) / 'models' / args.model / 'worker_environment.json', {
        'assigned_gpu': os.environ.get('COMPLAINTS_ASSIGNED_GPU'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'runtime': environment_record(), 'completed_signature': result['signature']})


if __name__ == '__main__':
    main()

import subprocess
import sys
import shlex
import json
from statistics import mean, median, stdev
from pathlib import Path
from math import isfinite, nan


def safe_stdev(values):
    if len(values) < 2 or not all(isfinite(value) for value in values):
        return nan
    return stdev(values)

if __name__ == '__main__':
    exps = [
        'python train.py --path data/test-00000-of-00001.txt --exp amp --dataloader base --kind fp32 --batch_size 32 --num_epochs 1',
        'python train.py --path data/test-00000-of-00001.txt --exp amp --dataloader base --kind fp16 --batch_size 32 --num_epochs 1',
        'python train.py --path data/test-00000-of-00001.txt --exp amp --dataloader base --kind static --batch_size 32 --num_epochs 1 --scale 65536',
        'python train.py --path data/test-00000-of-00001.txt --exp amp --dataloader base --kind dynamic --batch_size 32 --num_epochs 1 --scale 65536 --factor 2 --patience 80 --min_scale 1 --max_scale 16777216',
        
        'python train.py --path data/test-00000-of-00001.txt --exp loading --dataloader base --kind fp32 --batch_size 16 --num_epochs 1',
        'python train.py --path data/test-00000-of-00001.txt --exp loading --dataloader standard --kind fp32 --batch_size 16 --num_epochs 1',
        'python train.py --path data/test-00000-of-00001.txt --exp loading --dataloader balanced --kind fp32 --batch_size 16 --num_epochs 1 --k 20',
        'python train.py --path data/test-00000-of-00001.txt --exp loading --dataloader sequenced --kind fp32 --batch_size 16 --num_epochs 1',
        
        'python train.py --path data/test-00000-of-00001.txt --exp profiling --dataloader sequenced --kind dynamic --batch_size 48 --num_epochs 1 --scale 65536 --factor 2 --patience 80 --min_scale 1 --max_scale 16777216'
    ]

    for exp in exps:
        command = shlex.split(exp)
        command[0] = sys.executable
        subprocess.run(args=command, check=True)
    path_root = Path('results')
    amp_path = ['fp32', 'fp16', 'static', 'dynamic']
    loading_path = ['base', 'standard', 'balanced', 'sequenced']

    result = {}
    for p in amp_path:
        path = path_root / f'{p}.json'
        with open(path, encoding='utf-8') as f:
            p_res = json.load(f)
            result[p] = {
                p + '_loss_by_steps_mean': mean(p_res['loss_by_steps']),
                p + '_loss_by_steps_median': median(p_res['loss_by_steps']),
                p + '_loss_by_steps_std': safe_stdev(p_res['loss_by_steps']),

                p + '_accuracy_by_steps_mean': mean(p_res['accuracy_by_steps']),
                p + '_accuracy_by_steps_median': median(p_res['accuracy_by_steps']),
                p + '_accuracy_by_steps_std': safe_stdev(p_res['accuracy_by_steps']),

                p + '_grad_time_by_steps_mean': mean(p_res['grad_time_by_steps']),
                p + '_grad_time_by_steps_median': median(p_res['grad_time_by_steps']),
                p + '_grad_time_by_steps_std': safe_stdev(p_res['grad_time_by_steps']),

                p + '_all_steps': p_res['all_steps']
            }
    with open('results_stat_amp', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    result = {}
    for p in loading_path:
        path = path_root / f'{p}.json'
        with open(path, encoding='utf-8') as f:
            p_res = json.load(f)
            result[p] = {
                p + '_dataloader_time': p_res['dataloader_time'],
                p + '_peak_mem_allocation': p_res['peak_mem_allocation'],
                p + '_peak_mem_reserved': p_res['peak_mem_reserved'],
                p + '_all_train_time': p_res['all_train_time'],
                p + '_all_steps': p_res['all_steps']
            }
    with open('results_stat_loading', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

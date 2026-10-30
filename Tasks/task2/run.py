import subprocess
import sys
import shlex
import json
from statistics import mean, median, stdev
from pathlib import Path
from math import isfinite, nan
import matplotlib.pyplot as plt

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
        
    ]

    for exp in exps:
        command = shlex.split(exp)
        command[0] = sys.executable
        subprocess.run(args=command, check=True)
    path_root = Path('results')
    amp_path = ['fp32', 'fp16', 'static', 'dynamic']
    loading_path = ['base', 'standard', 'balanced', 'sequenced']

    result = {}
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
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
            

            axes[0].plot(range(1, len(p_res['loss_by_steps']) + 1), p_res['loss_by_steps'], label=p)
            axes[0].set_ylabel("Loss")
            axes[0].set_title("amp: качество и время вычисления градиентов")

            axes[1].plot(range(1, len(p_res['accuracy_by_steps']) + 1), p_res['accuracy_by_steps'], label=p)
            axes[1].set_ylabel("Accuracy")

            grad_time_ms = [seconds * 1000 for seconds in p_res['grad_time_by_steps']]
            axes[2].plot(range(1, len(grad_time_ms) + 1), grad_time_ms, label=p)
            axes[2].set_ylabel("Время, мс")

            axes[2].set_xlabel("Шаг обучения")
            for ax in axes:
                ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.legend()

    fig.savefig(f"all_results_amp.png", dpi=150)
    plt.close(fig)

    with open('results_stat_amp.json', 'w', encoding='utf-8') as f:
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
    with open('results_stat_loading.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

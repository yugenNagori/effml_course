import subprocess
import sys
import shlex

if __name__ == '__main__':
    exps = [
        'python train.py --exp amp --dataloader base --kind fp32 --batch_size 32 --num_epochs 1',
        'python train.py --exp amp --dataloader base --kind fp16 --batch_size 32 --num_epochs 1',
        'python train.py --exp amp --dataloader base --kind static --batch_size 32 --num_epochs 1 --scale 65536',
        'python train.py --exp amp --dataloader base --kind dynamic --batch_size 32 --num_epochs 1 --scale 65536 --factor 2 --patience 80 --min_scale 1 --max_scale 16777216',
        'python train.py --exp loading --dataloader base --kind fp32 --batch_size 32 --num_epochs 1',
        'python train.py --exp loading --dataloader standard --kind fp32 --batch_size 32 --num_epochs 1',
        'python train.py --exp loading --dataloader balanced --kind fp32 --batch_size 32 --num_epochs 1 --k 20',
        'python train.py --exp loading --dataloader sequenced --kind fp32 --batch_size 32 --num_epochs 1',
    ]

    for exp in exps:    
        command = shlex.split(exp)
        command[0] = sys.executable
        subprocess.run(args=command, check=True)
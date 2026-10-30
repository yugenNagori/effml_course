import subprocess
import sys
import shlex


if __name__ == "__main__":
    exps = [
        'python train.py --path data/test-00000-of-00001.txt --exp profiling --dataloader sequenced --kind dynamic --batch_size 48 --num_epochs 1 --scale 65536 --factor 2 --patience 80 --min_scale 1 --max_scale 16777216 --output_dir profiling_results/fix_all',
        'python train.py --path data/test-00000-of-00001.txt --exp loading --dataloader sequenced --kind dynamic --batch_size 48 --num_epochs 1 --scale 65536 --factor 2 --patience 80 --min_scale 1 --max_scale 16777216 --output_dir profiling_results/fix_all',
    ]

    for exp in exps:
        command = shlex.split(exp)
        command[0] = sys.executable
        subprocess.run(args=command, check=True)

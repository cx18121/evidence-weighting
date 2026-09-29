"""Wrapper around Kev's real trainer (not a substitute optimizer). Run after generate.py."""
import argparse
import subprocess
import sys
from pathlib import Path

BASE = {'4b': ('Qwen/Qwen3.5-4B-Base', 'jaredpalmer/kev-4b'),
        '08b': ('Qwen/Qwen3.5-0.8B-Base', 'jaredpalmer/kev-0.8b')}


def command(data, out, size='4b', steps=500, seed=155):
    base, checkpoint = BASE[size]
    return [sys.executable, '-m', 'kev.train', '--data', str(data), '--base', base,
            '--init_from', checkpoint, '--epochs', '2', '--lr', '2e-5',
            '--batch', '1', '--accum', '8', '--max_steps', str(steps),
            '--max_state', '2048', '--dtype', 'bf16',
            '--checkpointing', '1', '--device', 'cuda', '--seed', str(seed), '--out', str(out)]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--curriculum', choices=['varied','fixed8'], required=True)
    p.add_argument('--size', choices=BASE, default='4b'); p.add_argument('--data', default='results/training')
    p.add_argument('--out', default='results/trained'); p.add_argument('--steps', type=int, default=500)
    a = p.parse_args()
    file = Path(a.data) / (a.curriculum + '.jsonl')
    if not file.exists(): raise FileNotFoundError(file)
    out = Path(a.out) / (a.size + '-' + a.curriculum)
    if (out / 'head.pt').exists():
        print('Existing complete checkpoint:', out); return
    # Kev LoRA trainer does not support resumable optimizer state; incomplete jobs restart.
    if out.exists(): raise RuntimeError(f'Incomplete run at {out}: inspect before restarting')
    subprocess.run(command(file, out, a.size, a.steps), check=True)

if __name__ == '__main__': main()

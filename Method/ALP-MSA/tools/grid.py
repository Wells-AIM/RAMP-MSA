"""
Tiny job queue: run every (experiment, seed) pair, `per_gpu` jobs at a time on each GPU.
Skips pairs whose result.json already exists (safe to re-launch).

usage: python tools/grid.py EXP_FILE [--gpus 0,1,2] [--per_gpu 3] [--seeds 1111,2222,3333,4444,5555]
EXP_FILE: one experiment per line:  <name> <config> [key=value ...]
"""

import argparse
import os
import subprocess
import sys
import time


def free_mb(gpu):
    out = subprocess.run(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout
    return int(out.strip() or 0)


def already_running(name, seed):
    out = subprocess.run(['pgrep', '-f', f'train.py .*--seed {seed} --name {name} '],
                         capture_output=True, text=True).stdout
    return bool(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_file')
    ap.add_argument('--gpus', default='0,1,2')
    ap.add_argument('--per_gpu', type=int, default=2)
    ap.add_argument('--min_free_mb', type=int, default=10000)
    ap.add_argument('--seeds', default='1111,2222,3333,4444,5555')
    opt = ap.parse_args()

    jobs = []
    for line in open(opt.exp_file):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        name, cfg, *sets = line.split()
        for s in opt.seeds.split(','):
            if os.path.exists(f'runs/{name}/seed_{s}/result.json'):
                continue
            jobs.append((name, cfg, sets, int(s)))
    print(f'{len(jobs)} jobs', flush=True)

    slots = {int(g): [] for g in opt.gpus.split(',')}
    os.makedirs('logs', exist_ok=True)
    while jobs or any(slots.values()):
        for g in slots:
            slots[g] = [p for p in slots[g] if p.poll() is None]
            while jobs and len(slots[g]) < opt.per_gpu and free_mb(g) >= opt.min_free_mb:
                name, cfg, sets, s = jobs.pop(0)
                if already_running(name, s) or os.path.exists(f'runs/{name}/seed_{s}/result.json'):
                    continue
                cmd = [sys.executable, 'train.py', '--config', cfg, '--seed', str(s), '--name', name,
                       '--gpu', str(g)] + (['--set'] + sets if sets else [])
                log = open(f'logs/{name}_seed{s}.log', 'w')
                slots[g].append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
                print(time.strftime('%H:%M:%S'), 'start', name, s, 'gpu', g, flush=True)
                time.sleep(30)  # let the new job allocate its memory before re-checking
        if jobs and not any(slots.values()) and all(free_mb(g) < opt.min_free_mb for g in slots):
            time.sleep(60)      # GPUs busy with jobs from another launcher
            continue
        time.sleep(10)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()

"""
Convert an MMSA pickle (e.g. MOSEI unaligned_50.pkl, 13.6 GB) into per-split .npy
files that training processes can memory-map instead of each unpickling the whole file.

usage: python tools/prepare_npy.py SRC.pkl OUT_DIR
"""

import json
import os
import pickle
import sys

import numpy as np


def main(src, out):
    os.makedirs(out, exist_ok=True)
    with open(src, 'rb') as f:
        data = pickle.load(f)
    meta = {}
    for split, d in data.items():
        print(split, {k: getattr(v, 'shape', len(v) if hasattr(v, '__len__') else None) for k, v in d.items()}, flush=True)
        n = len(d['regression_labels'])
        audio = np.asarray(d['audio'], dtype=np.float32)
        vision = np.asarray(d['vision'], dtype=np.float32)
        audio[~np.isfinite(audio)] = 0
        vision[~np.isfinite(vision)] = 0
        arrays = {
            'text_bert': np.asarray(d['text_bert'], dtype=np.float32),
            'audio': audio,
            'vision': vision,
            'regression_labels': np.asarray(d['regression_labels'], dtype=np.float32),
        }
        for key in ('audio_lengths', 'vision_lengths'):
            if key in d and d[key] is not None and len(d[key]) == n:
                arrays[key] = np.asarray(d[key], dtype=np.int64)
        for k, v in arrays.items():
            np.save(os.path.join(out, f'{split}_{k}.npy'), v)
        with open(os.path.join(out, f'{split}_id.json'), 'w') as f:
            json.dump([str(x) for x in d['id']], f)
        meta[split] = {k: list(v.shape) for k, v in arrays.items()}
        del arrays, audio, vision
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('DONE', json.dumps(meta), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])

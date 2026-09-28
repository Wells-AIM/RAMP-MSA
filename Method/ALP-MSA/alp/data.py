"""MMSA-format unaligned MOSI / MOSEI loader that also returns A/V validity masks."""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_CACHE = {}


def _load_npy_dir(path):
    """Directory written by tools/prepare_npy.py -> {split: {key: memmapped array}}."""
    import json
    import os
    meta = json.load(open(os.path.join(path, 'meta.json')))
    data = {}
    for split, keys in meta.items():
        data[split] = {k: np.load(os.path.join(path, f'{split}_{k}.npy'), mmap_mode='r') for k in keys}
        data[split]['id'] = json.load(open(os.path.join(path, f'{split}_id.json')))
    return data


def _load(path):
    if path not in _CACHE:
        if path.endswith('.pkl'):
            with open(path, 'rb') as f:
                _CACHE[path] = pickle.load(f)
        else:
            _CACHE[path] = _load_npy_dir(path)
    return _CACHE[path]


def _lengths(split, key, feats):
    if key in split and split[key] is not None and len(split[key]) == len(feats):
        lens = np.asarray(split[key], dtype=np.int64)
    else:  # fall back to "last non-zero frame"
        nz = np.abs(feats).sum(-1) > 0
        lens = np.where(nz.any(1), feats.shape[1] - np.argmax(nz[:, ::-1], 1), 1)
    return np.clip(lens, 1, feats.shape[1])


class MMDataset(Dataset):
    def __init__(self, cfg, mode):
        split = _load(cfg.dataset.dataPath)[mode]
        if isinstance(split['audio'], np.memmap):   # prepared by tools/prepare_npy.py (already clean float32)
            self.text, self.vision, self.audio = split['text_bert'], split['vision'], split['audio']
        else:
            self.text = split['text_bert'].astype(np.float32)
            self.vision = split['vision'].astype(np.float32)
            self.audio = split['audio'].astype(np.float32)
            self.audio[~np.isfinite(self.audio)] = 0
            self.vision[~np.isfinite(self.vision)] = 0
        self.audio_len = _lengths(split, 'audio_lengths', self.audio)
        self.vision_len = _lengths(split, 'vision_lengths', self.vision)
        self.labels = split['regression_labels'].astype(np.float32)
        self.ids = list(split['id'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        la, lv = self.audio.shape[1], self.vision.shape[1]
        return {
            'text': torch.from_numpy(np.array(self.text[i], dtype=np.float32)),
            'audio': torch.from_numpy(np.array(self.audio[i], dtype=np.float32)),
            'vision': torch.from_numpy(np.array(self.vision[i], dtype=np.float32)),
            'audio_mask': torch.arange(la) < int(self.audio_len[i]),
            'vision_mask': torch.arange(lv) < int(self.vision_len[i]),
            'label': torch.tensor([self.labels[i]]),
            'index': i,
        }


def build_loaders(cfg, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    loaders = {}
    for mode in ('train', 'valid', 'test'):
        ds = MMDataset(cfg, mode)
        loaders[mode] = DataLoader(ds, batch_size=cfg.base.batch_size, shuffle=(mode == 'train'),
                                   num_workers=cfg.base.num_workers, generator=g if mode == 'train' else None,
                                   pin_memory=True)
    return loaders

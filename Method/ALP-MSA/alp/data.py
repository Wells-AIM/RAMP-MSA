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
        self.labels = np.asarray(split['regression_labels'], dtype=np.float32)
        self.ids = list(split['id'])
        self.norm = None
        self.drop = set(getattr(cfg.dataset, 'drop_modalities', None) or [])

        # Direction 3: teacher reasoning targets, attached to the TRAIN split only
        self.teacher = None
        d = getattr(cfg, 'distill', None)
        if mode == 'train' and d is not None and d.teacher_path and (d.w_label > 0 or d.w_reason > 0 or d.w_fields > 0):
            z = np.load(d.teacher_path)
            row = {str(i): k for k, i in enumerate(z['ids'])}
            idx = np.array([row.get(str(i), -1) for i in self.ids])
            has = idx >= 0
            safe = np.where(has, idx, 0)
            t = {'score': np.nan_to_num(z['score'][safe]).astype(np.float32),
                 'has_score': has & np.isfinite(z['score'][safe]),
                 'z_all': z['z_all'][safe].astype(np.float32),
                 'has_all': has & (z['valid_all'][safe] if 'valid_all' in z.files else True)}
            for f in ('semantic', 'prosody', 'facial'):
                t[f'z_{f}'] = z[f'z_{f}'][safe].astype(np.float32)
                t[f'has_{f}'] = has & z[f'valid_{f}'][safe]
            self.teacher = t
            print(f'teacher targets: {has.mean():.3f} of train utterances covered, score parsed '
                  f'{t["has_score"].mean():.3f}', flush=True)

    def __len__(self):
        return len(self.labels)

    def masked_stats(self, name, max_samples=4000):
        """Per-dim mean/std over valid frames (subsample of utterances for speed)."""
        x, lens = (self.audio, self.audio_len) if name == 'audio' else (self.vision, self.vision_len)
        idx = np.random.RandomState(0).permutation(len(lens))[:max_samples]
        frames = np.concatenate([np.asarray(x[i][:lens[i]], dtype=np.float64) for i in sorted(idx)])
        return frames.mean(0).astype(np.float32), (frames.std(0) + 1e-6).astype(np.float32)

    def __getitem__(self, i):
        la, lv = self.audio.shape[1], self.vision.shape[1]
        am = torch.arange(la) < int(self.audio_len[i])
        vm = torch.arange(lv) < int(self.vision_len[i])
        audio = np.array(self.audio[i], dtype=np.float32)
        vision = np.array(self.vision[i], dtype=np.float32)
        if 'audio' in self.drop:        # modality removed for training and testing (unimodal baselines)
            audio[:] = 0
            am = torch.arange(la) < 1
        if 'vision' in self.drop:
            vision[:] = 0
            vm = torch.arange(lv) < 1
        if self.norm is not None:
            audio = (audio - self.norm['audio'][0]) / self.norm['audio'][1] * am.numpy()[:, None]
            vision = (vision - self.norm['vision'][0]) / self.norm['vision'][1] * vm.numpy()[:, None]
        item = self._item(i, audio, vision, am, vm)
        if self.teacher is not None:
            item['teacher'] = {k: torch.as_tensor(v[i]) for k, v in self.teacher.items()}
        return item

    def _item(self, i, audio, vision, am, vm):
        return {
            'text': torch.from_numpy(np.array(self.text[i], dtype=np.float32)),
            'audio': torch.from_numpy(audio.astype(np.float32)),
            'vision': torch.from_numpy(vision.astype(np.float32)),
            'audio_mask': am,
            'vision_mask': vm,
            'label': torch.tensor([self.labels[i]]),
            'index': i,
        }


def build_loaders(cfg, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    loaders, norm = {}, None
    for mode in ('train', 'valid', 'test'):
        ds = MMDataset(cfg, mode)
        if getattr(cfg.dataset, 'input_norm', False):
            if mode == 'train':
                norm = {m: ds.masked_stats(m) for m in ('audio', 'vision')}
            ds.norm = norm
        loaders[mode] = DataLoader(ds, batch_size=cfg.base.batch_size, shuffle=(mode == 'train'),
                                   num_workers=cfg.base.num_workers, generator=g if mode == 'train' else None,
                                   pin_memory=True)
    return loaders

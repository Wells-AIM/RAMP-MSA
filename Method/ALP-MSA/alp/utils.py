import copy
from types import SimpleNamespace

import yaml


def dict_to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    return d


def apply_overrides(raw, overrides):
    """overrides: ['model.audio_mode=dynamic', 'base.lr=5e-5', ...] (values parsed as YAML)."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        path, val = item.split('=', 1)
        node = raw
        keys = path.split('.')
        for k in keys[:-1]:
            node = node[k]
        if keys[-1] not in node:
            raise KeyError(f'unknown config key: {path}')
        node[keys[-1]] = yaml.safe_load(val)
    return raw

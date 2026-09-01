from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


class AttrDict(dict):
    """Dict with attribute access; recursively wraps nested dicts."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    __setattr__ = dict.__setitem__


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path) -> AttrDict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _wrap(cfg)


def _parse_value(raw: str) -> Any:
    # YAML parser conveniently handles numbers, booleans, null, lists, dicts.
    return yaml.safe_load(raw)


def apply_overrides(cfg: AttrDict, overrides: Iterable[str] | None) -> AttrDict:
    cfg = _wrap(copy.deepcopy(dict(cfg)))
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        cur: Dict[str, Any] = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = AttrDict()
            cur = cur[p]
        cur[parts[-1]] = _wrap(_parse_value(raw))
    return cfg


def save_config(cfg: AttrDict, path: str | Path) -> None:
    def unwrap(x):
        if isinstance(x, dict):
            return {k: unwrap(v) for k, v in x.items()}
        if isinstance(x, list):
            return [unwrap(v) for v in x]
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(unwrap(cfg), f, sort_keys=False, allow_unicode=True)

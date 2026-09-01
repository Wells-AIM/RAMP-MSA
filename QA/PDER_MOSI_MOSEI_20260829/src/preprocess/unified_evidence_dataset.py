"""Unified evidence dataloader for one-network text+audio QA experiments.

All datasets are exposed to the model with the same batch schema:
- text_features: [batch, 12, 128, 768]
- audio_features: [batch, 12, 257, 768]

The on-disk payload may be direct (subjects/samples already contain tensors) or
indexed (MELD stores utterance features once and items reference them).
"""
import os
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler


FEATURE_TENSOR_DTYPES = {
    "text_features": torch.float32,
    "text_attention_mask": torch.bool,
    "audio_features": torch.float32,
    "audio_attention_mask": torch.bool,
}


def _pad_or_truncate_tensor(value, target_shape: Optional[Tuple[int, ...]], dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.as_tensor(value).to(dtype=dtype)
    if not target_shape or tuple(tensor.shape) == tuple(target_shape):
        return tensor
    if tensor.dim() != len(target_shape):
        return tensor
    fitted = tensor.new_zeros(tuple(target_shape))
    slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(tensor.shape, target_shape))
    fitted[slices] = tensor[slices]
    return fitted


def _infer_sample_tensor_shapes(sample: Dict[str, object]) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    for key in FEATURE_TENSOR_DTYPES:
        if key in sample:
            shapes[key] = tuple(torch.as_tensor(sample[key]).shape)
    return shapes


def _normalize_sample_tensor_shapes(
    sample: Dict[str, object],
    target_shapes: Dict[str, Tuple[int, ...]],
) -> Dict[str, object]:
    if not target_shapes:
        return sample
    normalized = dict(sample)
    for key, dtype in FEATURE_TENSOR_DTYPES.items():
        if key in normalized and key in target_shapes:
            normalized[key] = _pad_or_truncate_tensor(normalized[key], target_shapes[key], dtype)
    return normalized


class UnifiedEvidenceDataset(Dataset):
    def __init__(self, feature_path: str, split: str):
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Unified evidence feature file does not exist: {feature_path}")
        self.payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        self.split = split
        self.metadata = self.payload.get("metadata", {}) if isinstance(self.payload, dict) else {}
        self.indexed = bool(self.metadata.get("storage") == "indexed_unified_evidence")
        if self.indexed:
            self.items = self.payload["items"]
        elif isinstance(self.payload, dict) and "samples" in self.payload:
            self.items = self.payload["samples"]
        elif isinstance(self.payload, dict) and "subjects" in self.payload:
            self.items = self.payload["subjects"]
        else:
            self.items = self.payload
        print(
            f"Loaded unified evidence {split}: {len(self.items)} samples from {feature_path}, "
            f"indexed={self.indexed}"
        )

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _to_float_tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32)

    @staticmethod
    def _to_bool_tensor(value) -> torch.Tensor:
        return torch.as_tensor(value).bool()

    def _empty_source(self) -> Dict[str, object]:
        return {
            "text_features": self.payload["empty_text_features"],
            "text_attention_mask": self.payload["empty_text_attention_mask"],
            "audio_features": self.payload["empty_audio_features"],
            "audio_attention_mask": self.payload["empty_audio_attention_mask"],
        }

    def _source_from_ref(self, ref: Dict[str, object]) -> Dict[str, object]:
        ref_type = str(ref.get("type", "empty"))
        if ref_type == "utterance":
            index = int(ref.get("index", -1))
            if index >= 0:
                return self.payload["utterances"][index]
        if ref_type == "summary":
            key = str(ref.get("key", ""))
            source = self.payload.get("dialogue_summaries", {}).get(key)
            if source is not None:
                return source
        return self._empty_source()

    def _getitem_indexed(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        text_features = []
        text_masks = []
        audio_features = []
        audio_masks = []
        for ref in item["unit_refs"]:
            source = self._source_from_ref(ref)
            text_features.append(self._to_float_tensor(source["text_features"]))
            text_masks.append(self._to_bool_tensor(source["text_attention_mask"]))
            audio_features.append(self._to_float_tensor(source["audio_features"]))
            audio_masks.append(self._to_bool_tensor(source["audio_attention_mask"]))
        return {
            "text_features": torch.stack(text_features, dim=0),
            "text_attention_mask": torch.stack(text_masks, dim=0),
            "audio_features": torch.stack(audio_features, dim=0),
            "audio_attention_mask": torch.stack(audio_masks, dim=0),
            "emotion": torch.tensor(int(item["label_id"]), dtype=torch.long),
            "subject_id": str(item.get("subject_id", index)),
            "score": torch.tensor(float(item.get("score", 0.0)), dtype=torch.float32),
            "raw_score": torch.tensor(float(item.get("raw_score", item.get("score", 0.0))), dtype=torch.float32),
            "label": str(item.get("label", "")),
            "prompt_names": list(item.get("prompt_names", self.metadata.get("unit_names", []))),
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.indexed:
            return self._getitem_indexed(index)
        item = self.items[index]
        return {
            "text_features": self._to_float_tensor(item["text_features"]),
            "text_attention_mask": self._to_bool_tensor(item["text_attention_mask"]),
            "audio_features": self._to_float_tensor(item["audio_features"]),
            "audio_attention_mask": self._to_bool_tensor(item["audio_attention_mask"]),
            "emotion": torch.tensor(int(item["label_id"]), dtype=torch.long),
            "subject_id": str(item.get("subject_id", index)),
            "score": torch.tensor(float(item.get("score", 0.0)), dtype=torch.float32),
            "raw_score": torch.tensor(float(item.get("raw_score", item.get("score", 0.0))), dtype=torch.float32),
            "label": str(item.get("label", "")),
            "prompt_names": list(item.get("prompt_names", self.metadata.get("unit_names", []))),
        }


ORIGINAL_ID_KEYS = (
    "original_id",
    "source_subject_id",
    "source_sample_id",
    "clean_subject_id",
    "clean_sample_id",
)


def _sample_original_id(item: Dict[str, object], fallback: str) -> str:
    """Return the clean sample id used for fold-aware augmentation filtering."""
    for key in ORIGINAL_ID_KEYS:
        value = item.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(item.get("subject_id", fallback))


def _sample_subject_id(item: Dict[str, object], fallback: str) -> str:
    return str(item.get("subject_id", _sample_original_id(item, fallback)))


def _record_identity_sets(
    source_datasets: Sequence["UnifiedEvidenceDataset"],
    records: Sequence[Tuple[int, int]],
) -> Tuple[Set[str], Set[str]]:
    original_ids: Set[str] = set()
    subject_ids: Set[str] = set()
    for dataset_idx, item_idx in records:
        item = source_datasets[dataset_idx].items[item_idx]
        fallback = f"{dataset_idx}:{item_idx}"
        original_ids.add(_sample_original_id(item, fallback))
        subject_ids.add(_sample_subject_id(item, fallback))
    return original_ids, subject_ids


def _assert_disjoint_identities(
    named_records: Dict[str, Sequence[Tuple[int, int]]],
    source_datasets: Sequence["UnifiedEvidenceDataset"],
) -> None:
    identities = {
        name: _record_identity_sets(source_datasets, records)
        for name, records in named_records.items()
    }
    names = list(identities)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            original_overlap = identities[left][0] & identities[right][0]
            subject_overlap = identities[left][1] & identities[right][1]
            if original_overlap or subject_overlap:
                raise ValueError(
                    f"Data leakage detected between {left} and {right}: "
                    f"original_id_overlap={sorted(original_overlap)[:20]}, "
                    f"subject_id_overlap={sorted(subject_overlap)[:20]}"
                )


def _source_prosody_parameters(dataset: "UnifiedEvidenceDataset") -> Tuple[torch.Tensor, torch.Tensor]:
    mean = dataset.metadata.get("prosody_mean")
    std = dataset.metadata.get("prosody_std")
    if mean is None or std is None:
        raise ValueError(
            f"Fold-local prosody normalization requires prosody_mean/prosody_std in {dataset.split} metadata"
        )
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
    std_tensor = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
    if mean_tensor.numel() == 0 or mean_tensor.numel() != std_tensor.numel():
        raise ValueError(f"Invalid prosody statistics in {dataset.split} metadata")
    return mean_tensor, std_tensor


def _raw_prosody_from_item(
    item: Dict[str, object],
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
) -> torch.Tensor:
    audio = torch.as_tensor(item["audio_features"], dtype=torch.float32)
    mask = torch.as_tensor(item["audio_attention_mask"]).bool()
    if audio.dim() < 3 or mask.dim() < 2 or audio.size(-2) == 0:
        return audio.new_empty((0, source_mean.numel()))
    valid = mask[..., -1]
    if not valid.any():
        return audio.new_empty((0, source_mean.numel()))
    width = int(source_mean.numel())
    normalized = audio[..., -1, :width]
    return normalized[valid] * source_std.to(audio.device) + source_mean.to(audio.device)


def _fit_fold_prosody_statistics(
    source_datasets: Sequence["UnifiedEvidenceDataset"],
    clean_train_records: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    raw_values = []
    for dataset_idx, item_idx in clean_train_records:
        dataset = source_datasets[dataset_idx]
        source_mean, source_std = _source_prosody_parameters(dataset)
        raw = _raw_prosody_from_item(dataset.items[item_idx], source_mean, source_std)
        if raw.numel() > 0:
            raw_values.append(raw)
    if not raw_values:
        raise ValueError("Cannot fit fold-local prosody statistics: no valid train prosody tokens")
    matrix = torch.cat(raw_values, dim=0).float()
    return matrix.mean(dim=0), matrix.std(dim=0, unbiased=False)


def _renormalize_sample_prosody(
    sample: Dict[str, object],
    source_mean: torch.Tensor,
    source_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Dict[str, object]:
    audio = torch.as_tensor(sample["audio_features"], dtype=torch.float32)
    mask = torch.as_tensor(sample["audio_attention_mask"]).bool()
    if audio.dim() < 3 or mask.dim() < 2 or audio.size(-2) == 0:
        return sample
    valid = mask[..., -1]
    if not valid.any():
        return sample
    width = int(source_mean.numel())
    raw = audio[..., -1, :width] * source_std.to(audio.device) + source_mean.to(audio.device)
    normalized = (raw - target_mean.to(audio.device)) / target_std.to(audio.device).clamp_min(1e-6)
    repeats = int(math.ceil(audio.size(-1) / max(width, 1)))
    tiled = normalized.repeat(*([1] * (normalized.dim() - 1)), repeats)[..., : audio.size(-1)]
    updated_audio = audio.clone()
    updated_audio[..., -1, :] = torch.where(valid.unsqueeze(-1), tiled, updated_audio[..., -1, :])
    updated = dict(sample)
    updated["audio_features"] = updated_audio
    return updated


class UnifiedEvidenceKFoldDataset(Dataset):
    """A lightweight fold view over existing cached feature files."""

    def __init__(
        self,
        source_datasets: Sequence[UnifiedEvidenceDataset],
        records: Sequence[Tuple[int, int]],
        split: str,
        clean_record_count: Optional[int] = None,
        augmented_record_count: Optional[int] = None,
        fold_prosody_mean: Optional[torch.Tensor] = None,
        fold_prosody_std: Optional[torch.Tensor] = None,
    ):
        self.source_datasets = list(source_datasets)
        self.records = [(int(dataset_idx), int(item_idx)) for dataset_idx, item_idx in records]
        self.split = split
        self.metadata = self.source_datasets[0].metadata if self.source_datasets else {}
        self.indexed = False
        self.clean_record_count = int(clean_record_count if clean_record_count is not None else len(self.records))
        self.augmented_record_count = int(augmented_record_count if augmented_record_count is not None else 0)
        self.fold_prosody_mean = (
            None if fold_prosody_mean is None else fold_prosody_mean.detach().float().clone()
        )
        self.fold_prosody_std = (
            None if fold_prosody_std is None else fold_prosody_std.detach().float().clone()
        )
        self.items = [
            self.source_datasets[dataset_idx].items[item_idx]
            for dataset_idx, item_idx in self.records
        ]
        self._target_shapes: Dict[str, Tuple[int, ...]] = {}
        for source_dataset in self.source_datasets:
            if len(source_dataset) == 0:
                continue
            self._target_shapes = _infer_sample_tensor_shapes(source_dataset[0])
            if self._target_shapes:
                break
        print(
            f"Loaded unified evidence kfold {split}: {len(self.items)} samples "
            f"(clean={self.clean_record_count}, augmented={self.augmented_record_count})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, object]:
        dataset_idx, item_idx = self.records[index]
        sample = self.source_datasets[dataset_idx][item_idx]
        if self.fold_prosody_mean is not None and self.fold_prosody_std is not None:
            source_mean, source_std = _source_prosody_parameters(self.source_datasets[dataset_idx])
            sample = _renormalize_sample_prosody(
                sample,
                source_mean,
                source_std,
                self.fold_prosody_mean,
                self.fold_prosody_std,
            )
        return _normalize_sample_tensor_shapes(sample, self._target_shapes)


def _make_stratified_fold_indices(
    labels: Sequence[int],
    num_folds: int,
    fold_index: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    if num_folds < 2:
        raise ValueError(f"kfold.num_folds must be at least 2, got {num_folds}")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"kfold.fold_index must be in [0, {num_folds}), got {fold_index}")

    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.size < num_folds:
        raise ValueError(f"Cannot run {num_folds}-fold split with only {labels_array.size} samples")
    _, class_counts = np.unique(labels_array, return_counts=True)
    min_class_count = int(class_counts.min()) if class_counts.size else 0
    if min_class_count < num_folds:
        raise ValueError(
            f"Cannot run stratified {num_folds}-fold split because the smallest class has "
            f"{min_class_count} samples"
        )

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    all_indices = np.arange(labels_array.size)
    folds = list(splitter.split(all_indices, labels_array))
    train_indices, dev_indices = folds[fold_index]
    return train_indices.tolist(), dev_indices.tolist()


def _make_group_stratified_fold_indices(
    labels: Sequence[int],
    groups: Sequence[str],
    num_folds: int,
    fold_index: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split samples after augmentation while keeping each original_id in one fold."""
    if len(labels) != len(groups):
        raise ValueError(f"labels/groups length mismatch: {len(labels)} vs {len(groups)}")
    if num_folds < 2:
        raise ValueError(f"kfold.num_folds must be at least 2, got {num_folds}")
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"kfold.fold_index must be in [0, {num_folds}), got {fold_index}")

    import numpy as np

    labels_array = np.asarray(labels, dtype=np.int64)
    groups_array = np.asarray([str(group) for group in groups], dtype=object)
    if labels_array.size < num_folds:
        raise ValueError(f"Cannot run {num_folds}-fold split with only {labels_array.size} samples")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        all_indices = np.arange(labels_array.size)
        folds = list(splitter.split(all_indices, labels_array, groups_array))
        train_indices, dev_indices = folds[fold_index]
        return train_indices.tolist(), dev_indices.tolist()
    except Exception:
        pass

    # Fallback for older scikit-learn versions: stratify the unique original_id
    # groups, then expand group folds back to sample indices.
    from sklearn.model_selection import StratifiedKFold

    group_to_label: Dict[str, int] = {}
    group_order: List[str] = []
    for label, group in zip(labels_array.tolist(), groups_array.tolist()):
        group = str(group)
        if group not in group_to_label:
            group_to_label[group] = int(label)
            group_order.append(group)
        elif group_to_label[group] != int(label):
            raise ValueError(f"original_id group {group} contains multiple labels")

    group_labels = np.asarray([group_to_label[group] for group in group_order], dtype=np.int64)
    if group_labels.size < num_folds:
        raise ValueError(f"Cannot run {num_folds}-fold split with only {group_labels.size} original_id groups")
    _, class_counts = np.unique(group_labels, return_counts=True)
    min_class_count = int(class_counts.min()) if class_counts.size else 0
    if min_class_count < num_folds:
        raise ValueError(
            f"Cannot run group-stratified {num_folds}-fold split because the smallest "
            f"group class has {min_class_count} groups"
        )

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    group_indices = np.arange(group_labels.size)
    folds = list(splitter.split(group_indices, group_labels))
    train_group_indices, dev_group_indices = folds[fold_index]
    train_groups = {group_order[index] for index in train_group_indices.tolist()}
    dev_groups = {group_order[index] for index in dev_group_indices.tolist()}
    train_indices = [idx for idx, group in enumerate(groups_array.tolist()) if str(group) in train_groups]
    dev_indices = [idx for idx, group in enumerate(groups_array.tolist()) if str(group) in dev_groups]
    return train_indices, dev_indices


class CrossSampleBalancedBatchSampler(Sampler[List[int]]):
    """JEST-style train batch curation over sample relationships.

    The sampler keeps batch-level class balance while avoiding duplicate subject/sample
    ids inside a batch whenever possible. It replaces weighted sampling with a more
    stable cross-sample batch organization and does not use dev/test information.
    """

    def __init__(self, dataset: Dataset, data_config: Dict[str, object]):
        self.dataset = dataset
        self.batch_size = int(data_config["batch_size"])
        labels = [int(item["label_id"]) for item in getattr(dataset, "items", [])]
        if not labels:
            raise ValueError("CrossSampleBalancedBatchSampler requires dataset.items with label_id")
        self.labels = labels
        configured_classes = int(data_config.get("num_classes", max(labels) + 1))
        self.classes = [cls for cls in range(configured_classes) if any(label == cls for label in labels)]
        self.indices_by_class = {
            cls: [idx for idx, label in enumerate(labels) if label == cls]
            for cls in self.classes
        }
        self.subject_ids = [
            str(item.get("subject_id", idx))
            for idx, item in enumerate(getattr(dataset, "items", []))
        ]
        multiplier = float(data_config.get("samples_per_epoch_multiplier", 1.0))
        self.num_samples = max(int(round(len(labels) * multiplier)), len(labels))
        self.num_batches = max(1, math.ceil(self.num_samples / max(self.batch_size, 1)))
        self.seed = int(data_config.get("sampler_seed", 2026))
        self.epoch = 0
        self.drop_last = bool(data_config.get("batch_curation_drop_last", False))

    def __len__(self) -> int:
        return self.num_batches

    def _class_plan(self, batch_idx: int) -> List[int]:
        if not self.classes:
            return []
        class_count = len(self.classes)
        base = self.batch_size // class_count
        rem = self.batch_size % class_count
        rotated = self.classes[batch_idx % class_count:] + self.classes[:batch_idx % class_count]
        plan: List[int] = []
        for offset, cls in enumerate(rotated):
            quota = base + (1 if offset < rem else 0)
            plan.extend([cls] * quota)
        return plan[: self.batch_size]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        pools = {cls: list(indices) for cls, indices in self.indices_by_class.items()}
        for indices in pools.values():
            rng.shuffle(indices)
        ptrs = {cls: 0 for cls in self.classes}

        def next_index(cls: int, used_subjects: set) -> int:
            pool = pools[cls]
            if not pool:
                raise StopIteration
            # Prefer a subject not already present in this batch.
            for _ in range(2):
                start_ptr = ptrs[cls]
                for step in range(len(pool)):
                    pos = (start_ptr + step) % len(pool)
                    idx = pool[pos]
                    if self.subject_ids[idx] not in used_subjects:
                        ptrs[cls] = (pos + 1) % len(pool)
                        if ptrs[cls] == 0:
                            rng.shuffle(pool)
                        return idx
                rng.shuffle(pool)
            idx = pool[ptrs[cls] % len(pool)]
            ptrs[cls] = (ptrs[cls] + 1) % len(pool)
            if ptrs[cls] == 0:
                rng.shuffle(pool)
            return idx

        for batch_idx in range(self.num_batches):
            batch: List[int] = []
            used_subjects = set()
            for cls in self._class_plan(batch_idx):
                idx = next_index(cls, used_subjects)
                batch.append(idx)
                used_subjects.add(self.subject_ids[idx])
            if batch and (not self.drop_last or len(batch) == self.batch_size):
                yield batch


def get_kfold_unified_evidence_dataloaders(config: Dict[str, object]) -> Dict[str, DataLoader]:
    data_config = config["data"]
    cache_path = data_config["cache_path"]
    kfold_config = data_config.get("kfold", {})
    num_folds = int(kfold_config.get("num_folds", data_config.get("num_folds", 10)))
    fold_index = int(kfold_config.get("fold_index", data_config.get("fold_index", 0)))
    fold_seed = int(kfold_config.get("seed", config.get("random_seed", 2026)))
    source_splits = list(kfold_config.get("source_splits", ("train", "dev", "test")))
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config.get("num_workers", 0))

    source_datasets: List[UnifiedEvidenceDataset] = []
    for source_split in source_splits:
        feature_path = os.path.join(cache_path, f"{source_split}_qa_features.pt")
        if not os.path.exists(feature_path):
            continue
        source_datasets.append(UnifiedEvidenceDataset(feature_path, f"kfold_source_{source_split}"))
    if not source_datasets:
        raise FileNotFoundError(f"No kfold source features found under {cache_path}: {source_splits}")

    all_records: List[Tuple[int, int]] = []
    all_labels: List[int] = []
    all_original_ids: List[str] = []
    all_is_augmented: List[bool] = []
    for dataset_idx, dataset in enumerate(source_datasets):
        for item_idx, item in enumerate(dataset.items):
            all_records.append((dataset_idx, item_idx))
            all_labels.append(int(item["label_id"]))
            all_original_ids.append(_sample_original_id(item, f"{dataset_idx}:{item_idx}"))
            all_is_augmented.append(False)

    augmentation_cache_path = (
        kfold_config.get("augmentation_cache_path")
        or data_config.get("augmentation_cache_path")
    )
    augmentation_split_policy = str(
        kfold_config.get(
            "augmentation_split_policy",
            data_config.get("augmentation_split_policy", ""),
        )
        or ""
    ).lower()
    full_augmented_split = augmentation_split_policy in {
        "group_full",
        "augmented_full",
        "full",
        "resplit_augmented",
        "pre_split",
    }
    augmentation_source_splits = list(
        kfold_config.get("augmentation_source_splits", source_splits)
    )

    if augmentation_cache_path and full_augmented_split:
        loaded_augmented_splits = 0
        augmented_count = 0
        for source_split in augmentation_source_splits:
            feature_path = os.path.join(str(augmentation_cache_path), f"{source_split}_qa_features.pt")
            if not os.path.exists(feature_path):
                continue
            dataset_idx = len(source_datasets)
            augmented_dataset = UnifiedEvidenceDataset(feature_path, f"kfold_full_augmented_{source_split}")
            source_datasets.append(augmented_dataset)
            loaded_augmented_splits += 1
            for item_idx, item in enumerate(augmented_dataset.items):
                all_records.append((dataset_idx, item_idx))
                all_labels.append(int(item["label_id"]))
                all_original_ids.append(_sample_original_id(item, f"aug:{dataset_idx}:{item_idx}"))
                all_is_augmented.append(True)
                augmented_count += 1
        if loaded_augmented_splits == 0:
            raise FileNotFoundError(
                f"kfold augmentation_cache_path is configured but no features were found under "
                f"{augmentation_cache_path}: {augmentation_source_splits}"
            )
        print(
            "Augmented-full kfold loaded: "
            f"path={augmentation_cache_path}, clean={len(all_records) - augmented_count}, "
            f"augmented={augmented_count}, policy={augmentation_split_policy}"
        )

    splitter_name = str(kfold_config.get("splitter", "") or "").lower()
    grouped_split = full_augmented_split or splitter_name in {
        "stratifiedgroupkfold",
        "groupstratifiedkfold",
    }
    if grouped_split:
        outer_train_indices, outer_test_indices = _make_group_stratified_fold_indices(
            all_labels,
            all_original_ids,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=fold_seed,
        )
    else:
        outer_train_indices, outer_test_indices = _make_stratified_fold_indices(
            all_labels,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=fold_seed,
        )

    evaluation_protocol = str(kfold_config.get("evaluation_protocol", "legacy_cv") or "legacy_cv").lower()
    nested_cv = evaluation_protocol in {"nested", "nested_cv", "strict_nested_cv"}
    if nested_cv and full_augmented_split:
        raise ValueError("nested_cv requires train-only augmentation, not a full augmented split")
    if nested_cv:
        inner_num_folds = int(kfold_config.get("inner_num_folds", 5))
        inner_fold_index = int(kfold_config.get("inner_fold_index", fold_index % inner_num_folds))
        inner_labels = [all_labels[index] for index in outer_train_indices]
        inner_groups = [all_original_ids[index] for index in outer_train_indices]
        inner_splitter_name = str(
            kfold_config.get("inner_splitter", kfold_config.get("splitter", "")) or ""
        ).lower()
        inner_grouped_split = inner_splitter_name in {
            "stratifiedgroupkfold",
            "groupstratifiedkfold",
        }
        if inner_grouped_split:
            inner_train_relative, inner_dev_relative = _make_group_stratified_fold_indices(
                inner_labels,
                inner_groups,
                num_folds=inner_num_folds,
                fold_index=inner_fold_index,
                seed=int(kfold_config.get("inner_seed", fold_seed + 1009)),
            )
        else:
            inner_train_relative, inner_dev_relative = _make_stratified_fold_indices(
                inner_labels,
                num_folds=inner_num_folds,
                fold_index=inner_fold_index,
                seed=int(kfold_config.get("inner_seed", fold_seed + 1009)),
            )
        train_indices = [outer_train_indices[index] for index in inner_train_relative]
        dev_indices = [outer_train_indices[index] for index in inner_dev_relative]
        test_indices = list(outer_test_indices)
    else:
        train_indices = list(outer_train_indices)
        dev_indices = list(outer_test_indices)
        test_indices = list(outer_test_indices)

    train_records = [all_records[index] for index in train_indices]
    dev_records = [all_records[index] for index in dev_indices]
    test_records = [all_records[index] for index in test_indices]
    train_original_ids: Set[str] = {all_original_ids[index] for index in train_indices}
    dev_original_ids: Set[str] = {all_original_ids[index] for index in dev_indices}
    test_original_ids: Set[str] = {all_original_ids[index] for index in test_indices}
    train_augmented_count = sum(1 for index in train_indices if all_is_augmented[index])
    dev_augmented_count = sum(1 for index in dev_indices if all_is_augmented[index])
    test_augmented_count = sum(1 for index in test_indices if all_is_augmented[index])
    train_clean_count = len(train_records) - train_augmented_count
    dev_clean_count = len(dev_records) - dev_augmented_count
    test_clean_count = len(test_records) - test_augmented_count
    train_clean_records = list(train_records)

    if bool(kfold_config.get("assert_disjoint_subjects", nested_cv)):
        identity_splits = {"train": train_clean_records, "dev": dev_records}
        if nested_cv:
            identity_splits["test"] = test_records
        _assert_disjoint_identities(identity_splits, source_datasets)

    augmented_records: List[Tuple[int, int]] = []
    if augmentation_cache_path and not full_augmented_split:
        loaded_augmented_splits = 0
        dev_augmented_skipped = 0
        test_augmented_skipped = 0
        ignored_augmented = 0
        for source_split in augmentation_source_splits:
            feature_path = os.path.join(str(augmentation_cache_path), f"{source_split}_qa_features.pt")
            if not os.path.exists(feature_path):
                continue
            dataset_idx = len(source_datasets)
            augmented_dataset = UnifiedEvidenceDataset(feature_path, f"kfold_augmented_{source_split}")
            source_datasets.append(augmented_dataset)
            loaded_augmented_splits += 1
            for item_idx, item in enumerate(augmented_dataset.items):
                original_id = _sample_original_id(item, f"aug:{dataset_idx}:{item_idx}")
                if original_id in train_original_ids:
                    augmented_records.append((dataset_idx, item_idx))
                elif original_id in dev_original_ids:
                    dev_augmented_skipped += 1
                elif original_id in test_original_ids:
                    test_augmented_skipped += 1
                else:
                    ignored_augmented += 1
        if loaded_augmented_splits == 0:
            raise FileNotFoundError(
                f"kfold augmentation_cache_path is configured but no features were found under "
                f"{augmentation_cache_path}: {augmentation_source_splits}"
            )
        if ignored_augmented and bool(kfold_config.get("fail_on_unknown_augmentation", False)):
            raise ValueError(
                f"Found {ignored_augmented} augmented samples without a matching clean fold identity"
            )
        print(
            "Fold-aware augmentation loaded: "
            f"path={augmentation_cache_path}, train_added={len(augmented_records)}, "
            f"dev_skipped={dev_augmented_skipped}, test_skipped={test_augmented_skipped}, "
            f"out_of_fold_skipped={ignored_augmented}"
        )
        train_records = train_records + augmented_records
        train_augmented_count += len(augmented_records)

    fold_prosody_mean = None
    fold_prosody_std = None
    if bool(kfold_config.get("fold_local_prosody_normalization", False)):
        fold_prosody_mean, fold_prosody_std = _fit_fold_prosody_statistics(
            source_datasets,
            train_clean_records,
        )
        print(
            "Fold-local prosody normalization enabled: "
            f"fit_clean_train={len(train_clean_records)}, width={fold_prosody_mean.numel()}"
        )

    train_dataset = UnifiedEvidenceKFoldDataset(
        source_datasets,
        train_records,
        "train",
        clean_record_count=train_clean_count,
        augmented_record_count=train_augmented_count,
        fold_prosody_mean=fold_prosody_mean,
        fold_prosody_std=fold_prosody_std,
    )
    dev_dataset = UnifiedEvidenceKFoldDataset(
        source_datasets,
        dev_records,
        "dev",
        clean_record_count=dev_clean_count,
        augmented_record_count=dev_augmented_count,
        fold_prosody_mean=fold_prosody_mean,
        fold_prosody_std=fold_prosody_std,
    )
    test_dataset = dev_dataset
    if nested_cv:
        test_dataset = UnifiedEvidenceKFoldDataset(
            source_datasets,
            test_records,
            "test",
            clean_record_count=test_clean_count,
            augmented_record_count=test_augmented_count,
            fold_prosody_mean=fold_prosody_mean,
            fold_prosody_std=fold_prosody_std,
        )
    train_dataset.kfold_train_original_ids = train_original_ids
    train_dataset.kfold_dev_original_ids = dev_original_ids
    train_dataset.kfold_test_original_ids = test_original_ids
    dev_dataset.kfold_train_original_ids = train_original_ids
    dev_dataset.kfold_dev_original_ids = dev_original_ids
    dev_dataset.kfold_test_original_ids = test_original_ids
    test_dataset.kfold_train_original_ids = train_original_ids
    test_dataset.kfold_dev_original_ids = dev_original_ids
    test_dataset.kfold_test_original_ids = test_original_ids
    print(
        f"Unified evidence kfold split: protocol={evaluation_protocol}, fold={fold_index + 1}/{num_folds}, "
        f"seed={fold_seed}, train={len(train_dataset)}, dev={len(dev_dataset)}, "
        f"test={len(test_dataset)}, train_aug={train_augmented_count}, "
        f"dev_aug={dev_augmented_count}, test_aug={test_augmented_count}"
    )

    batch_sampler = _build_cross_sample_batch_sampler(train_dataset, data_config)
    sampler = None if batch_sampler is not None else _build_balanced_sampler(train_dataset, data_config)
    if batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = dev_loader
    if nested_cv:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    return {"train": train_loader, "dev": dev_loader, "test": test_loader}


def resolve_label_space_auto_value(kind: str, value, labels: torch.Tensor, num_classes: Optional[int] = None) -> float:
    """Resolve unified auto imbalance knobs from train-label statistics only.

    label_space_auto is intentionally separate from the older sampling ``auto``
    behavior so old experiment configs keep their original meaning.
    """
    if labels.numel() == 0:
        return float(value)
    inferred_classes = int(labels.max().item()) + 1
    configured_classes = int(num_classes or inferred_classes)
    effective_classes = max(configured_classes, inferred_classes)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "label_space_auto":
            if effective_classes <= 2:
                return 1.0 if kind == "sampling_alpha" else 0.35
            return 0.50 if kind == "sampling_alpha" else 0.10
        if lowered == "auto" and kind == "sampling_alpha":
            counts = torch.bincount(labels, minlength=effective_classes).float().clamp_min(1.0)
            present_counts = counts[counts > 0]
            imbalance_ratio = float((present_counts.max() / present_counts.min()).item()) if present_counts.numel() else 1.0
            # Legacy auto rule: keep for backward-compatible probe configs.
            ratio_progress = max(0.0, min(1.0, (imbalance_ratio - 4.0) / 12.0))
            return 0.75 - 0.25 * ratio_progress
    return float(value)


def _build_cross_sample_batch_sampler(dataset: UnifiedEvidenceDataset, data_config: Dict[str, object]) -> Optional[CrossSampleBalancedBatchSampler]:
    mode = str(data_config.get("batch_sampling", data_config.get("cross_sample_batch_sampling", "")) or "").lower()
    if mode not in {"joint_batch_curation", "balanced_no_replacement", "cross_sample_balanced"}:
        return None
    sampler = CrossSampleBalancedBatchSampler(dataset, data_config)
    labels = torch.tensor([int(item["label_id"]) for item in dataset.items], dtype=torch.long)
    num_classes = int(data_config.get("num_classes", labels.max().item() + 1)) if labels.numel() else 0
    counts = torch.bincount(labels, minlength=num_classes).int().tolist() if labels.numel() else []
    print(
        f"Cross-sample batch curation enabled for {dataset.split}: "
        f"mode={mode}, counts={counts}, batch_size={sampler.batch_size}, "
        f"num_batches={len(sampler)}, sampled_epoch_size~={sampler.num_samples}"
    )
    return sampler


def _build_balanced_sampler(dataset: UnifiedEvidenceDataset, data_config: Dict[str, object]) -> Optional[WeightedRandomSampler]:
    if not data_config.get("balanced_sampling", False):
        return None
    labels = torch.tensor([int(item["label_id"]) for item in dataset.items], dtype=torch.long)
    if labels.numel() == 0:
        return None
    num_classes = int(data_config.get("num_classes", labels.max().item() + 1))
    label_mapping = data_config.get("label_mapping")
    if isinstance(label_mapping, dict) and label_mapping:
        num_classes = max(num_classes, max(int(v) for v in label_mapping.values()) + 1)
    counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    alpha_config = data_config.get("sampling_alpha", 1.0)
    sampling_alpha = resolve_label_space_auto_value("sampling_alpha", alpha_config, labels, num_classes)
    sampling_alpha = max(0.0, min(1.0, sampling_alpha))
    dataset.resolved_sampling_alpha = sampling_alpha
    dataset.class_counts = counts
    dataset.imbalance_control = str(alpha_config)
    # alpha=0 keeps the natural distribution; alpha=1 makes class sampling roughly uniform.
    sample_weights = torch.tensor(
        [float(counts[int(label)].pow(-sampling_alpha).item()) for label in labels],
        dtype=torch.double,
    )
    multiplier = float(data_config.get("samples_per_epoch_multiplier", 1.0))
    num_samples = max(int(round(len(dataset) * multiplier)), len(dataset))
    generator = torch.Generator()
    generator.manual_seed(int(data_config.get("sampler_seed", 2026)))
    print(
        f"Balanced unified sampler enabled for {dataset.split}: "
        f"counts={counts.int().tolist()}, alpha={sampling_alpha:.2f}, "
        f"alpha_config={alpha_config}, sampled_epoch_size={num_samples}"
    )
    return WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True, generator=generator)


def get_unified_evidence_dataloaders(config: Dict[str, object]) -> Dict[str, DataLoader]:
    data_config = config["data"]
    kfold_config = data_config.get("kfold", {})
    if bool(kfold_config.get("enabled", False) or data_config.get("kfold_enabled", False)):
        return get_kfold_unified_evidence_dataloaders(config)

    cache_path = data_config["cache_path"]
    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config.get("num_workers", 0))
    datasets: Dict[str, Dataset] = {}
    clean_datasets: Dict[str, UnifiedEvidenceDataset] = {}
    for split in ("train", "dev", "test"):
        feature_path = os.path.join(cache_path, f"{split}_qa_features.pt")
        if os.path.exists(feature_path):
            clean_datasets[split] = UnifiedEvidenceDataset(feature_path, split)
            datasets[split] = clean_datasets[split]
    if "train" not in clean_datasets or "dev" not in clean_datasets:
        raise FileNotFoundError(f"Expected train/dev unified evidence features under {cache_path}")

    if bool(data_config.get("assert_disjoint_subjects", False)):
        identities = {}
        for split, dataset in clean_datasets.items():
            original_ids = {
                _sample_original_id(item, f"{split}:{index}")
                for index, item in enumerate(dataset.items)
            }
            subject_ids = {
                _sample_subject_id(item, f"{split}:{index}")
                for index, item in enumerate(dataset.items)
            }
            identities[split] = (original_ids, subject_ids)
        splits = list(identities)
        for left_index, left in enumerate(splits):
            for right in splits[left_index + 1 :]:
                original_overlap = identities[left][0] & identities[right][0]
                subject_overlap = identities[left][1] & identities[right][1]
                if original_overlap or subject_overlap:
                    raise ValueError(
                        f"Data leakage detected between official {left}/{right}: "
                        f"original_ids={sorted(original_overlap)[:20]}, "
                        f"subject_ids={sorted(subject_overlap)[:20]}"
                    )

    augmentation_cache_path = data_config.get("augmentation_cache_path")
    if augmentation_cache_path:
        source_datasets: List[UnifiedEvidenceDataset] = [clean_datasets["train"]]
        train_records = [(0, index) for index in range(len(clean_datasets["train"]))]
        train_ids = {
            _sample_original_id(item, f"train:{index}")
            for index, item in enumerate(clean_datasets["train"].items)
        }
        heldout_ids = set()
        for split in ("dev", "test"):
            dataset = clean_datasets.get(split)
            if dataset is not None:
                heldout_ids.update(
                    _sample_original_id(item, f"{split}:{index}")
                    for index, item in enumerate(dataset.items)
                )
        augmented_records = []
        heldout_skipped = 0
        unknown_skipped = 0
        loaded_augmented_splits = 0
        for source_split in data_config.get("augmentation_source_splits", ("train",)):
            feature_path = os.path.join(
                str(augmentation_cache_path), f"{source_split}_qa_features.pt"
            )
            if not os.path.exists(feature_path):
                continue
            dataset_idx = len(source_datasets)
            augmented_dataset = UnifiedEvidenceDataset(
                feature_path, f"official_train_augmented_{source_split}"
            )
            source_datasets.append(augmented_dataset)
            loaded_augmented_splits += 1
            for item_idx, item in enumerate(augmented_dataset.items):
                original_id = _sample_original_id(item, f"aug:{dataset_idx}:{item_idx}")
                if original_id in train_ids:
                    augmented_records.append((dataset_idx, item_idx))
                elif original_id in heldout_ids:
                    heldout_skipped += 1
                else:
                    unknown_skipped += 1
        if loaded_augmented_splits == 0:
            raise FileNotFoundError(
                f"augmentation_cache_path is configured but no features were found under "
                f"{augmentation_cache_path}"
            )
        if unknown_skipped and bool(data_config.get("fail_on_unknown_augmentation", False)):
            raise ValueError(
                f"Found {unknown_skipped} augmented samples without a matching official split identity"
            )
        train_records.extend(augmented_records)
        datasets["train"] = UnifiedEvidenceKFoldDataset(
            source_datasets,
            train_records,
            "train",
            clean_record_count=len(clean_datasets["train"]),
            augmented_record_count=len(augmented_records),
        )
        print(
            "Official-split train augmentation loaded: "
            f"train_added={len(augmented_records)}, heldout_skipped={heldout_skipped}, "
            f"unknown_skipped={unknown_skipped}"
        )

    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        batch_sampler = _build_cross_sample_batch_sampler(dataset, data_config) if split == "train" else None
        sampler = None if batch_sampler is not None else (_build_balanced_sampler(dataset, data_config) if split == "train" else None)
        if batch_sampler is not None:
            dataloaders[split] = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train" and sampler is None),
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
    return dataloaders

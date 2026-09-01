#!/usr/bin/env python3
"""Convert the aligned HME MOSI/MOSEI pickle into RAMP dense features.

The source sample format is::

    ((words, vision, audio), regression_label, sample_id)

Text is contextualized once with a frozen local BERT model.  The original
vision/audio arrays, labels, IDs, and split membership are preserved.
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


SPLIT_ALIASES = {"train": "train", "dev": "valid", "valid": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bert-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-text-tokens", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def tokenize_words(tokenizer: BertTokenizer, words: list[Any], max_length: int) -> list[int]:
    pieces: list[str] = []
    for word in words:
        current = tokenizer.tokenize(str(word))
        pieces.extend(current if current else [tokenizer.unk_token])
    pieces = pieces[: max(max_length - 2, 0)]
    tokens = [tokenizer.cls_token, *pieces, tokenizer.sep_token]
    return tokenizer.convert_tokens_to_ids(tokens)


def clean_sequence(value: Any, valid_length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a [time, dim] feature array, got {array.shape}")
    length = max(1, min(int(valid_length), array.shape[0]))
    array = array[:length].copy()
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


@torch.inference_mode()
def convert_split(
    examples: list[Any],
    tokenizer: BertTokenizer,
    bert: BertModel,
    device: torch.device,
    batch_size: int,
    max_text_tokens: int,
    split_name: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "text": [],
        "audio": [],
        "vision": [],
        "text_mask": [],
        "audio_mask": [],
        "vision_mask": [],
        "regression_labels": [],
        "id": [],
        "raw_text": [],
    }
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for start in tqdm(range(0, len(examples), batch_size), desc=split_name):
        chunk = examples[start : start + batch_size]
        token_ids: list[list[int]] = []
        parsed: list[tuple[list[Any], np.ndarray, np.ndarray, float, str]] = []
        for example in chunk:
            if not isinstance(example, (tuple, list)) or len(example) != 3:
                raise ValueError(f"Unexpected sample format: {type(example).__name__}")
            modalities, label, sample_id = example
            if not isinstance(modalities, (tuple, list)) or len(modalities) != 3:
                raise ValueError("Expected modalities=(words, vision, audio)")
            words, vision, audio = modalities
            words = list(words)
            valid_length = len(words)
            vision_array = clean_sequence(vision, valid_length)
            audio_array = clean_sequence(audio, valid_length)
            ids = tokenize_words(tokenizer, words, max_text_tokens)
            token_ids.append(ids)
            parsed.append((words, vision_array, audio_array, float(label), str(sample_id)))

        max_len = max(len(ids) for ids in token_ids)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=device)
        for row, ids in enumerate(token_ids):
            length = len(ids)
            input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row, :length] = 1

        hidden = bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.detach().cpu()

        for row, (words, vision, audio, label, sample_id) in enumerate(parsed):
            text_length = len(token_ids[row])
            text = hidden[row, :text_length].numpy().astype(np.float16, copy=False)
            result["text"].append(text)
            result["vision"].append(vision)
            result["audio"].append(audio)
            result["text_mask"].append(np.ones(text.shape[0], dtype=np.bool_))
            result["vision_mask"].append(np.ones(vision.shape[0], dtype=np.bool_))
            result["audio_mask"].append(np.ones(audio.shape[0], dtype=np.bool_))
            result["regression_labels"].append(label)
            result["id"].append(sample_id)
            result["raw_text"].append(" ".join(str(word) for word in words))

    result["regression_labels"] = np.asarray(result["regression_labels"], dtype=np.float32)
    result["id"] = np.asarray(result["id"])
    result["raw_text"] = np.asarray(result["raw_text"])
    return result


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with source.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Source pickle root must be a dictionary")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path, local_files_only=True)
    bert = BertModel.from_pretrained(args.bert_path, local_files_only=True).to(device).eval()

    converted: dict[str, Any] = {}
    for source_name, target_name in SPLIT_ALIASES.items():
        if source_name not in raw:
            continue
        if target_name in converted:
            raise ValueError(f"Duplicate split mapping for {target_name}")
        converted[target_name] = convert_split(
            raw[source_name],
            tokenizer,
            bert,
            device,
            args.batch_size,
            args.max_text_tokens,
            target_name,
        )
    missing = {"train", "valid", "test"} - converted.keys()
    if missing:
        raise KeyError(f"Missing required splits after conversion: {sorted(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, output)
    print(f"Saved {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    for name, split in converted.items():
        first = 0
        print(
            name,
            len(split["regression_labels"]),
            split["text"][first].shape,
            split["audio"][first].shape,
            split["vision"][first].shape,
        )


if __name__ == "__main__":
    main()

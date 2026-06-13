"""Download and prepare small Chinese text datasets for miniGPT Studio.

The goal is to provide practical online Chinese corpus presets that can be
converted into the same parquet shard layout used by pretraining.
"""

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat_mlx.common import get_base_dir


DATASET_PRESETS = {
    "chinese_wiki": {
        "repo": "pleisto/wikipedia-cn-20230720-filtered",
        "split": "train",
        "kind": "wiki",
        "default_rows": 5000,
        "label": "中文百科文本",
    },
    "chinese_alpaca": {
        "repo": "shibing624/alpaca-zh",
        "split": "train",
        "kind": "instruction",
        "default_rows": 3000,
        "label": "中文问答指令文本",
    },
}


def _first_text(record, keys):
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _format_wiki_record(record):
    title = _first_text(record, ["title"])
    text = _first_text(record, ["text", "content", "completion", "article"])
    if title and text and not text.startswith(title):
        return f"{title}\n{text}"
    return text


def _format_instruction_record(record):
    instruction = _first_text(record, ["instruction", "prompt", "question", "query"])
    input_text = _first_text(record, ["input", "context"])
    output = _first_text(record, ["output", "completion", "answer", "response"])
    if not instruction and not output:
        return _format_wiki_record(record)
    prompt = instruction
    if input_text:
        prompt = f"{prompt}\n{input_text}" if prompt else input_text
    if output:
        return f"用户：{prompt}\n助手：{output}"
    return prompt


def _format_record(record, kind):
    text = _format_instruction_record(record) if kind == "instruction" else _format_wiki_record(record)
    return " ".join(text.split()).strip()


def _load_texts(preset_id, max_rows):
    if preset_id not in DATASET_PRESETS:
        raise ValueError(f"Unknown Chinese dataset preset: {preset_id}")

    preset = DATASET_PRESETS[preset_id]
    print(f"Loading {preset['label']} from Hugging Face: {preset['repo']} ({preset['split']})")
    dataset = load_dataset(preset["repo"], split=preset["split"], streaming=True)
    texts = []
    for record in dataset:
        text = _format_record(record, preset["kind"])
        if text:
            texts.append(text)
        if len(texts) >= max_rows:
            break
    if len(texts) < 2:
        raise ValueError("Downloaded dataset produced fewer than 2 usable rows")
    return texts


def _write_shards(texts, output_dir, val_ratio):
    os.makedirs(output_dir, exist_ok=True)
    val_count = max(1, int(len(texts) * val_ratio))
    train_texts = texts[:-val_count] or texts[:1]
    val_texts = texts[-val_count:]
    for idx, shard_texts in enumerate([train_texts, val_texts]):
        path = os.path.join(output_dir, f"shard_{idx:05d}.parquet")
        pq.write_table(pa.table({"text": shard_texts}), path)
        print(f"Wrote {len(shard_texts):,} rows to {path}")
    return {"train_rows": len(train_texts), "val_rows": len(val_texts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare online Chinese datasets for miniGPT Studio")
    parser.add_argument("--preset", choices=sorted(DATASET_PRESETS), required=True)
    parser.add_argument("--max-rows", type=int, default=0, help="Max rows to download; 0 uses preset default")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args(argv)

    preset = DATASET_PRESETS[args.preset]
    max_rows = args.max_rows if args.max_rows > 0 else preset["default_rows"]
    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "base_data")

    print(f"Preparing Chinese online dataset preset: {args.preset}")
    print(f"Target miniGPT Studio base dir: {base_dir}")
    texts = _load_texts(args.preset, max_rows)
    stats = _write_shards(texts, output_dir, args.val_ratio)

    meta_path = os.path.join(base_dir, "chinese_dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preset": args.preset,
                "repo": preset["repo"],
                "label": preset["label"],
                "examples": len(texts),
                **stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Prepared {len(texts):,} Chinese examples")
    print(f"Saved metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

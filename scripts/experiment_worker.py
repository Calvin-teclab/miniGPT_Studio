"""Run one Studio experiment, keeping completed stages on disk."""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scripts import quickstart as qs
from scripts.experiments import PROJECT, checkpoint, read, save


def identity(folder, depth, source):
    directory = (
        folder
        / "base"
        / "mlx_checkpoints"
        / (f"d{depth}" + ("_sft" if source == "sft" else ""))
    )
    (meta,) = directory.glob("*_meta.json")
    return checkpoint(
        qs._encode_checkpoint_id(
            "general", depth, source, meta.name, "exp_" + folder.name
        )
    )


def strict_json(text):
    def reject(value):
        raise ValueError(value)

    def unique(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ValueError("duplicate key")
        return obj

    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)


def assess(record, stage):
    from nanochat_mlx.common import set_memory_limit
    from nanochat_mlx.engine import Engine
    from nanochat_mlx.tokenizer import get_tokenizer
    from scripts.chat import load_model

    cp = record["base" if stage == "before" else "target"]
    # Resolve the snapshot again before use so a stale checkpoint cannot acquire a new score.
    from scripts.experiments import file_hash

    for path, key in [
        (cp["weights"], "weight_hash"),
        (cp["meta_path"], "meta_hash"),
        (Path(cp["base_dir"]) / "tokenizer/tokenizer.pkl", "tokenizer_hash"),
    ]:
        if file_hash(path) != cp[key]:
            raise ValueError("模型或分词器已改变，请恢复原文件。")
    set_memory_limit(8)
    model = load_model(
        cp["depth"],
        cp["step"],
        cp["source"],
        cp["model_name"],
        meta_path=cp["meta_path"],
    )
    tokenizer = get_tokenizer()
    engine = Engine(model, tokenizer)
    results = []
    for q in record["config"]["questions"]:
        tokens, stop = qs.build_chat_completion_tokens(
            tokenizer, [{"role": "user", "content": q["prompt"]}]
        )
        if len(tokens) + record["generation"]["max_tokens"] > model.config.sequence_len:
            raise ValueError("题目超出模型的生成范围，请缩短提示词。")
        generated = []
        for column, _ in engine.generate(
            tokens, enable_tools=False, **record["generation"]
        ):
            if column[0] in stop:
                break
            generated.append(column[0])
        raw = tokenizer.decode(generated)
        try:
            parsed = strict_json(raw)
            valid = True
        except ValueError:
            parsed, valid = None, False
        exact = (
            json.dumps(parsed, sort_keys=True)
            == json.dumps(q["expected"], sort_keys=True)
            if q["expected"] is not None
            else None
        )
        results.append({"text": raw, "valid": valid, "exact": exact})
    return results


def run_child(folder, stage):
    import random

    import mlx.core as mx
    import numpy as np

    from nanochat_mlx.common import get_peak_memory_mb

    random.seed(42)
    np.random.seed(42)
    mx.random.seed(42)
    record = read(folder / "record.json")
    if stage in ("before", "after"):
        answers = assess(record, stage)
        (folder / (stage + ".json")).write_text(json.dumps(answers, ensure_ascii=False))
    elif stage == "prepare":
        from scripts.music_data import write_parquet_shards
        from scripts.tok_train import main

        rows = [
            json.loads(line)
            for line in (folder / "train.jsonl").read_text().splitlines()
        ]
        docs = [r["prompt"] + " " + r["completion"] for r in rows] * 64
        write_parquet_shards(docs, folder / "base/base_data")
        if main(["--vocab-size=512"]):
            raise RuntimeError("分词器训练失败")
    else:
        if stage == "pretrain":
            from scripts.train import build_parser
            from scripts.train import train as train_fn

            args = build_parser().parse_args(
                [
                    "--depth=2",
                    "--head-dim=64",
                    "--num-iterations=12",
                    "--device-batch-size=1",
                    "--total-batch-size=128",
                    "--max-seq-len=128",
                    "--sample-every=-1",
                    "--eval-steps=0",
                    "--use-simple-adamw",
                    "--memory-limit-gb=4",
                    "--model-name=colors-base-" + folder.name[:8],
                ]
            )
        else:
            from scripts.sft import build_parser
            from scripts.sft import sft as train_fn

            cp = record["base"]
            args = build_parser().parse_args(
                [
                    f"--depth={cp['depth']}",
                    f"--step={cp['step']}",
                    f"--base-model-name={cp['model_name']}",
                    f"--num-iterations={record['config']['steps']}",
                    "--max-seq-len=128",
                    "--eval-steps=0",
                    "--eval-every=-1",
                    "--load-optimizer=0",
                    "--simple-lr=0.0005",
                    "--custom-sft-epochs=40",
                    "--memory-limit-gb=8",
                    f"--custom-sft-path={folder / 'train.jsonl'}",
                    "--model-name=colors-sft-" + folder.name[:8],
                ]
            )
        (folder / (stage + "-params.json")).write_text(
            json.dumps({**vars(args), "seed": 42})
        )
        train_fn(args)
    print(f"Peak memory: {get_peak_memory_mb():.0f}MB")


def work(folder):
    r = read(folder / "record.json")

    if r["kind"] == "training":
        from argparse import Namespace

        from scripts.training_record import recorded_training

        try:
            os.environ.update(
                NANOCHAT_BASE_DIR=r["base_dir"],
                MINIGPT_REPLAY_ID=r["id"],
                MINIGPT_DATA_DOMAIN=r["domain"],
                MINIGPT_DATA_SCOPE=r["scope"],
            )
            stage_name = r["training_stage"]
            if stage_name == "train":
                from nanochat_mlx.train import train
            else:
                from nanochat_mlx.sft import sft as train
            recorded_training(
                stage_name, Namespace(**r["parameters"][stage_name]), train
            )
        except Exception as exc:
            r = read(folder / "record.json")
            r.update(status="failed", error=str(exc), ended=time.time())
            save(folder, r)
        return

    def stage(name, base_dir):
        r["stage"] = name
        save(folder, r)
        start = time.monotonic()
        env = dict(
            os.environ,
            NANOCHAT_BASE_DIR=str(base_dir),
            PYTHONUNBUFFERED="1",
            HF_DATASETS_OFFLINE="1",
        )
        with subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.experiment_worker",
                str(folder),
                name,
            ],
            cwd=PROJECT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as child:
            tail = []
            for line in child.stdout:
                print(line, end="", flush=True)
                tail = (tail + [line.strip()])[-5:]
                match = qs.METRIC_RE.search(line)
                if match and name == "sft":
                    r["curve"].append(
                        {
                            "step": int(match[1]),
                            "loss": float(match[3]),
                            "tok_s": int(match[4].replace(",", "")),
                        }
                    )
                    save(folder, r)
                peak = re.search(r"Peak memory: (\d+)MB", line)
                if peak:
                    r["peak_mb"] = max(r["peak_mb"], int(peak[1]))
            if child.wait():
                raise RuntimeError("\n".join(tail))
        r["timings"][name] = round(time.monotonic() - start, 3)
        if name in ("sft", "pretrain"):
            r["parameters"][name] = read(folder / (name + "-params.json"))
        if name in ("before", "after"):
            r[name] = {"at": time.time(), "answers": read(folder / (name + ".json"))}
        save(folder, r)

    try:
        base_dir = folder / "base"
        if not r["base"]:
            stage("prepare", base_dir)
            stage("pretrain", base_dir)
            r["base"] = identity(folder, 2, "base")
        if r["before"] is None:
            stage("before", r["base"]["base_dir"])
        if not r["target"]:
            cp = r["base"]
            if Path(cp["base_dir"]) != base_dir:
                # Give each run its own output directory and the exact input weights.
                dest = base_dir / "mlx_checkpoints" / f"d{cp['depth']}"
                dest.mkdir(parents=True, exist_ok=True)
                for path in (cp["weights"], cp["meta_path"]):
                    shutil.copy2(path, dest)
                (dest / ".input-copy").touch()
                shutil.copytree(
                    Path(cp["base_dir"]) / "tokenizer",
                    base_dir / "tokenizer",
                    dirs_exist_ok=True,
                )
                from scripts.experiments import file_hash

                if (
                    any(
                        file_hash(dest / Path(cp[k]).name) != cp[h]
                        for k, h in [
                            ("weights", "weight_hash"),
                            ("meta_path", "meta_hash"),
                        ]
                    )
                    or file_hash(base_dir / "tokenizer/tokenizer.pkl")
                    != cp["tokenizer_hash"]
                ):
                    raise ValueError("复制基座时文件发生变化，请重试。")
            stage("sft", base_dir)
            r["target"] = identity(folder, cp["depth"], "sft")
            save(folder, r)
        stage("after", r["target"]["base_dir"])
        r.update(status="done", stage="done")
    except Exception as exc:
        traceback.print_exc()
        r.update(status="failed", error=str(exc))
    finally:
        r["ended"] = time.time()
        save(folder, r)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    if len(sys.argv) == 2:
        work(path)
    else:
        run_child(path, sys.argv[2])

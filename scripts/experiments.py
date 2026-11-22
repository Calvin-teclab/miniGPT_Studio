"""Disk-backed Studio experiments. The worker outlives the HTTP request."""

import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scripts import quickstart as qs

router = APIRouter(prefix="/experiments")
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "sample_datasets/colors_sft.jsonl"
GENERATION = {"temperature": 0, "top_k": 1, "max_tokens": 40, "seed": 42}
RULE = "json-fields-v1"


class Question(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    expected: dict[str, str | int] | None = None


class Config(BaseModel):
    questions: list[Question] = Field(min_length=1, max_length=24)
    base: str = ""
    target: str = ""
    steps: int = Field(default=20, ge=1, le=200)


def defaults():
    return [
        {
            "prompt": f"把{color}色、{n}件转成含颜色和数量的 JSON。",
            "expected": {"颜色": color + "色", "数量": n},
        }
        for i, color in enumerate("红黄蓝绿")
        for n in range(1, 10)
        if (i + n) % 5 == 0
    ]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def save(folder, record):
    temp = folder / "record.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(folder / "record.json")


def folder_for(key):
    if not re.fullmatch("[0-9a-f]{32}", key):
        raise HTTPException(404, "实验不存在")
    return Path(qs.experiments_dir()) / key


@contextmanager
def reserve():
    from scripts import quickstart

    root = Path(qs.experiments_dir())
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise HTTPException(409, "另一个实验正在运行，请等待完成。")
        if quickstart.running_process_claimed:
            raise HTTPException(409, "训练流程正在运行，请等待完成。")
        yield lock


def status(key):
    folder = folder_for(key)
    if not (folder / "record.json").is_file():
        raise HTTPException(404, "实验不存在")
    record = read(folder / "record.json")
    if record["status"] == "running":
        with (folder / "active.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return record
            record = read(folder / "record.json")
            if record["status"] == "running":
                record.update(
                    status="failed", error="工作进程已中断。请重试评测或按原配置复跑。"
                )
    return record


def checkpoint(key):
    cp = qs.resolve_checkpoint_id(key)
    meta = Path(cp["meta_path"])
    weights = Path(str(meta).removesuffix("_meta.json") + ".safetensors")
    tokenizer = Path(cp["base_dir"]) / "tokenizer/tokenizer.pkl"
    if not weights.is_file() or not tokenizer.is_file():
        raise ValueError("模型权重或分词器缺失，请恢复文件或选择其他模型。")
    return {
        **cp,
        "weights": str(weights),
        "weight_hash": file_hash(weights),
        "tokenizer_hash": file_hash(tokenizer),
        "meta_hash": file_hash(meta),
    }


def preflight(config, folder):
    import mlx.core as mx

    if platform.machine() != "arm64" or not mx.metal.is_available():
        raise ValueError("请使用支持 Metal 的 Apple Silicon Mac，并安装 MLX。")
    try:
        info = mx.device_info()
    except RuntimeError as exc:
        raise ValueError(
            "Metal GPU 无法访问，请在本机终端重新启动 Studio 后端。"
        ) from exc
    if not os.access(folder, os.W_OK):
        raise ValueError("实验目录无法写入，请检查目录权限。")
    cp = checkpoint(config.base) if config.base else None
    target = checkpoint(config.target) if config.target else None
    if target and (not cp or config.base == config.target):
        raise ValueError("请选择两个不同的模型。")
    if any(m and m["data_domain"] != "general" for m in (cp, target)):
        raise ValueError("请为文字题集选择文本模型。")
    if cp and not target and cp["source"] != "base":
        raise ValueError("SFT 需要预训练基座。")
    from nanochat_mlx.tokenizer import RustBPETokenizer

    for m in (cp, target):
        if not m:
            continue
        tokenizer = RustBPETokenizer.from_directory(Path(m["base_dir"]) / "tokenizer")
        meta = read(Path(m["meta_path"]))
        if tokenizer.get_vocab_size() != meta.get("vocab_size"):
            raise ValueError(
                "分词器与模型词表不匹配，请恢复训练时的分词器或选择其他模型。"
            )
        for q in config.questions:
            ids, _ = qs.build_chat_completion_tokens(
                tokenizer, [{"role": "user", "content": q.prompt}]
            )
            if len(ids) + GENERATION["max_tokens"] > meta["sequence_len"]:
                raise ValueError("题目过长，请缩短提示词或选择上下文更长的模型。")

    size = max([Path(m["weights"]).stat().st_size for m in (cp, target) if m] or [0])
    required = max(512 << 20, size * (2 if target else 6))
    vm = subprocess.check_output(["vm_stat"], text=True)
    page_size = int(re.search(r"page size of (\d+) bytes", vm)[1])
    available = page_size * sum(
        int(n) for n in re.findall(r"Pages (?:free|inactive):\s+(\d+)", vm)
    )
    if available < required:
        raise ValueError(
            f"预计需要 {required // 2**20} MB 可回收内存，请关闭其他应用或选择较小模型。"
        )
    if shutil.disk_usage(folder).free < max(128 << 20, size * 2):
        raise ValueError("磁盘空间不足，请清理实验目录所在磁盘。")
    return (
        cp,
        target,
        {
            "chip": info.get("device_name"),
            "memory": info.get("memory_size"),
            "available": available,
        },
    )


def launch(folder, record, lock):
    # Both file descriptions are inherited by the detached worker.
    with (
        (folder / "active.lock").open("a+") as active,
        (folder / "run.log").open("a") as log,
    ):
        fcntl.flock(active, fcntl.LOCK_EX)
        record.update(status="running", error="")
        save(folder, record)
        try:
            subprocess.Popen(
                [sys.executable, "-u", "-m", "scripts.experiment_worker", str(folder)],
                cwd=PROJECT,
                env={
                    **os.environ,
                    **(
                        {"NANOCHAT_BASE_DIR": record["base_dir"]}
                        if record["kind"] == "training"
                        else {}
                    ),
                },
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(active.fileno(), lock.fileno()),
            )
        except OSError as exc:
            record.update(status="failed", error=str(exc))
            save(folder, record)
            raise
    return record


@router.get("/preset")
def preset():
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    return {"questions": defaults(), "sample": rows[0], "count": len(rows), "steps": 20}


def start(config, previous=None):
    with reserve() as lock:
        root = Path(qs.experiments_dir())
        try:
            cp, target, device = preflight(config, root)
            training = DATA.read_bytes()
            rows = [json.loads(line) for line in training.decode().splitlines()]
            prompts = [q.prompt.strip() for q in config.questions]
            if not all(prompts) or len(prompts) != len(set(prompts)):
                raise ValueError("题目不能为空或重复。")
            if not target and set(prompts) & {row["prompt"] for row in rows}:
                raise ValueError("留出题与训练提示重复，请换一道题。")
            if previous:
                if digest(training) != previous["data_hash"]:
                    raise ValueError("训练文件内容已改变，请新建实验。")
                for actual, old in [
                    (cp, previous["base"]),
                    (
                        target,
                        previous["target"] if previous["kind"] == "compare" else None,
                    ),
                ]:
                    if old and (
                        not actual
                        or any(
                            actual[k] != old[k]
                            for k in ("weight_hash", "tokenizer_hash", "meta_hash")
                        )
                    ):
                        raise ValueError("模型或分词器内容已改变，请新建实验。")
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        folder = folder_for(uuid.uuid4().hex)
        folder.mkdir()
        (folder / "train.jsonl").write_bytes(training)
        record = {
            "id": folder.name,
            "kind": "compare" if target else "sft",
            "config": config.model_dump(),
            "base": cp,
            "target": target,
            "created": time.time(),
            "data_hash": digest(training),
            "data_file": DATA.name,
            "generation": GENERATION,
            "rule": RULE,
            "device": device,
            "versions": {n: version(n) for n in ("mlx", "tiktoken", "rustbpe")},
            "python": platform.python_version(),
            "parent": previous["id"] if previous else None,
            "stage": "queued",
            "timings": {},
            "before": None,
            "after": None,
            "curve": [],
            "parameters": {},
            "peak_mb": 0,
        }
        return launch(folder, record, lock)


@router.post("")
def create(config: Config):
    return start(config)


@router.get("")
def history():
    root = Path(qs.experiments_dir())
    records = [status(p.parent.name) for p in root.glob("*/record.json")]
    return sorted(
        (
            {
                k: r.get(k)
                for k in ("id", "kind", "created", "status", "stage", "training_stage")
            }
            for r in records
        ),
        key=lambda r: r["created"],
        reverse=True,
    )


@router.get("/{key}")
def detail(key: str):
    record = status(key)
    log = folder_for(key) / "run.log"
    if record["status"] == "running" and log.is_file():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4096))
            record["logs"] = stream.read().decode("utf-8", "replace").splitlines()[-12:]
    return record


@router.post("/{key}/rerun")
def rerun(key: str):
    old = status(key)
    if old["kind"] == "training":
        return replay_training(old)
    config = Config(**old["config"])
    if old["base"]:
        config.base = old["base"]["checkpoint_id"]
    return start(config, old)


@router.post("/{key}/retry")
def retry(key: str):
    with reserve() as lock:
        old = status(key)
        if old["status"] != "failed" or not old["target"] or old["before"] is None:
            raise HTTPException(409, "只有后评测失败时才需要单独重试。")
        return launch(folder_for(key), old, lock)


def replay_training(old):
    from argparse import Namespace

    from scripts.training_record import inventory

    with reserve() as lock:
        stage = old["training_stage"]
        if stage not in old["parameters"]:
            raise HTTPException(409, "原训练未完成输入检查，请在训练流程重新配置。")
        params = old["parameters"][stage].copy()
        base = Path(old["base_dir"])
        try:
            current = inventory(stage, Namespace(**params), base)
            if (
                current != old["inputs"]
                or file_hash(base / "tokenizer/tokenizer.pkl") != old["tokenizer_hash"]
            ):
                raise ValueError("训练数据或分词器发生变化，请新建训练。")
            if old["base"]:
                now = checkpoint(old["base"]["checkpoint_id"])
                if any(
                    now[k] != old["base"][k]
                    for k in ("weight_hash", "meta_hash", "tokenizer_hash")
                ):
                    raise ValueError("基座已变化，请重新选择模型。")
        except (OSError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc
        key = uuid.uuid4().hex
        folder = folder_for(key)
        folder.mkdir()
        params["model_name"] = "replay-" + key
        record = {
            **old,
            "id": key,
            "created": time.time(),
            "parent": old["id"],
            "stage": "queued",
            "target": None,
            "parameters": {stage: params},
            "curve": [],
            "timings": {},
            "peak_mb": 0,
        }
        record.pop("ended", None)
        return launch(folder, record, lock)

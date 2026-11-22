"""Record ordinary Studio training jobs while preserving their live console output."""

import fcntl
import json
import os
import sys
import time
import uuid
from contextlib import redirect_stdout
from importlib.metadata import version
from pathlib import Path

from nanochat_mlx.preflight import require_checkpoint
from scripts import quickstart as qs
from scripts.experiments import checkpoint, digest, file_hash, folder_for, save


def inventory(stage, args, base):
    if stage == "train":
        files = sorted((base / "base_data").glob("*.parquet"))
    else:
        files = [Path(args.custom_sft_path)] if args.custom_sft_path else []
    return {str(p): file_hash(p) for p in files}


def source_key(base):
    for domain, key, candidate in qs.iter_training_bases():
        if Path(candidate).resolve() == base.resolve() and (
            key != "legacy" or domain == "music"
        ):
            return domain, key
    return "general", "legacy"


class LiveLog:
    def __init__(self, stream, record, folder):
        self.stream, self.record, self.folder = stream, record, folder

    def write(self, text):
        n = self.stream.write(text)
        match = qs.METRIC_RE.search(text)
        if match:
            self.record["curve"].append(
                {
                    "step": int(match[1]),
                    "loss": float(match[3]),
                    "tok_s": int(match[4].replace(",", "")),
                }
            )
            save(self.folder, self.record)
        return n

    def flush(self):
        self.stream.flush()


def recorded_training(stage, args, train):
    import mlx.core as mx

    from nanochat_mlx.common import get_peak_memory_mb

    base = Path(qs.get_base_dir()).resolve()
    key = os.environ.get("MINIGPT_REPLAY_ID") or uuid.uuid4().hex
    folder = folder_for(key)
    folder.mkdir(parents=True, exist_ok=True)
    domain, scope = source_key(base)
    # A legacy base directory may be passed through the child environment only.
    domain = os.environ.get("MINIGPT_DATA_DOMAIN", domain)
    scope = os.environ.get("MINIGPT_DATA_SCOPE", scope)
    rerun = "MINIGPT_REPLAY_ID" in os.environ
    if rerun:
        r = json.loads((folder / "record.json").read_text())
    else:
        r = {
            "id": key,
            "kind": "training",
            "config": {"steps": args.num_iterations, "questions": []},
            "base": None,
            "target": None,
            "data_file": args.custom_sft_path
            if stage == "sft" and args.custom_sft_path
            else str(base / "base_data")
            if stage == "train"
            else "默认 SFT 混合数据",
            "created": time.time(),
            "timings": {},
            "before": None,
            "after": None,
            "generation": {},
            "rule": "",
            "curve": [],
            "parameters": {},
            "python": sys.version.split()[0],
            "versions": {"mlx": version("mlx")},
            "device": {},
            "peak_mb": 0,
            "error": "",
            "base_dir": str(base),
            "domain": domain,
            "scope": scope,
            "training_stage": stage,
            "data_hash": "",
            "seed": 42,
        }
    lock = (folder / "active.lock").open("a+")
    if not rerun:
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        r.update(stage=stage, status="running")
        save(folder, r)
        r["inputs"] = inventory(stage, args, base)
        r["data_hash"] = digest(json.dumps(r["inputs"], sort_keys=True).encode())
        r["tokenizer_hash"] = file_hash(base / "tokenizer/tokenizer.pkl")
        r["device"] = {"chip": mx.device_info().get("device_name")}
        if stage == "sft":
            # Match the SFT loader's latest-checkpoint selection and pin it for replay.
            if args.step is None and args.base_model_name is None:
                from nanochat_mlx.sft import _find_latest_checkpoint

                _, path, _ = _find_latest_checkpoint(
                    str(base / "mlx_checkpoints" / f"d{args.depth}")
                )
            else:
                _, path = require_checkpoint(
                    args.depth,
                    "base",
                    args.step,
                    base_dir=str(base),
                    model_name=args.base_model_name,
                )
            cp = checkpoint(
                qs._encode_checkpoint_id(
                    domain, args.depth, "base", Path(path).name, scope
                )
            )
            r["base"] = cp
            args.step, args.base_model_name = cp["step"], cp["model_name"]
        r["parameters"][stage] = vars(args).copy()
        save(folder, r)
        mx.random.seed(r["seed"])
        with redirect_stdout(LiveLog(sys.stdout, r, folder)):
            train(args)
        paths = sorted(
            (
                base
                / "mlx_checkpoints"
                / (f"d{args.depth}" + ("_sft" if stage == "sft" else ""))
            ).glob("*_meta.json"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        for path in paths:
            if json.loads(path.read_text()).get("model_name", "") == args.model_name:
                r["target"] = checkpoint(
                    qs._encode_checkpoint_id(
                        domain,
                        args.depth,
                        "sft" if stage == "sft" else "base",
                        path.name,
                        scope,
                    )
                )
                break
        if not r["target"]:
            raise RuntimeError("训练结束，但未找到对应的模型产物。")
        r.update(status="done", stage="done", peak_mb=round(get_peak_memory_mb()))
    except BaseException as exc:
        r.update(status="failed", error=str(getattr(exc, "detail", exc)))
        raise
    finally:
        r["ended"] = time.time()
        r["timings"][stage] = round(r["ended"] - r["created"], 3)
        save(folder, r)
        lock.close()

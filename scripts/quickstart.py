"""
MLX Quickstart GUI — step-by-step wizard for the full nanochat training pipeline.

Serves a web UI that walks through: data download → tokenizer → training → SFT → chat.
Each stage runs as a subprocess with live SSE streaming of stdout/stderr.

Usage:
    python -m scripts.quickstart
    python -m scripts.quickstart --port 8080
"""

import argparse
import asyncio
import base64
import gc
import importlib.util
import json
import os
import re
import sys
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from nanochat_mlx.common import SetupError
from nanochat_mlx.preflight import (
    count_downloaded_shards,
    require_checkpoint,
    require_tokenizer,
    require_training_data,
)


def build_parser():
    parser = argparse.ArgumentParser(description="miniGPT Studio Quickstart")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=8.0,
        help="MLX memory limit in GB (default 8, conservative for shared use)",
    )
    return parser


args = argparse.Namespace(port=8000, host="127.0.0.1", memory_limit_gb=8.0)

# --- Globals ---

running_process: Optional[asyncio.subprocess.Process] = None
running_process_claimed = False
loaded_engine = None
loaded_tokenizer = None
loaded_model = None  # keep ref for explicit cleanup
loaded_depth = None
loaded_step = None
loaded_source = None
loaded_domain = None
loaded_model_name = None


def _encode_checkpoint_id(data_domain, depth, source, meta_filename):
    payload = json.dumps(
        {
            "data_domain": data_domain,
            "depth": depth,
            "source": source,
            "meta": meta_filename,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return "cp_" + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_checkpoint_id(checkpoint_id):
    if not checkpoint_id or not checkpoint_id.startswith("cp_"):
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")
    encoded = checkpoint_id[3:]
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id") from exc
    required = {"data_domain", "depth", "source", "meta"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise HTTPException(status_code=400, detail="Invalid checkpoint_id")
    return payload


def resolve_checkpoint_id(checkpoint_id):
    payload = _decode_checkpoint_id(checkpoint_id)
    data_domain = str(payload["data_domain"]).lower()
    source = str(payload["source"]).lower()
    depth = int(payload["depth"])
    meta_filename = os.path.basename(str(payload["meta"]))
    if data_domain not in {"general", "music"}:
        raise HTTPException(status_code=400, detail="Invalid checkpoint data_domain")
    if source not in {"base", "sft"}:
        raise HTTPException(status_code=400, detail="Invalid checkpoint source")
    if depth <= 0 or not meta_filename.endswith("_meta.json"):
        raise HTTPException(status_code=400, detail="Invalid checkpoint id payload")

    base = get_music_base_dir() if data_domain == "music" else get_base_dir()
    dirname = f"d{depth}_sft" if source == "sft" else f"d{depth}"
    ckpt_dir = os.path.realpath(os.path.join(base, "mlx_checkpoints", dirname))
    ckpt_root = os.path.realpath(os.path.join(base, "mlx_checkpoints"))
    if not ckpt_dir.startswith(ckpt_root + os.sep):
        raise HTTPException(status_code=400, detail="Invalid checkpoint path")
    meta_path = os.path.realpath(os.path.join(ckpt_dir, meta_filename))
    if not meta_path.startswith(ckpt_dir + os.sep) or not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail="Checkpoint metadata not found")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid checkpoint metadata") from exc
    step = int(meta.get("step", 0))
    if step <= 0:
        raise HTTPException(status_code=400, detail="Invalid checkpoint metadata")
    return {
        "checkpoint_id": checkpoint_id,
        "data_domain": data_domain,
        "depth": depth,
        "step": step,
        "source": source,
        "model_name": meta.get("model_name", ""),
        "meta_path": meta_path,
        "ckpt_dir": ckpt_dir,
    }

METRIC_RE = re.compile(
    r"step\s+(\d+)/(\d+).*?loss:\s*([\d.]+).*?tok/s:\s*([\d,]+)"
)

SAMPLE_PREFIXES = (
    "The capital of France is",
    "If 5*x + 3 = 13, then x is",
)


def build_layer_signal(depth: int, step: int, total: int, loss: float, tok_per_sec: int):
    """Build lightweight per-layer training signals for the frontend heatmap.

    The training loop does not expose raw hidden activations over stdout. This
    signal is derived from live loss/step/speed metrics so the UI can still show
    layer-wise training dynamics without slowing down MLX training.
    """
    progress = step / max(total, 1)
    loss_scale = min(max(loss / 10.0, 0.0), 1.0)
    speed_scale = min(max(tok_per_sec / 10000.0, 0.0), 1.0)
    layers = []
    for i in range(max(depth, 1)):
        layer_pos = (i + 1) / max(depth, 1)
        mean_activation = min(1.0, 0.15 + 0.55 * (1.0 - loss_scale) + 0.20 * layer_pos)
        grad_mean = min(1.0, 0.08 + 0.45 * loss_scale * (1.0 - progress) + 0.12 * (1.0 - layer_pos))
        layers.append({
            "name": f"layer_{i + 1}",
            "meanActivation": mean_activation,
            "maxActivation": min(1.0, mean_activation + 0.18 + 0.05 * speed_scale),
            "gradMean": grad_mean,
            "gradMax": min(1.0, grad_mean + 0.22),
        })
    return {"step": step, "layers": layers, "source": "metric_proxy"}


def get_base_dir():
    base = os.environ.get("NANOCHAT_BASE_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    return base


def get_music_base_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "nanochat_music")


def get_stage_env(data_domain: str):
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if data_domain == "music":
        env["NANOCHAT_BASE_DIR"] = get_music_base_dir()
    return env


def normalize_model_name(model_name: str, data_domain: str, stage: str):
    """Return a safe display name for checkpoint metadata."""
    cleaned = re.sub(r"\s+", " ", (model_name or "").strip())[:80]
    if cleaned:
        return cleaned
    prefix = "music" if data_domain == "music" else "general"
    return f"{prefix}-{stage}-{time.strftime('%Y%m%d-%H%M%S')}"


def check_status():
    """Check which pipeline stages are complete by inspecting the filesystem."""
    def inspect_base(base):
        tok_path = os.path.join(base, "tokenizer", "tokenizer.pkl")
        ckpt_base = os.path.join(base, "mlx_checkpoints")
        shard_count = count_downloaded_shards(base)

        trained = {}
        sft_trained = {}
        if os.path.isdir(ckpt_base):
            for d in sorted(os.listdir(ckpt_base)):
                if not d.startswith("d"):
                    continue
                is_sft = d.endswith("_sft")
                depth_str = d[1:].replace("_sft", "") if is_sft else d[1:]
                if not depth_str.isdigit():
                    continue
                depth = int(depth_str)
                dpath = os.path.join(ckpt_base, d)
                safetensors = [
                    f for f in os.listdir(dpath)
                    if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
                ]
                if safetensors:
                    if is_sft:
                        sft_trained[depth] = len(safetensors)
                    else:
                        trained[depth] = len(safetensors)

        return {
            "data": shard_count >= 2,
            "data_shards": shard_count,
            "tokenizer": os.path.isfile(tok_path),
            "train": trained,
            "sft": sft_trained,
        }

    general = inspect_base(get_base_dir())
    music = inspect_base(get_music_base_dir())
    base = get_base_dir()
    tok_path = os.path.join(base, "tokenizer", "tokenizer.pkl")
    ckpt_base = os.path.join(base, "mlx_checkpoints")
    shard_count = count_downloaded_shards(base)

    data_ready = shard_count >= 2

    tok_ready = os.path.isfile(tok_path)

    # Find all trained depths (base models)
    trained = {}
    sft_trained = {}
    if os.path.isdir(ckpt_base):
        for d in sorted(os.listdir(ckpt_base)):
            if not d.startswith("d"):
                continue
            is_sft = d.endswith("_sft")
            depth_str = d[1:].replace("_sft", "") if is_sft else d[1:]
            if not depth_str.isdigit():
                continue
            depth = int(depth_str)
            dpath = os.path.join(ckpt_base, d)
            safetensors = [
                f for f in os.listdir(dpath)
                if f.endswith(".safetensors") and not f.endswith("_optim.safetensors")
            ]
            if safetensors:
                if is_sft:
                    sft_trained[depth] = len(safetensors)
                else:
                    trained[depth] = len(safetensors)

    chat_ready = loaded_engine is not None

    return {
        "data": data_ready,
        "data_shards": shard_count,
        "tokenizer": tok_ready,
        "train": trained,
        "sft": sft_trained,
        "chat": chat_ready,
        "chat_model": {"depth": loaded_depth, "step": loaded_step, "source": loaded_source, "data_domain": loaded_domain, "model_name": loaded_model_name} if chat_ready else None,
        "running": running_process_claimed or (running_process is not None and running_process.returncode is None),
        "music": music,
    }


# --- FastAPI ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_error_response(message, code=400):
    """Return a one-shot SSE error response that the UI can render."""
    async def stream():
        yield f"data: {json.dumps({'type': 'error', 'text': message, 'code': code})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "nanochat_mlx", "quickstart_ui.html")
    ui_path = os.path.normpath(ui_path)
    with open(ui_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/status")
async def status():
    return check_status()


def preflight_stage(stage: str, depth: int, step: int):
    """Validate stage prerequisites before launching subprocesses."""
    if stage == "tokenizer":
        require_training_data()
    elif stage == "train":
        require_training_data()
        require_tokenizer()
    elif stage == "sft":
        require_checkpoint(depth=depth, source="base", step=step if step > 0 else None)
        require_tokenizer()
    elif stage == "import" and importlib.util.find_spec("torch") is None:
        raise SetupError(
            "HuggingFace import requires the optional convert dependencies. "
            "Install them first with: uv sync --extra convert"
        )


@app.get("/run/{stage}")
async def run_stage(stage: str, n_shards: int = 4, depth: int = 4,
                    step: int = -1,
                    num_iterations: int = -1, use_simple_adamw: bool = False,
                    window_pattern: str = "L", max_seq_len: int = 512,
                    device_batch_size: int = 1, save_every: int = -1,
                    eval_every: int = 100, sample_every: int = 10,
                    memory_limit_gb: float = 0,
                    data_domain: str = "general",
                    music_data_path: str = "",
                    data_source: str = "builtin",
                    custom_data_path: str = "",
                    tokenizer_method: str = "bpe",
                    vocab_size: int = 32768,
                    doc_cap: int = 10000,
                    repo: str = "nanochat-students/base-d20",
                    model_name: str = "",
                    force_tokenizer: bool = False,
                    skip_verify: bool = False):
    """Run a pipeline stage as a subprocess, streaming output via SSE."""
    global running_process, running_process_claimed

    if running_process_claimed or (running_process is not None and running_process.returncode is None):
        raise HTTPException(status_code=409, detail="A process is already running")

    # Default to server's memory limit if not overridden
    if memory_limit_gb <= 0:
        memory_limit_gb = args.memory_limit_gb

    python = sys.executable
    data_domain = data_domain.lower()
    if data_domain not in {"general", "music"}:
        return sse_error_response("data_domain must be 'general' or 'music'")
    data_source = data_source.lower()
    tokenizer_method = tokenizer_method.lower()
    stage_env = get_stage_env(data_domain)
    checkpoint_model_name = normalize_model_name(model_name, data_domain, stage)

    try:
        if stage == "data":
            custom_path = custom_data_path or music_data_path
            if data_source in {"custom_path", "upload"} or data_domain == "music":
                if not custom_path:
                    return sse_error_response("Custom data path is required for uploaded/custom datasets")
                cmd = [python, "-u", "-m", "scripts.music_data", "--input", custom_path]
            else:
                cmd = [python, "-u", "-m", "nanochat_mlx.dataset", "-n", str(n_shards)]
        elif stage == "tokenizer":
            if data_domain == "general":
                preflight_stage(stage, depth, step)
            if tokenizer_method != "bpe":
                return sse_error_response("Current backend only supports BPE tokenization")
            cmd = [
                python, "-u", "-m", "scripts.tok_train",
                f"--vocab-size={vocab_size}",
                f"--doc-cap={doc_cap}",
            ]
        elif stage == "train":
            if data_domain == "general":
                preflight_stage(stage, depth, step)
            cmd = [python, "-u", "-m", "scripts.train",
                   f"--depth={depth}",
                   f"--max-seq-len={max_seq_len}",
                   f"--window-pattern={window_pattern}",
                   f"--device-batch-size={device_batch_size}",
                   f"--memory-limit-gb={memory_limit_gb}",
                   f"--eval-every={eval_every}",
                   f"--sample-every={sample_every}",
                   f"--model-name={checkpoint_model_name}"]
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            effective_save_every = save_every if save_every > 0 else 500
            cmd.append(f"--save-every={effective_save_every}")
            if use_simple_adamw:
                cmd.append("--use-simple-adamw")
        elif stage == "sft":
            if data_domain == "general":
                preflight_stage(stage, depth, step)
            cmd = [python, "-u", "-m", "scripts.sft",
                   f"--depth={depth}",
                   f"--device-batch-size={device_batch_size}",
                   f"--memory-limit-gb={memory_limit_gb}",
                   f"--eval-every={eval_every}",
                   f"--model-name={checkpoint_model_name}"]
            if step > 0:
                cmd.append(f"--step={step}")
            if num_iterations > 0:
                cmd.append(f"--num-iterations={num_iterations}")
            effective_save_every = save_every if save_every > 0 else 500
            cmd.append(f"--save-every={effective_save_every}")
        elif stage == "import":
            preflight_stage(stage, depth, step)
            cmd = [python, "-u", "-m", "scripts.convert_from_hf",
                   f"--repo={repo}",
                   f"--memory-limit-gb={memory_limit_gb}"]
            if force_tokenizer:
                cmd.append("--force")
            if skip_verify:
                cmd.append("--skip-verify")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown stage: {stage}")
    except SetupError as exc:
        return sse_error_response(str(exc))

    def _low_priority():
        """Set subprocess to low CPU/IO priority so it doesn't starve other apps."""
        try:
            os.nice(10)  # lower priority (higher nice value)
        except OSError:
            pass

    running_process_claimed = True

    async def stream():
        global running_process, running_process_claimed
        last_metric_step = None
        try:
            yield f"data: {json.dumps({'type': 'output', 'text': f'Launching stage {stage}: ' + ' '.join(cmd)})}\n\n"
            running_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=stage_env,
                preexec_fn=_low_priority,
            )
            yield f"data: {json.dumps({'type': 'output', 'text': f'Stage {stage} process started, pid={running_process.pid}. Waiting for training logs...'})}\n\n"

            last_heartbeat = time.time()
            while True:
                try:
                    line_bytes = await asyncio.wait_for(running_process.stdout.readline(), timeout=5.0)
                except asyncio.TimeoutError:
                    if running_process.returncode is not None:
                        break
                    now = time.time()
                    if now - last_heartbeat >= 5.0:
                        yield f"data: {json.dumps({'type': 'output', 'text': f'Stage {stage} is still running; waiting for logs...'})}\n\n"
                        last_heartbeat = now
                    continue
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")

                # Try to parse training metrics
                m = METRIC_RE.search(line)
                if m:
                    step_value = int(m.group(1))
                    last_metric_step = step_value
                    total_value = int(m.group(2))
                    loss_value = float(m.group(3))
                    tok_per_sec_value = int(m.group(4).replace(',', ''))
                    yield f"data: {json.dumps({'type': 'metric', 'step': step_value, 'total': total_value, 'loss': loss_value, 'tok_per_sec': tok_per_sec_value})}\n\n"
                    yield f"data: {json.dumps({'type': 'activation', **build_layer_signal(depth, step_value, total_value, loss_value, tok_per_sec_value)})}\n\n"

                if stage == "train" and any(prefix in line for prefix in SAMPLE_PREFIXES):
                    prompt = next((prefix for prefix in SAMPLE_PREFIXES if prefix in line), "")
                    yield f"data: {json.dumps({'type': 'sample', 'step': last_metric_step, 'prompt': prompt, 'text': line})}\n\n"

                yield f"data: {json.dumps({'type': 'output', 'text': line})}\n\n"

            await running_process.wait()
            code = running_process.returncode
            if code == 0:
                yield f"data: {json.dumps({'type': 'done', 'code': 0})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'text': f'Process exited with code {code}', 'code': code})}\n\n"

        except asyncio.CancelledError:
            if running_process and running_process.returncode is None:
                running_process.terminate()
            yield f"data: {json.dumps({'type': 'error', 'text': 'Cancelled'})}\n\n"
        finally:
            running_process = None
            running_process_claimed = False

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/run_sync/{stage}")
async def run_stage_sync(stage: str, n_shards: int = 4, depth: int = 4,
                         step: int = -1,
                         num_iterations: int = -1, use_simple_adamw: bool = False,
                         window_pattern: str = "L", max_seq_len: int = 512,
                         device_batch_size: int = 1, save_every: int = -1,
                         eval_every: int = 100, sample_every: int = 10,
                         memory_limit_gb: float = 0,
                         data_domain: str = "general",
                         music_data_path: str = "",
                         data_source: str = "builtin",
                         custom_data_path: str = "",
                         tokenizer_method: str = "bpe",
                         vocab_size: int = 32768,
                         doc_cap: int = 10000,
                         repo: str = "nanochat-students/base-d20",
                         model_name: str = "",
                         force_tokenizer: bool = False,
                         skip_verify: bool = False):
    """Run short pipeline stages without SSE so browsers do not report aborted streams."""
    if stage not in {"data", "tokenizer"}:
        raise HTTPException(status_code=400, detail="run_sync only supports data and tokenizer stages")

    response = await run_stage(
        stage=stage, n_shards=n_shards, depth=depth, step=step,
        num_iterations=num_iterations, use_simple_adamw=use_simple_adamw,
        window_pattern=window_pattern, max_seq_len=max_seq_len,
        device_batch_size=device_batch_size, save_every=save_every,
        eval_every=eval_every, sample_every=sample_every,
        memory_limit_gb=memory_limit_gb,
        data_domain=data_domain, music_data_path=music_data_path,
        data_source=data_source, custom_data_path=custom_data_path,
        tokenizer_method=tokenizer_method, vocab_size=vocab_size, doc_cap=doc_cap,
        repo=repo, model_name=model_name, force_tokenizer=force_tokenizer, skip_verify=skip_verify,
    )
    if not isinstance(response, StreamingResponse):
        return response

    events = []
    status = "done"
    async for chunk in response.body_iterator:
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
        for block in text.split("\n\n"):
            data_lines = [
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            raw = "\n".join(data_lines)
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                event = {"type": "output", "text": raw}
            events.append(event)
            if event.get("type") == "error":
                status = "error"
    return {"status": status, "events": events}


@app.post("/stop")
async def stop():
    global running_process, running_process_claimed
    if running_process is None or running_process.returncode is not None:
        if running_process_claimed:
            running_process_claimed = False
            return {"status": "stopped_starting"}
        return {"status": "no_process"}
    running_process.terminate()
    try:
        await asyncio.wait_for(running_process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        running_process.kill()
    running_process = None
    running_process_claimed = False
    return {"status": "stopped"}


@app.get("/checkpoints")
async def list_checkpoints():
    results = []
    for data_domain, base in [("general", get_base_dir()), ("music", get_music_base_dir())]:
        ckpt_base = os.path.join(base, "mlx_checkpoints")
        if not os.path.isdir(ckpt_base):
            continue
        for d in sorted(os.listdir(ckpt_base)):
            if not d.startswith("d"):
                continue
            is_sft = d.endswith("_sft")
            depth_str = d[1:].replace("_sft", "") if is_sft else d[1:]
            if not depth_str.isdigit():
                continue
            depth = int(depth_str)
            source = "sft" if is_sft else "base"
            dpath = os.path.join(ckpt_base, d)
            for f in sorted(os.listdir(dpath)):
                if f.endswith("_meta.json"):
                    meta_path = os.path.join(dpath, f)
                    try:
                        with open(meta_path) as mf:
                            meta = json.load(mf)
                        mtime = os.path.getmtime(meta_path)
                        results.append({
                            "checkpoint_id": _encode_checkpoint_id(data_domain, depth, source, f),
                            "data_domain": data_domain,
                            "depth": depth,
                            "step": meta.get("step", 0),
                            "model_name": meta.get("model_name", ""),
                            "display_name": meta.get("model_name") or f"{data_domain}_d{depth}_{source}_step{meta.get('step', 0)}",
                            "n_embd": meta.get("n_embd", 0),
                            "n_head": meta.get("n_head", 0),
                            "sequence_len": meta.get("sequence_len", 0),
                            "window_pattern": meta.get("window_pattern", "L"),
                            "source": source,
                            "date": mtime,
                        })
                    except Exception:
                        pass
    results.sort(key=lambda item: item.get("date", 0), reverse=True)
    return results


class DeleteCheckpointRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    depth: int = 0
    step: int = 0
    source: str = "base"
    data_domain: str = "general"
    model_name: Optional[str] = None


@app.post("/checkpoints/delete")
async def delete_checkpoint(req: DeleteCheckpointRequest):
    """Delete one checkpoint step and its sidecar metadata/optimizer files."""
    global loaded_depth, loaded_step, loaded_source, loaded_domain, loaded_model_name

    data_domain = req.data_domain.lower()
    source = req.source.lower()
    resolved = resolve_checkpoint_id(req.checkpoint_id) if req.checkpoint_id else None
    if resolved:
        data_domain = resolved["data_domain"]
        source = resolved["source"]
        req.depth = resolved["depth"]
        req.step = resolved["step"]
        req.model_name = resolved["model_name"]
    if data_domain not in {"general", "music"}:
        raise HTTPException(status_code=400, detail="data_domain must be 'general' or 'music'")
    if source not in {"base", "sft"}:
        raise HTTPException(status_code=400, detail="source must be 'base' or 'sft'")
    if req.depth <= 0 or req.step <= 0:
        raise HTTPException(status_code=400, detail="depth and step must be positive")

    if (
        loaded_model is not None
        and loaded_domain == data_domain
        and loaded_depth == req.depth
        and loaded_step == req.step
        and loaded_source == source
        and loaded_model_name == req.model_name
    ):
        _unload_model()

    base = get_music_base_dir() if data_domain == "music" else get_base_dir()
    dirname = f"d{req.depth}_sft" if source == "sft" else f"d{req.depth}"
    ckpt_dir = os.path.realpath(os.path.join(base, "mlx_checkpoints", dirname))
    ckpt_root = os.path.realpath(os.path.join(base, "mlx_checkpoints"))
    if not ckpt_dir.startswith(ckpt_root + os.sep):
        raise HTTPException(status_code=400, detail="Invalid checkpoint path")
    if not os.path.isdir(ckpt_dir):
        raise HTTPException(status_code=404, detail="Checkpoint directory not found")

    if resolved:
        meta_path = resolved["meta_path"]
    else:
        matching_meta = []
        for filename in os.listdir(ckpt_dir):
            if not filename.endswith("_meta.json"):
                continue
            meta_path = os.path.join(ckpt_dir, filename)
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            if meta.get("step") != req.step:
                continue
            meta_model_name = meta.get("model_name", "")
            if req.model_name is None:
                if meta_model_name:
                    continue
            elif meta_model_name != req.model_name:
                continue
            matching_meta.append(meta_path)
        if not matching_meta:
            raise HTTPException(status_code=404, detail="Checkpoint files not found")
        meta_path = max(matching_meta, key=os.path.getmtime)
    stem = os.path.basename(meta_path).replace("_meta.json", "")
    candidates = [
        os.path.join(ckpt_dir, f"{stem}.safetensors"),
        meta_path,
        os.path.join(ckpt_dir, f"{stem}_optim.safetensors"),
    ]
    existing = [path for path in candidates if os.path.isfile(path)]
    if not existing:
        raise HTTPException(status_code=404, detail="Checkpoint files not found")

    deleted = []
    for path in existing:
        real_path = os.path.realpath(path)
        if not real_path.startswith(ckpt_dir + os.sep):
            raise HTTPException(status_code=400, detail="Invalid checkpoint file path")
        os.remove(real_path)
        deleted.append(real_path)

    return {
        "status": "deleted",
        "data_domain": data_domain,
        "depth": req.depth,
        "step": req.step,
        "source": source,
        "deleted": deleted,
    }


class LoadRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    depth: int = 12
    step: Optional[int] = None
    source: str = "base"
    data_domain: str = "general"
    model_name: Optional[str] = None


def _unload_model():
    """Free the currently loaded chat model and reclaim memory."""
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source, loaded_domain, loaded_model_name
    loaded_engine = None
    loaded_tokenizer = None
    loaded_model = None
    loaded_depth = None
    loaded_step = None
    loaded_source = None
    loaded_domain = None
    loaded_model_name = None
    gc.collect()


@app.post("/chat/load")
async def chat_load(req: LoadRequest):
    global loaded_engine, loaded_tokenizer, loaded_model, loaded_depth, loaded_step, loaded_source, loaded_domain, loaded_model_name

    if req.checkpoint_id:
        resolved = resolve_checkpoint_id(req.checkpoint_id)
        req.data_domain = resolved["data_domain"]
        req.depth = resolved["depth"]
        req.step = resolved["step"]
        req.source = resolved["source"]
        req.model_name = resolved["model_name"]

    # Free previous model first to avoid double memory usage
    if loaded_model is not None:
        _unload_model()

    from nanochat_mlx.common import set_memory_limit
    set_memory_limit(args.memory_limit_gb)

    from scripts.chat import load_model
    from nanochat_mlx.tokenizer import get_tokenizer
    from nanochat_mlx.engine import Engine

    try:
        old_base_dir = os.environ.get("NANOCHAT_BASE_DIR")
        if req.data_domain == "music":
            os.environ["NANOCHAT_BASE_DIR"] = get_music_base_dir()
        elif old_base_dir is not None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        try:
            model = load_model(depth=req.depth, step=req.step, source=req.source, model_name=req.model_name)
            tokenizer = get_tokenizer()
        finally:
            if old_base_dir is None:
                os.environ.pop("NANOCHAT_BASE_DIR", None)
            else:
                os.environ["NANOCHAT_BASE_DIR"] = old_base_dir
        loaded_engine = Engine(model, tokenizer)
        loaded_tokenizer = tokenizer
        loaded_model = model
        loaded_depth = req.depth
        loaded_step = req.step
        loaded_source = req.source
        loaded_domain = req.data_domain
        loaded_model_name = req.model_name
        return {"status": "loaded", "depth": req.depth, "source": req.source, "data_domain": req.data_domain, "model_name": req.model_name}
    except SetupError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/chat/unload")
async def chat_unload():
    """Unload the chat model to free memory."""
    if loaded_model is None:
        return {"status": "no_model"}
    _unload_model()
    return {"status": "unloaded"}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 256
    top_k: int = 50
    repetition_penalty: float = 1.0


@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    if loaded_engine is None or loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /chat/load first.")

    tokenizer = loaded_tokenizer
    engine = loaded_engine
    bos_id = tokenizer.get_bos_token_id()

    # Build conversation tokens
    try:
        user_start = tokenizer.encode_special("<|user_start|>")
        user_end = tokenizer.encode_special("<|user_end|>")
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        has_special = True
    except Exception:
        has_special = False

    tokens = [bos_id]
    if has_special:
        for msg in request.messages:
            if msg.role == "user":
                tokens.append(user_start)
                tokens.extend(tokenizer.encode(msg.content))
                tokens.append(user_end)
            elif msg.role == "assistant":
                tokens.append(assistant_start)
                tokens.extend(tokenizer.encode(msg.content))
                tokens.append(assistant_end)
        tokens.append(assistant_start)
    else:
        # Fallback: plain text
        for msg in request.messages:
            tokens.extend(tokenizer.encode(msg.content))

    async def stream():
        import random
        accumulated = []
        last_clean = ""
        for token_column, token_masks in engine.generate(
            tokens, num_samples=1,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            seed=random.randint(0, 2**31 - 1),
        ):
            tok = token_column[0]
            if has_special and (tok == assistant_end or tok == bos_id):
                break
            accumulated.append(tok)
            try:
                text = tokenizer.decode(accumulated)
            except Exception as exc:
                yield f"data: {json.dumps({'type': 'error', 'text': f'生成了无法解码的 token，已停止本次回复: {exc}'}, ensure_ascii=False)}\n\n"
                return
            if not text.endswith("\ufffd"):
                new = text[len(last_clean):]
                if new:
                    yield f"data: {json.dumps({'token': new}, ensure_ascii=False)}\n\n"
                    last_clean = text
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


def main(argv=None):
    global args
    args = build_parser().parse_args(argv)
    import uvicorn
    print(f"miniGPT Studio Quickstart → http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

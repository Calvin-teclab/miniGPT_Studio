"""
Extended miniGPT Studio training API.

Wraps the existing quickstart.py backend and adds:
- All original endpoints re-mounted under /api/
- /api/params — parameter descriptions for the frontend
- /api/eval/benchmark — SSE benchmark evaluation
- /api/eval/external — external model comparison

Usage:
    uvicorn scripts.training_api:app --port 8000
"""

import asyncio
import json
import os
import re
import sys
from typing import List, Optional

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Import shared state and helpers from quickstart
from scripts.quickstart import (
    LoadRequest,
    ChatRequest,
    app as _original_app,
    args,
    check_status,
    get_base_dir,
    get_music_base_dir,
    get_stage_env,
    list_checkpoints as _list_checkpoints,
    delete_checkpoint as _delete_checkpoint,
    DeleteCheckpointRequest,
    chat_load as _chat_load,
    chat_unload as _chat_unload,
    chat_completions as _chat_completions,
    run_stage as _run_stage,
    run_stage_sync as _run_stage_sync,
    stop as _stop,
    sse_error_response,
)

# ---------------------------------------------------------------------------
# App setup — we create a new app and include the original app's routes
# ---------------------------------------------------------------------------

app = FastAPI(title="miniGPT Studio Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all original routes (/, /status, /run/{stage}, etc.) so they still work
for route in _original_app.routes:
    app.routes.append(route)

# ---------------------------------------------------------------------------
# /api router — mirrors existing endpoints + adds new ones
# ---------------------------------------------------------------------------

api = APIRouter(prefix="/api")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_DATASET_DIR = os.path.join(PROJECT_ROOT, "sample_datasets")
EXPORT_DIR = os.path.join(PROJECT_ROOT, "exports")
SAMPLE_DATASET_META = {
    "text_general_raw.txt": {
        "id": "text_general_raw.txt",
        "label": "通用文本 TXT 原始数据",
        "data_domain": "general",
        "format": "txt",
    },
    "text_instruction_raw.jsonl": {
        "id": "text_instruction_raw.jsonl",
        "label": "指令文本 JSONL 原始数据",
        "data_domain": "general",
        "format": "jsonl",
    },
    "music_starter_chords_raw.jsonl": {
        "id": "music_starter_chords_raw.jsonl",
        "label": "音乐和弦 JSONL 原始数据",
        "data_domain": "music",
        "format": "jsonl",
    },
    "music_chords_raw.csv": {
        "id": "music_chords_raw.csv",
        "label": "音乐和弦 CSV 原始数据",
        "data_domain": "music",
        "format": "csv",
    },
    "music_chords_raw.txt": {
        "id": "music_chords_raw.txt",
        "label": "音乐和弦 TXT 原始数据",
        "data_domain": "music",
        "format": "txt",
    },
}


def _sample_dataset_path(dataset_id: str):
    if dataset_id not in SAMPLE_DATASET_META:
        raise HTTPException(status_code=404, detail="Unknown sample dataset")
    path = os.path.abspath(os.path.join(SAMPLE_DATASET_DIR, dataset_id))
    if not path.startswith(os.path.abspath(SAMPLE_DATASET_DIR) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid sample dataset path")
    return path


def _safe_filename(filename: str, default: str):
    basename = os.path.basename(filename or default)
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", basename).strip("._")
    return safe or default


@api.get("/status")
async def api_status():
    return check_status()


@api.get("/checkpoints")
async def api_checkpoints():
    return await _list_checkpoints()


@api.post("/checkpoints/delete")
async def api_delete_checkpoint(req: DeleteCheckpointRequest):
    return await _delete_checkpoint(req)


class ReportExportRequest(BaseModel):
    filename: str
    content: str
    content_type: str = "text/markdown"


@api.post("/reports/export")
async def api_export_report(req: ReportExportRequest):
    filename = _safe_filename(req.filename, "nanochat-report.md")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return {
        "filename": filename,
        "path": path,
        "bytes": len(req.content.encode("utf-8")),
        "download_url": f"/api/reports/download/{filename}",
    }


@api.get("/reports/download/{filename}")
async def api_download_report(filename: str):
    safe = _safe_filename(filename, "nanochat-report.md")
    path = os.path.join(EXPORT_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Report file not found")
    return FileResponse(
        path,
        media_type="text/markdown; charset=utf-8",
        filename=safe,
    )


@api.get("/datasets/samples")
async def api_list_sample_datasets():
    results = []
    for dataset_id, meta in SAMPLE_DATASET_META.items():
        path = _sample_dataset_path(dataset_id)
        results.append({
            **meta,
            "path": path,
            "exists": os.path.isfile(path),
            "bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
        })
    return results


@api.get("/datasets/samples/{dataset_id}")
async def api_get_sample_dataset(dataset_id: str):
    path = _sample_dataset_path(dataset_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Sample dataset file not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {**SAMPLE_DATASET_META[dataset_id], "path": path, "content": content}


class SampleDatasetSaveRequest(BaseModel):
    content: str


@api.put("/datasets/samples/{dataset_id}")
async def api_save_sample_dataset(dataset_id: str, req: SampleDatasetSaveRequest):
    path = _sample_dataset_path(dataset_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)
    return {
        **SAMPLE_DATASET_META[dataset_id],
        "path": path,
        "bytes": len(req.content.encode("utf-8")),
    }


class DatasetUploadRequest(BaseModel):
    filename: str
    content: str
    data_domain: str = "general"


@api.post("/datasets/upload")
async def api_upload_dataset(req: DatasetUploadRequest):
    data_domain = req.data_domain.lower()
    if data_domain not in {"general", "music"}:
        raise HTTPException(status_code=400, detail="data_domain must be 'general' or 'music'")

    suffix = os.path.splitext(req.filename)[1].lower()
    if suffix not in {".jsonl", ".csv", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .jsonl, .csv, .txt, and .md datasets are supported")

    safe_stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", os.path.splitext(req.filename)[0]).strip("._")
    if not safe_stem:
        safe_stem = "dataset"

    base_dir = get_music_base_dir() if data_domain == "music" else get_base_dir()
    upload_dir = os.path.join(base_dir, "uploaded_datasets")
    os.makedirs(upload_dir, exist_ok=True)
    path = os.path.join(upload_dir, f"{safe_stem}{suffix}")
    with open(path, "w", encoding="utf-8") as f:
        f.write(req.content)

    return {"path": path, "filename": os.path.basename(path), "bytes": len(req.content.encode("utf-8"))}


@api.get("/run/{stage}")
async def api_run_stage(stage: str, n_shards: int = 4, depth: int = 4,
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
    return await _run_stage(
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


@api.get("/run_sync/{stage}")
async def api_run_stage_sync(stage: str, n_shards: int = 4, depth: int = 4,
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
    return await _run_stage_sync(
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


@api.post("/stop")
async def api_stop():
    return await _stop()


@api.post("/chat/load")
async def api_chat_load(req: LoadRequest):
    return await _chat_load(req)


@api.post("/chat/unload")
async def api_chat_unload():
    return await _chat_unload()


@api.post("/chat/completions")
async def api_chat_completions(request: ChatRequest):
    return await _chat_completions(request)


@api.post("/chat/generate")
async def api_chat_generate(request: ChatRequest):
    """Generate a complete chat response as JSON to avoid browser abort noise."""
    import random
    import scripts.quickstart as qs

    if qs.loaded_engine is None or qs.loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /api/chat/load first.")

    tokenizer = qs.loaded_tokenizer
    engine = qs.loaded_engine
    bos_id = tokenizer.get_bos_token_id()

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
        for msg in request.messages:
            tokens.extend(tokenizer.encode(msg.content))

    accumulated = []
    last_clean = ""
    for token_column, _token_masks in engine.generate(
        tokens,
        num_samples=1,
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
            raise HTTPException(
                status_code=500,
                detail=f"生成了无法解码的 token，已停止本次回复: {exc}",
            ) from exc
        if not text.endswith("\ufffd"):
            last_clean = text

    return {"text": last_clean}


@api.post("/music/generate")
async def api_music_generate(request: ChatRequest):
    """Generate a complete music response as JSON to avoid browser SSE abort noise."""
    import random
    import scripts.quickstart as qs

    if qs.loaded_engine is None or qs.loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /api/chat/load first.")

    tokenizer = qs.loaded_tokenizer
    engine = qs.loaded_engine
    bos_id = tokenizer.get_bos_token_id()

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
        for msg in request.messages:
            tokens.extend(tokenizer.encode(msg.content))

    accumulated = []
    last_clean = ""
    for token_column, _token_masks in engine.generate(
        tokens,
        num_samples=1,
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
            raise HTTPException(
                status_code=500,
                detail=f"生成了无法解码的 token，已停止本次回复: {exc}",
            ) from exc
        if not text.endswith("\ufffd"):
            last_clean = text

    return {"text": last_clean}


@api.get("/chat/completions")
async def api_chat_completions_get(
    messages: str = "[]",
    temperature: float = 0.8,
    max_tokens: int = 256,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
):
    try:
        parsed_messages = json.loads(messages)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid messages JSON") from exc

    request = ChatRequest(
        messages=parsed_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    return await _chat_completions(request)


# ---------------------------------------------------------------------------
# NEW: /api/params — parameter descriptions for the frontend
# ---------------------------------------------------------------------------

PARAM_DEFINITIONS = [
    {
        "key": "depth",
        "label": "Model Depth",
        "description": "Controls overall model scale. Each depth unit adds a transformer layer. Higher = more capable but slower.",
        "type": "slider",
        "default": 4,
        "min": 1,
        "max": 26,
        "group": "model",
    },
    {
        "key": "max_seq_len",
        "label": "Max Sequence Length",
        "description": "Maximum context length in tokens the model can process.",
        "type": "slider",
        "default": 512,
        "min": 128,
        "max": 2048,
        "step": 128,
        "group": "model",
    },
    {
        "key": "window_pattern",
        "label": "Window Pattern",
        "description": "Attention window pattern. L=full attention, S=half window. e.g. 'SSSL' alternates short and long attention.",
        "type": "text",
        "default": "L",
        "group": "model",
    },
    {
        "key": "num_iterations",
        "label": "Training Steps",
        "description": "Number of training iterations. -1 for automatic based on data/param ratio.",
        "type": "number",
        "default": -1,
        "min": -1,
        "max": 100000,
        "group": "training",
    },
    {
        "key": "device_batch_size",
        "label": "Device Batch Size",
        "description": "Per-device batch size. Reduce if running out of memory.",
        "type": "slider",
        "default": 1,
        "min": 1,
        "max": 16,
        "group": "training",
    },
    {
        "key": "save_every",
        "label": "Save Every N Steps",
        "description": "Save a checkpoint every N training steps. -1 saves only at end.",
        "type": "number",
        "default": 500,
        "min": -1,
        "max": 10000,
        "group": "training",
    },
    {
        "key": "eval_every",
        "label": "Eval Every N Steps",
        "description": "Evaluate validation loss every N steps. -1 to disable.",
        "type": "number",
        "default": 100,
        "min": -1,
        "max": 10000,
        "group": "training",
    },
    {
        "key": "memory_limit_gb",
        "label": "Memory Limit (GB)",
        "description": "MLX memory limit in GB. Keep conservative for shared use.",
        "type": "slider",
        "default": 8.0,
        "min": 2.0,
        "max": 64.0,
        "step": 1.0,
        "group": "system",
    },
    {
        "key": "use_simple_adamw",
        "label": "Use Simple AdamW",
        "description": "Use plain AdamW optimizer instead of Muon+AdamW hybrid.",
        "type": "boolean",
        "default": False,
        "group": "training",
    },
    {
        "key": "n_shards",
        "label": "Data Shards",
        "description": "Number of data shards to download for training.",
        "type": "slider",
        "default": 4,
        "min": 2,
        "max": 32,
        "group": "data",
    },
]


@api.get("/params")
async def api_params():
    return {"params": PARAM_DEFINITIONS}


# ---------------------------------------------------------------------------
# NEW: /api/eval/benchmark — SSE benchmark evaluation
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "arc_easy": {"name": "ARC-Easy", "task_name": "ARC-Easy"},
    "arc_challenge": {"name": "ARC-Challenge", "task_name": "ARC-Challenge"},
    "mmlu": {"name": "MMLU", "task_name": "MMLU"},
    "gsm8k": {"name": "GSM8K", "task_name": "GSM8K"},
    "humaneval": {"name": "HumanEval", "task_name": "HumanEval"},
    "spellingbee": {"name": "SpellingBee", "task_name": "SpellingBee"},
}


@api.get("/eval/benchmark")
async def api_eval_benchmark(checkpoint: str = "", benchmarks: str = "arc_easy"):
    """
    SSE endpoint that runs benchmark evaluations on a checkpoint.
    Query params:
      - checkpoint: path or "depth:step:source" descriptor
      - benchmarks: comma-separated benchmark ids (arc_easy,mmlu,gsm8k,...)
    """
    benchmark_ids = [b.strip() for b in benchmarks.split(",") if b.strip()]

    # Validate benchmark ids
    for bid in benchmark_ids:
        if bid not in BENCHMARK_REGISTRY:
            return sse_error_response(f"Unknown benchmark: {bid}. Available: {', '.join(BENCHMARK_REGISTRY.keys())}")

    async def stream():
        try:
            yield f"data: {json.dumps({'type': 'log', 'text': 'Starting benchmark evaluation...'})}\n\n"

            # Parse checkpoint descriptor (format: "depth:step:source" or just use loaded model)
            data_domain, depth, step, source = "general", 4, None, "sft"
            if checkpoint:
                parts = checkpoint.split(":")
                offset = 1 if parts and not parts[0].isdigit() else 0
                if offset:
                    data_domain = parts[0]
                if len(parts) > offset and parts[offset].isdigit():
                    depth = int(parts[offset])
                if len(parts) > offset + 1 and parts[offset + 1] and parts[offset + 1] != "latest":
                    step = int(parts[offset + 1])
                if len(parts) > offset + 2:
                    source = parts[offset + 2]

            yield f"data: {json.dumps({'type': 'log', 'text': f'Loading model domain={data_domain} depth={depth} step={step} source={source}...'})}\n\n"

            # Run evaluation as subprocess to avoid blocking
            python = sys.executable

            for bid in benchmark_ids:
                info = BENCHMARK_REGISTRY[bid]
                task_name = info["task_name"]
                display_name = info["name"]

                yield f"data: {json.dumps({'type': 'log', 'text': f'Running {display_name}...'})}\n\n"

                # Build command
                cmd = [
                    python, "-m", "scripts.chat_eval",
                    f"--depth={depth}",
                    f"--source={source}",
                    f"--task-name={task_name}",
                    "--max-problems=100",
                    "--batch-size=8",
                ]
                if step is not None:
                    cmd.append(f"--step={step}")

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=get_stage_env(data_domain),
                )

                output_lines = []
                async for line_bytes in proc.stdout:
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                    output_lines.append(line)
                    yield f"data: {json.dumps({'type': 'log', 'text': line})}\n\n"

                await proc.wait()

                if proc.returncode != 0:
                    yield f"data: {json.dumps({'type': 'log', 'text': f'ERROR: {display_name} evaluation failed (exit code {proc.returncode})'})}\n\n"
                    continue

                # Parse accuracy from output — look for "Final: X/Y (Z%)" pattern
                import re
                score, total, accuracy = 0, 0, 0.0
                for line in output_lines:
                    m = re.search(r"Final:\s*(\d+)/(\d+)\s*\(([\d.]+)%\)", line)
                    if m:
                        score = int(m.group(1))
                        total = int(m.group(2))
                        accuracy = float(m.group(3)) / 100.0
                        break

                result_data = {
                    "type": "result",
                    "result": {
                        "benchmark": display_name,
                        "score": score,
                        "total": total,
                        "accuracy": accuracy,
                    },
                }
                yield f"data: {json.dumps(result_data)}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': str(exc)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# NEW: /api/eval/external — external model comparison
# ---------------------------------------------------------------------------

class ExternalEvalConfig(BaseModel):
    provider: str = "openai"
    apiKey: str = ""
    endpoint: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"


class ExternalEvalRequest(BaseModel):
    checkpoint: str = ""
    config: ExternalEvalConfig
    prompts: List[str]


def _parse_checkpoint_descriptor(checkpoint: str):
    data_domain, depth, step, source = "general", 4, None, "sft"
    if checkpoint:
        parts = checkpoint.split(":")
        offset = 1 if parts and not parts[0].isdigit() else 0
        if offset:
            data_domain = parts[0]
        if len(parts) > offset and parts[offset].isdigit():
            depth = int(parts[offset])
        if len(parts) > offset + 1 and parts[offset + 1] and parts[offset + 1] != "latest":
            step = int(parts[offset + 1])
        if len(parts) > offset + 2 and parts[offset + 2]:
            source = parts[offset + 2]
    return data_domain, depth, step, source


@api.post("/eval/external")
async def api_eval_external(request: ExternalEvalRequest):
    """
    Compare local model against an external OpenAI-compatible model.
    For each prompt:
      1. Generate local model response
      2. Get external model response
      3. Ask external model to score local response (1-10)
    """
    import scripts.quickstart as qs

    data_domain, depth, step, source = _parse_checkpoint_descriptor(request.checkpoint)
    needs_load = (
        qs.loaded_engine is None
        or qs.loaded_tokenizer is None
        or qs.loaded_depth != depth
        or qs.loaded_step != step
        or qs.loaded_source != source
        or qs.loaded_domain != data_domain
    )
    if needs_load:
        await _chat_load(LoadRequest(data_domain=data_domain, depth=depth, step=step, source=source))

    if qs.loaded_engine is None or qs.loaded_tokenizer is None:
        raise HTTPException(status_code=400, detail="No model loaded. POST /api/chat/load first.")

    engine = qs.loaded_engine
    tokenizer = qs.loaded_tokenizer

    # Helper to generate from local model
    def _generate_local(prompt_text: str) -> str:
        import hashlib
        bos_id = tokenizer.get_bos_token_id()

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
            tokens.append(user_start)
            tokens.extend(tokenizer.encode(prompt_text))
            tokens.append(user_end)
            tokens.append(assistant_start)
        else:
            tokens.extend(tokenizer.encode(prompt_text))

        seed = int(hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8], 16)
        accumulated = []
        for token_column, _ in engine.generate(
            tokens, num_samples=1,
            max_tokens=256,
            temperature=0.0,
            top_k=1,
            seed=seed,
        ):
            tok = token_column[0]
            if has_special and (tok == assistant_end or tok == bos_id):
                break
            accumulated.append(tok)

        return tokenizer.decode(accumulated)

    # Helper to call external API
    async def _call_external(messages: list) -> str:
        import httpx
        endpoint = request.config.endpoint
        if not endpoint:
            if request.config.provider == "openai":
                endpoint = "https://api.openai.com/v1"
            else:
                raise HTTPException(status_code=400, detail="Custom/Anthropic endpoint is required")
        url = endpoint.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {request.config.apiKey}",
        }
        body = {
            "model": request.config.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    results = []
    for prompt in request.prompts:
        try:
            # 1. Generate local response
            local_response = _generate_local(prompt)

            # 2. Get external model response
            external_response = await _call_external([
                {"role": "user", "content": prompt}
            ])

            # 3. Ask external model to score local response
            scoring_prompt = (
                f"You are evaluating an AI assistant's response.\n\n"
                f"User prompt: {prompt}\n\n"
                f"Assistant response: {local_response}\n\n"
                f"Score this response from 1 to 10 (10=excellent, 1=terrible). "
                f"Consider accuracy, helpfulness, and coherence.\n"
                f"Reply in this exact JSON format: {{\"score\": <number>, \"feedback\": \"<brief feedback>\"}}"
            )
            scoring_raw = await _call_external([
                {
                    "role": "system",
                    "content": "You are a strict evaluator. Return only valid JSON and do not include markdown.",
                },
                {"role": "user", "content": scoring_prompt}
            ])

            # Parse score
            score = 5
            feedback = scoring_raw
            try:
                parsed = json.loads(scoring_raw)
                score = max(1, min(10, int(parsed.get("score", 5))))
                feedback = parsed.get("feedback", scoring_raw)
            except (json.JSONDecodeError, ValueError):
                # Try to extract score from text
                import re
                m = re.search(r'"score"\s*:\s*(\d+)', scoring_raw)
                if m:
                    score = max(1, min(10, int(m.group(1))))

            results.append({
                "prompt": prompt,
                "localResponse": local_response,
                "externalResponse": external_response,
                "externalScore": score,
                "externalFeedback": feedback,
                "local_response": local_response,
                "external_response": external_response,
                "score": score,
                "feedback": feedback,
                "evalConfig": {
                    "local_temperature": 0.0,
                    "local_top_k": 1,
                    "external_temperature": 0.0,
                    "seed_strategy": "sha256(prompt)",
                },
            })
        except Exception as exc:
            results.append({
                "prompt": prompt,
                "localResponse": "",
                "externalResponse": "",
                "externalScore": 0,
                "externalFeedback": f"Error: {str(exc)}",
                "local_response": "",
                "external_response": "",
                "score": 0,
                "feedback": f"Error: {str(exc)}",
                "error": True,
            })

    return {"results": results}


# ---------------------------------------------------------------------------
# Include the API router
# ---------------------------------------------------------------------------

app.include_router(api)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="miniGPT Studio Training API")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--memory-limit-gb", type=float, default=8.0)
    parsed = parser.parse_args(argv)

    # Update the shared args from quickstart
    args.port = parsed.port
    args.host = parsed.host
    args.memory_limit_gb = parsed.memory_limit_gb

    print(f"miniGPT Studio Training API → http://{parsed.host}:{parsed.port}")
    uvicorn.run(app, host=parsed.host, port=parsed.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import type {
  SystemStatus,
  Checkpoint,
  ChatMessage,
  ChatSettings,
  ExternalEvalConfig,
} from '@/types';

const BASE = '/api';

type JsonRecord = Record<string, unknown>;

interface StatusDomain {
  data?: unknown;
  tokenizer?: unknown;
  train?: JsonRecord;
  sft?: JsonRecord;
}

interface StatusResponse extends StatusDomain {
  music?: StatusDomain;
  chat?: unknown;
  running?: unknown;
}

interface RawCheckpoint {
  checkpoint_id?: string;
  data_domain?: string;
  depth?: number;
  step?: number;
  source?: string;
  n_embd?: number;
  window_pattern?: string;
  model_name?: string;
  display_name?: string;
  date?: number;
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === 'object' && value !== null;
}

export interface SampleDataset {
  id: string;
  label: string;
  data_domain: 'general' | 'music';
  format: string;
  path: string;
  exists?: boolean;
  bytes?: number;
  content?: string;
}

async function parseJsonOrThrow<T>(res: Response): Promise<T> {
  const data: unknown = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = isRecord(data) && typeof data.detail === 'string' ? data.detail : res.statusText;
    throw new Error(detail || `Request failed with status ${res.status}`);
  }
  return data as T;
}

// Helper for SSE connections
export function createSSE(url: string, onMessage: (data: unknown) => void, onError?: (err: Event) => void): EventSource {
  const es = new EventSource(`${BASE}${url}`);
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      onMessage(data);
    } catch {
      onMessage(e.data);
    }
  };
  if (onError) es.onerror = onError;
  return es;
}

// System status — maps backend format to frontend SystemStatus
export async function getStatus(): Promise<SystemStatus> {
  const res = await fetch(`${BASE}/status`);
  const raw = await parseJsonOrThrow<StatusResponse>(res);
  let selected: StatusDomain = raw;
  try {
    const config = JSON.parse(localStorage.getItem('nanochat_config') || '{}') as JsonRecord;
    if (config.data_domain === 'music' && raw.music) {
      selected = raw.music;
    }
  } catch {
    selected = raw;
  }
  // Backend returns: {data, data_shards, tokenizer, train: {depth: step}, sft: {depth: step}, chat, chat_model, running}
  return {
    dataReady: !!selected.data,
    tokenizerReady: !!selected.tokenizer,
    modelReady: !!selected.train && Object.keys(selected.train).length > 0,
    sftReady: !!selected.sft && Object.keys(selected.sft).length > 0,
    chatLoaded: !!raw.chat,
    running: !!raw.running,
    currentStage: raw.running ? 'unknown' : undefined,
  };
}

// Get checkpoints — maps backend format to frontend Checkpoint type
function checkpointPreference(checkpoint: Checkpoint) {
  // For one named training run, prefer the finetuned model over its base checkpoint.
  return checkpoint.isSft ? 2 : 1;
}

export async function getCheckpoints(): Promise<Checkpoint[]> {
  const res = await fetch(`${BASE}/checkpoints`);
  const raw = await parseJsonOrThrow<RawCheckpoint[]>(res);
  const checkpoints = raw.map((cp) => {
    const fallbackName = `${cp.data_domain === 'music' ? 'music_' : ''}d${cp.depth}_${cp.source}_step${cp.step}`;
    const displayName = cp.display_name || cp.model_name || fallbackName;
    const legacyDescriptor = [
      cp.data_domain || 'general',
      cp.depth || 0,
      cp.step || 0,
      cp.source || 'base',
      encodeURIComponent(cp.model_name || ''),
    ].join(':');
    return {
      path: cp.checkpoint_id || legacyDescriptor,
      checkpointId: cp.checkpoint_id,
      name: displayName,
      modelName: cp.model_name,
      displayName,
      depth: cp.depth || 0,
      step: cp.step || 0,
      nEmbd: cp.n_embd || 0,
      isSft: cp.source === 'sft',
      windowStrategy: cp.window_pattern,
      dataDomain: cp.data_domain || 'general',
      source: cp.source,
      date: cp.date,
    };
  });
  const latestByNamedModel = new Map<string, Checkpoint>();
  const unnamed: Checkpoint[] = [];

  for (const checkpoint of checkpoints) {
    if (!checkpoint.modelName) {
      unnamed.push(checkpoint);
      continue;
    }
    const key = [
      checkpoint.dataDomain || 'general',
      checkpoint.depth,
      checkpoint.modelName,
    ].join(':');
    const current = latestByNamedModel.get(key);
    const currentPreference = current ? checkpointPreference(current) : 0;
    const nextPreference = checkpointPreference(checkpoint);
    if (
      !current ||
      nextPreference > currentPreference ||
      (nextPreference === currentPreference && (
        checkpoint.step > current.step ||
        (checkpoint.step === current.step && (checkpoint.date || 0) > (current.date || 0))
      ))
    ) {
      latestByNamedModel.set(key, checkpoint);
    }
  }

  return [...latestByNamedModel.values(), ...unnamed].sort((a, b) => (b.date || 0) - (a.date || 0));
}

// Run a pipeline stage with SSE - returns EventSource for streaming
export function runStage(
  stage: string,
  params?: Record<string, unknown>
): EventSource {
  const query = params ? '?' + new URLSearchParams(
    Object.entries(params).reduce((acc, [k, v]) => {
      if (v !== undefined && v !== null && v !== '') acc[k] = String(v);
      return acc;
    }, {} as Record<string, string>)
  ).toString() : '';
  
  return createSSE(`/run/${stage}${query}`, () => {});
}

// Stop current running process
export async function stopProcess(): Promise<void> {
  await fetch(`${BASE}/stop`, { method: 'POST' });
}

// Chat - load model (backend expects {depth, step, source})
export async function loadModel(checkpointDescriptor: string): Promise<{ status: string }> {
  if (checkpointDescriptor.startsWith('cp_')) {
    const res = await fetch(`${BASE}/chat/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint_id: checkpointDescriptor }),
    });
    return parseJsonOrThrow<{ status: string }>(res);
  }
  // Descriptor format: "data_domain:depth:step:source" (old "depth:step:source" is also supported)
  const parts = checkpointDescriptor.split(':');
  const hasDomain = Number.isNaN(parseInt(parts[0]));
  const dataDomain = hasDomain ? parts[0] : 'general';
  const offset = hasDomain ? 1 : 0;
  const depth = parseInt(parts[offset]) || 4;
  const step = parts[offset + 1] ? parseInt(parts[offset + 1]) : undefined;
  const source = parts[offset + 2] || 'base';
  const modelName = parts.length > offset + 3 ? decodeURIComponent(parts[offset + 3] || '') : undefined;

  const res = await fetch(`${BASE}/chat/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ data_domain: dataDomain, depth, step: step || null, source, model_name: modelName ?? null }),
  });
  return parseJsonOrThrow<{ status: string }>(res);
}

function parseCheckpointDescriptor(checkpointDescriptor: string) {
  const parts = checkpointDescriptor.split(':');
  const hasDomain = Number.isNaN(parseInt(parts[0]));
  const dataDomain = hasDomain ? parts[0] : 'general';
  const offset = hasDomain ? 1 : 0;
  return {
    data_domain: dataDomain,
    depth: parseInt(parts[offset]) || 4,
    step: parts[offset + 1] ? parseInt(parts[offset + 1]) : 0,
    source: parts[offset + 2] || 'base',
    model_name: parts.length > offset + 3 ? decodeURIComponent(parts[offset + 3] || '') : null,
  };
}

export async function deleteCheckpoint(checkpointDescriptor: string): Promise<{ status: string; deleted: string[] }> {
  const body = checkpointDescriptor.startsWith('cp_')
    ? { checkpoint_id: checkpointDescriptor }
    : parseCheckpointDescriptor(checkpointDescriptor);
  const res = await fetch(`${BASE}/checkpoints/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return parseJsonOrThrow<{ status: string; deleted: string[] }>(res);
}

// Chat - unload model
export async function unloadModel(): Promise<void> {
  const res = await fetch(`${BASE}/chat/unload`, { method: 'POST' });
  await parseJsonOrThrow(res);
}

// Chat - send message (SSE streaming)
export function chatCompletions(
  messages: ChatMessage[],
  settings: ChatSettings
): EventSource {
  const params = new URLSearchParams({
    messages: JSON.stringify(messages),
    temperature: String(settings.temperature),
    top_k: String(settings.topK),
    repetition_penalty: String(settings.repetitionPenalty),
    max_tokens: String(settings.maxTokens),
  });
  return createSSE(`/chat/completions?${params.toString()}`, () => {});
}

// Get training parameter descriptions
export async function getParamDescriptions(): Promise<unknown> {
  const res = await fetch(`${BASE}/params`);
  return parseJsonOrThrow<unknown>(res);
}

export async function uploadDatasetFile(
  file: File,
  dataDomain: 'general' | 'music'
): Promise<{ path: string; filename: string; bytes: number }> {
  const content = await file.text();
  const res = await fetch(`${BASE}/datasets/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename: file.name,
      content,
      data_domain: dataDomain,
    }),
  });
  return parseJsonOrThrow<{ path: string; filename: string; bytes: number }>(res);
}

export async function uploadDatasetContent(
  filename: string,
  content: string,
  dataDomain: 'general' | 'music'
): Promise<{ path: string; filename: string; bytes: number }> {
  const res = await fetch(`${BASE}/datasets/upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename,
      content,
      data_domain: dataDomain,
    }),
  });
  return parseJsonOrThrow<{ path: string; filename: string; bytes: number }>(res);
}

export async function getSampleDatasets(): Promise<SampleDataset[]> {
  const res = await fetch(`${BASE}/datasets/samples`);
  return parseJsonOrThrow<SampleDataset[]>(res);
}

export async function getSampleDataset(datasetId: string): Promise<SampleDataset> {
  const res = await fetch(`${BASE}/datasets/samples/${encodeURIComponent(datasetId)}`);
  return parseJsonOrThrow<SampleDataset>(res);
}

export async function saveSampleDataset(
  datasetId: string,
  content: string
): Promise<SampleDataset> {
  const res = await fetch(`${BASE}/datasets/samples/${encodeURIComponent(datasetId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  return parseJsonOrThrow<SampleDataset>(res);
}

export async function exportReportFile(
  filename: string,
  content: string,
  contentType = 'text/markdown'
): Promise<{ filename: string; path: string; bytes: number; download_url: string }> {
  const res = await fetch(`${BASE}/reports/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      filename,
      content,
      content_type: contentType,
    }),
  });
  return parseJsonOrThrow<{ filename: string; path: string; bytes: number; download_url: string }>(res);
}

// Run benchmark evaluation
export async function runBenchmark(
  checkpoint: string,
  benchmarks: string[]
): Promise<EventSource> {
  const params = new URLSearchParams({
    checkpoint,
    benchmarks: benchmarks.join(','),
  });
  return createSSE(`/eval/benchmark?${params.toString()}`, () => {});
}

// Run external model evaluation
export async function runExternalEval(
  checkpoint: string,
  config: ExternalEvalConfig,
  prompts: string[]
): Promise<unknown> {
  const res = await fetch(`${BASE}/eval/external`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ checkpoint, config, prompts }),
  });
  return parseJsonOrThrow<unknown>(res);
}

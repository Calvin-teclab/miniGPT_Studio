import { useState, useEffect, useRef, useCallback, useSyncExternalStore, type SetStateAction } from 'react';
import { useLocation } from 'react-router-dom';
import {
  Activity,
  Play,
  Square,
  Terminal,
  Layers,
  MessageSquareText,
  TrendingDown,
  Zap,
  Database,
  Cpu,
  Brain,
  Repeat2,
  Save,
  CheckCircle2,
  Download,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import MetricCard from '@/components/MetricCard';
import { cn } from '@/lib/utils';
import type { LossPoint, TrainingSample, LayerActivation, TrainingMetrics } from '@/types';

// Regex to parse training metrics from log lines (matching backend format)
const METRIC_RE = /step\s+(\d+)\/(\d+)\s+\|\s+loss\s+([\d.]+)\s+\|\s+lr\s+([\d.e-]+)\s+\|\s+([\d.]+)\s+tok\/s\s+\|\s+ETA\s+(.+)/;
const VAL_RE = /val loss\s+([\d.]+)\s+\|\s+bpb\s+([\d.]+)/;
const SAMPLE_RE = /--- sample ---\n([\s\S]*?)--- end sample ---/;
const ACTIVATION_RE = /\[ACTIVATIONS\]\s+(.+)/;
const RUN_PARAM_KEYS = new Set([
  'depth',
  'num_iterations',
  'use_simple_adamw',
  'window_pattern',
  'max_seq_len',
  'device_batch_size',
  'save_every',
  'eval_every',
  'sample_every',
  'memory_limit_gb',
  'data_domain',
  'music_data_path',
  'model_name',
]);

const DEFAULT_TRAINING_PARAMS: Record<string, string> = {
  depth: '4',
  num_iterations: '500',
  max_seq_len: '512',
  device_batch_size: '1',
  window_pattern: 'L',
  eval_every: '100',
  sample_every: '10',
  save_every: '500',
  memory_limit_gb: '8',
  data_domain: 'general',
  music_data_path: '',
  model_name: '',
};
const TRAINING_STATE_KEY = 'minigpt_training_monitor_state';

type PhaseId = 'prepare' | 'data' | 'forward' | 'loss' | 'backward' | 'update' | 'eval' | 'sample' | 'save' | 'done';

interface TrainingMonitorSnapshot {
  isRunning: boolean;
  stage: 'train' | 'sft';
  metrics: TrainingMetrics | null;
  lossHistory: LossPoint[];
  samples: TrainingSample[];
  activations: LayerActivation | null;
  logs: string[];
  currentPhase: PhaseId;
  phaseEvents: { phase: PhaseId; text: string; time: string }[];
  errorMessage: string;
  modelName: string;
  updatedAt: number;
}

const phaseDetails: Record<PhaseId, { label: string; description: string }> = {
  prepare: {
    label: '准备模型与优化器',
    description: '加载 tokenizer、构建 GPT 结构、统计参数量、设置优化器。',
  },
  data: {
    label: '读取训练批次',
    description: '从当前数据域的 parquet 分片中取出 token 序列，组成当前训练 batch。',
  },
  forward: {
    label: '前向传播',
    description: '模型根据当前参数预测每个位置的下一个 token。',
  },
  loss: {
    label: '计算 Loss',
    description: '比较模型预测和真实答案的差距，loss 越低说明预测越接近。',
  },
  backward: {
    label: '反向传播',
    description: '根据 loss 计算每个参数应该往哪个方向调整。',
  },
  update: {
    label: '更新参数',
    description: '优化器把梯度应用到模型权重上，模型完成一次学习。',
  },
  eval: {
    label: '验证评估',
    description: '用验证集检查模型是否真的学会，而不是只记住训练数据。',
  },
  sample: {
    label: '生成样本',
    description: '用当前模型生成文本，直观看到模型能力变化。',
  },
  save: {
    label: '保存检查点',
    description: '把当前模型权重保存到本地，后续可继续训练、评测或对话。',
  },
  done: {
    label: '训练完成',
    description: '本轮训练结束，模型 checkpoint 已准备好用于评测或对话。',
  },
};

const stepLoop = [
  { id: 'data' as PhaseId, icon: Database },
  { id: 'forward' as PhaseId, icon: Brain },
  { id: 'loss' as PhaseId, icon: TrendingDown },
  { id: 'backward' as PhaseId, icon: Repeat2 },
  { id: 'update' as PhaseId, icon: Cpu },
];

function downloadTextFile(filename: string, content: string, type = 'text/markdown') {
  const blob = new Blob([content], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function getSavedModelName() {
  try {
    const saved = JSON.parse(localStorage.getItem('nanochat_config') || '{}') as Record<string, unknown>;
    return typeof saved.model_name === 'string' ? saved.model_name.trim() : '';
  } catch {
    return '';
  }
}

function createDefaultModelName(stage: 'train' | 'sft', dataDomain = 'general') {
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
  ].join('');
  return `${dataDomain}-${stage}-${stamp}`;
}

function loadTrainingSnapshot(): TrainingMonitorSnapshot | null {
  try {
    const raw = sessionStorage.getItem(TRAINING_STATE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as Partial<TrainingMonitorSnapshot>;
    if (data.stage !== 'train' && data.stage !== 'sft') return null;
    return {
      isRunning: !!data.isRunning,
      stage: data.stage,
      metrics: data.metrics || null,
      lossHistory: Array.isArray(data.lossHistory) ? data.lossHistory : [],
      samples: Array.isArray(data.samples) ? data.samples : [],
      activations: data.activations || null,
      logs: Array.isArray(data.logs) ? data.logs : [],
      currentPhase: data.currentPhase || 'prepare',
      phaseEvents: Array.isArray(data.phaseEvents) ? data.phaseEvents : [],
      errorMessage: data.errorMessage || '',
      modelName: data.modelName || getSavedModelName(),
      updatedAt: typeof data.updatedAt === 'number' ? data.updatedAt : Date.now(),
    };
  } catch {
    return null;
  }
}

function createEmptyTrainingSnapshot(stage: 'train' | 'sft' = 'train'): TrainingMonitorSnapshot {
  return {
    isRunning: false,
    stage,
    metrics: null,
    lossHistory: [],
    samples: [],
    activations: null,
    logs: [],
    currentPhase: 'prepare',
    phaseEvents: [],
    errorMessage: '',
    modelName: getSavedModelName() || createDefaultModelName(stage),
    updatedAt: Date.now(),
  };
}

let trainingMonitorSnapshot: TrainingMonitorSnapshot | null = null;
const trainingMonitorListeners = new Set<() => void>();
let activeTrainingConnection: { close: () => void } | null = null;
let waitingLogTimer: number | null = null;
let localHeartbeatTimer: number | null = null;

function getTrainingMonitorSnapshot(preferredStage?: 'train' | 'sft') {
  if (!trainingMonitorSnapshot) {
    trainingMonitorSnapshot = loadTrainingSnapshot() || createEmptyTrainingSnapshot(preferredStage || 'train');
    if (preferredStage && !trainingMonitorSnapshot.isRunning && trainingMonitorSnapshot.stage !== preferredStage) {
      trainingMonitorSnapshot = { ...trainingMonitorSnapshot, stage: preferredStage };
    }
  }
  return trainingMonitorSnapshot;
}

function subscribeTrainingMonitor(listener: () => void) {
  trainingMonitorListeners.add(listener);
  return () => trainingMonitorListeners.delete(listener);
}

function updateTrainingMonitor(mutator: (snapshot: TrainingMonitorSnapshot) => TrainingMonitorSnapshot) {
  const next = mutator(getTrainingMonitorSnapshot());
  trainingMonitorSnapshot = {
    ...next,
    logs: next.logs.slice(-500),
    phaseEvents: next.phaseEvents.slice(-200),
    updatedAt: Date.now(),
  };
  sessionStorage.setItem(TRAINING_STATE_KEY, JSON.stringify(trainingMonitorSnapshot));
  trainingMonitorListeners.forEach((listener) => listener());
}

function resolveState<T>(value: SetStateAction<T>, previous: T): T {
  return typeof value === 'function' ? (value as (prev: T) => T)(previous) : value;
}

function clearWaitingLogTimer() {
  if (waitingLogTimer !== null) {
    window.clearTimeout(waitingLogTimer);
    waitingLogTimer = null;
  }
}

function clearLocalHeartbeatTimer() {
  if (localHeartbeatTimer !== null) {
    window.clearInterval(localHeartbeatTimer);
    localHeartbeatTimer = null;
  }
}

function closeActiveTrainingConnection() {
  activeTrainingConnection?.close();
  activeTrainingConnection = null;
  clearWaitingLogTimer();
  clearLocalHeartbeatTimer();
}

export default function TrainingPage() {
  const location = useLocation();
  const autoStart = (location.state as { autoStart?: 'train' | 'sft' } | null)?.autoStart;
  const monitorSnapshot = useSyncExternalStore(
    subscribeTrainingMonitor,
    () => getTrainingMonitorSnapshot(autoStart),
    () => getTrainingMonitorSnapshot(autoStart)
  );
  const initialSnapshotRef = useRef<TrainingMonitorSnapshot | null>(monitorSnapshot);

  const isRunning = monitorSnapshot.isRunning;
  const stage = monitorSnapshot.stage;
  const metrics = monitorSnapshot.metrics;
  const lossHistory = monitorSnapshot.lossHistory;
  const samples = monitorSnapshot.samples;
  const activations = monitorSnapshot.activations;
  const logs = monitorSnapshot.logs;
  const [activeTab, setActiveTab] = useState<'chart' | 'heatmap' | 'samples' | 'logs'>('chart');
  const currentPhase = monitorSnapshot.currentPhase;
  const phaseEvents = monitorSnapshot.phaseEvents;
  const errorMessage = monitorSnapshot.errorMessage;
  const modelName = monitorSnapshot.modelName;

  const setIsRunning = useCallback((value: SetStateAction<boolean>) => {
    updateTrainingMonitor((prev) => ({ ...prev, isRunning: resolveState(value, prev.isRunning) }));
  }, []);
  const setStage = useCallback((value: SetStateAction<'train' | 'sft'>) => {
    updateTrainingMonitor((prev) => ({ ...prev, stage: resolveState(value, prev.stage) }));
  }, []);
  const setMetrics = useCallback((value: SetStateAction<TrainingMetrics | null>) => {
    updateTrainingMonitor((prev) => ({ ...prev, metrics: resolveState(value, prev.metrics) }));
  }, []);
  const setLossHistory = useCallback((value: SetStateAction<LossPoint[]>) => {
    updateTrainingMonitor((prev) => ({ ...prev, lossHistory: resolveState(value, prev.lossHistory) }));
  }, []);
  const setSamples = useCallback((value: SetStateAction<TrainingSample[]>) => {
    updateTrainingMonitor((prev) => ({ ...prev, samples: resolveState(value, prev.samples) }));
  }, []);
  const setActivations = useCallback((value: SetStateAction<LayerActivation | null>) => {
    updateTrainingMonitor((prev) => ({ ...prev, activations: resolveState(value, prev.activations) }));
  }, []);
  const setLogs = useCallback((value: SetStateAction<string[]>) => {
    updateTrainingMonitor((prev) => ({ ...prev, logs: resolveState(value, prev.logs) }));
  }, []);
  const setCurrentPhase = useCallback((value: SetStateAction<PhaseId>) => {
    updateTrainingMonitor((prev) => ({ ...prev, currentPhase: resolveState(value, prev.currentPhase) }));
  }, []);
  const setPhaseEvents = useCallback((value: SetStateAction<{ phase: PhaseId; text: string; time: string }[]>) => {
    updateTrainingMonitor((prev) => ({ ...prev, phaseEvents: resolveState(value, prev.phaseEvents) }));
  }, []);
  const setErrorMessage = useCallback((value: SetStateAction<string>) => {
    updateTrainingMonitor((prev) => ({ ...prev, errorMessage: resolveState(value, prev.errorMessage) }));
  }, []);
  const setModelName = useCallback((value: SetStateAction<string>) => {
    updateTrainingMonitor((prev) => ({ ...prev, modelName: resolveState(value, prev.modelName) }));
  }, []);

  const logsEndRef = useRef<HTMLDivElement>(null);
  const autoStartedRef = useRef(false);
  const autoStartTimerRef = useRef<number | null>(null);
  const startTrainingRef = useRef<((stageOverride?: 'train' | 'sft') => void) | null>(null);

  useEffect(() => {
    if (isRunning) return;
    if (autoStart && stage !== autoStart) {
      setStage(autoStart);
    }
    const savedModelName = getSavedModelName();
    if (savedModelName && modelName !== savedModelName) {
      setModelName(savedModelName);
    }
  }, [autoStart, isRunning, modelName, setModelName, setStage, stage]);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    const snapshot: TrainingMonitorSnapshot = {
      isRunning,
      stage,
      metrics,
      lossHistory,
      samples,
      activations,
      logs: logs.slice(-500),
      currentPhase,
      phaseEvents: phaseEvents.slice(-200),
      errorMessage,
      modelName,
      updatedAt: Date.now(),
    };
    sessionStorage.setItem(TRAINING_STATE_KEY, JSON.stringify(snapshot));
  }, [isRunning, stage, metrics, lossHistory, samples, activations, logs, currentPhase, phaseEvents, errorMessage, modelName]);

  useEffect(() => {
    if (autoStart) return;
    fetch('/api/status')
      .then((res) => res.json())
      .then((status) => {
        const backendRunning = !!status.running;
        if (backendRunning) {
          setIsRunning(true);
          setLogs((prev) => {
            const text = activeTrainingConnection
              ? '训练监控会话仍在运行：页面切换不会停止训练，日志和指标会继续实时刷新。'
              : '已恢复训练监控快照：后端训练仍在运行；如果是整页刷新后返回，实时日志只能从重新连接后的新任务开始完整收集。';
            if (prev.includes(text)) return prev;
            return [...prev.slice(-500), text];
          });
          return;
        }
        if (initialSnapshotRef.current?.isRunning) {
          setIsRunning(false);
          setLogs((prev) => {
            const text = '已恢复训练监控快照：后端当前没有运行中的训练，之前的训练可能已完成或已停止。';
            if (prev.includes(text)) return prev;
            return [...prev.slice(-500), text];
          });
        }
      })
      .catch(() => {});
  }, [autoStart, setIsRunning, setLogs]);

  const recordPhase = useCallback((phase: PhaseId, text: string) => {
    setCurrentPhase(phase);
    setPhaseEvents((prev) => [
      ...prev.slice(-11),
      { phase, text, time: new Date().toLocaleTimeString() },
    ]);
  }, [setCurrentPhase, setPhaseEvents]);

  const processLine = useCallback((line: string) => {
    if (line.includes('Vocab size') || line.includes('Model config') || line.includes('Parameter counts') || line.includes('Using ')) {
      recordPhase('prepare', line);
    } else if (line.includes('Starting training') || line.includes('Training:')) {
      recordPhase('data', line);
    } else if (line.includes('Val loss') || line.includes('Val BPB')) {
      recordPhase('eval', line);
    } else if (line.includes('Saved checkpoint')) {
      recordPhase('save', line);
    } else if (line.includes('Training complete')) {
      recordPhase('done', line);
    }

    // Parse metrics
    const metricMatch = line.match(METRIC_RE);
    if (metricMatch) {
      const m: TrainingMetrics = {
        step: parseInt(metricMatch[1]),
        totalSteps: parseInt(metricMatch[2]),
        loss: parseFloat(metricMatch[3]),
        lr: parseFloat(metricMatch[4]),
        tokensPerSec: parseFloat(metricMatch[5]),
        eta: metricMatch[6],
        elapsed: '',
      };
      setMetrics(m);
      setLossHistory((prev) => {
        const next = [...prev, { step: m.step, trainLoss: m.loss, lr: m.lr }];
        return next.length > 2000 ? next.slice(-2000) : next;
      });
      recordPhase('update', `完成第 ${m.step} 步：loss=${m.loss.toFixed(4)}，速度=${m.tokensPerSec.toFixed(0)} tok/s`);
      return;
    }

    // Parse validation metrics
    const valMatch = line.match(VAL_RE);
    if (valMatch) {
      const valLoss = parseFloat(valMatch[1]);
      const bpb = parseFloat(valMatch[2]);
      setMetrics((prev) => prev ? { ...prev, valLoss, bpb } : prev);
      setLossHistory((prev) => {
        if (prev.length === 0) return prev;
        const last = { ...prev[prev.length - 1], valLoss };
        return [...prev.slice(0, -1), last];
      });
      return;
    }

    // Parse activation data
    const actMatch = line.match(ACTIVATION_RE);
    if (actMatch) {
      try {
        const data = JSON.parse(actMatch[1]);
        setActivations(data);
      } catch {
        setLogs((prev) => [...prev.slice(-500), '警告: 激活数据解析失败']);
      }
      return;
    }

    // Parse sample
    const sampleMatch = line.match(SAMPLE_RE);
    if (sampleMatch) {
      setSamples((prev) => [
        ...prev,
        {
          step: metrics?.step || 0,
          prompt: '',
          generated: sampleMatch[1].trim(),
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    }
  }, [metrics?.step, recordPhase, setActivations, setLogs, setLossHistory, setMetrics, setSamples]);

  const startTraining = useCallback(
    (stageOverride?: 'train' | 'sft') => {
      const s = stageOverride || stage;
      closeActiveTrainingConnection();
      setIsRunning(true);
      setLossHistory([]);
      setSamples([]);
      setLogs([]);
      setMetrics(null);
      setCurrentPhase('prepare');
      setPhaseEvents([]);
      setErrorMessage('');

      // Load saved config
      const savedConfig = localStorage.getItem('nanochat_config');
      const config = savedConfig ? JSON.parse(savedConfig) : {};
      const savedModelName = typeof config.model_name === 'string' ? config.model_name.trim() : '';
      const resolvedModelName = savedModelName || modelName.trim() || createDefaultModelName(s, String(config.data_domain || DEFAULT_TRAINING_PARAMS.data_domain));
      if (resolvedModelName !== modelName) {
        setModelName(resolvedModelName);
      }
      localStorage.setItem('nanochat_config', JSON.stringify({ ...config, model_name: resolvedModelName }));
      const params = new URLSearchParams();
      const mergedConfig = { ...DEFAULT_TRAINING_PARAMS, ...config, model_name: resolvedModelName };
      Object.entries(mergedConfig).forEach(([k, v]) => {
        if (RUN_PARAM_KEYS.has(k) && v !== undefined && v !== null && v !== '') {
          params.set(k, String(v));
        }
      });
      if (params.get('num_iterations') === '-1') {
        params.set('num_iterations', DEFAULT_TRAINING_PARAMS.num_iterations);
      }
      if (s === 'train' && params.get('num_iterations') === '500' && params.get('save_every') === '250') {
        params.set('save_every', '500');
      }
      if (s === 'sft' && params.get('num_iterations') === DEFAULT_TRAINING_PARAMS.num_iterations) {
        params.set('num_iterations', '200');
      }
      params.set('_ts', String(Date.now()));
      recordPhase('prepare', `启动${s === 'train' ? '预训练' : 'SFT 微调'}：${params.toString()}`);
      setLogs([
        `正在连接训练服务：/api/run/${s}`,
        '如果首次加载模型或准备数据较慢，后端日志可能需要几秒钟才会出现。',
      ]);

      const controller = new AbortController();
      const connection = { close: () => controller.abort() };
      activeTrainingConnection = connection;
      waitingLogTimer = window.setTimeout(() => {
        setLogs((prev) => [
          ...prev.slice(-500),
          '仍在等待后端训练输出：如果长时间无响应，请查看后端终端，或点击停止后重试。',
        ]);
      }, 8000);
      localHeartbeatTimer = window.setInterval(() => {
        setLogs((prev) => {
          const last = prev[prev.length - 1] || '';
          if (last.includes('本地等待心跳')) return prev;
          return [
            ...prev.slice(-500),
            '本地等待心跳：训练请求仍在等待后端输出，模型初始化或首批 step 可能需要一段时间。',
          ];
        });
      }, 10000);

      const handlePayload = (payload: string) => {
        clearWaitingLogTimer();
        try {
          const data = JSON.parse(payload);
          if (data.type === 'log' || data.type === 'output') {
            setLogs((prev) => [...prev.slice(-500), data.text]);
            processLine(data.text);
          } else if (data.type === 'metric') {
            // Backend sends structured metrics: {step, total, loss, tok_per_sec}
            const m: TrainingMetrics = {
              step: data.step,
              totalSteps: data.total,
              loss: data.loss,
              lr: 0,
              tokensPerSec: data.tok_per_sec,
              eta: '',
              elapsed: '',
            };
            setMetrics(m);
            setLossHistory((prev) => {
              const next = [...prev, { step: m.step, trainLoss: m.loss }];
              return next.length > 2000 ? next.slice(-2000) : next;
            });
            recordPhase('update', `完成第 ${m.step} 步：loss=${m.loss.toFixed(4)}，速度=${m.tokensPerSec.toFixed(0)} tok/s`);
          } else if (data.type === 'sample') {
            recordPhase('sample', `第 ${data.step || 0} 步生成了文本样本`);
            setSamples((prev) => [
              ...prev,
              {
                step: data.step || 0,
                prompt: data.prompt || '',
                generated: data.text || '',
                timestamp: new Date().toLocaleTimeString(),
              },
            ]);
          } else if (data.type === 'activation') {
            setActivations(data);
          } else if (data.type === 'done' || data.type === 'error') {
            clearLocalHeartbeatTimer();
            if (data.type === 'error') {
              const text = data.text || data.message || '训练任务出错，请查看训练日志。';
              setErrorMessage(text);
              setLogs((prev) => [...prev.slice(-500), `错误: ${text}`]);
            }
            recordPhase(data.type === 'done' ? 'done' : 'prepare', data.type === 'done' ? '训练任务完成' : '训练任务出错');
            setIsRunning(false);
            if (activeTrainingConnection === connection) {
              activeTrainingConnection = null;
            }
          }
        } catch {
          setLogs((prev) => [...prev.slice(-500), payload]);
          processLine(payload);
        }
      };

      const finishWithError = (text: string) => {
        clearWaitingLogTimer();
        setErrorMessage(text);
        setLogs((prev) => [...prev.slice(-500), `错误: ${text}`]);
        setIsRunning(false);
        if (activeTrainingConnection === connection) {
          activeTrainingConnection = null;
        }
        clearLocalHeartbeatTimer();
      };

      void (async () => {
        try {
          const response = await fetch(`/api/run/${s}?${params.toString()}`, {
            signal: controller.signal,
          });
          setLogs((prev) => [...prev.slice(-500), '训练流已连接，等待后端输出...']);
          if (!response.ok || !response.body) {
            throw new Error(`后端返回 ${response.status}`);
          }
          const reader = response.body.getReader();
          const decoder = new TextDecoder();
          let buffer = '';
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split('\n\n');
            buffer = events.pop() || '';
            for (const event of events) {
              const dataLines = event
                .split('\n')
                .filter((line) => line.startsWith('data:'))
                .map((line) => line.slice(5).trimStart());
              if (dataLines.length > 0) {
                handlePayload(dataLines.join('\n'));
              }
            }
          }
          const tail = buffer.trim();
          if (tail.startsWith('data:')) {
            handlePayload(tail.slice(5).trimStart());
          }
          if (activeTrainingConnection === connection) {
            activeTrainingConnection = null;
          }
          setIsRunning(false);
          clearLocalHeartbeatTimer();
        } catch (error) {
          if ((error as Error).name === 'AbortError') return;
          finishWithError(
            `训练连接中断：${error instanceof Error ? error.message : '请确认后端服务正在运行，或查看后端终端中的错误日志。'}`
          );
        }
      })();
    },
    [
      modelName,
      processLine,
      recordPhase,
      setActivations,
      setCurrentPhase,
      setErrorMessage,
      setIsRunning,
      setLogs,
      setLossHistory,
      setMetrics,
      setModelName,
      setPhaseEvents,
      setSamples,
      stage,
    ]
  );
  startTrainingRef.current = startTraining;

  // Auto-start once when navigated from the pipeline page.
  useEffect(() => {
    if (!autoStart || autoStartedRef.current) return;
    autoStartedRef.current = true;
    updateTrainingMonitor(() => ({
      ...createEmptyTrainingSnapshot(autoStart),
      logs: ['已进入新的训练任务，正在准备启动本轮训练...'],
    }));
    autoStartTimerRef.current = window.setTimeout(() => {
      autoStartTimerRef.current = null;
      startTrainingRef.current?.(autoStart);
    }, 150);
    return () => {
      if (autoStartTimerRef.current !== null) {
        window.clearTimeout(autoStartTimerRef.current);
        autoStartTimerRef.current = null;
      }
    };
  }, [autoStart]);

  useEffect(() => {
    return () => {
      if (autoStartTimerRef.current !== null) {
        window.clearTimeout(autoStartTimerRef.current);
        autoStartTimerRef.current = null;
      }
    };
  }, []);

  const stopTraining = useCallback(() => {
    closeActiveTrainingConnection();
    fetch('/api/stop', { method: 'POST' });
    setIsRunning(false);
  }, [setIsRunning]);

  const exportTrainingReport = useCallback(() => {
    const savedConfig = localStorage.getItem('nanochat_config');
    const config = savedConfig ? JSON.parse(savedConfig) : DEFAULT_TRAINING_PARAMS;
    const finalPoint = lossHistory[lossHistory.length - 1];
    const bestTrainLoss = lossHistory.reduce(
      (best, point) => Math.min(best, point.trainLoss ?? Number.POSITIVE_INFINITY),
      Number.POSITIVE_INFINITY
    );
    const report = [
      '# miniGPT Studio 训练报告',
      '',
      `- 导出时间：${new Date().toLocaleString()}`,
      `- 训练阶段：${stage === 'train' ? '预训练' : 'SFT 微调'}`,
      `- 当前状态：${isRunning ? '训练中' : errorMessage ? '失败/中断' : currentPhase === 'done' ? '完成' : '未运行或已停止'}`,
      `- 当前步骤：${metrics?.step ?? finalPoint?.step ?? '-'}`,
      `- 总步骤：${metrics?.totalSteps ?? '-'}`,
      `- 最终训练 Loss：${metrics?.loss?.toFixed(4) ?? finalPoint?.trainLoss?.toFixed(4) ?? '-'}`,
      `- 最优训练 Loss：${Number.isFinite(bestTrainLoss) ? bestTrainLoss.toFixed(4) : '-'}`,
      `- 验证 Loss：${metrics?.valLoss?.toFixed(4) ?? '-'}`,
      `- BPB：${metrics?.bpb?.toFixed(4) ?? '-'}`,
      `- 训练速度：${metrics?.tokensPerSec?.toFixed(0) ?? '-'} tok/s`,
      '',
      '## 训练参数',
      '',
      '```json',
      JSON.stringify(config, null, 2),
      '```',
      '',
      '## 训练阶段事件',
      '',
      ...phaseEvents.map((event) => `- ${event.time} ${phaseDetails[event.phase].label}：${event.text}`),
      '',
      '## Loss 曲线数据',
      '',
      '| step | trainLoss | valLoss | lr |',
      '|---:|---:|---:|---:|',
      ...lossHistory.map((point) => `| ${point.step} | ${point.trainLoss ?? ''} | ${point.valLoss ?? ''} | ${point.lr ?? ''} |`),
      '',
      '## 生成样本',
      '',
      ...(samples.length > 0
        ? samples.map((sample) => `### Step ${sample.step} · ${sample.timestamp}\n\n\`\`\`text\n${sample.generated}\n\`\`\``)
        : ['暂无生成样本。']),
      '',
      '## 最近训练日志',
      '',
      '```text',
      ...logs.slice(-80),
      '```',
    ].join('\n');
    downloadTextFile(`minigpt-studio-training-report-${Date.now()}.md`, report);
  }, [currentPhase, errorMessage, isRunning, logs, lossHistory, metrics, phaseEvents, samples, stage]);

  const progress = metrics ? (metrics.step / metrics.totalSteps) * 100 : 0;

  return (
    <div className="px-5 py-6 lg:px-6 w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Activity className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">训练监控</h1>
            {isRunning && (
              <span className="px-2 py-0.5 bg-success/20 text-success text-xs rounded-full animate-pulse">
                训练中
              </span>
            )}
          </div>
          <p className="text-text-muted">实时监控模型训练过程，包括损失变化、层激活和生成质量</p>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={stage}
            onChange={(e) => setStage(e.target.value as 'train' | 'sft')}
            disabled={isRunning}
            className="bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
          >
            <option value="train">预训练</option>
            <option value="sft">SFT 微调</option>
          </select>
          {!isRunning ? (
            <>
              <button
                onClick={exportTrainingReport}
                disabled={lossHistory.length === 0 && logs.length === 0}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm font-medium transition-all',
                  lossHistory.length === 0 && logs.length === 0
                    ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                    : 'bg-surface-light border border-border text-text hover:bg-surface-lighter'
                )}
              >
                <Download className="w-4 h-4" />
                导出报告
              </button>
              <button
                onClick={() => startTraining()}
                className="flex items-center gap-1.5 px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-dark transition-all"
              >
                <Play className="w-4 h-4" />
                开始训练
              </button>
            </>
          ) : (
            <button
              onClick={stopTraining}
              className="flex items-center gap-1.5 px-4 py-2 bg-error/20 text-error rounded-lg text-sm font-medium hover:bg-error/30 transition-all"
            >
              <Square className="w-4 h-4" />
              停止
            </button>
          )}
        </div>
      </div>

      {/* Progress bar */}
      {errorMessage && (
        <div className="mb-6 border border-error/30 bg-error/10 text-error rounded-xl px-4 py-3 text-sm">
          <div className="font-medium mb-1">训练任务未完成</div>
          <div className="text-error/90">{errorMessage}</div>
        </div>
      )}

      <div className="mb-6 bg-surface-light border border-border rounded-xl p-4">
        <div className="space-y-1.5">
          <span className="text-xs font-medium text-text-muted">模型名称</span>
          <div className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text">
            {modelName}
          </div>
          <p className="text-[11px] text-text-muted">
            该名称来自“训练流程”的模型名称配置，并会写入 checkpoint 元数据。需要修改时，请返回“训练流程”调整。
          </p>
        </div>
      </div>

      {/* Progress bar */}
      {isRunning && metrics && (
        <div className="mb-6 bg-surface-light border border-border rounded-xl p-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-text-muted">
              步骤 {metrics.step.toLocaleString()} / {metrics.totalSteps.toLocaleString()}
            </span>
            <span className="text-text-muted">ETA: {metrics.eta}</span>
          </div>
          <div className="h-2 bg-surface-lighter rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-primary-light rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Training process overview */}
      <div className="mb-6 bg-surface-light border border-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-border flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold flex items-center gap-2">
              <Brain className="w-4 h-4 text-primary" />
              模型正在如何学习
            </h2>
            <p className="text-xs text-text-muted mt-1">
              每一个训练 step 都会重复：读取数据 → 前向预测 → 计算 loss → 反向传播 → 更新参数。
            </p>
          </div>
          <div className="text-xs text-text-muted">
            当前阶段：<span className="text-primary-light font-medium">{phaseDetails[currentPhase].label}</span>
          </div>
        </div>

        <div className="p-5 grid grid-cols-1 lg:grid-cols-[1.3fr_1fr] gap-5">
          <div>
            <div className="grid grid-cols-5 gap-2">
              {stepLoop.map((item, index) => {
                const Icon = item.icon;
                const activeIndex = metrics ? metrics.step % stepLoop.length : 0;
                const isActivePhase = isRunning && (currentPhase === item.id || activeIndex === index);
                return (
                  <div
                    key={item.id}
                    className={cn(
                      'rounded-xl border p-3 transition-all',
                      isActivePhase
                        ? 'border-primary/60 bg-primary/10 shadow-lg shadow-primary/5'
                        : 'border-border bg-surface'
                    )}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <Icon className={cn('w-4 h-4', isActivePhase ? 'text-primary-light' : 'text-text-muted')} />
                      {isActivePhase && (
                        <span className="w-2 h-2 rounded-full bg-success animate-pulse" />
                      )}
                    </div>
                    <div className="text-xs font-medium text-text">{phaseDetails[item.id].label}</div>
                    <p className="text-[11px] text-text-muted mt-1 leading-relaxed">
                      {phaseDetails[item.id].description}
                    </p>
                  </div>
                );
              })}
            </div>

            <div className="mt-4 grid grid-cols-3 gap-2">
              {(['eval', 'sample', 'save'] as PhaseId[]).map((phase) => {
                const Icon = phase === 'eval' ? CheckCircle2 : phase === 'sample' ? MessageSquareText : Save;
                const isActivePhase = currentPhase === phase;
                return (
                  <div
                    key={phase}
                    className={cn(
                      'rounded-lg border px-3 py-2',
                      isActivePhase ? 'border-warning/50 bg-warning/10' : 'border-border bg-surface'
                    )}
                  >
                    <div className="flex items-center gap-2 text-xs font-medium">
                      <Icon className={cn('w-3.5 h-3.5', isActivePhase ? 'text-warning' : 'text-text-muted')} />
                      {phaseDetails[phase].label}
                    </div>
                    <p className="text-[11px] text-text-muted mt-1">{phaseDetails[phase].description}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="bg-surface border border-border rounded-xl p-4">
            <h3 className="text-sm font-medium mb-3">训练事件流</h3>
            {phaseEvents.length > 0 ? (
              <div className="space-y-2 max-h-56 overflow-y-auto">
                {phaseEvents.map((event, i) => (
                  <div key={`${event.time}-${i}`} className="text-xs border-l-2 border-primary/50 pl-3">
                    <div className="flex items-center justify-between">
                      <span className="text-primary-light">{phaseDetails[event.phase].label}</span>
                      <span className="text-text-muted">{event.time}</span>
                    </div>
                    <div className="text-text-muted mt-0.5 truncate" title={event.text}>
                      {event.text}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-40 flex flex-col items-center justify-center text-text-muted">
                <Activity className="w-8 h-8 mb-2 opacity-30" />
                <p className="text-sm">点击开始训练后，这里会展示模型学习的关键事件</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
        <MetricCard
          label="当前步骤"
          value={metrics?.step?.toLocaleString() || '-'}
          tooltip="当前训练到第几步"
        />
        <MetricCard
          label="训练损失"
          value={metrics?.loss?.toFixed(4) || '-'}
          trend={lossHistory.length > 1 && lossHistory[lossHistory.length - 1]?.trainLoss < lossHistory[Math.max(0, lossHistory.length - 10)]?.trainLoss ? 'down' : 'neutral'}
          tooltip="训练损失（Loss）衡量模型预测与实际值的差距，越低越好。Loss 持续下降说明模型在学习。"
        />
        <MetricCard
          label="验证损失"
          value={metrics?.valLoss?.toFixed(4) || '-'}
          tooltip="在未参与训练的数据上评估的损失值，可以判断模型是否过拟合"
        />
        <MetricCard
          label="学习率"
          value={metrics?.lr?.toExponential(2) || '-'}
          tooltip="当前学习率，通常随训练进程按余弦退火策略逐步减小"
        />
        <MetricCard
          label="速度"
          value={metrics?.tokensPerSec?.toFixed(0) || '-'}
          unit="tok/s"
          tooltip="每秒处理的 token 数，反映训练速度"
        />
        <MetricCard
          label="BPB"
          value={metrics?.bpb?.toFixed(4) || '-'}
          tooltip="Bits Per Byte，信息论指标，衡量模型压缩文本的效率，越低越好"
        />
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 mb-4 bg-surface-light border border-border rounded-lg p-1">
        {([
          { key: 'chart', icon: TrendingDown, label: '损失曲线' },
          { key: 'heatmap', icon: Layers, label: '层级训练信号' },
          { key: 'samples', icon: MessageSquareText, label: '生成样本' },
          { key: 'logs', icon: Terminal, label: '训练日志' },
        ] as const).map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              'flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-all',
              activeTab === tab.key
                ? 'bg-primary/15 text-primary font-medium'
                : 'text-text-muted hover:text-text hover:bg-surface-lighter'
            )}
          >
            <tab.icon className="w-3.5 h-3.5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="bg-surface-light border border-border rounded-xl overflow-hidden min-h-[400px]">
        {/* Loss chart */}
        {activeTab === 'chart' && (
          <div className="p-5">
            {lossHistory.length > 0 ? (
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={lossHistory}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis
                    dataKey="step"
                    stroke="#94a3b8"
                    fontSize={12}
                    tickFormatter={(v) => v.toLocaleString()}
                  />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <RechartsTooltip
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      fontSize: '12px',
                    }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="trainLoss"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={false}
                    name="训练损失"
                  />
                  <Line
                    type="monotone"
                    dataKey="valLoss"
                    stroke="#22c55e"
                    strokeWidth={2}
                    dot={false}
                    name="验证损失"
                    connectNulls
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-text-muted">
                <TrendingDown className="w-12 h-12 mb-3 opacity-30" />
                <p>等待训练数据...</p>
                <p className="text-xs mt-1">开始训练后，损失曲线将在此实时绘制</p>
              </div>
            )}
          </div>
        )}

        {/* Heatmap */}
        {activeTab === 'heatmap' && (
          <div className="p-5">
            {activations ? (
              <div>
                <h3 className="text-sm font-medium mb-1">
                  各层训练信号分布 (Step {activations.step})
                </h3>
                <p className="text-xs text-text-muted mb-3">
                  当前训练脚本未直接暴露隐藏层原始激活，这里展示由实时 loss、step、tok/s 推导的层级训练信号，用于观察训练动态。
                </p>
                <div className="space-y-1.5">
                  {activations.layers.map((layer, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <span className="text-xs text-text-muted w-20 text-right shrink-0 font-mono">
                        {layer.name}
                      </span>
                      <div className="flex-1 flex gap-1">
                        <div
                          className="h-6 rounded"
                          style={{
                            width: `${Math.min(100, layer.meanActivation * 100)}%`,
                            backgroundColor: `rgba(99, 102, 241, ${Math.min(1, layer.meanActivation)})`,
                          }}
                          title={`mean: ${layer.meanActivation.toFixed(4)}`}
                        />
                        <div
                          className="h-6 rounded"
                          style={{
                            width: `${Math.min(100, layer.gradMean * 200)}%`,
                            backgroundColor: `rgba(234, 179, 8, ${Math.min(1, layer.gradMean * 10)})`,
                          }}
                          title={`grad: ${layer.gradMean.toFixed(6)}`}
                        />
                      </div>
                      <div className="text-xs text-text-muted w-32 shrink-0 font-mono">
                        act:{layer.meanActivation.toFixed(3)} grad:{layer.gradMean.toFixed(5)}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="flex gap-6 mt-4 text-xs text-text-muted">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-primary" />
                    <span>学习信号</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-warning" />
                    <span>梯度压力</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-text-muted">
                <Layers className="w-12 h-12 mb-3 opacity-30" />
                <p>等待层级训练信号...</p>
                <p className="text-xs mt-1">训练开始并产生 metric 后，各层训练信号将在此展示</p>
              </div>
            )}
          </div>
        )}

        {/* Samples */}
        {activeTab === 'samples' && (
          <div className="p-5">
            {samples.length > 0 ? (
              <div className="space-y-4">
                {samples.map((sample, i) => (
                  <div
                    key={i}
                    className="bg-surface border border-border rounded-lg p-4"
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-3.5 h-3.5 text-warning" />
                      <span className="text-xs text-text-muted">
                        Step {sample.step} · {sample.timestamp}
                      </span>
                    </div>
                    <pre className="text-sm text-text whitespace-pre-wrap font-mono leading-relaxed">
                      {sample.generated}
                    </pre>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-[400px] text-text-muted">
                <MessageSquareText className="w-12 h-12 mb-3 opacity-30" />
                <p>{stage === 'sft' ? 'SFT 阶段暂不生成样本' : '等待生成样本...'}</p>
                <p className="text-xs mt-1">
                  {stage === 'sft'
                    ? '当前 SFT 脚本主要输出 loss、验证和 checkpoint；需要看模型回答可到“模型对话”页加载 SFT checkpoint。'
                    : `预训练会按 sample_every 间隔生成文本样本，当前默认约每 ${DEFAULT_TRAINING_PARAMS.sample_every} 步生成一次。`}
                </p>
                {stage === 'train' && metrics?.step !== undefined && Number(DEFAULT_TRAINING_PARAMS.sample_every) > metrics.step && (
                  <p className="text-xs mt-1 text-primary-light">
                    当前第 {metrics.step} 步，预计第 {DEFAULT_TRAINING_PARAMS.sample_every} 步开始出现第一批样本。
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Logs */}
        {activeTab === 'logs' && (
          <div className="p-4 max-h-[500px] overflow-y-auto font-mono text-xs text-text-muted space-y-0.5">
            {logs.length > 0 ? (
              logs.map((log, i) => (
                <div key={i} className="hover:text-text transition-colors">
                  {log}
                </div>
              ))
            ) : (
              <div className="text-center py-20 text-text-muted">
                等待日志输出...
              </div>
            )}
            <div ref={logsEndRef} />
          </div>
        )}
      </div>
    </div>
  );
}

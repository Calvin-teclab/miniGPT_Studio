import { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { BookOpen, Database, Scissors, Sparkles, Upload } from 'lucide-react';
import StepCard, { type StepDetail } from '@/components/StepCard';
import type { PipelineStep } from '@/types';
import {
  getSampleDataset,
  getSampleDatasets,
  getStatus,
  saveSampleDataset,
  uploadDatasetFile,
  type SampleDataset,
} from '@/api/client';

const initialSteps: PipelineStep[] = [
  {
    id: 'data',
    name: '准备训练数据',
    description: '准备通用文本或音乐/和弦训练数据',
    detailedHelp:
      '训练一个模型需要先准备教材。默认使用 FineWeb-Edu 文本数据；如果在参数配置中选择“音乐/和弦数据”，系统会把您提供的 JSONL/CSV/TXT 和弦数据转换为训练分片，并放入独立缓存目录，不影响原来的通用文本流程。',
    status: 'pending',
    stage: 'data',
  },
  {
    id: 'tokenizer',
    name: '训练分词器',
    description: '基于训练数据构建 BPE 分词器',
    detailedHelp:
      '分词器（Tokenizer）负责将文本转换为模型能理解的数字序列。我们使用 BPE（字节对编码）算法，从训练数据中学习最优的分词方式。好的分词器能让模型更高效地学习语言规律。训练过程自动完成，约需 1-2 分钟。',
    status: 'pending',
    stage: 'tokenizer',
  },
  {
    id: 'train',
    name: '预训练模型',
    description: '从零开始训练 GPT 语言模型',
    detailedHelp:
      '这是核心步骤！模型会从随机初始化的参数开始，通过阅读大量文本来学习语言的规律。训练过程中，模型反复尝试预测下一个词，并根据预测结果调整自身参数（就像学生做练习题后对答案改进）。您可以在训练监控页面实时观察 Loss（损失值，越低越好）的下降趋势。',
    status: 'pending',
    stage: 'train',
  },
  {
    id: 'sft',
    name: 'SFT 微调',
    description: '使用对话数据进行有监督微调',
    detailedHelp:
      'SFT（Supervised Fine-Tuning，有监督微调）是让模型学会"对话"的关键步骤。预训练后的模型虽然理解语言，但还不擅长按照指令回答问题。通过在高质量的问答对话数据上微调，模型将学会理解用户意图并生成有帮助的回答。这就像一个学过很多知识的人，还需要经过面试训练才能好好地回答问题。',
    status: 'pending',
    stage: 'sft',
  },
  {
    id: 'chat',
    name: '开始对话',
    description: '加载训练好的模型并开始交互',
    detailedHelp:
      '恭喜！模型训练完成后，您可以直接在这里与它对话。模型会被加载到内存中，利用 Apple Silicon 的统一内存架构进行高速推理。您可以测试模型的各种能力，调整生成参数，感受自己从零训练出的 AI 模型。',
    status: 'pending',
    stage: 'chat',
  },
];

type DataDomain = 'general' | 'music';
type DataSource = 'builtin' | 'custom_path' | 'upload';

interface DataPrepConfig {
  model_name: string;
  data_domain: DataDomain;
  data_source: DataSource;
  text_dataset: string;
  music_dataset: string;
  custom_data_path: string;
  music_data_path: string;
  tokenizer_method: string;
  vocab_size: number;
  doc_cap: number;
  n_shards: number;
}

interface RunSyncEvent {
  type?: string;
  text?: string;
  step?: number;
  total?: number;
  loss?: number;
  tok_per_sec?: number;
}

const defaultDataPrepConfig: DataPrepConfig = {
  model_name: '',
  data_domain: 'general',
  data_source: 'builtin',
  text_dataset: 'text_general_raw.txt',
  music_dataset: 'music_starter_chords_raw.jsonl',
  custom_data_path: '',
  music_data_path: '',
  tokenizer_method: 'bpe',
  vocab_size: 32768,
  doc_cap: 10000,
  n_shards: 4,
};

function createDefaultFlowModelName(dataDomain: DataDomain = 'general') {
  const now = new Date();
  const stamp = [
    now.getFullYear(),
    String(now.getMonth() + 1).padStart(2, '0'),
    String(now.getDate()).padStart(2, '0'),
    String(now.getHours()).padStart(2, '0'),
    String(now.getMinutes()).padStart(2, '0'),
    String(now.getSeconds()).padStart(2, '0'),
  ].join('');
  return `${dataDomain}-train-${stamp}`;
}

const loadDataPrepConfig = ({ freshModelName = false } = {}): DataPrepConfig => {
  try {
    const saved = JSON.parse(localStorage.getItem('nanochat_config') || '{}');
    const merged = { ...defaultDataPrepConfig, ...saved };
    if (merged.music_dataset === 'starter_chords') {
      merged.music_dataset = 'music_starter_chords_raw.jsonl';
    }
    if (freshModelName) {
      const dataDomain: DataDomain = merged.data_domain === 'music' ? 'music' : 'general';
      merged.model_name = createDefaultFlowModelName(dataDomain);
    }
    return merged;
  } catch {
    return {
      ...defaultDataPrepConfig,
      model_name: freshModelName ? createDefaultFlowModelName(defaultDataPrepConfig.data_domain) : '',
    };
  }
};

const stepDetails: Record<string, StepDetail> = {
  data: {
    purpose:
      '把原始数据准备成模型可学习的训练材料。没有这一步，后面的分词器和模型都没有“教材”。',
    inputs: [
      'FineWeb-Edu 数据源，或本地音乐/和弦 JSONL、CSV、TXT 数据文件',
      '数据分片数量配置 n_shards',
      '音乐数据文件路径 music_data_path（仅音乐数据模式）',
      '本地缓存目录 ~/.cache/nanochat',
    ],
    process: [
      {
        title: '连接数据源',
        description: '通用模式会请求 FineWeb-Edu 数据分片；音乐模式会读取您提供的本地和弦数据文件。',
        result: '确认可下载/可复用的数据分片',
      },
      {
        title: '转换训练分片',
        description: '把原始文本或和弦样本转换成统一的 parquet 分片，后续 tokenizer 和模型训练都读取这个格式。',
        result: '本地 parquet shard 文件',
      },
      {
        title: '切分训练/验证数据',
        description: '保留一部分数据作为验证集，用来判断模型是否真的学会，而不是只记住训练文本。',
        result: 'train / val 数据',
      },
      {
        title: '写入缓存状态',
        description: '后端会记录已下载分片数量，页面刷新时可识别“已检测到本地缓存”。',
        result: 'dataReady 状态',
      },
    ],
    outputs: [
      '可用于训练 tokenizer 的原始文本数据',
      '可用于模型训练的训练集',
      '可用于验证 loss 的验证集',
    ],
    successResult: '后续步骤可以直接读取本地数据，不需要重新下载。',
  },
  tokenizer: {
    purpose:
      '训练分词器，把人类文本拆成模型能处理的 token 数字序列。它相当于模型的“识字表”。',
    inputs: [
      '第 1 步准备好的训练文本',
      '特殊 token 定义（用户开始、助手开始、结束符等）',
      '目标词表大小配置',
    ],
    process: [
      {
        title: '扫描文本语料',
        description: '统计训练文本里字符、字节和常见片段的分布，找出高频组合。',
        result: '文本频率统计',
      },
      {
        title: '训练 BPE 规则',
        description: '用字节对编码反复合并高频片段，让常见词更短、生僻词也能表示。',
        result: 'BPE 合并表',
      },
      {
        title: '加入特殊 token',
        description: '为对话场景添加 user/assistant/system 等边界 token，方便后续 SFT 和聊天。',
        result: '对话边界能力',
      },
      {
        title: '保存 tokenizer',
        description: '把 tokenizer 保存到本地，训练、评测、对话都必须使用同一个 tokenizer。',
        result: 'tokenizer.pkl',
      },
    ],
    outputs: [
      'tokenizer.pkl',
      '词表和 BPE 合并规则',
      '文本 encode/decode 能力',
    ],
    successResult: '任意文本都能被稳定转换成 token id，并可还原为文本。',
  },
  train: {
    purpose:
      '从随机初始化开始训练 GPT 语言模型，让模型通过预测下一个 token 学习语言规律。',
    inputs: [
      '训练/验证 token 数据',
      'tokenizer 词表大小',
      '模型参数配置（depth、序列长度、窗口模式等）',
      '训练超参（步数、batch、保存间隔、验证间隔）',
    ],
    process: [
      {
        title: '构建模型结构',
        description: '根据 depth 推导层数、嵌入维度、注意力头数，初始化 Transformer 参数。',
        result: 'GPT 模型骨架',
      },
      {
        title: '读取 batch',
        description: '每一步从训练集中读取一批 token 序列，作为模型的输入和目标答案。',
        result: 'input / target',
      },
      {
        title: '前向预测',
        description: '模型根据前面的 token 预测下一个 token 的概率分布。',
        result: 'logits 概率',
      },
      {
        title: '计算 loss',
        description: '比较预测概率和真实 token，loss 越低说明模型预测越准确。',
        result: '训练损失',
      },
      {
        title: '反向传播并更新参数',
        description: '根据 loss 计算梯度，优化器把梯度应用到模型权重，模型完成一次学习。',
        result: '新模型权重',
      },
      {
        title: '验证/采样/保存',
        description: '定期用验证集测 loss、生成样本文本、保存 checkpoint。',
        result: 'checkpoint',
      },
    ],
    outputs: [
      'base checkpoint 权重文件',
      '训练日志、loss、tok/s、显存占用',
      '可继续 SFT 或评测的基础语言模型',
    ],
    successResult: '得到一个会续写文本的 base 模型 checkpoint。',
  },
  sft: {
    purpose:
      '用问答/对话数据微调 base 模型，让模型从“会续写”变成“会按用户指令回答”。',
    inputs: [
      '第 3 步训练出的 base checkpoint',
      '对话格式训练数据',
      'tokenizer 的特殊对话 token',
      'SFT 训练步数、batch、验证间隔',
    ],
    process: [
      {
        title: '加载 base 模型',
        description: '读取预训练 checkpoint 作为起点，而不是从随机参数重新开始。',
        result: 'base 权重',
      },
      {
        title: '渲染对话样本',
        description: '把 user/assistant 多轮消息转换成模型能学习的 token 序列。',
        result: '对话 token',
      },
      {
        title: '只监督助手回答',
        description: '训练时重点让模型学习 assistant 部分如何回答，避免学习用户输入本身。',
        result: 'SFT loss mask',
      },
      {
        title: '微调参数',
        description: '继续执行前向、loss、反向传播和参数更新，但目标变成“更像助手回答”。',
        result: '指令跟随能力',
      },
      {
        title: '保存 SFT 模型',
        description: '把微调后的权重保存到 d{depth}_sft 目录，用于聊天和外部评测。',
        result: 'SFT checkpoint',
      },
    ],
    outputs: [
      'SFT checkpoint',
      '更适合问答和聊天的模型',
      '可在前端直接加载对话的模型版本',
    ],
    successResult: '得到一个可以进入聊天页面交互的对话模型。',
  },
  chat: {
    purpose:
      '加载训练好的模型到内存中，使用 tokenizer 和推理引擎进行流式对话。',
    inputs: [
      'base 或 SFT checkpoint',
      'tokenizer.pkl',
      '聊天采样参数（temperature、top-k、重复惩罚等）',
      '用户输入的消息',
    ],
    process: [
      {
        title: '选择 checkpoint',
        description: '从本地已训练模型中选择一个版本，优先建议选择 SFT 模型。',
        result: '模型版本',
      },
      {
        title: '加载模型权重',
        description: '把 safetensors 权重加载到 MLX 模型结构中，并创建推理引擎。',
        result: '内存中的模型',
      },
      {
        title: '渲染对话上下文',
        description: '把用户消息和历史对话转换成模型能理解的 token 序列。',
        result: 'prompt tokens',
      },
      {
        title: '逐 token 生成',
        description: '模型每次预测下一个 token，前端通过 SSE 流式显示生成内容。',
        result: '流式回复',
      },
    ],
    outputs: [
      '可交互的聊天会话',
      '模型实时生成的回答',
      '可调节的推理参数',
    ],
    successResult: '你可以在前端直接和自己训练出来的小模型对话。',
  },
};

export default function PipelinePage() {
  const [steps, setSteps] = useState<PipelineStep[]>(initialSteps);
  const [activeStep, setActiveStep] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [cachedStepIds, setCachedStepIds] = useState<Set<string>>(new Set());
  const [dataPrepConfig, setDataPrepConfig] = useState<DataPrepConfig>(() =>
    loadDataPrepConfig({ freshModelName: true })
  );
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [sampleDatasets, setSampleDatasets] = useState<SampleDataset[]>([]);
  const [editorContent, setEditorContent] = useState('');
  const [editorStatus, setEditorStatus] = useState('');
  const [editorLoading, setEditorLoading] = useState(false);
  const [showDatasetEditor, setShowDatasetEditor] = useState(false);
  const runningStepRef = useRef(false);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const hasRunningStep = steps.some((item) => item.status === 'running');

  const updateDataPrepConfig = useCallback((patch: Partial<DataPrepConfig>) => {
    setDataPrepConfig((prev) => {
      const next = { ...prev, ...patch };
      const saved = loadDataPrepConfig();
      localStorage.setItem('nanochat_config', JSON.stringify({ ...saved, ...next }));
      return next;
    });
  }, []);

  // Check initial status
  useEffect(() => {
    getStatus().then((status) => {
      setSteps((prev) => {
        const next = [...prev];
        const detected = new Set<string>();
        if (status.dataReady) {
          next[0] = { ...next[0], status: 'completed' };
          detected.add(next[0].id);
        }
        if (status.tokenizerReady) {
          next[1] = { ...next[1], status: 'completed' };
          detected.add(next[1].id);
        }
        if (status.modelReady) {
          next[2] = { ...next[2], status: 'completed' };
          detected.add(next[2].id);
        }
        if (status.sftReady) {
          next[3] = { ...next[3], status: 'completed' };
          detected.add(next[3].id);
        }
        setCachedStepIds(detected);
        // Find first pending step
        const firstPending = next.findIndex((s) => s.status === 'pending');
        if (firstPending >= 0) setActiveStep(firstPending);
        return next;
      });
    }).catch(() => {});
  }, []);

  const selectedSampleId =
    dataPrepConfig.data_domain === 'music'
      ? dataPrepConfig.music_dataset
      : dataPrepConfig.text_dataset;
  const selectedSample = sampleDatasets.find((dataset) => dataset.id === selectedSampleId);
  const visibleSampleDatasets = sampleDatasets.filter(
    (dataset) => dataset.data_domain === dataPrepConfig.data_domain
  );

  useEffect(() => {
    getSampleDatasets()
      .then((datasets) => setSampleDatasets(datasets))
      .catch((error) => setEditorStatus(`读取示例数据集失败: ${error instanceof Error ? error.message : String(error)}`));
  }, []);

  useEffect(() => {
    if (dataPrepConfig.data_source !== 'builtin' || selectedSampleId === 'fineweb_edu') {
      setEditorContent('');
      return;
    }
    setEditorLoading(true);
    getSampleDataset(selectedSampleId)
      .then((dataset) => {
        setEditorContent(dataset.content || '');
        setEditorStatus(`已加载: ${dataset.path}`);
      })
      .catch((error) => {
        setEditorContent('');
        setEditorStatus(`读取数据集失败: ${error instanceof Error ? error.message : String(error)}`);
      })
      .finally(() => setEditorLoading(false));
  }, [dataPrepConfig.data_source, selectedSampleId]);

  const saveCurrentSampleDataset = useCallback(async () => {
    if (!selectedSample || selectedSampleId === 'fineweb_edu') return;
    setEditorLoading(true);
    try {
      const saved = await saveSampleDataset(selectedSampleId, editorContent);
      setSampleDatasets((prev) =>
        prev.map((dataset) =>
          dataset.id === selectedSampleId ? { ...dataset, bytes: saved.bytes, path: saved.path } : dataset
        )
      );
      setEditorStatus(`已保存: ${saved.path}`);
    } catch (error) {
      setEditorStatus(`保存失败: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setEditorLoading(false);
    }
  }, [editorContent, selectedSample, selectedSampleId]);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    return () => {
      runningStepRef.current = false;
    };
  }, []);

  const startStep = useCallback(
    async (index: number) => {
      const step = steps[index];

      if (runningStepRef.current || steps.some((item) => item.status === 'running')) {
        setLogs((prev) => [
          ...prev,
          '已有流程步骤正在运行，请等待它完成后再继续下一步。',
        ]);
        return;
      }

      if (step.stage === 'chat') {
        navigate('/chat');
        return;
      }

      // If it's the train step, redirect to training monitor
      if (step.stage === 'train' || step.stage === 'sft') {
        navigate('/training', { state: { autoStart: step.stage } });
        return;
      }

      runningStepRef.current = true;
      setActiveStep(index);
      setSteps((prev) =>
        prev.map((s, i) => {
          if (i === index) return { ...s, status: 'running' };
          if (i > index) return { ...s, status: 'pending', progress: undefined };
          return s;
        })
      );
      setCachedStepIds((prev) => {
        const next = new Set(prev);
        steps.slice(index).forEach((item) => next.delete(item.id));
        return next;
      });
      setLogs([]);

      const params = new URLSearchParams();
      const config = { ...loadDataPrepConfig(), ...dataPrepConfig };
      localStorage.setItem('nanochat_config', JSON.stringify(config));
      if (config.data_domain) {
        params.set('data_domain', String(config.data_domain));
      }
      if (config.data_source) params.set('data_source', String(config.data_source));
      if (config.custom_data_path) params.set('custom_data_path', String(config.custom_data_path));
      if (config.music_data_path) params.set('music_data_path', String(config.music_data_path));
      if (config.tokenizer_method) params.set('tokenizer_method', String(config.tokenizer_method));
      if (config.vocab_size) params.set('vocab_size', String(config.vocab_size));
      if (config.doc_cap) params.set('doc_cap', String(config.doc_cap));
      if (step.stage === 'data' && config.n_shards) {
        params.set('n_shards', String(config.n_shards));
      }

      if (step.stage === 'data') {
        const dataDomain = config.data_domain === 'music' ? 'music' : 'general';
        try {
          if (config.data_source === 'upload') {
            if (!uploadFile) {
              throw new Error('请先选择要上传的数据文件。支持 .jsonl、.csv、.txt、.md。');
            }
            setLogs((prev) => [...prev, `正在上传数据集: ${uploadFile.name}`]);
            const uploaded = await uploadDatasetFile(uploadFile, dataDomain);
            params.set('data_source', 'upload');
            params.set('custom_data_path', uploaded.path);
            if (dataDomain === 'music') params.set('music_data_path', uploaded.path);
            setLogs((prev) => [...prev, `上传完成: ${uploaded.path}`]);
          } else if (config.data_source === 'builtin') {
            const datasetId = dataDomain === 'music' ? config.music_dataset : config.text_dataset;
            const sample = sampleDatasets.find((dataset) => dataset.id === datasetId);
            if (sample) {
              params.set('data_source', 'custom_path');
              params.set('custom_data_path', sample.path);
              if (dataDomain === 'music') params.set('music_data_path', sample.path);
              setLogs((prev) => [...prev, `使用可编辑示例数据集: ${sample.path}`]);
            }
          }
        } catch (error) {
          const message = error instanceof Error ? error.message : '数据集准备失败';
          setSteps((prev) =>
            prev.map((s, i) => (i === index ? { ...s, status: 'error' } : s))
          );
          setLogs((prev) => [...prev, `错误: ${message}`]);
          runningStepRef.current = false;
          return;
        }
      }

      const query = params.toString();
      const url = `/api/run/${step.stage}${query ? `?${query}` : ''}`;
      let heartbeatTimer: number | null = null;
      let stepFailed = false;

      const applyDone = () => {
        setSteps((prev) =>
          prev.map((s, i) => (i === index ? { ...s, status: 'completed' } : s))
        );
        setActiveStep(index + 1);
      };

      const applyError = (text: string) => {
        stepFailed = true;
        setSteps((prev) =>
          prev.map((s, i) => (i === index ? { ...s, status: 'error' } : s))
        );
        setLogs((prev) => [...prev, `错误: ${text}`]);
      };

      const handleMessage = (raw: string) => {
        try {
          const data = JSON.parse(raw);
          if (data.type === 'log' || data.type === 'output') {
            setLogs((prev) => [...prev, data.text]);
          } else if (data.type === 'metric') {
            setLogs((prev) => [...prev, `step ${data.step}/${data.total} | loss ${data.loss} | ${data.tok_per_sec} tok/s`]);
          } else if (data.type === 'done') {
            applyDone();
          } else if (data.type === 'error') {
            applyError(data.text || '流程步骤执行失败');
          }
        } catch {
          if (raw) setLogs((prev) => [...prev, raw]);
        }
      };

      try {
        if (step.stage === 'data' || step.stage === 'tokenizer') {
          const syncUrl = `/api/run_sync/${step.stage}${query ? `?${query}` : ''}`;
          setLogs((prev) => [
            ...prev,
            `正在执行${step.stage === 'data' ? '数据准备' : '分词器训练'}，完成后会一次性展示完整日志...`,
          ]);
          heartbeatTimer = window.setInterval(() => {
            setLogs((prev) => {
              const last = prev[prev.length - 1] || '';
              if (last.includes('流程仍在执行')) return prev;
              return [...prev, '流程仍在执行：数据准备/分词可能需要几十秒，请稍候...'];
            });
          }, 10000);
          const response = await fetch(syncUrl);
          if (heartbeatTimer !== null) {
            window.clearInterval(heartbeatTimer);
            heartbeatTimer = null;
          }
          const result = await response.json().catch(() => ({}));
          if (!response.ok) {
            throw new Error(typeof result.detail === 'string' ? result.detail : `后端返回 ${response.status}`);
          }
          const events = Array.isArray(result.events) ? result.events : [];
          events.forEach((event: RunSyncEvent) => handleMessage(JSON.stringify(event)));
          if (result.status === 'error') {
            if (!stepFailed) applyError('流程步骤执行失败，请查看日志。');
          } else {
            applyDone();
          }
          return;
        }

        const response = await fetch(url);
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
              handleMessage(dataLines.join('\n'));
            }
          }
        }
        const tail = buffer.trim();
        if (tail.startsWith('data:')) {
          handleMessage(tail.slice(5).trimStart());
        }
      } catch (error) {
        setSteps((prev) =>
          prev.map((s, i) => (i === index ? { ...s, status: 'error' } : s))
        );
        setLogs((prev) => [
          ...prev,
          `错误: 与后端的连接断开。${error instanceof Error ? error.message : '请确认后端服务已启动。'}`,
        ]);
      } finally {
        if (heartbeatTimer !== null) {
          window.clearInterval(heartbeatTimer);
        }
        runningStepRef.current = false;
      }
    },
    [steps, navigate, dataPrepConfig, uploadFile, sampleDatasets]
  );

  const skipStep = useCallback((index: number) => {
    setSteps((prev) =>
      prev.map((s, i) => (i === index ? { ...s, status: 'skipped' } : s))
    );
    setActiveStep(index + 1);
  }, []);

  const getRerunLabel = (stage: string) => {
    if (stage === 'train') return '重新训练';
    if (stage === 'sft') return '重新微调';
    if (stage === 'tokenizer') return '重新训练分词器';
    if (stage === 'data') return '重新准备数据';
    return '重新运行';
  };

  const dataPrepPanel = (
    <div className="border border-border rounded-xl bg-surface/70 overflow-hidden">
      <div className="px-3 py-2 border-b border-border flex items-center gap-2">
        <Database className="w-4 h-4 text-primary" />
        <div>
          <div className="text-sm font-medium">数据准备选项</div>
          <p className="text-xs text-text-muted">先选择训练材料，再决定 tokenizer 如何把内容拆成 token。</p>
        </div>
      </div>

      <div className="p-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
        <label className="space-y-1 xl:col-span-2">
          <span className="text-xs font-medium text-text-muted">模型名称</span>
          <input
            type="text"
            value={dataPrepConfig.model_name}
            onChange={(e) => updateDataPrepConfig({ model_name: e.target.value })}
            className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
            placeholder="例如：客服助手-v1、和弦生成实验-0501"
            maxLength={80}
          />
          <p className="text-[11px] text-text-muted">
            训练保存后会显示在模型选择下拉框；不填写时系统会按数据类型和时间自动命名。
          </p>
        </label>

        <label className="space-y-1">
          <span className="text-xs font-medium text-text-muted">数据类型</span>
          <select
            value={dataPrepConfig.data_domain}
            onChange={(e) => updateDataPrepConfig({ data_domain: e.target.value as DataDomain })}
            className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
          >
            <option value="general">文字 / 通用文本</option>
            <option value="music">音乐 / 和弦数据</option>
          </select>
        </label>

        <label className="space-y-1">
          <span className="text-xs font-medium text-text-muted">数据来源</span>
          <select
            value={dataPrepConfig.data_source}
            onChange={(e) => updateDataPrepConfig({ data_source: e.target.value as DataSource })}
            className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
          >
            <option value="builtin">选择内置/推荐数据集</option>
            <option value="custom_path">填写本地数据集路径</option>
            <option value="upload">从浏览器上传数据集</option>
          </select>
        </label>

        {dataPrepConfig.data_source === 'builtin' && dataPrepConfig.data_domain === 'general' && (
          <label className="space-y-1">
            <span className="text-xs font-medium text-text-muted">可选文字数据集</span>
            <select
              value={dataPrepConfig.text_dataset}
              onChange={(e) => updateDataPrepConfig({ text_dataset: e.target.value })}
              className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
            >
              <option value="fineweb_edu">FineWeb-Edu 教育文本（在线下载）</option>
              {visibleSampleDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.label}
                </option>
              ))}
            </select>
            <p className="text-[11px] text-text-muted">
              选择 `sample_datasets` 中的原始数据后，可以在下方直接编辑并保存。
            </p>
          </label>
        )}

        {dataPrepConfig.data_source === 'builtin' && dataPrepConfig.data_domain === 'music' && (
          <label className="space-y-1">
            <span className="text-xs font-medium text-text-muted">可选音乐数据集</span>
            <select
              value={dataPrepConfig.music_dataset}
              onChange={(e) => updateDataPrepConfig({ music_dataset: e.target.value })}
              className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
            >
              {visibleSampleDatasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.label}
                </option>
              ))}
            </select>
            <p className="text-[11px] text-text-muted">
              这些数据来自 `sample_datasets`，可以在下方编辑后直接作为训练输入。
            </p>
          </label>
        )}

        {dataPrepConfig.data_source === 'builtin' && selectedSample && (
          <div className="xl:col-span-2 rounded-xl border border-border bg-surface overflow-hidden">
            <div className="px-3 py-2 border-b border-border flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-xs font-medium">编辑当前数据集：{selectedSample.label}</div>
                <div className="text-[11px] text-text-muted truncate">{selectedSample.path}</div>
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {showDatasetEditor && (
                  <button
                    type="button"
                    onClick={saveCurrentSampleDataset}
                    disabled={editorLoading}
                    className="px-3 py-1.5 rounded-md bg-primary text-white text-xs font-medium hover:bg-primary-dark disabled:opacity-60"
                  >
                    {editorLoading ? '处理中...' : '保存修改'}
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => setShowDatasetEditor((value) => !value)}
                  className="px-3 py-1.5 rounded-md border border-border text-xs text-text-muted hover:text-text hover:bg-surface-lighter"
                >
                  {showDatasetEditor ? '收起' : '编辑数据'}
                </button>
              </div>
            </div>
            {showDatasetEditor && (
              <textarea
                value={editorContent}
                onChange={(e) => setEditorContent(e.target.value)}
                spellCheck={false}
                className="w-full min-h-32 bg-surface p-3 font-mono text-xs text-text focus:outline-none"
                placeholder="这里会显示可编辑的原始数据内容"
              />
            )}
            {editorStatus && (
              <div className="px-3 py-2 border-t border-border text-[11px] text-text-muted">
                {editorStatus}
              </div>
            )}
          </div>
        )}

        {dataPrepConfig.data_source === 'custom_path' && (
          <label className="space-y-1 xl:col-span-2">
            <span className="text-xs font-medium text-text-muted">本地数据集路径</span>
            <input
              type="text"
              value={dataPrepConfig.data_domain === 'music' ? dataPrepConfig.music_data_path : dataPrepConfig.custom_data_path}
              onChange={(e) => {
                const value = e.target.value;
                updateDataPrepConfig(
                  dataPrepConfig.data_domain === 'music'
                    ? { music_data_path: value, custom_data_path: value }
                    : { custom_data_path: value }
                );
              }}
              className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
              placeholder="/Users/you/data/chords.jsonl 或 /Users/you/data/text.txt"
            />
            <p className="text-[11px] text-text-muted">支持 .jsonl、.csv、.txt、.md；后端会转换为训练 parquet 分片。</p>
          </label>
        )}

        {dataPrepConfig.data_source === 'upload' && (
          <label className="space-y-1 xl:col-span-2">
            <span className="text-xs font-medium text-text-muted">上传自定义数据集</span>
            <div className="flex items-center gap-3 rounded-lg border border-dashed border-border bg-surface px-3 py-2">
              <Upload className="w-4 h-4 text-primary" />
              <input
                type="file"
                accept=".jsonl,.csv,.txt,.md"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="text-sm text-text-muted file:mr-3 file:rounded-md file:border-0 file:bg-primary file:px-3 file:py-1.5 file:text-sm file:text-white"
              />
            </div>
            <p className="text-[11px] text-text-muted">
              文件会上传到本机缓存目录，只保存本地路径给训练流程使用。
            </p>
          </label>
        )}

        <details className="xl:col-span-2 rounded-xl border border-border bg-surface px-3 py-2">
          <summary className="cursor-pointer text-xs font-medium text-text-muted hover:text-text">
            高级设置：分片、Tokenizer 与词表
          </summary>
          <div className="mt-3 grid grid-cols-1 xl:grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">数据分片数</span>
              <input
                type="number"
                min={2}
                max={32}
                value={dataPrepConfig.n_shards}
                onChange={(e) => updateDataPrepConfig({ n_shards: Number(e.target.value) })}
                className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
              />
              <p className="text-[11px] text-text-muted">主要用于 FineWeb-Edu 下载；自定义数据会自动切分训练/验证 shard。</p>
            </label>

            <div className="space-y-1">
              <span className="text-xs font-medium text-text-muted flex items-center gap-1.5">
                <Scissors className="w-3.5 h-3.5" />
                Token 拆分方式
              </span>
              <select
                value={dataPrepConfig.tokenizer_method}
                onChange={(e) => updateDataPrepConfig({ tokenizer_method: e.target.value })}
                className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
              >
                <option value="bpe">BPE 字节对编码（当前可训练）</option>
                <option value="byte" disabled>Byte-level 字节级（展示中，暂未启用）</option>
                <option value="char" disabled>Char-level 字符级（展示中，暂未启用）</option>
              </select>
              <p className="text-[11px] text-text-muted">BPE 会把高频字符片段合并成 token，兼顾中文、英文、符号与和弦记号。</p>
            </div>

            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">词表大小</span>
              <input
                type="number"
                min={2048}
                max={65536}
                step={1024}
                value={dataPrepConfig.vocab_size}
                onChange={(e) => updateDataPrepConfig({ vocab_size: Number(e.target.value) })}
                className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
              />
              <p className="text-[11px] text-text-muted">越大表达更细，但小数据集容易浪费；和弦小数据可适当调低。</p>
            </label>

            <label className="space-y-1">
              <span className="text-xs font-medium text-text-muted">单文档最大字符数</span>
              <input
                type="number"
                min={512}
                max={100000}
                step={512}
                value={dataPrepConfig.doc_cap}
                onChange={(e) => updateDataPrepConfig({ doc_cap: Number(e.target.value) })}
                className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary"
              />
              <p className="text-[11px] text-text-muted">训练 tokenizer 前会截断超长文档，避免单篇文本主导分词规则。</p>
            </label>
          </div>
        </details>
      </div>
    </div>
  );

  return (
    <div className="px-5 py-6 lg:px-6 w-full">
      {/* Header */}
      <div className="mb-5">
        <div className="flex items-center gap-3 mb-2">
          <Sparkles className="w-6 h-6 text-primary" />
          <h1 className="text-2xl font-bold">AI 模型训练流程</h1>
        </div>
        <p className="text-text-muted">
          按照以下步骤，从零开始训练属于您自己的 AI 语言模型。若本机已有缓存产物，步骤会标记为“已检测到本地缓存”，您可以继续使用或重新运行。
        </p>
      </div>

      {/* Info banner */}
      <div className="mb-4 p-3 bg-info/10 border border-info/20 rounded-xl flex items-start gap-3">
        <BookOpen className="w-5 h-5 text-info shrink-0 mt-0.5" />
        <div className="text-sm">
          <p className="font-medium text-info">新手指南</p>
          <p className="text-text-muted mt-1">
            训练一个 AI 模型就像培养一个学生：先给它教材（数据），再教它识字（分词器），然后让它大量阅读学习（预训练），最后教它如何正确回答问题（SFT微调）。鼠标悬停在每个步骤的 
            <span className="text-primary"> ⓘ </span>
            图标上可以查看详细说明。
          </p>
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {steps.map((step, index) => (
          <StepCard
            key={step.id}
            name={step.name}
            description={step.description}
            detailedHelp={step.detailedHelp}
            detail={stepDetails[step.id]}
            logs={index === activeStep ? logs : []}
            status={step.status}
            stepNumber={index + 1}
            isActive={index === activeStep}
            onStart={() => startStep(index)}
            onSkip={step.stage === 'sft' ? () => skipStep(index) : undefined}
            detectedFromCache={cachedStepIds.has(step.id)}
            rerunLabel={getRerunLabel(step.stage)}
            extra={step.id === 'data' ? dataPrepPanel : undefined}
            disabled={
              (hasRunningStep && step.status !== 'running') ||
              (
                index > 0 &&
                steps[index - 1].status !== 'completed' &&
                steps[index - 1].status !== 'skipped'
              )
            }
          />
        ))}
      </div>

      {/* Log output */}
      {logs.length > 0 && (
        <div className="mt-4 bg-surface-light border border-border rounded-xl overflow-hidden">
          <div className="px-4 py-2.5 border-b border-border text-xs font-medium text-text-muted">
            运行日志
          </div>
          <div className="p-4 max-h-64 overflow-y-auto font-mono text-xs text-text-muted space-y-0.5">
            {logs.map((log, i) => (
              <div key={i}>{log}</div>
            ))}
            <div ref={logsEndRef} />
          </div>
        </div>
      )}
    </div>
  );
}

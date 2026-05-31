import { useState } from 'react';
import { Settings, Save, RotateCcw, Layers, Cpu, Database, Gauge } from 'lucide-react';
import ParamTooltip from '@/components/ParamTooltip';
import { cn } from '@/lib/utils';
import type { ParamConfig } from '@/types';

const paramConfigs: ParamConfig[] = [
  {
    key: 'depth',
    label: '模型深度 (Depth)',
    description: '控制模型整体规模的"旋钮"。后端会根据 depth 自动推导层数、嵌入维度和注意力头数。深度越大，能力越强，但训练时间和内存需求也越高。',
    type: 'slider',
    default: 4,
    min: 1,
    max: 26,
    step: 1,
    group: 'model',
  },
  {
    key: 'max_seq_len',
    label: '序列长度 (Max Seq Len)',
    description: '模型一次能处理的最大 token 数量，也就是模型的上下文窗口。更长的序列能理解更长上下文，但需要更多内存和计算。',
    type: 'select',
    options: [
      { value: 256, label: '256' },
      { value: 512, label: '512' },
      { value: 1024, label: '1024' },
      { value: 2048, label: '2048' },
    ],
    default: 512,
    group: 'model',
  },
  {
    key: 'window_pattern',
    label: '注意力窗口模式',
    description: '控制注意力计算方式。"S" 表示滑动窗口，"L" 表示全局注意力。必须使用后端支持的 S/L 字符串。',
    type: 'select',
    options: [
      { value: 'L', label: 'L (全局)' },
      { value: 'S', label: 'S (滑动)' },
      { value: 'SSSL', label: 'SSSL (混合)' },
    ],
    default: 'L',
    group: 'model',
  },
  {
    key: 'device_batch_size',
    label: '设备批大小',
    description: '每个设备一次处理的样本数。增大可提升吞吐，但更占内存；内存不足时优先调小这个值。',
    type: 'number',
    default: 1,
    min: 1,
    max: 16,
    step: 1,
    group: 'training',
  },
  {
    key: 'num_iterations',
    label: '训练步数',
    description: '总共训练多少步。之前默认自动模式在小模型上只跑 50 步，会很快结束；现在默认 500 步，让您能更清楚地观察 loss、速度和保存过程。',
    type: 'number',
    default: 500,
    min: -1,
    max: 100000,
    step: 100,
    group: 'training',
  },
  {
    key: 'save_every',
    label: '保存间隔',
    description: '每隔多少训练步保存一个 checkpoint。设为 -1 时只在训练结束保存。',
    type: 'number',
    default: 500,
    min: -1,
    max: 10000,
    step: 100,
    group: 'training',
  },
  {
    key: 'eval_every',
    label: '验证间隔',
    description: '每隔多少步运行一次验证。较小的值反馈更及时，但训练会更慢。',
    type: 'number',
    default: 100,
    min: -1,
    max: 10000,
    step: 100,
    group: 'training',
  },
  {
    key: 'sample_every',
    label: '样本生成间隔',
    description: '预训练阶段每隔多少步让当前模型生成文本样本。设为 10 可以更快看到模型输出变化；SFT 当前主要展示 loss，暂不生成样本。',
    type: 'number',
    default: 10,
    min: -1,
    max: 10000,
    step: 10,
    group: 'training',
  },
  {
    key: 'use_simple_adamw',
    label: '优化器',
    description: '默认使用 Muon + AdamW 混合优化器。选择 AdamW 会切换到更简单、更保守的优化方式。',
    type: 'select',
    options: [
      { value: 'false', label: 'Muon + AdamW (默认)' },
      { value: 'true', label: 'AdamW' },
    ],
    default: 'false',
    group: 'optimizer',
  },
  {
    key: 'data_domain',
    label: '数据类型',
    description: '选择训练数据来源。通用文本保持原 FineWeb-Edu 流程；音乐和弦会使用独立缓存目录，不影响原有模型与数据。',
    type: 'select',
    options: [
      { value: 'general', label: '通用文本 (FineWeb-Edu)' },
      { value: 'music', label: '音乐/和弦数据' },
    ],
    default: 'general',
    group: 'data',
  },
  {
    key: 'n_shards',
    label: '数据分片数',
    description: '下载多少个 FineWeb-Edu 数据分片。数量越多数据越丰富，但下载和预处理时间越长。',
    type: 'number',
    default: 4,
    min: 2,
    max: 32,
    step: 1,
    group: 'data',
  },
  {
    key: 'music_data_path',
    label: '音乐数据文件路径',
    description: '本地音乐/和弦数据文件路径，支持 .jsonl、.csv、.txt、.md。仅在数据类型选择“音乐/和弦数据”时使用。',
    type: 'text',
    default: '',
    group: 'data',
  },
  {
    key: 'memory_limit_gb',
    label: '内存限制 (GB)',
    description: '限制 MLX 使用的内存大小。机器内存较小或同时运行其他程序时，建议使用保守值。',
    type: 'slider',
    default: 8,
    min: 2,
    max: 64,
    step: 1,
    group: 'optimizer',
  },
];

const groupInfo = {
  model: { icon: Layers, label: '模型架构', description: '定义模型的结构与规模' },
  training: { icon: Cpu, label: '训练超参数', description: '控制训练过程的行为' },
  data: { icon: Database, label: '数据配置', description: '训练数据的设置' },
  optimizer: { icon: Gauge, label: '优化器配置', description: '参数更新策略' },
};

const defaultValues = () => Object.fromEntries(paramConfigs.map((p) => [p.key, p.default]));
type ConfigValue = string | number | boolean;

const loadSavedValues = () => {
  const defaults = defaultValues();
  try {
    const saved = localStorage.getItem('nanochat_config');
    return saved ? { ...defaults, ...(JSON.parse(saved) as Record<string, ConfigValue>) } : defaults;
  } catch {
    return defaults;
  }
};

export default function ConfigPage() {
  const [values, setValues] = useState<Record<string, ConfigValue>>(loadSavedValues);
  const [saved, setSaved] = useState(false);
  const [hasSavedConfig, setHasSavedConfig] = useState(() => !!localStorage.getItem('nanochat_config'));

  const updateValue = (key: string, value: ConfigValue) => {
    setValues((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const resetDefaults = () => {
    const defaults = defaultValues();
    setValues(defaults);
    localStorage.setItem('nanochat_config', JSON.stringify(defaults));
    setHasSavedConfig(true);
    setSaved(false);
  };

  const saveConfig = () => {
    localStorage.setItem('nanochat_config', JSON.stringify(values));
    setHasSavedConfig(true);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const groups = ['model', 'training', 'data', 'optimizer'] as const;

  return (
    <div className="px-5 py-6 lg:px-6 w-full">
      <div className="flex items-center justify-between mb-5">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Settings className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">参数配置</h1>
          </div>
          <p className="text-text-muted">
            配置模型架构与训练参数。悬停在参数名旁的图标上查看详细说明。
          </p>
          <p className="text-xs text-success mt-2">
            {hasSavedConfig
              ? '已加载本地保存的训练配置，下一次训练会使用这些参数。'
              : '当前显示默认配置，保存后会用于下一次训练。'}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={resetDefaults}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm text-text-muted hover:text-text hover:bg-surface-lighter transition-all"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            重置默认
          </button>
          <button
            onClick={saveConfig}
            className={cn(
              'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              saved
                ? 'bg-success/20 text-success'
                : 'bg-primary text-white hover:bg-primary-dark'
            )}
          >
            <Save className="w-3.5 h-3.5" />
            {saved ? '已保存' : '保存配置'}
          </button>
        </div>
      </div>

      <div className="space-y-5">
        {groups.map((group) => {
          const info = groupInfo[group];
          const params = paramConfigs.filter((p) => p.group === group);

          return (
            <div key={group} className="bg-surface-light border border-border rounded-xl overflow-hidden">
              <div className="px-4 py-3 border-b border-border flex items-center gap-2.5">
                <info.icon className="w-4.5 h-4.5 text-primary" />
                <div>
                  <h2 className="text-sm font-semibold">{info.label}</h2>
                  <p className="text-xs text-text-muted">{info.description}</p>
                </div>
              </div>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-x-6 gap-y-3 p-4">
                {params.map((param) => (
                  <div key={param.key} className="flex items-center gap-3 min-w-0">
                    <div className="w-40 shrink-0">
                      <div className="flex items-center gap-1.5">
                        <label className="truncate text-sm font-medium">{param.label}</label>
                        <ParamTooltip content={param.description} />
                      </div>
                    </div>
                    <div className="min-w-0 flex-1">
                      {param.type === 'select' && (
                        <select
                          value={String(values[param.key] ?? '')}
                          onChange={(e) => updateValue(param.key, e.target.value)}
                          className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary transition-colors"
                        >
                          {param.options?.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      )}
                      {param.type === 'number' && (
                        <input
                          type="number"
                          value={Number(values[param.key] ?? 0)}
                          onChange={(e) => updateValue(param.key, Number(e.target.value))}
                          min={param.min}
                          max={param.max}
                          step={param.step}
                          className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary transition-colors"
                        />
                      )}
                      {param.type === 'text' && (
                        <input
                          type="text"
                          value={String(values[param.key] ?? '')}
                          onChange={(e) => updateValue(param.key, e.target.value)}
                          className="w-full bg-surface border border-border rounded-lg px-3 py-1.5 text-sm text-text focus:outline-none focus:border-primary transition-colors"
                          placeholder="/Users/you/data/chords.jsonl"
                        />
                      )}
                      {param.type === 'slider' && (
                        <div className="flex items-center gap-4">
                          <input
                            type="range"
                            value={Number(values[param.key] ?? 0)}
                            onChange={(e) => updateValue(param.key, Number(e.target.value))}
                            min={param.min}
                            max={param.max}
                            step={param.step}
                            className="flex-1 accent-primary"
                          />
                          <span className="text-sm font-mono text-text-muted w-16 text-right">
                            {values[param.key]}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

import { useState, useRef } from 'react';
import {
  FlaskConical,
  Play,
  Loader2,
  Award,
  ExternalLink,
  Key,
  Globe,
  Bot,
  AlertCircle,
  Download,
} from 'lucide-react';
import ParamTooltip from '@/components/ParamTooltip';
import { cn } from '@/lib/utils';
import type { EvalResult, ExternalEvalResult, ExternalEvalConfig, Checkpoint } from '@/types';
import { exportReportFile, getCheckpoints } from '@/api/client';
import { useEffect } from 'react';

const benchmarks = [
  {
    id: 'arc_easy',
    name: 'ARC-Easy',
    description: '小学科学选择题，测试基础推理能力',
  },
  {
    id: 'arc_challenge',
    name: 'ARC-Challenge',
    description: '困难版科学选择题，需要更深层推理',
  },
  {
    id: 'mmlu',
    name: 'MMLU',
    description: '大规模多任务语言理解，覆盖 57 个学科领域',
  },
  {
    id: 'gsm8k',
    name: 'GSM8K',
    description: '小学数学应用题，测试多步推理和计算能力',
  },
  {
    id: 'humaneval',
    name: 'HumanEval',
    description: 'Python 代码生成测试，评估编程能力',
  },
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
  setTimeout(() => URL.revokeObjectURL(url), 3000);
}

export default function EvalPage() {
  const [activeTab, setActiveTab] = useState<'benchmark' | 'external'>('benchmark');
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>(['arc_easy']);
  const [benchmarkResults, setBenchmarkResults] = useState<EvalResult[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [benchmarkError, setBenchmarkError] = useState('');
  const [exportMessage, setExportMessage] = useState('');

  // External eval state
  const [externalConfig, setExternalConfig] = useState<ExternalEvalConfig>({
    provider: 'openai',
    apiKey: '',
    endpoint: '',
    model: 'gpt-4o-mini',
  });
  const [evalPrompts, setEvalPrompts] = useState<string>(
    '请解释什么是机器学习？\n写一首关于春天的诗。\n1+1等于几？为什么？'
  );
  const [externalResults, setExternalResults] = useState<ExternalEvalResult[]>([]);
  const [isExternalRunning, setIsExternalRunning] = useState(false);
  const [externalError, setExternalError] = useState('');

  const esRef = useRef<EventSource | null>(null);

  const closeBenchmarkStream = () => {
    esRef.current?.close();
    esRef.current = null;
  };

  const exportReport = async (filename: string, report: string) => {
    setExportMessage('');
    try {
      const saved = await exportReportFile(filename, report);
      const link = document.createElement('a');
      link.href = saved.download_url;
      link.download = saved.filename;
      link.rel = 'noopener';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setExportMessage(`报告已保存到本地项目: ${saved.path}`);
    } catch (error) {
      downloadTextFile(filename, report);
      setExportMessage(
        `后端保存失败，已尝试浏览器下载: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  };

  useEffect(() => {
    getCheckpoints()
      .then((cps) => {
        setCheckpoints(cps);
        if (cps.length > 0) setSelectedCheckpoint(cps[0].path);
      })
      .catch(() => {});
    return () => closeBenchmarkStream();
  }, []);

  const toggleBenchmark = (id: string) => {
    setSelectedBenchmarks((prev) =>
      prev.includes(id) ? prev.filter((b) => b !== id) : [...prev, id]
    );
  };

  const runBenchmarks = () => {
    if (!selectedCheckpoint || selectedBenchmarks.length === 0) return;
    closeBenchmarkStream();
    setIsRunning(true);
    setBenchmarkResults([]);
    setLogs([]);
    setBenchmarkError('');

    const params = new URLSearchParams({
      checkpoint: selectedCheckpoint,
      benchmarks: selectedBenchmarks.join(','),
    });

    let streamFinished = false;
    const es = new EventSource(`/api/eval/benchmark?${params.toString()}`);
    esRef.current = es;

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'result') {
          setBenchmarkResults((prev) => [...prev, data.result]);
        } else if (data.type === 'log') {
          setLogs((prev) => [...prev, data.text]);
        } else if (data.type === 'error') {
          streamFinished = true;
          const text = data.text || '标准评测失败，请查看日志。';
          setBenchmarkError(text);
          setLogs((prev) => [...prev, `错误: ${text}`]);
          setIsRunning(false);
          closeBenchmarkStream();
        } else if (data.type === 'done') {
          streamFinished = true;
          setIsRunning(false);
          closeBenchmarkStream();
        }
      } catch {
        setLogs((prev) => [...prev, e.data]);
      }
    };

    es.onerror = () => {
      if (streamFinished || es.readyState === EventSource.CLOSED) {
        return;
      }
      const text = '评测连接中断：请确认后端服务正在运行，或查看后端终端中的错误日志。';
      setBenchmarkError(text);
      setLogs((prev) => [...prev, `错误: ${text}`]);
      setIsRunning(false);
      closeBenchmarkStream();
    };
  };

  const runExternalEval = async () => {
    if (!selectedCheckpoint || !externalConfig.apiKey) return;
    setIsExternalRunning(true);
    setExternalResults([]);
    setExternalError('');

    try {
      const prompts = evalPrompts.split('\n').filter((p) => p.trim());
      const res = await fetch('/api/eval/external', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          checkpoint: selectedCheckpoint,
          config: externalConfig,
          prompts,
        }),
      });
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        const detail = typeof errorData.detail === 'string' ? errorData.detail : res.statusText;
        throw new Error(detail || `外部评测请求失败 (${res.status})`);
      }
      const data = await res.json();
      setExternalResults(data.results || []);
    } catch (err) {
      setExternalError(err instanceof Error ? err.message : '外部模型评测失败，请检查 API Key、Endpoint 和后端日志。');
    } finally {
      setIsExternalRunning(false);
    }
  };

  const exportBenchmarkReport = () => {
    const report = [
      '# miniGPT Studio 标准评测报告',
      '',
      `- 导出时间：${new Date().toLocaleString()}`,
      `- Checkpoint：${selectedCheckpoint || '-'}`,
      `- 评测任务：${selectedBenchmarks.join(', ')}`,
      '',
      '## 结果汇总',
      '',
      '| Benchmark | Score | Total | Accuracy |',
      '|---|---:|---:|---:|',
      ...benchmarkResults.map((result) =>
        `| ${result.benchmark} | ${result.score} | ${result.total} | ${(result.accuracy * 100).toFixed(2)}% |`
      ),
      '',
      '## 运行日志',
      '',
      '```text',
      ...logs.slice(-120),
      '```',
    ].join('\n');
    void exportReport(`minigpt-studio-benchmark-report-${Date.now()}.md`, report);
  };

  const exportExternalReport = () => {
    const safeConfig = {
      ...externalConfig,
      apiKey: externalConfig.apiKey ? `${externalConfig.apiKey.slice(0, 4)}***` : '',
    };
    const report = [
      '# miniGPT Studio 外部模型对比评测报告',
      '',
      `- 导出时间：${new Date().toLocaleString()}`,
      `- Checkpoint：${selectedCheckpoint || '-'}`,
      `- 外部模型：${externalConfig.model}`,
      `- Endpoint：${externalConfig.endpoint || 'https://api.openai.com/v1'}`,
      `- 评测策略：本地模型与外部模型分别回答，外部模型按 1-10 分评分`,
      '',
      '## 评测配置',
      '',
      '```json',
      JSON.stringify(safeConfig, null, 2),
      '```',
      '',
      '## 逐题结果',
      '',
      ...externalResults.map((result, index) => [
        `### ${index + 1}. ${result.prompt}`,
        '',
        `- 评分：${result.externalScore}/10`,
        `- 评语：${result.externalFeedback || '-'}`,
        '',
        '#### 本地模型回答',
        '',
        '```text',
        result.localResponse || '',
        '```',
        '',
        '#### 外部模型回答',
        '',
        '```text',
        result.externalResponse || '',
        '```',
      ].join('\n')),
    ].join('\n');
    void exportReport(`minigpt-studio-external-eval-report-${Date.now()}.md`, report);
  };

  return (
    <div className="px-5 py-6 lg:px-6 w-full">
      <div className="mb-6">
        <div className="flex items-center gap-3 mb-2">
          <FlaskConical className="w-6 h-6 text-primary" />
          <h1 className="text-2xl font-bold">模型评测</h1>
        </div>
        <p className="text-text-muted">
          使用标准基准测试或接入外部模型对您训练的模型进行全方位评估
        </p>
      </div>

      {/* Checkpoint selector */}
      <div className="mb-6 bg-surface-light border border-border rounded-xl p-4">
        <div className="flex items-center gap-3">
          <label className="text-sm font-medium shrink-0">选择检查点</label>
          <ParamTooltip content="选择要评测的模型检查点。每个检查点对应训练到某一步的模型快照。" />
          <select
            value={selectedCheckpoint}
            onChange={(e) => setSelectedCheckpoint(e.target.value)}
            className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
          >
            {checkpoints.map((cp) => (
              <option key={cp.path} value={cp.path}>
                {cp.name} (depth={cp.depth}, step={cp.step}, {cp.isSft ? 'SFT' : '预训练'})
              </option>
            ))}
            {checkpoints.length === 0 && <option>暂无可用检查点</option>}
          </select>
        </div>
      </div>

      {exportMessage && (
        <div className="mb-6 rounded-xl border border-success/30 bg-success/10 px-4 py-3 text-sm text-success">
          {exportMessage}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-6 bg-surface-light border border-border rounded-lg p-1">
        <button
          onClick={() => setActiveTab('benchmark')}
          className={cn(
            'flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-all flex-1 justify-center',
            activeTab === 'benchmark'
              ? 'bg-primary/15 text-primary font-medium'
              : 'text-text-muted hover:text-text'
          )}
        >
          <Award className="w-4 h-4" />
          标准基准测试
        </button>
        <button
          onClick={() => setActiveTab('external')}
          className={cn(
            'flex items-center gap-1.5 px-4 py-2 rounded-md text-sm transition-all flex-1 justify-center',
            activeTab === 'external'
              ? 'bg-primary/15 text-primary font-medium'
              : 'text-text-muted hover:text-text'
          )}
        >
          <ExternalLink className="w-4 h-4" />
          外部模型对比评测
        </button>
      </div>

      {/* Benchmark tab */}
      {activeTab === 'benchmark' && (
        <div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-6">
            {benchmarks.map((b) => (
              <button
                key={b.id}
                onClick={() => toggleBenchmark(b.id)}
                className={cn(
                  'p-4 rounded-xl border text-left transition-all',
                  selectedBenchmarks.includes(b.id)
                    ? 'border-primary/50 bg-primary/5'
                    : 'border-border bg-surface-light hover:border-surface-lighter'
                )}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="font-medium text-sm">{b.name}</span>
                  <div
                    className={cn(
                      'w-4 h-4 rounded border flex items-center justify-center',
                      selectedBenchmarks.includes(b.id)
                        ? 'bg-primary border-primary'
                        : 'border-border'
                    )}
                  >
                    {selectedBenchmarks.includes(b.id) && (
                      <span className="text-white text-xs">✓</span>
                    )}
                  </div>
                </div>
                <p className="text-xs text-text-muted">{b.description}</p>
              </button>
            ))}
          </div>

          <div className="flex gap-2 mb-6">
            <button
              onClick={runBenchmarks}
              disabled={isRunning || !selectedCheckpoint || selectedBenchmarks.length === 0}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all',
                isRunning
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary-dark'
              )}
            >
              {isRunning ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isRunning ? '评测中...' : '开始评测'}
            </button>
            <button
              onClick={exportBenchmarkReport}
              disabled={benchmarkResults.length === 0}
              className={cn(
                'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all',
                benchmarkResults.length === 0
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-surface-light border border-border text-text hover:bg-surface-lighter'
              )}
            >
              <Download className="w-4 h-4" />
              导出报告
            </button>
          </div>

          {benchmarkError && (
            <div className="mb-6 flex gap-2 rounded-xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-medium">标准评测未完成</div>
                <div className="text-error/90">{benchmarkError}</div>
              </div>
            </div>
          )}

          {/* Results */}
          {benchmarkResults.length > 0 && (
            <div className="bg-surface-light border border-border rounded-xl overflow-hidden">
              <div className="px-5 py-3 border-b border-border">
                <h3 className="text-sm font-semibold">评测结果</h3>
              </div>
              <div className="divide-y divide-border">
                {benchmarkResults.map((result, i) => (
                  <div key={i} className="px-5 py-4 flex items-center justify-between">
                    <div>
                      <div className="font-medium text-sm">{result.benchmark}</div>
                      <div className="text-xs text-text-muted mt-0.5">
                        {result.score}/{result.total} 正确
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-primary">
                        {(result.accuracy * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Logs */}
          {logs.length > 0 && (
            <div className="mt-4 bg-surface border border-border rounded-xl p-4 max-h-48 overflow-y-auto font-mono text-xs text-text-muted space-y-0.5">
              {logs.map((log, i) => (
                <div key={i}>{log}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* External eval tab */}
      {activeTab === 'external' && (
        <div className="space-y-6">
          <div className="bg-surface-light border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
              <Bot className="w-4 h-4 text-primary" />
              外部模型配置
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="flex items-center gap-1.5 text-sm text-text-muted mb-1.5">
                  <Globe className="w-3.5 h-3.5" />
                  服务商
                  <ParamTooltip content="选择您要对接的 AI 服务商。选择自定义可以填写任意 OpenAI 兼容 API 的地址。" />
                </label>
                <select
                  value={externalConfig.provider}
                  onChange={(e) =>
                    setExternalConfig((prev) => ({
                      ...prev,
                      provider: e.target.value as ExternalEvalConfig['provider'],
                    }))
                  }
                  className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
                >
                  <option value="openai">OpenAI</option>
                  <option value="custom">自定义 (OpenAI 兼容)</option>
                </select>
              </div>
              <div>
                <label className="flex items-center gap-1.5 text-sm text-text-muted mb-1.5">
                  模型名称
                  <ParamTooltip content="要使用的模型标识符，如 gpt-4o-mini, claude-3-haiku 等" />
                </label>
                <input
                  type="text"
                  value={externalConfig.model}
                  onChange={(e) =>
                    setExternalConfig((prev) => ({ ...prev, model: e.target.value }))
                  }
                  className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
                  placeholder="gpt-4o-mini"
                />
              </div>
              <div>
                <label className="flex items-center gap-1.5 text-sm text-text-muted mb-1.5">
                  <Key className="w-3.5 h-3.5" />
                  API Key
                  <ParamTooltip content="您的 API 密钥，仅在本地使用，不会上传到任何服务器" />
                </label>
                <input
                  type="password"
                  value={externalConfig.apiKey}
                  onChange={(e) =>
                    setExternalConfig((prev) => ({ ...prev, apiKey: e.target.value }))
                  }
                  className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
                  placeholder="sk-..."
                />
              </div>
              {externalConfig.provider === 'custom' && (
                <div>
                  <label className="flex items-center gap-1.5 text-sm text-text-muted mb-1.5">
                    API Endpoint
                    <ParamTooltip content="自定义 API 地址，需要兼容 OpenAI 的 /v1/chat/completions 接口" />
                  </label>
                  <input
                    type="text"
                    value={externalConfig.endpoint}
                    onChange={(e) =>
                      setExternalConfig((prev) => ({
                        ...prev,
                        endpoint: e.target.value,
                      }))
                    }
                    className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
                    placeholder="https://api.example.com/v1"
                  />
                </div>
              )}
            </div>
          </div>

          <div className="bg-surface-light border border-border rounded-xl p-5">
            <h3 className="text-sm font-semibold mb-3">评测提示词</h3>
            <p className="text-xs text-text-muted mb-3">
              每行一个提示词。系统会分别用您训练的模型和外部模型回答这些问题，然后让外部模型对比评分。
            </p>
            <textarea
              value={evalPrompts}
              onChange={(e) => setEvalPrompts(e.target.value)}
              rows={5}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text font-mono resize-y"
              placeholder="每行输入一个评测提示词..."
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={runExternalEval}
              disabled={isExternalRunning || !externalConfig.apiKey || !selectedCheckpoint}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all',
                isExternalRunning
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary-dark'
              )}
            >
              {isExternalRunning ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isExternalRunning ? '评测中...' : '开始对比评测'}
            </button>
            <button
              onClick={exportExternalReport}
              disabled={externalResults.length === 0}
              className={cn(
                'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all',
                externalResults.length === 0
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-surface-light border border-border text-text hover:bg-surface-lighter'
              )}
            >
              <Download className="w-4 h-4" />
              导出报告
            </button>
          </div>

          {externalError && (
            <div className="flex gap-2 rounded-xl border border-error/30 bg-error/10 px-4 py-3 text-sm text-error">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div>
                <div className="font-medium">外部模型评测失败</div>
                <div className="text-error/90">{externalError}</div>
              </div>
            </div>
          )}

          {/* Results */}
          {externalResults.length > 0 && (
            <div className="space-y-4">
              {externalResults.map((result, i) => (
                <div
                  key={i}
                  className="bg-surface-light border border-border rounded-xl overflow-hidden"
                >
                  <div className="px-5 py-3 border-b border-border flex items-center justify-between">
                    <span className="text-sm font-medium">提示词: {result.prompt}</span>
                    <span
                      className={cn(
                        'px-2 py-0.5 rounded-full text-xs font-medium',
                        result.externalScore >= 7
                          ? 'bg-success/20 text-success'
                          : result.externalScore >= 4
                          ? 'bg-warning/20 text-warning'
                          : 'bg-error/20 text-error'
                      )}
                    >
                      {result.externalScore}/10
                    </span>
                  </div>
                  <div className="grid grid-cols-2 divide-x divide-border">
                    <div className="p-4">
                      <div className="text-xs text-primary font-medium mb-2">
                        🤖 您的模型
                      </div>
                      <p className="text-sm text-text-muted whitespace-pre-wrap">
                        {result.localResponse}
                      </p>
                    </div>
                    <div className="p-4">
                      <div className="text-xs text-success font-medium mb-2">
                        🌐 {externalConfig.model}
                      </div>
                      <p className="text-sm text-text-muted whitespace-pre-wrap">
                        {result.externalResponse}
                      </p>
                    </div>
                  </div>
                  {result.externalFeedback && (
                    <div className="px-5 py-3 border-t border-border bg-surface">
                      <div className="text-xs text-text-muted">
                        <span className="font-medium text-warning">评语:</span>{' '}
                        {result.externalFeedback}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

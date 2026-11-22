import { useEffect, useState } from 'react';
import { Link, useLocation, useSearchParams } from 'react-router-dom';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getAllCheckpoints } from '@/api/client';
import type { Checkpoint } from '@/types';

type Question = {
  prompt: string;
  expected: Record<string, string | number> | null;
};
type Answer = { text: string; valid: boolean; exact: boolean | null };
type Model = {
  checkpoint_id: string;
  model_name: string;
  step: number;
  depth: number;
  weights: string;
  weight_hash: string;
  tokenizer_hash: string;
};
type Summary = {
  id: string;
  kind: 'compare' | 'sft' | 'training';
  status: string;
  stage: string;
  training_stage?: string;
  domain?: string;
  created: number;
};
type Assessment = { at: number; answers: Answer[] };
type Experiment = Summary & {
  base: Model | null;
  target: Model | null;
  before: Assessment | null;
  after: Assessment | null;
  config: { questions: Question[]; steps: number };
  ended?: number;
  error?: string;
  logs?: string[];
  data_file: string;
  data_hash: string;
  timings: Record<string, number>;
  curve: { step: number; loss: number }[];
  parameters: Record<string, unknown>;
  versions: Record<string, string>;
  generation: Record<string, number>;
  device: { chip: string };
  peak_mb: number;
  rule: string;
};
type Preset = {
  questions: Question[];
  count: number;
  sample: { prompt: string; completion: string };
  steps: number;
};
const section =
  'rounded-xl border border-border bg-surface-light p-4 space-y-3';
const input =
  'w-full rounded-lg bg-surface border border-border px-3 py-2 text-sm';
const button =
  'rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white disabled:opacity-50';
const stages: Record<string, string> = {
  train: '预训练',
  queued: '准备启动',
  prepare: '准备数据与分词器',
  pretrain: '预训练示例基座',
  before: '基线评测',
  sft: '微调',
  after: '后评测',
  done: '完成',
};
const time = (t: number) => new Date(t * 1000).toLocaleString();
const model = (m: Model | null) =>
  m
    ? (m.model_name || '未命名模型') + ' · d' + m.depth + ' · step ' + m.step
    : '尚未生成';
const title = (r: Summary) =>
  (r.kind === 'training'
    ? r.training_stage === 'sft'
      ? '常规微调'
      : '常规预训练'
    : r.kind === 'sft'
      ? '微调实验'
      : '模型对比') +
  ' · ' +
  time(r.created) +
  ' · ' +
  r.id.slice(0, 6);

async function api<T>(path: string, post?: object): Promise<T> {
  const response = await fetch(
    '/api/experiments' + path,
    post
      ? {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(post),
        }
      : {},
  );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === 'string'
        ? data.detail
        : '请求失败，请检查输入与后端日志。',
    );
  return data as T;
}

function NewExperiment({
  preset,
  models,
  compare,
  launch,
  busy,
}: {
  preset: Preset;
  models: Checkpoint[];
  compare: boolean;
  busy: boolean;
  launch: (path: string, body: object) => void;
}) {
  const bases = models.filter((m) => compare || !m.isSft);
  const [base, setBase] = useState(bases[0]?.path || '');
  const [target, setTarget] = useState(
    models.find((m) => m.path !== base)?.path || '',
  );
  const [steps, setSteps] = useState(preset.steps);
  const [rows, setRows] = useState(
    preset.questions.map((q) => ({
      prompt: q.prompt,
      expected: JSON.stringify(q.expected),
    })),
  );
  const [error, setError] = useState('');
  function submit() {
    try {
      const questions = rows.map((row) => ({
        prompt: row.prompt.trim(),
        expected: row.expected.trim() ? JSON.parse(row.expected) : null,
      }));
      if (
        questions.some(
          (q) =>
            !q.prompt ||
            (q.expected !== null &&
              (typeof q.expected !== 'object' || Array.isArray(q.expected))),
        )
      )
        throw new Error('请填写题目，并使用 JSON 对象作为标准答案。');
      setError('');
      launch('', { questions, base, target: compare ? target : '', steps });
    } catch (e) {
      setError(String(e));
    }
  }
  const options = (list: Checkpoint[]) =>
    list.map((m) => (
      <option key={m.path} value={m.path}>
        {m.name} · {m.isSft ? 'SFT' : '基座'} · d{m.depth} · step {m.step} ·{' '}
        {m.datasetScope}
      </option>
    ));
  return (
    <section className={section}>
      <h2 className="font-semibold">
        {compare ? '用固定题集比较两个模型' : '颜色与数量 · 格式转换'}
      </h2>
      {!compare && (
        <div className="text-sm text-text-muted">
          训练样例（共 {preset.count} 条）
          <p className="mt-1 text-text">
            {preset.sample.prompt} → <code>{preset.sample.completion}</code>
          </p>
        </div>
      )}
      <label className="block text-sm">
        {compare ? '模型 A' : '预训练基座'}
        <select
          className={input}
          value={base}
          onChange={(e) => setBase(e.target.value)}
        >
          <option value="">
            {compare ? '请选择' : '用内置数据预训练小模型（12 步）'}
          </option>
          {options(bases)}
        </select>
      </label>
      {compare ? (
        <label className="block text-sm">
          模型 B
          <select
            className={input}
            value={target}
            onChange={(e) => setTarget(e.target.value)}
          >
            <option value="">请选择</option>
            {options(models)}
          </select>
        </label>
      ) : (
        <label className="block max-w-48 text-sm">
          微调步数
          <input
            className={input}
            type="number"
            min="1"
            max="200"
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </label>
      )}
      <details>
        <summary className="cursor-pointer text-sm text-primary-light">
          查看和编辑 {rows.length} 道留出题
        </summary>
        <p className="my-2 text-xs text-text-muted">
          标准答案留空时只展示回答。
        </p>
        {rows.map((row, i) => (
          <div key={i} className="mb-2 grid gap-2 md:grid-cols-[2fr_1fr_auto]">
            {(['prompt', 'expected'] as const).map((key) => (
              <input
                className={input}
                key={key}
                aria-label={
                  (key === 'prompt' ? '问题 ' : '标准答案 ') + (i + 1)
                }
                value={row[key]}
                onChange={(e) =>
                  setRows(
                    rows.map((r, j) =>
                      j === i ? { ...r, [key]: e.target.value } : r,
                    ),
                  )
                }
              />
            ))}
            <button
              className="text-xs text-text-muted"
              onClick={() => setRows(rows.filter((_, j) => j !== i))}
            >
              移除
            </button>
          </div>
        ))}
        <button
          className="text-sm text-primary-light"
          disabled={rows.length >= 24}
          onClick={() => setRows([...rows, { prompt: '', expected: '' }])}
        >
          添加问题
        </button>
      </details>
      {error && <p role="alert">{error}</p>}
      <button
        className={button}
        disabled={
          busy ||
          !rows.length ||
          (!compare && (steps < 1 || steps > 200)) ||
          (compare && (!base || !target || base === target))
        }
        onClick={submit}
      >
        {compare ? '开始模型对比' : '检查环境并开始实验'}
      </button>
    </section>
  );
}

function ResponseCell({ answer }: { answer?: Answer }) {
  if (!answer) return <p className="text-text-muted">待评测</p>;
  return (
    <div>
      <p className="mb-1 text-xs text-text-muted">
        {answer.exact === null
          ? '开放题'
          : answer.exact
            ? '字段全对'
            : answer.valid
              ? 'JSON 合法，字段不符'
              : 'JSON 格式不符'}
      </p>
      <pre className="whitespace-pre-wrap break-all text-sm">
        {answer.text || '（空回答）'}
      </pre>
    </div>
  );
}

function score(result: Assessment | null) {
  if (!result) return '待评测';
  const scored = result.answers.filter((a) => a.exact !== null);
  if (!scored.length) return '仅开放题';
  return (
    'JSON 合法 ' +
    scored.filter((a) => a.valid).length +
    '/' +
    scored.length +
    ' · 字段全对 ' +
    scored.filter((a) => a.exact).length +
    '/' +
    scored.length
  );
}

function LossChart({ r }: { r: Experiment }) {
  return r.curve.length > 0 ? (
    <div>
      <h3 className="text-sm">训练损失 · 按训练步数</h3>
      <ResponsiveContainer width="100%" height={160}>
        <LineChart data={r.curve}>
          <CartesianGrid stroke="#334155" />
          <XAxis dataKey="step" />
          <YAxis width={40} />
          <Tooltip />
          <Line dataKey="loss" stroke="#818cf8" isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  ) : null;
}

function Monitor({ r }: { r: Experiment }) {
  const order =
    r.kind === 'training'
      ? [r.training_stage || 'train']
      : r.kind === 'compare'
        ? ['before', 'after']
        : [
            ...(r.config &&
            ('pretrain' in r.timings ||
              ['prepare', 'pretrain'].includes(r.stage))
              ? ['prepare', 'pretrain']
              : []),
            'before',
            'sft',
            'after',
          ];
  return (
    <section className={section}>
      <h2>{title(r)}</h2>
      <div className="divide-y divide-border">
        {order.map((k) => (
          <div className="flex justify-between py-2 text-sm" key={k}>
            <span>{stages[k]}</span>
            <span>
              {k in r.timings
                ? r.timings[k] + ' 秒'
                : r.stage === k
                  ? r.status === 'failed'
                    ? '失败'
                    : '运行中'
                  : '等待'}
            </span>
          </div>
        ))}
      </div>
      <LossChart r={r} />
      {r.logs?.length ? (
        <pre className="whitespace-pre-wrap break-all text-xs text-text-muted">
          {r.logs.join('\n')}
        </pre>
      ) : null}
      {r.error && <p role="alert">{r.error}</p>}
      <Link className="text-primary-light" to={'/experiments?id=' + r.id}>
        查看实验报告
      </Link>
    </section>
  );
}

function Report({
  r,
  busy,
  launch,
}: {
  r: Experiment;
  busy: boolean;
  launch: (path: string, body: object) => void;
}) {
  const labels = r.kind === 'sft' ? ['微调前', '微调后'] : ['模型 A', '模型 B'];
  return (
    <section className={section}>
      <h2 className="font-semibold">{title(r)}</h2>
      <p role="status" className="text-sm">
        {stages[r.stage]}
        {r.status === 'running'
          ? '进行中…'
          : r.status === 'failed'
            ? '失败'
            : ''}{' '}
        {r.ended && ' · 总耗时 ' + Math.round(r.ended - r.created) + ' 秒'}
      </p>
      {r.error && (
        <p
          role="alert"
          className="whitespace-pre-wrap break-words text-sm text-error"
        >
          {r.error}
        </p>
      )}
      {r.kind === 'training' ? (
        <p className="text-sm">训练产物 · {model(r.target)}</p>
      ) : (
        <>
          <div className="grid gap-3 md:grid-cols-2">
            {[r.before, r.after].map((result, i) => (
              <div key={i} className="rounded-lg bg-surface p-3 text-sm">
                <strong>
                  {labels[i]} · {model(i ? r.target : r.base)}
                </strong>
                <p>{score(result)}</p>
                {result && <small>{time(result.at)}</small>}
              </div>
            ))}
          </div>
          <p className="text-xs text-text-muted">
            逐题判定使用完整
            JSON。字段名称、类型和值均须符合标准答案。开放题不计分。
          </p>
        </>
      )}
      {r.config.questions.map((q, i) => {
        const a = r.before?.answers[i],
          b = r.after?.answers[i];
        const change =
          a && b && a.exact !== null
            ? a.exact !== b.exact
              ? b.exact
                ? '字段改善'
                : '字段退步'
              : a.valid !== b.valid
                ? b.valid
                  ? '格式改善'
                  : '格式退步'
                : '持平'
            : '';
        return (
          <details key={i} className="rounded-lg border border-border p-3">
            <summary className="cursor-pointer text-sm">
              {q.prompt} <span className="text-primary-light">{change}</span>
            </summary>
            <p className="my-2 text-xs text-text-muted">
              {q.expected ? '标准答案 ' + JSON.stringify(q.expected) : '开放题'}
            </p>
            <div className="grid gap-3 md:grid-cols-2">
              {[a, b].map((item, j) => (
                <div key={j}>
                  <strong className="text-xs">{labels[j]}</strong>
                  <ResponseCell answer={item} />
                </div>
              ))}
            </div>
          </details>
        );
      })}
      <LossChart r={r} />
      <details>
        <summary className="text-sm cursor-pointer">
          数据、参数与运行环境
        </summary>
        <pre className="mt-2 whitespace-pre-wrap break-all text-xs">
          {JSON.stringify(
            {
              训练文件: r.kind !== 'compare' ? r.data_file : undefined,
              SHA256: r.kind !== 'compare' ? r.data_hash : undefined,
              基座: r.base,
              目标: r.target,
              参数: r.parameters,
              生成: r.generation,
              评分: r.rule,
              环境: r.device,
              版本: r.versions,
              阶段耗时秒: r.timings,
              MLX峰值MB: r.peak_mb,
            },
            null,
            2,
          )}
        </pre>
      </details>
      <div className="flex flex-wrap gap-4 text-sm text-primary-light">
        {r.status !== 'running' && (
          <button
            disabled={busy}
            onClick={() => launch('/' + r.id + '/rerun', {})}
          >
            按此配置再运行
          </button>
        )}
        {r.status === 'failed' && r.before && r.target && (
          <button
            disabled={busy}
            onClick={() => launch('/' + r.id + '/retry', {})}
          >
            仅重试后评测
          </button>
        )}
        {r.target && r.domain !== 'music' && (
          <Link
            to={
              '/chat?checkpoint=' +
              encodeURIComponent(r.target.checkpoint_id) +
              '&experiment=' +
              r.id
            }
          >
            使用此模型对话
          </Link>
        )}
        {r.domain === 'music' && r.target && (
          <Link to="/music">打开音乐生成</Link>
        )}
        <Link to={'/training?experiment=' + r.id}>训练监控</Link>
        <a
          href={'/api/experiments/' + r.id}
          download={'experiment-' + r.id + '.json'}
        >
          下载实验记录
        </a>
      </div>
    </section>
  );
}

function Compare({
  left,
  right,
  launch,
  busy,
}: {
  left: Experiment;
  right: Experiment;
  launch: (path: string, body: object) => void;
  busy: boolean;
}) {
  const pairs = left.config.questions.flatMap((q, i) => {
    const j = right.config.questions.findIndex(
      (x) =>
        x.prompt === q.prompt &&
        JSON.stringify(Object.entries(x.expected || {}).sort()) ===
          JSON.stringify(Object.entries(q.expected || {}).sort()) &&
        (x.expected === null) === (q.expected === null),
    );
    return j < 0
      ? []
      : [{ q, a: left.after?.answers[i], b: right.after?.answers[j] }];
  });
  return (
    <section className={section}>
      <h2 className="font-semibold">两次实验对比</h2>
      <div className="grid gap-3 md:grid-cols-2">
        {[left, right].map((r) => (
          <div key={r.id} className="rounded-lg bg-surface p-3 text-sm">
            <b>{title(r)}</b>
            <p>基座 {model(r.base)}</p>
            <p>产物 {model(r.target)}</p>
            <p>
              训练集{' '}
              {r.kind !== 'compare'
                ? r.data_file + ' · ' + r.data_hash.slice(0, 12)
                : '仅评测'}
            </p>
            <p>
              {r.kind === 'sft' && '微调 ' + r.config.steps + ' 步 · '}
              {score(r.after)}
            </p>
            <details>
              <summary>训练参数及生成配置</summary>
              <pre className="whitespace-pre-wrap break-all">
                {JSON.stringify(
                  { 参数: r.parameters, 生成: r.generation, 规则: r.rule },
                  null,
                  2,
                )}
              </pre>
            </details>
          </div>
        ))}
      </div>
      <div className="text-sm">
        {left.base?.weight_hash !== right.base?.weight_hash && (
          <p>基座权重不同</p>
        )}
        {left.data_hash !== right.data_hash && <p>训练数据摘要不同</p>}
        {Object.entries({
          ...(left.parameters.sft as Record<string, unknown>),
          ...(right.parameters.sft as Record<string, unknown>),
        })
          .filter(
            ([key]) =>
              !['model_name', 'custom_sft_path'].includes(key) &&
              (left.parameters.sft as Record<string, unknown>)?.[key] !==
                (right.parameters.sft as Record<string, unknown>)?.[key],
          )
          .map(([key]) => (
            <p key={key}>
              {key} ·{' '}
              {String(
                (left.parameters.sft as Record<string, unknown>)?.[key] ?? '—',
              )}{' '}
              →{' '}
              {String(
                (right.parameters.sft as Record<string, unknown>)?.[key] ?? '—',
              )}
            </p>
          ))}
      </div>
      <p className="text-sm">
        共有 {pairs.length} 道题目和标准答案相同。
        {left.rule !== right.rule && '两次评分规则不同。'}
      </p>
      {pairs.map(({ q, a, b }) => (
        <details key={q.prompt}>
          <summary>{q.prompt}</summary>
          <div className="grid gap-3 md:grid-cols-2">
            <ResponseCell answer={a} />
            <ResponseCell answer={b} />
          </div>
        </details>
      ))}
      <button
        className={button}
        disabled={
          busy ||
          !left.target ||
          !right.target ||
          left.domain === 'music' ||
          right.domain === 'music' ||
          (!left.config.questions.length && !right.config.questions.length) ||
          left.target.checkpoint_id === right.target.checkpoint_id
        }
        onClick={() =>
          launch('', {
            base: left.target?.checkpoint_id,
            target: right.target?.checkpoint_id,
            questions: left.config.questions.length
              ? left.config.questions
              : right.config.questions,
          })
        }
      >
        使用共同题集重新评测两个产物
      </button>
    </section>
  );
}

export default function ExperimentsPage() {
  const [params, setParams] = useSearchParams();
  const location = useLocation();
  const id = params.get('id') || params.get('experiment') || '';
  const mode = params.get('new');
  const [other, setOther] = useState('');
  const [history, setHistory] = useState<Summary[]>([]);
  const [r, setRecord] = useState<Experiment | null>(null);
  const [second, setSecond] = useState<Experiment | null>(null);
  const [preset, setPreset] = useState<Preset | null>(null);
  const [models, setModels] = useState<Checkpoint[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [filter, setFilter] = useState('');
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const refresh = async () => {
      try {
        const [list, rec, comparison] = await Promise.all([
          api<Summary[]>(''),
          id ? api<Experiment>('/' + id) : null,
          other ? api<Experiment>('/' + other) : null,
        ]);
        if (alive) {
          setHistory(list);
          setRecord(rec);
          setSecond(comparison);
        }
      } catch (e) {
        if (alive) setError(String(e));
      }
      if (alive) timer = setTimeout(refresh, 2500);
    };
    void refresh();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id, other]);
  useEffect(() => {
    let alive = true;
    if (mode)
      Promise.all([api<Preset>('/preset'), getAllCheckpoints()])
        .then(([p, c]) => {
          if (alive) {
            setPreset(p);
            setModels(c.filter((m) => m.dataDomain !== 'music'));
          }
        })
        .catch((e) => {
          if (alive) setError(String(e));
        });
    return () => {
      alive = false;
    };
  }, [mode]);
  async function launch(path: string, body: object) {
    setBusy(true);
    setError('');
    try {
      const rec = await api<Experiment>(path, body);
      setRecord(rec);
      setOther('');
      setParams({ id: rec.id });
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="mx-auto max-w-6xl p-4 md:p-6 space-y-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold">
          {location.pathname === '/training' ? '实验训练监控' : '实验记录'}
        </h1>
        <div className="flex gap-3 text-sm text-primary-light">
          <Link to="/experiments">历史记录</Link>
          <Link to="/experiments?new=sft">示例微调实验</Link>
          <Link to="/experiments?new=compare">模型对比</Link>
        </div>
      </header>
      {error && (
        <p role="alert" className="text-error text-sm">
          {error}
        </p>
      )}
      {mode && preset && (
        <NewExperiment
          key={mode}
          preset={preset}
          models={models}
          compare={mode === 'compare'}
          busy={busy}
          launch={launch}
        />
      )}
      {location.pathname === '/training' && r?.id === id && <Monitor r={r} />}
      {location.pathname !== '/training' && id && r?.id === id && (
        <>
          <Report r={r} launch={launch} busy={busy} />
          <label className="block text-sm">
            选择另一条记录对比
            <select
              className={input}
              value={other}
              onChange={(e) => setOther(e.target.value)}
            >
              <option value="">请选择</option>
              {history
                .filter((x) => x.id !== id)
                .map((x) => (
                  <option key={x.id} value={x.id}>
                    {title(x)}
                  </option>
                ))}
            </select>
          </label>
          {other && second?.id === other && other !== id && (
            <Compare left={r} right={second} launch={launch} busy={busy} />
          )}
        </>
      )}
      {!mode && !id && (
        <section className={section}>
          <label className="text-sm">
            运行状态{' '}
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="bg-surface p-2"
            >
              <option value="">全部</option>
              <option value="done">已完成</option>
              <option value="running">运行中</option>
              <option value="failed">失败</option>
            </select>
          </label>
          {history
            .filter((x) => !filter || x.status === filter)
            .map((x) => (
              <Link
                className="block rounded-lg border border-border p-3 hover:bg-surface"
                key={x.id}
                to={'/experiments?id=' + x.id}
              >
                {title(x)} · {x.status === 'failed' ? '失败' : stages[x.stage]}
              </Link>
            ))}
          {!history.length && (
            <p className="text-sm text-text-muted">
              尚无记录。运行一次示例微调，即可在这里查看前后回答。
            </p>
          )}
        </section>
      )}
    </div>
  );
}

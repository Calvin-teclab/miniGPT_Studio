import { cn } from '@/lib/utils';
import { type ReactNode, useState } from 'react';
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Circle,
  FileText,
  Loader2,
  PackageCheck,
  SkipForward,
} from 'lucide-react';
import ParamTooltip from './ParamTooltip';

export interface StepDetail {
  purpose: string;
  inputs: string[];
  process: {
    title: string;
    description: string;
    result: string;
  }[];
  outputs: string[];
  successResult: string;
}

interface StepCardProps {
  name: string;
  description: string;
  detailedHelp: string;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  stepNumber: number;
  progress?: number;
  isActive: boolean;
  onStart: () => void;
  onSkip?: () => void;
  disabled?: boolean;
  detectedFromCache?: boolean;
  rerunLabel?: string;
  detail?: StepDetail;
  logs?: string[];
  extra?: ReactNode;
}

const statusConfig = {
  pending: { icon: Circle, color: 'text-text-muted', bg: 'bg-surface-lighter/50' },
  running: { icon: Loader2, color: 'text-info', bg: 'bg-info/5' },
  completed: { icon: CheckCircle2, color: 'text-success', bg: 'bg-success/5' },
  error: { icon: AlertCircle, color: 'text-error', bg: 'bg-error/5' },
  skipped: { icon: SkipForward, color: 'text-text-muted', bg: 'bg-surface-lighter/30' },
};

export default function StepCard({
  name,
  description,
  detailedHelp,
  status,
  stepNumber,
  progress,
  isActive,
  onStart,
  onSkip,
  disabled,
  detectedFromCache,
  rerunLabel = '重新运行',
  detail,
  logs = [],
  extra,
}: StepCardProps) {
  const isCachedCompleted = status === 'completed' && !!detectedFromCache;
  const config = statusConfig[status];
  const Icon = isCachedCompleted ? PackageCheck : config.icon;
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={cn(
        'rounded-xl border transition-all duration-200',
        isActive ? 'border-primary/50 shadow-lg shadow-primary/5' : 'border-border',
        isCachedCompleted ? 'bg-info/5' : config.bg
      )}
    >
      <div className="p-4">
        <div className="flex items-start justify-between mb-2.5">
          <div className="flex items-center gap-2.5">
            <div
              className={cn(
                'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold',
                status === 'completed' && !isCachedCompleted
                  ? 'bg-success/20 text-success'
                  : isCachedCompleted
                  ? 'bg-info/15 text-info'
                  : isActive
                  ? 'bg-primary/20 text-primary'
                  : 'bg-surface-lighter text-text-muted'
              )}
            >
              {status === 'completed' && !isCachedCompleted ? (
                <CheckCircle2 className="w-4.5 h-4.5" />
              ) : (
                stepNumber
              )}
            </div>
            <div>
              <h3 className="font-semibold text-text flex items-center gap-2">
                {name}
                <ParamTooltip content={detailedHelp} />
                {detectedFromCache && (
                  <span className="px-2 py-0.5 rounded-full bg-info/10 text-info text-[11px] font-normal">
                    可复用本地缓存
                  </span>
                )}
              </h3>
              <p className="text-xs text-text-muted mt-0.5">{description}</p>
              {detectedFromCache && (
                <p className="text-[11px] text-info/90 mt-1">
                  这是之前留下的本地产物，不代表本轮已重新执行；需要新训练时请点击“{rerunLabel}”。
                </p>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {detail && (
              <button
                type="button"
                onClick={() => setExpanded((value) => !value)}
                className="p-1 rounded-md text-text-muted hover:text-text hover:bg-surface-lighter transition-colors"
                aria-label={expanded ? '收起详情' : '展开详情'}
              >
                {expanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
              </button>
            )}
            <Icon
              className={cn(
                'w-5 h-5 shrink-0',
                config.color,
                isCachedCompleted && 'text-info',
                status === 'running' && 'animate-spin'
              )}
            />
          </div>
        </div>

        {detail && expanded && (
          <div className="mt-3 border border-border rounded-xl bg-surface/60 overflow-hidden">
            <div className="p-3 border-b border-border">
              <div className="text-xs uppercase tracking-wide text-primary-light mb-1">这一步的目标</div>
              <p className="text-sm text-text-muted leading-relaxed">{detail.purpose}</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-0 border-b border-border">
              <div className="p-3 border-b md:border-b-0 md:border-r border-border">
                <div className="flex items-center gap-2 text-sm font-medium mb-2">
                  <FileText className="w-4 h-4 text-info" />
                  输入/前置条件
                </div>
                <ul className="space-y-1.5">
                  {detail.inputs.map((input) => (
                    <li key={input} className="flex gap-2 text-xs text-text-muted">
                      <span className="mt-1 w-1.5 h-1.5 rounded-full bg-info shrink-0" />
                      <span>{input}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="p-3">
                <div className="flex items-center gap-2 text-sm font-medium mb-2">
                  <PackageCheck className="w-4 h-4 text-success" />
                  产物/结果
                </div>
                <ul className="space-y-1.5">
                  {detail.outputs.map((output) => (
                    <li key={output} className="flex gap-2 text-xs text-text-muted">
                      <span className="mt-1 w-1.5 h-1.5 rounded-full bg-success shrink-0" />
                      <span>{output}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="p-3">
              <div className="text-sm font-medium mb-2">具体处理环节</div>
              <div className="space-y-2">
                {detail.process.map((item, index) => (
                  <div key={item.title} className="grid grid-cols-[28px_1fr] gap-3">
                    <div className="flex flex-col items-center">
                      <div
                        className={cn(
                          'w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold',
                          status === 'completed' && !detectedFromCache
                            ? 'bg-success/20 text-success'
                            : detectedFromCache
                            ? 'bg-info/15 text-info'
                            : status === 'running'
                            ? 'bg-primary/20 text-primary'
                            : 'bg-surface-lighter text-text-muted'
                        )}
                      >
                        {index + 1}
                      </div>
                      {index < detail.process.length - 1 && <div className="w-px flex-1 bg-border mt-1" />}
                    </div>
                    <div className="pb-3">
                      <div className="flex items-center gap-2 text-sm font-medium">
                        {item.title}
                        <ArrowRight className="w-3.5 h-3.5 text-text-muted" />
                        <span className="text-xs text-success">{item.result}</span>
                      </div>
                      <p className="text-xs text-text-muted mt-1 leading-relaxed">{item.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="px-3 py-2 bg-success/5 border-t border-success/20 text-xs text-success">
              完成后你会得到：{detail.successResult}
            </div>

            {logs.length > 0 && (
              <div className="p-3 border-t border-border">
                <div className="text-sm font-medium mb-2">实时过程日志</div>
                <div className="max-h-36 overflow-y-auto rounded-lg bg-surface border border-border p-3 font-mono text-[11px] text-text-muted space-y-1">
                  {logs.slice(-12).map((log, i) => (
                    <div key={`${log}-${i}`}>{log}</div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {extra && (
          <div className="mt-3">
            {extra}
          </div>
        )}

        {/* Progress bar */}
        {status === 'running' && progress !== undefined && (
          <div className="mt-3 mb-3">
            <div className="flex justify-between text-xs text-text-muted mb-1">
              <span>进度</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="h-1.5 bg-surface-lighter rounded-full overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Actions */}
        {(status === 'pending' || status === 'error') && (
          <div className="flex gap-2 mt-3">
            <button
              onClick={onStart}
              disabled={disabled}
              className={cn(
                'px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all',
                disabled
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary-dark active:scale-[0.98]'
              )}
            >
              {status === 'error' ? '重试' : '开始'}
            </button>
            {onSkip && (
              <button
                onClick={onSkip}
                disabled={disabled}
                className={cn(
                  'px-3.5 py-1.5 rounded-lg text-sm transition-all',
                  disabled
                    ? 'text-text-muted/60 cursor-not-allowed'
                    : 'text-text-muted hover:text-text hover:bg-surface-lighter'
                )}
              >
                跳过
              </button>
            )}
          </div>
        )}

        {status === 'completed' && detectedFromCache && (
          <div className="flex gap-2 mt-3">
            <button
              onClick={onStart}
              disabled={disabled}
              className={cn(
                'px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all',
                disabled
                  ? 'bg-surface-lighter text-text-muted cursor-not-allowed'
                  : 'bg-primary text-white hover:bg-primary-dark active:scale-[0.98]'
              )}
            >
              {rerunLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

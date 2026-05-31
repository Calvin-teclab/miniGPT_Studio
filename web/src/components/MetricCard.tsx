import { cn } from '@/lib/utils';
import ParamTooltip from './ParamTooltip';

interface MetricCardProps {
  label: string;
  value: string | number;
  unit?: string;
  tooltip?: string;
  trend?: 'up' | 'down' | 'neutral';
  className?: string;
}

export default function MetricCard({ label, value, unit, tooltip, trend, className }: MetricCardProps) {
  return (
    <div className={cn('bg-surface-light rounded-lg border border-border p-4', className)}>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-xs text-text-muted">{label}</span>
        {tooltip && <ParamTooltip content={tooltip} />}
      </div>
      <div className="flex items-baseline gap-1">
        <span
          className={cn(
            'text-xl font-bold',
            trend === 'down' ? 'text-success' : trend === 'up' ? 'text-error' : 'text-text'
          )}
        >
          {value}
        </span>
        {unit && <span className="text-xs text-text-muted">{unit}</span>}
      </div>
    </div>
  );
}

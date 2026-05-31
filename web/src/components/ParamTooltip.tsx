import * as Tooltip from '@radix-ui/react-tooltip';
import { HelpCircle } from 'lucide-react';

interface ParamTooltipProps {
  content: string;
  children?: React.ReactNode;
  side?: 'top' | 'right' | 'bottom' | 'left';
}

export default function ParamTooltip({ content, children, side = 'right' }: ParamTooltipProps) {
  return (
    <Tooltip.Provider delayDuration={200}>
      <Tooltip.Root>
        <Tooltip.Trigger asChild>
          {children || (
            <button type="button" className="inline-flex items-center text-text-muted hover:text-primary transition-colors">
              <HelpCircle className="w-3.5 h-3.5" />
            </button>
          )}
        </Tooltip.Trigger>
        <Tooltip.Portal>
          <Tooltip.Content
            side={side}
            sideOffset={5}
            className="max-w-xs px-3 py-2 text-xs leading-relaxed text-text bg-surface-lighter border border-border rounded-lg shadow-xl z-50"
            style={{ animation: 'tooltip-fade-in 0.15s ease-out' }}
          >
            {content}
            <Tooltip.Arrow className="fill-surface-lighter" />
          </Tooltip.Content>
        </Tooltip.Portal>
      </Tooltip.Root>
    </Tooltip.Provider>
  );
}

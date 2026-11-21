import { NavLink, Outlet, useMatch } from 'react-router-dom';
import { Activity, FlaskConical, MessageCircle, Music2, Settings, Workflow } from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { to: '/', icon: Workflow, label: '训练流程' },
  { to: '/config', icon: Settings, label: '参数配置' },
  { to: '/training', icon: Activity, label: '训练监控' },
  { to: '/eval', icon: FlaskConical, label: '模型评测' },
  { to: '/chat', icon: MessageCircle, label: '模型对话' },
  { to: '/music', icon: Music2, label: '音乐生成' },
];

export default function Layout() {
  const onPipelineAlias = useMatch('/pipeline');

  return (
    <div className="flex h-dvh overflow-hidden">
      <aside className="flex w-16 shrink-0 flex-col border-r border-border bg-surface-light md:w-52">
        <div className="flex h-16 shrink-0 items-center justify-center gap-2.5 border-b border-border md:justify-start md:px-4">
          <img src="/favicon.svg" alt="" className="size-8" />
          <span className="sr-only whitespace-nowrap text-[15px] font-semibold md:not-sr-only">miniGPT Studio</span>
        </div>
        <nav aria-label="主导航" className="flex-1 space-y-1 overflow-y-auto px-1 py-3 md:px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to === '/' && onPipelineAlias ? '/pipeline' : item.to}
              end
              className={({ isActive }) =>
                cn(
                  'flex flex-col items-center gap-1 rounded-lg py-2 text-[11px] transition-colors md:flex-row md:gap-3 md:px-3 md:py-2.5 md:text-sm',
                  isActive
                    ? 'bg-primary/15 font-medium text-primary-light'
                    : 'text-text-muted hover:bg-surface-lighter hover:text-text'
                )
              }
            >
              <item.icon className="size-4.5 shrink-0" />
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="hidden border-t border-border p-4 text-xs leading-relaxed text-text-muted md:block">
          <div>Apple Silicon MLX</div>
          <div>本地训练 · 作者：xuc</div>
        </div>
      </aside>
      <main className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

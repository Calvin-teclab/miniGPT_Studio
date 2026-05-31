import { NavLink, Outlet } from 'react-router-dom';
import {
  Settings,
  Workflow,
  Activity,
  FlaskConical,
  MessageCircle,
  Music2,
  Brain,
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navItems = [
  { to: '/', icon: Workflow, label: '训练流程', description: '引导式训练步骤' },
  { to: '/config', icon: Settings, label: '参数配置', description: '模型与训练参数' },
  { to: '/training', icon: Activity, label: '训练监控', description: '实时训练过程' },
  { to: '/eval', icon: FlaskConical, label: '模型评测', description: '基准与对比评测' },
  { to: '/chat', icon: MessageCircle, label: '模型对话', description: '与模型交互' },
  { to: '/music', icon: Music2, label: '音乐生成', description: '和弦生成' },
];

export default function Layout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-surface-light border-r border-border flex flex-col shrink-0">
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-2">
            <Brain className="w-6 h-6 text-primary" />
            <div>
              <h1 className="text-base font-bold text-text">miniGPT Studio</h1>
              <p className="text-xs text-text-muted">AI Training Lab</p>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-2.5 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-2.5 px-2.5 py-2 rounded-lg transition-all text-sm',
                  isActive
                    ? 'bg-primary/15 text-primary-light font-medium'
                    : 'text-text-muted hover:bg-surface-lighter hover:text-text'
                )
              }
            >
              <item.icon className="w-4.5 h-4.5 shrink-0" />
              <div>
                <div>{item.label}</div>
                <div className="text-[11px] opacity-60">{item.description}</div>
              </div>
            </NavLink>
          ))}
        </nav>
        <div className="p-3.5 border-t border-border text-xs text-text-muted">
          <div>Apple Silicon MLX</div>
          <div className="mt-0.5 opacity-60">本地训练 · 作者：xuc</div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}

import { useState, useEffect, useRef, useCallback } from 'react';
import {
  MessageCircle,
  Send,
  Trash2,
  Settings,
  Loader2,
  Bot,
  User,
  Download,
  Upload,
  RefreshCw,
} from 'lucide-react';
import ParamTooltip from '@/components/ParamTooltip';
import { cn } from '@/lib/utils';
import type { ChatMessage, ChatSettings, Checkpoint } from '@/types';
import { deleteCheckpoint, getCheckpoints, loadModel, unloadModel } from '@/api/client';

export default function ChatPage() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [loadedCheckpoint, setLoadedCheckpoint] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [pendingDelete, setPendingDelete] = useState<{ path: string; label: string } | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState<ChatSettings>({
    temperature: 0.7,
    topK: 50,
    repetitionPenalty: 1.2,
    maxTokens: 512,
  });
  const [showSettings, setShowSettings] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const activeRequestRef = useRef(0);
  const activeAbortRef = useRef<AbortController | null>(null);

  const cancelActiveGeneration = useCallback(() => {
    activeAbortRef.current?.abort();
    activeAbortRef.current = null;
    activeRequestRef.current += 1;
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const refreshCheckpoints = useCallback(() => {
    getCheckpoints()
      .then((cps) => {
        const chatCheckpoints = cps.filter((c) => c.dataDomain !== 'music');
        setCheckpoints(chatCheckpoints);
        setSelectedCheckpoint((current) => {
          if (current && chatCheckpoints.some((c) => c.path === current)) return current;
          // Prefer SFT checkpoints in the general chat domain.
          const sft = chatCheckpoints.find((c) => c.isSft);
          if (sft) return sft.path;
          if (chatCheckpoints.length > 0) return chatCheckpoints[0].path;
          return '';
        });
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    refreshCheckpoints();
    window.addEventListener('focus', refreshCheckpoints);
    return () => {
      window.removeEventListener('focus', refreshCheckpoints);
      cancelActiveGeneration();
    };
  }, [refreshCheckpoints, cancelActiveGeneration]);

  useEffect(() => {
    if (isLoaded) return;
    refreshCheckpoints();
  }, [isLoaded, refreshCheckpoints]);

  const handleLoad = async () => {
    if (!selectedCheckpoint) return;
    cancelActiveGeneration();
    setIsLoading(true);
    setLoadError('');
    try {
      await loadModel(selectedCheckpoint);
      setMessages([]);
      setInput('');
      setIsGenerating(false);
      setLoadedCheckpoint(selectedCheckpoint);
      setIsLoaded(true);
    } catch (err) {
      console.error(err);
      setIsLoaded(false);
      setLoadedCheckpoint('');
      setLoadError(err instanceof Error ? err.message : '模型加载失败');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUnload = async () => {
    cancelActiveGeneration();
    await unloadModel();
    setIsLoaded(false);
    setIsGenerating(false);
    setLoadedCheckpoint('');
    setMessages([]);
    setInput('');
  };

  const handleDeleteCheckpoint = async () => {
    if (!selectedCheckpoint || isDeleting) return;
    const checkpoint = checkpoints.find((item) => item.path === selectedCheckpoint);
    const label = checkpoint ? `${checkpoint.name} (${checkpoint.isSft ? 'SFT' : '预训练'}, step=${checkpoint.step})` : selectedCheckpoint;
    setPendingDelete({ path: selectedCheckpoint, label });
  };

  const confirmDeleteCheckpoint = async () => {
    if (!pendingDelete || isDeleting) return;
    setIsDeleting(true);
    setLoadError('');
    try {
      cancelActiveGeneration();
      await deleteCheckpoint(pendingDelete.path);
      setIsLoaded(false);
      setLoadedCheckpoint('');
      setMessages([]);
      setInput('');
      setIsGenerating(false);
      setPendingDelete(null);
      refreshCheckpoints();
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : '删除模型失败');
    } finally {
      setIsDeleting(false);
    }
  };

  const sendMessage = useCallback(async () => {
    if (!input.trim() || isGenerating || !isLoaded) return;
    activeAbortRef.current?.abort();
    const controller = new AbortController();
    activeAbortRef.current = controller;
    const requestId = activeRequestRef.current + 1;
    activeRequestRef.current = requestId;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: Date.now(),
    };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setIsGenerating(true);

    // Add empty assistant message
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };
    setMessages([...updatedMessages, assistantMessage]);

    const updateAssistantMessage = (content: string) => {
      if (activeRequestRef.current !== requestId) return;
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const next = [...prev];
        next[next.length - 1] = {
          ...next[next.length - 1],
          content,
        };
        return next;
      });
    };

    try {
      let accumulated = '';
      let terminalEventReceived = false;

      const finishStream = () => {
        terminalEventReceived = true;
        if (activeRequestRef.current === requestId) {
          setIsGenerating(false);
          setTimeout(() => inputRef.current?.focus(), 0);
        }
      };

      const handleStreamPayload = (payload: string) => {
        if (activeRequestRef.current !== requestId) return;
        try {
          const data = JSON.parse(payload);
          if (data.type === 'token' || typeof data.token === 'string') {
            accumulated += data.token;
            updateAssistantMessage(accumulated);
          } else if (data.type === 'done' || data.done === true) {
            finishStream();
          } else if (data.type === 'error' || data.detail) {
            accumulated += `\n[错误] ${data.text || data.detail}`;
            updateAssistantMessage(accumulated);
            finishStream();
          }
        } catch {
          accumulated += payload;
          updateAssistantMessage(accumulated);
        }
      };

      const response = await fetch('/api/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          messages: updatedMessages,
          temperature: settings.temperature,
          top_k: settings.topK,
          repetition_penalty: settings.repetitionPenalty,
          max_tokens: settings.maxTokens,
        }),
      });

      if (!response.ok || !response.body) {
        const data = await response.json().catch(() => ({}));
        throw new Error(typeof data.detail === 'string' ? data.detail : `后端返回 ${response.status}`);
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
            handleStreamPayload(dataLines.join('\n'));
          }
        }
      }
      const tail = buffer.trim();
      if (tail.startsWith('data:')) {
        handleStreamPayload(tail.slice(5).trimStart());
      }
      if (!terminalEventReceived) {
        finishStream();
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      updateAssistantMessage(`[错误] 对话请求失败：${error instanceof Error ? error.message : '请重试。'}`);
    } finally {
      if (activeAbortRef.current === controller) {
        activeAbortRef.current = null;
      }
      if (activeRequestRef.current === requestId) {
        setIsGenerating(false);
        setTimeout(() => inputRef.current?.focus(), 0);
      }
    }
  }, [input, isGenerating, isLoaded, messages, settings, cancelActiveGeneration]);

  const clearChat = () => {
    cancelActiveGeneration();
    setMessages([]);
    setIsGenerating(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <div className="px-5 py-6 lg:px-6 border-b border-border bg-surface-light flex items-center justify-between shrink-0">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <MessageCircle className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">模型对话</h1>
            {isLoaded && (
              <span className="px-2 py-0.5 bg-success/20 text-success text-xs rounded-full">
                已加载
              </span>
            )}
          </div>
          <p className="text-text-muted">
            加载本地模型进行多轮对话，验证回答质量、指令跟随和推理表现。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={cn(
              'p-2 rounded-lg transition-all',
              showSettings ? 'bg-primary/15 text-primary' : 'text-text-muted hover:text-text hover:bg-surface-lighter'
            )}
          >
            <Settings className="w-4 h-4" />
          </button>
          <button
            onClick={clearChat}
            className="p-2 rounded-lg text-text-muted hover:text-error hover:bg-error/10 transition-all"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Model loader */}
      {!isLoaded && (
        <div className="px-6 py-4 border-b border-border bg-surface-light/50">
          <div className="flex items-center gap-3">
            <select
              value={selectedCheckpoint}
              onChange={(e) => setSelectedCheckpoint(e.target.value)}
              className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
            >
              {checkpoints.map((cp) => (
                <option key={cp.path} value={cp.path}>
                  {cp.name} ({cp.isSft ? 'SFT' : '预训练'}, step={cp.step})
                </option>
              ))}
              {checkpoints.length === 0 && <option>暂无可用通用模型</option>}
            </select>
            <button
              onClick={refreshCheckpoints}
              disabled={isLoading || isDeleting}
              className="flex items-center gap-1.5 px-3 py-2 bg-surface border border-border text-text-muted rounded-lg text-sm font-medium hover:bg-surface-lighter hover:text-text transition-all disabled:opacity-50"
              title="刷新模型列表"
            >
              <RefreshCw className="w-4 h-4" />
              刷新
            </button>
            <button
              onClick={handleDeleteCheckpoint}
              disabled={isLoading || isDeleting || !selectedCheckpoint}
              className="flex items-center gap-1.5 px-3 py-2 bg-error/10 border border-error/20 text-error rounded-lg text-sm font-medium hover:bg-error/20 transition-all disabled:opacity-50"
              title="删除当前选中的本地 checkpoint"
            >
              {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
              删除
            </button>
            <button
              onClick={handleLoad}
              disabled={isLoading || isDeleting || !selectedCheckpoint}
              className="flex items-center gap-1.5 px-4 py-2 bg-primary text-white rounded-lg text-sm font-medium hover:bg-primary-dark transition-all disabled:opacity-50"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Download className="w-4 h-4" />
              )}
              加载模型
            </button>
          </div>
          {loadError && (
            <div className="mt-3 text-xs text-error bg-error/10 border border-error/20 rounded-lg px-3 py-2">
              {loadError}
            </div>
          )}
        </div>
      )}

      {isLoaded && (
        <div className="px-6 py-2 border-b border-border bg-surface-light/30 flex items-center justify-between">
          <span className="text-xs text-text-muted">
            当前模型: {checkpoints.find((c) => c.path === loadedCheckpoint)?.name || checkpoints.find((c) => c.path === selectedCheckpoint)?.name}
          </span>
          <button
            onClick={handleUnload}
            className="flex items-center gap-1 text-xs text-text-muted hover:text-error transition-colors"
          >
            <Upload className="w-3 h-3" />
            卸载
          </button>
        </div>
      )}

      {/* Settings panel */}
      {showSettings && (
        <div className="px-6 py-4 border-b border-border bg-surface-light/50 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="flex items-center gap-1 text-xs text-text-muted mb-1">
              温度
              <ParamTooltip content="控制生成文本的随机性。较高的温度产生更多样但可能不太连贯的文本，较低则更保守。" />
            </label>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={settings.temperature}
              onChange={(e) =>
                setSettings((prev) => ({ ...prev, temperature: Number(e.target.value) }))
              }
              className="w-full accent-primary"
            />
            <span className="text-xs font-mono text-text-muted">{settings.temperature}</span>
          </div>
          <div>
            <label className="flex items-center gap-1 text-xs text-text-muted mb-1">
              Top-K
              <ParamTooltip content="只从概率最高的 K 个 token 中采样。较小的值使输出更确定，较大的值更多样。" />
            </label>
            <input
              type="number"
              min={1}
              max={200}
              value={settings.topK}
              onChange={(e) =>
                setSettings((prev) => ({ ...prev, topK: Number(e.target.value) }))
              }
              className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-text"
            />
          </div>
          <div>
            <label className="flex items-center gap-1 text-xs text-text-muted mb-1">
              重复惩罚
              <ParamTooltip content="惩罚模型重复已生成的内容。大于 1 减少重复，等于 1 不惩罚。" />
            </label>
            <input
              type="number"
              min={1}
              max={3}
              step={0.1}
              value={settings.repetitionPenalty}
              onChange={(e) =>
                setSettings((prev) => ({
                  ...prev,
                  repetitionPenalty: Number(e.target.value),
                }))
              }
              className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-text"
            />
          </div>
          <div>
            <label className="flex items-center gap-1 text-xs text-text-muted mb-1">
              最大 Token
              <ParamTooltip content="模型最多生成多少个 token。控制回复长度的上限。" />
            </label>
            <input
              type="number"
              min={32}
              max={4096}
              step={32}
              value={settings.maxTokens}
              onChange={(e) =>
                setSettings((prev) => ({ ...prev, maxTokens: Number(e.target.value) }))
              }
              className="w-full bg-surface border border-border rounded px-2 py-1 text-xs text-text"
            />
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-text-muted">
            <Bot className="w-16 h-16 mb-4 opacity-20" />
            <h2 className="text-lg font-medium mb-1">开始对话</h2>
            <p className="text-sm text-center max-w-md">
              {isLoaded
                ? '模型已就绪，在下方输入消息开始对话吧！'
                : '请先加载一个训练好的模型检查点，然后就可以开始对话了。'}
            </p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              'flex gap-3 w-full max-w-3xl min-w-0',
              msg.role === 'user' ? 'ml-auto flex-row-reverse' : ''
            )}
          >
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center shrink-0',
                msg.role === 'user' ? 'bg-primary/20' : 'bg-surface-lighter'
              )}
            >
              {msg.role === 'user' ? (
                <User className="w-4 h-4 text-primary" />
              ) : (
                <Bot className="w-4 h-4 text-text-muted" />
              )}
            </div>
            <div
              className={cn(
                'rounded-2xl px-4 py-3 text-sm max-w-[80%] min-w-0 overflow-hidden',
                msg.role === 'user'
                  ? 'bg-primary text-white rounded-br-sm'
                  : 'bg-surface-light border border-border rounded-bl-sm'
              )}
            >
              <div className="whitespace-pre-wrap break-words [overflow-wrap:anywhere]">{msg.content}</div>
              {msg.role === 'assistant' && isGenerating && i === messages.length - 1 && (
                <span className="inline-block w-1.5 h-4 bg-primary ml-0.5 animate-pulse" />
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 border-t border-border bg-surface-light shrink-0">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isLoaded ? '输入消息...' : '请先加载模型'}
            disabled={!isLoaded}
            rows={1}
            className="flex-1 bg-surface border border-border rounded-xl px-4 py-3 text-sm text-text resize-none focus:outline-none focus:border-primary transition-colors disabled:opacity-50"
          />
          <button
            onClick={sendMessage}
            disabled={!input.trim() || isGenerating || !isLoaded}
            className="px-4 py-3 bg-primary text-white rounded-xl hover:bg-primary-dark transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      {pendingDelete && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
          onClick={() => !isDeleting && setPendingDelete(null)}
        >
          <div
            className="w-full max-w-md rounded-xl border border-border bg-surface-light p-5 shadow-xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full bg-error/10 p-2 text-error">
                <Trash2 className="w-5 h-5" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-text">确认删除模型？</h2>
                <p className="mt-2 text-sm text-text-muted">
                  将删除本地 checkpoint，无法在界面中恢复。
                </p>
                <p className="mt-3 rounded-lg border border-border bg-surface px-3 py-2 text-xs text-text-muted">
                  {pendingDelete.label}
                </p>
              </div>
            </div>
            <div className="mt-5 flex justify-end gap-2">
              <button
                onClick={() => setPendingDelete(null)}
                disabled={isDeleting}
                className="px-4 py-2 rounded-lg text-sm text-text-muted hover:text-text hover:bg-surface-lighter disabled:opacity-50"
              >
                取消
              </button>
              <button
                onClick={confirmDeleteCheckpoint}
                disabled={isDeleting}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-error text-white text-sm font-medium hover:bg-error/80 disabled:opacity-50"
              >
                {isDeleting && <Loader2 className="w-4 h-4 animate-spin" />}
                确认删除
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

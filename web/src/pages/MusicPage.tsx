import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  Download,
  FileMusic,
  Loader2,
  Music2,
  Play,
  RefreshCw,
  Share2,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  Upload,
} from 'lucide-react';
import { deleteCheckpoint, getCheckpoints, loadModel, unloadModel } from '@/api/client';
import { cn } from '@/lib/utils';
import type { ChatMessage, Checkpoint } from '@/types';

type MusicPreset = {
  id: string;
  label: string;
  description: string;
  style: string;
  mood: string;
  key: string;
  seed: string;
};

const presets: MusicPreset[] = [
  {
    id: 'pop-axis',
    label: '流行循环',
    description: 'I-V-vi-IV，适合主歌/副歌快速验证',
    style: 'pop anthem',
    mood: 'uplifting',
    key: 'C major',
    seed: 'I V vi IV',
  },
  {
    id: 'mandopop-ballad',
    label: '华语抒情',
    description: '温暖、怀旧的中速流行和声',
    style: 'mandopop ballad',
    mood: 'nostalgic',
    key: 'A minor',
    seed: 'i VI III VII',
  },
  {
    id: 'jazz-turnaround',
    label: '爵士转接',
    description: 'ii-V-I 与扩展和弦感',
    style: 'jazz turnaround',
    mood: 'smooth',
    key: 'C major',
    seed: 'ii V I I',
  },
  {
    id: 'neo-soul',
    label: 'Neo-soul',
    description: '温暖、丝滑、适合 Rhodes/电钢',
    style: 'neo-soul',
    mood: 'warm',
    key: 'D major',
    seed: 'I iii IV V',
  },
  {
    id: 'cinematic-minor',
    label: '电影感小调',
    description: '史诗、悬疑或情绪推进',
    style: 'cinematic trailer',
    mood: 'dramatic',
    key: 'E minor',
    seed: 'i VI III VII',
  },
];

const NOTE_TO_SEMITONE: Record<string, number> = {
  C: 0,
  'C#': 1,
  Db: 1,
  D: 2,
  'D#': 3,
  Eb: 3,
  E: 4,
  F: 5,
  'F#': 6,
  Gb: 6,
  G: 7,
  'G#': 8,
  Ab: 8,
  A: 9,
  'A#': 10,
  Bb: 10,
  B: 11,
};

const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10];
const ROMAN_DEGREE: Record<string, number> = {
  I: 0,
  II: 1,
  III: 2,
  IV: 3,
  V: 4,
  VI: 5,
  VII: 6,
};

function parseProgression(text: string): string[] {
  const candidates = [
    /Progression:\s*([\s\S]*?)(?:\s+\|\s*(?:Roman|Detected Key|Key|Style|Mood):|$)/i,
    /Roman:\s*([\s\S]*?)(?:\s+\|\s*(?:Progression|Detected Key|Key|Style|Mood):|$)/i,
  ];
  for (const pattern of candidates) {
    const match = text.match(pattern);
    const chords = match?.[1] ? splitChordLine(match[1]) : [];
    if (chords.length > 0) {
      return chords;
    }
  }
  return splitChordLine(text);
}

function splitChordLine(text: string): string[] {
  return text
    .replace(/\b(?:Roman|Progression|Detected Key|Key|Style|Mood):/gi, ' ')
    .replace(/[，,]/g, ' ')
    .split(/\s*(?:\||-|–|—|→|,|\n)\s*|\s+/)
    .map((token) => token.trim())
    .filter((token) => isChordSyntax(token));
}

function isChordSyntax(token: string) {
  const cleaned = token.replace(/[()[\]{}]/g, '').trim();
  if (!cleaned || cleaned === '//') return false;
  if (/^(major|minor|maj|min|style|mood|key|detected)$/i.test(cleaned)) return false;
  if (/^[A-Ga-g](?:#|b)?(?:maj|min|m(?!aj)|dim|aug|sus|add|\+|°|o|\d|\/|#|b)*$/i.test(cleaned)) return true;
  return /^[#b]?[ivIV]+(?:maj|min|m(?!aj)|dim|aug|sus|add|\+|°|o|\d|\/|#|b)*$/i.test(cleaned);
}

function buildPrompt(style: string, mood: string, key: string, seed: string) {
  return [
    'Generate one concise chord progression for a music producer.',
    `Style: ${style}`,
    `Mood: ${mood}`,
    `Key: ${key}`,
    `Seed progression or roman numerals: ${seed}`,
    'Return exactly one line in this format:',
    'Roman: <roman numerals with | bar separators> | Progression: <playable chords>',
  ].join('\n');
}

function getInitialMusicConfig() {
  const fallback = presets[0];
  if (typeof window === 'undefined') {
    return {
      selectedPreset: fallback.id,
      style: fallback.style,
      mood: fallback.mood,
      musicKey: fallback.key,
      seed: fallback.seed,
      bpm: 90,
    };
  }
  const params = new URLSearchParams(window.location.search);
  const preset = presets.find((item) => item.id === params.get('preset')) || fallback;
  return {
    selectedPreset: preset.id,
    style: params.get('style') || preset.style,
    mood: params.get('mood') || preset.mood,
    musicKey: params.get('key') || preset.key,
    seed: params.get('seed') || preset.seed,
    bpm: Number(params.get('bpm')) || 90,
  };
}

function midiToFrequency(midi: number) {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

function normalizeRoot(root: string) {
  return root.charAt(0).toUpperCase() + root.slice(1);
}

function getKeyRoot(key: string) {
  const match = key.trim().match(/^([A-Ga-g](?:#|b)?)/);
  return match ? normalizeRoot(match[1]) : 'C';
}

function isMinorKey(key: string) {
  const normalized = key.trim();
  return /\b(minor|min|m)\b/i.test(normalized) || /^[A-G](?:#|b)?m$/i.test(normalized);
}

function chordTokenToFrequencies(token: string, key: string): number[] {
  return chordTokenToMidiNotes(token, key).map(midiToFrequency);
}

function chordTokenToMidiNotes(token: string, key: string): number[] {
  const cleaned = token.replace(/[()[\]{}]/g, '').trim();
  if (!cleaned || cleaned === '//') return [];

  const keyRoot = NOTE_TO_SEMITONE[getKeyRoot(key)] ?? 0;
  const scale = isMinorKey(key) ? MINOR_SCALE : MAJOR_SCALE;
  const romanMatch = cleaned.match(/^([#b]?)([ivIV]+)(?:[°o]|dim|aug|\+|m|maj|min|sus|add|\d|#|b)*/);
  const absoluteMatch = cleaned.match(/^([A-Ga-g](?:#|b)?)(.*)$/);
  let rootSemitone: number | null = null;
  let quality = 'major';

  if (romanMatch) {
    const accidental = romanMatch[1];
    const roman = romanMatch[2];
    const degree = ROMAN_DEGREE[roman.toUpperCase()];
    if (degree !== undefined) {
      const accidentalOffset = accidental === '#' ? 1 : accidental === 'b' ? -1 : 0;
      rootSemitone = (keyRoot + scale[degree] + accidentalOffset + 12) % 12;
      quality = roman === roman.toLowerCase() ? 'minor' : 'major';
      if (/[°o]|dim/i.test(cleaned)) quality = 'dim';
      if (/aug|\+/i.test(cleaned)) quality = 'aug';
    }
  } else if (absoluteMatch) {
    const root = normalizeRoot(absoluteMatch[1]);
    rootSemitone = NOTE_TO_SEMITONE[root] ?? null;
    const suffix = absoluteMatch[2] || '';
    if (/^m(?!aj)|min/i.test(suffix)) quality = 'minor';
    if (/dim|°|o/i.test(suffix)) quality = 'dim';
    if (/aug|\+/i.test(suffix)) quality = 'aug';
  }

  if (rootSemitone === null) return [];
  const intervals = quality === 'minor' ? [0, 3, 7] : quality === 'dim' ? [0, 3, 6] : quality === 'aug' ? [0, 4, 8] : [0, 4, 7];
  const rootMidi = 60 + rootSemitone;
  return intervals.map((interval) => rootMidi + interval);
}

function playChordInContext(context: AudioContext, frequencies: number[], startAt: number, duration: number) {
  const master = context.createGain();
  master.gain.setValueAtTime(0.0001, startAt);
  master.gain.exponentialRampToValueAtTime(0.18, startAt + 0.03);
  master.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);
  master.connect(context.destination);

  frequencies.forEach((frequency) => {
    const oscillator = context.createOscillator();
    oscillator.type = 'triangle';
    oscillator.frequency.setValueAtTime(frequency, startAt);
    oscillator.connect(master);
    oscillator.start(startAt);
    oscillator.stop(startAt + duration + 0.05);
  });
}

function createAudioContext() {
  const AudioContextCtor =
    window.AudioContext ||
    (window as Window & typeof globalThis & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!AudioContextCtor) throw new Error('当前浏览器不支持 Web Audio。');
  return new AudioContextCtor();
}

function writeUint32(value: number) {
  return [(value >> 24) & 255, (value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function writeUint16(value: number) {
  return [(value >> 8) & 255, value & 255];
}

function writeVarLen(value: number) {
  let buffer = value & 0x7f;
  const bytes = [];
  while ((value >>= 7)) {
    buffer <<= 8;
    buffer |= (value & 0x7f) | 0x80;
  }
  while (true) {
    bytes.push(buffer & 0xff);
    if (buffer & 0x80) buffer >>= 8;
    else break;
  }
  return bytes;
}

function buildMidiBytes(chords: string[], key: string, bpm: number) {
  const ticksPerQuarter = 480;
  const chordTicks = ticksPerQuarter * 2;
  const tempo = Math.round(60_000_000 / Math.max(40, Math.min(220, bpm)));
  const track: number[] = [
    0x00,
    0xff,
    0x51,
    0x03,
    (tempo >> 16) & 255,
    (tempo >> 8) & 255,
    tempo & 255,
  ];

  chords.forEach((chord) => {
    const notes = chordTokenToMidiNotes(chord, key);
    if (notes.length === 0) return;
    notes.forEach((note) => {
      track.push(...writeVarLen(0), 0x90, note, 84);
    });
    notes.forEach((note, index) => {
      track.push(...writeVarLen(index === 0 ? chordTicks : 0), 0x80, note, 0);
    });
  });
  track.push(0x00, 0xff, 0x2f, 0x00);

  return new Uint8Array([
    0x4d,
    0x54,
    0x68,
    0x64,
    ...writeUint32(6),
    ...writeUint16(0),
    ...writeUint16(1),
    ...writeUint16(ticksPerQuarter),
    0x4d,
    0x54,
    0x72,
    0x6b,
    ...writeUint32(track.length),
    ...track,
  ]);
}

export default function MusicPage() {
  const initialConfig = useMemo(() => getInitialMusicConfig(), []);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [loadedCheckpoint, setLoadedCheckpoint] = useState('');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<{ path: string; label: string } | null>(null);
  const [selectedPreset, setSelectedPreset] = useState(initialConfig.selectedPreset);
  const [style, setStyle] = useState(initialConfig.style);
  const [mood, setMood] = useState(initialConfig.mood);
  const [musicKey, setMusicKey] = useState(initialConfig.musicKey);
  const [seed, setSeed] = useState(initialConfig.seed);
  const [bpm, setBpm] = useState(initialConfig.bpm);
  const [rawOutput, setRawOutput] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const generatingRef = useRef(false);
  const audioContextRef = useRef<AudioContext | null>(null);
  const playbackTimerRef = useRef<number | null>(null);

  const timeline = useMemo(() => parseProgression(rawOutput), [rawOutput]);
  const prompt = useMemo(() => buildPrompt(style, mood, musicKey, seed), [style, mood, musicKey, seed]);

  const refreshCheckpoints = useCallback(() => {
    const params = new URLSearchParams(window.location.search);
    getCheckpoints()
      .then((items) => {
        const musicCheckpoints = items.filter((item) => item.dataDomain === 'music');
        setCheckpoints(musicCheckpoints);
        const checkpointFromUrl = params.get('checkpoint');
        const music = musicCheckpoints.find((item) => item.isSft) || musicCheckpoints[0];
        setSelectedCheckpoint((current) => {
          if (current && musicCheckpoints.some((item) => item.path === current)) return current;
          if (checkpointFromUrl && musicCheckpoints.some((item) => item.path === checkpointFromUrl)) {
            return checkpointFromUrl;
          }
          if (music) return music.path;
          setNotice('暂无音乐模型，请先在训练流程中选择音乐数据并完成训练。');
          return '';
        });
      })
      .catch((err) => setError(err instanceof Error ? err.message : '读取模型列表失败'));
  }, []);

  useEffect(() => {
    refreshCheckpoints();
    window.addEventListener('focus', refreshCheckpoints);
    return () => window.removeEventListener('focus', refreshCheckpoints);
  }, [refreshCheckpoints]);

  useEffect(() => {
    return () => {
      if (playbackTimerRef.current !== null) {
        window.clearTimeout(playbackTimerRef.current);
        playbackTimerRef.current = null;
      }
      audioContextRef.current?.close();
      audioContextRef.current = null;
    };
  }, []);

  const applyPreset = (presetId: string) => {
    const preset = presets.find((item) => item.id === presetId) || presets[0];
    setSelectedPreset(preset.id);
    setStyle(preset.style);
    setMood(preset.mood);
    setMusicKey(preset.key);
    setSeed(preset.seed);
  };

  const handleLoad = async () => {
    if (!selectedCheckpoint) return;
    setIsLoading(true);
    setError('');
    setNotice('');
    try {
      await loadModel(selectedCheckpoint);
      setLoadedCheckpoint(selectedCheckpoint);
      setIsLoaded(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : '模型加载失败');
      setIsLoaded(false);
      setLoadedCheckpoint('');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUnload = async () => {
    await unloadModel();
    setIsLoaded(false);
    setLoadedCheckpoint('');
    setIsGenerating(false);
    setRawOutput('');
    stopPlayback();
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
    setError('');
    try {
      await deleteCheckpoint(pendingDelete.path);
      setIsLoaded(false);
      setLoadedCheckpoint('');
      setRawOutput('');
      stopPlayback();
      setNotice('模型 checkpoint 已删除。');
      setPendingDelete(null);
      refreshCheckpoints();
    } catch (err) {
      setError(err instanceof Error ? err.message : '删除模型失败');
    } finally {
      setIsDeleting(false);
    }
  };

  const generate = useCallback(async () => {
    if (!isLoaded || generatingRef.current) return;
    generatingRef.current = true;
    setIsGenerating(true);
    setError('');
    setRawOutput('');

    const messages: ChatMessage[] = [{ role: 'user', content: prompt, timestamp: Date.now() }];

    try {
      const response = await fetch('/api/music/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages,
          temperature: 0.7,
          top_k: 50,
          repetition_penalty: 1.15,
          max_tokens: 160,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.detail || `后端返回 ${response.status}`);
      }
      setRawOutput(typeof data.text === 'string' ? data.text : '');
    } catch (err) {
      setError(err instanceof Error ? err.message : '生成失败');
    } finally {
      generatingRef.current = false;
      setIsGenerating(false);
    }
  }, [isLoaded, prompt]);

  const stopPlayback = useCallback(() => {
    if (playbackTimerRef.current !== null) {
      window.clearTimeout(playbackTimerRef.current);
      playbackTimerRef.current = null;
    }
    audioContextRef.current?.close();
    audioContextRef.current = null;
    setIsPlaying(false);
  }, []);

  const playChord = useCallback((chord: string) => {
    stopPlayback();
    try {
      const frequencies = chordTokenToFrequencies(chord, musicKey);
      if (frequencies.length === 0) {
        setError(`无法试听和弦：${chord}`);
        return;
      }
      const context = createAudioContext();
      audioContextRef.current = context;
      setIsPlaying(true);
      playChordInContext(context, frequencies, context.currentTime + 0.02, 1.1);
      playbackTimerRef.current = window.setTimeout(stopPlayback, 1250);
    } catch (err) {
      setError(err instanceof Error ? err.message : '试听失败');
      setIsPlaying(false);
    }
  }, [musicKey, stopPlayback]);

  const playTimeline = useCallback(() => {
    if (timeline.length === 0) return;
    stopPlayback();
    try {
      const context = createAudioContext();
      audioContextRef.current = context;
      const beatSeconds = 60 / Math.max(40, Math.min(220, bpm));
      const chordSeconds = beatSeconds * 2;
      let playableCount = 0;
      timeline.forEach((chord, index) => {
        const frequencies = chordTokenToFrequencies(chord, musicKey);
        if (frequencies.length > 0) {
          playableCount += 1;
          playChordInContext(context, frequencies, context.currentTime + 0.05 + index * chordSeconds, chordSeconds * 0.9);
        }
      });
      if (playableCount === 0) {
        setError('当前时间线没有可试听的和弦。');
        context.close();
        audioContextRef.current = null;
        return;
      }
      setIsPlaying(true);
      playbackTimerRef.current = window.setTimeout(stopPlayback, timeline.length * chordSeconds * 1000 + 300);
    } catch (err) {
      setError(err instanceof Error ? err.message : '试听失败');
      setIsPlaying(false);
    }
  }, [bpm, musicKey, stopPlayback, timeline]);

  const exportMidi = useCallback(() => {
    if (timeline.length === 0) return;
    try {
      const bytes = buildMidiBytes(timeline, musicKey, bpm);
      const blob = new Blob([bytes], { type: 'audio/midi' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `minigpt-studio-chords-${Date.now()}.mid`;
      link.click();
      URL.revokeObjectURL(url);
      setNotice('MIDI 已导出。');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'MIDI 导出失败');
    }
  }, [bpm, musicKey, timeline]);

  const shareLink = useCallback(async () => {
    const url = new URL(window.location.href);
    url.pathname = '/music';
    url.searchParams.set('preset', selectedPreset);
    url.searchParams.set('style', style);
    url.searchParams.set('mood', mood);
    url.searchParams.set('key', musicKey);
    url.searchParams.set('seed', seed);
    url.searchParams.set('bpm', String(bpm));
    if (selectedCheckpoint) url.searchParams.set('checkpoint', selectedCheckpoint);
    try {
      await navigator.clipboard.writeText(url.toString());
      setNotice('分享链接已复制到剪贴板。');
    } catch {
      setNotice(url.toString());
    }
  }, [bpm, mood, musicKey, seed, selectedCheckpoint, selectedPreset, style]);

  return (
    <div className="px-5 py-6 lg:px-6 w-full space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Music2 className="w-6 h-6 text-primary" />
            <h1 className="text-2xl font-bold">音乐生成</h1>
            {isLoaded && (
              <span className="px-2 py-0.5 bg-success/20 text-success text-xs rounded-full">
                已加载
              </span>
            )}
          </div>
          <p className="text-text-muted">
            用音乐 checkpoint 生成和弦进行，并把结果解析成可读的和弦时间线。
          </p>
        </div>
      </div>

      {!isLoaded && (
        <div className="rounded-xl border border-border bg-surface-light/50 px-4 py-4">
          <div className="flex items-center gap-3">
            <select
              value={selectedCheckpoint}
              onChange={(event) => setSelectedCheckpoint(event.target.value)}
              className="flex-1 bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text"
            >
              {checkpoints.map((checkpoint) => (
                <option key={checkpoint.path} value={checkpoint.path}>
                  {checkpoint.name} ({checkpoint.isSft ? 'SFT' : '预训练'}, step={checkpoint.step})
                </option>
              ))}
              {checkpoints.length === 0 && <option>暂无可用音乐模型</option>}
            </select>
            <button
              onClick={refreshCheckpoints}
              disabled={isLoading || isDeleting}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-surface border border-border text-text-muted text-sm font-medium hover:bg-surface-lighter hover:text-text disabled:opacity-50 transition-all"
              title="刷新模型列表"
            >
              <RefreshCw className="w-4 h-4" />
              刷新
            </button>
            <button
              onClick={handleDeleteCheckpoint}
              disabled={isLoading || isDeleting || !selectedCheckpoint}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-error/10 border border-error/20 text-error text-sm font-medium hover:bg-error/20 disabled:opacity-50 transition-all"
              title="删除当前选中的本地 checkpoint"
            >
              {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
              删除
            </button>
            <button
              onClick={handleLoad}
              disabled={isLoading || isDeleting || !selectedCheckpoint}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-dark disabled:opacity-50 transition-all"
            >
              {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
              加载模型
            </button>
          </div>
        </div>
      )}

      {isLoaded && (
        <div className="rounded-xl border border-border bg-surface-light/30 px-4 py-2 flex items-center justify-between">
          <span className="text-xs text-text-muted">
            当前模型: {checkpoints.find((item) => item.path === loadedCheckpoint)?.name || checkpoints.find((item) => item.path === selectedCheckpoint)?.name}
          </span>
          <button
            onClick={handleUnload}
            disabled={isGenerating}
            className="flex items-center gap-1 text-xs text-text-muted hover:text-error transition-colors disabled:opacity-50"
          >
            <Upload className="w-3 h-3" />
            卸载
          </button>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-error/20 bg-error/10 px-4 py-3 text-sm text-error">
          {error}
        </div>
      )}
      {notice && (
        <div className="rounded-xl border border-primary/20 bg-primary/10 px-4 py-3 text-sm text-primary-light">
          {notice}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6">
        <section className="rounded-xl border border-border bg-surface/70 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            <div>
              <div className="text-sm font-medium">Prompt Presets</div>
              <div className="text-xs text-text-muted">选择一个音乐方向，然后按需微调。</div>
            </div>
          </div>
          <div className="p-4 space-y-3">
            {presets.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset.id)}
                className={cn(
                  'w-full text-left rounded-lg border px-3 py-2 transition-all',
                  selectedPreset === preset.id
                    ? 'border-primary bg-primary/10 text-primary-light'
                    : 'border-border bg-surface hover:bg-surface-lighter text-text'
                )}
              >
                <div className="text-sm font-medium">{preset.label}</div>
                <div className="text-xs text-text-muted mt-0.5">{preset.description}</div>
              </button>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-border bg-surface/70 overflow-hidden">
          <div className="px-4 py-3 border-b border-border flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-primary" />
            <div>
              <div className="text-sm font-medium">生成参数</div>
              <div className="text-xs text-text-muted">当前先生成文本和弦时间线，试听/MIDI 在下一步加入。</div>
            </div>
          </div>
          <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted">风格</span>
              <input value={style} onChange={(event) => setStyle(event.target.value)} className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text" />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted">情绪</span>
              <input value={mood} onChange={(event) => setMood(event.target.value)} className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text" />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted">调性</span>
              <input value={musicKey} onChange={(event) => setMusicKey(event.target.value)} className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text" />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted">起始和弦 / 罗马数字</span>
              <input value={seed} onChange={(event) => setSeed(event.target.value)} className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text" />
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-text-muted">BPM</span>
              <input type="number" min={40} max={220} value={bpm} onChange={(event) => setBpm(Number(event.target.value) || 90)} className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text" />
            </label>
            <div className="md:col-span-2 flex items-center justify-between gap-3 rounded-lg border border-border bg-surface px-3 py-2">
              <div className="text-xs text-text-muted font-mono truncate">{prompt.replace(/\n/g, ' / ')}</div>
              <button
                onClick={generate}
                disabled={!isLoaded || isGenerating}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-primary text-white text-sm font-medium hover:bg-primary-dark disabled:opacity-50 shrink-0"
              >
                {isGenerating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                生成和弦
              </button>
            </div>
          </div>
        </section>
      </div>

      <section className="rounded-xl border border-border bg-surface/70 overflow-hidden">
        <div className="px-4 py-3 border-b border-border flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-medium">和弦时间线</div>
            <div className="text-xs text-text-muted">优先解析 `Roman:`，否则解析 `Progression:` 或原始输出。</div>
          </div>
          <button
            type="button"
            onClick={isPlaying ? stopPlayback : playTimeline}
            disabled={timeline.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-light border border-border text-xs text-text hover:bg-surface-lighter disabled:opacity-50"
          >
            {isPlaying ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            {isPlaying ? '停止' : '试听整段'}
          </button>
          <button
            type="button"
            onClick={exportMidi}
            disabled={timeline.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-light border border-border text-xs text-text hover:bg-surface-lighter disabled:opacity-50"
          >
            <FileMusic className="w-3.5 h-3.5" />
            导出 MIDI
          </button>
          <button
            type="button"
            onClick={shareLink}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-light border border-border text-xs text-text hover:bg-surface-lighter"
          >
            <Share2 className="w-3.5 h-3.5" />
            分享配置
          </button>
        </div>
        <div className="p-4">
          {timeline.length > 0 ? (
            <div className="flex gap-2 overflow-x-auto pb-2">
              {timeline.map((chord, index) => (
                <button
                  key={`${chord}-${index}`}
                  type="button"
                  onClick={() => playChord(chord)}
                  className="min-w-20 rounded-lg border border-border bg-surface-light px-3 py-3 text-center hover:border-primary/60 hover:bg-primary/10 transition-colors"
                >
                  <div className="text-[10px] text-text-muted mb-1">#{index + 1}</div>
                  <div className="font-mono text-sm text-text">{chord}</div>
                  <div className="text-[10px] text-text-muted mt-1">试听</div>
                </button>
              ))}
            </div>
          ) : (
            <div className="text-sm text-text-muted">还没有生成结果。加载音乐模型后点击“生成和弦”。</div>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-border bg-surface/70 overflow-hidden">
        <div className="px-4 py-3 border-b border-border text-sm font-medium">原始输出</div>
        <pre className="p-4 min-h-32 whitespace-pre-wrap text-xs text-text-muted font-mono">
          {rawOutput || '—'}
        </pre>
      </section>
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
                <h2 className="text-base font-semibold text-text">确认删除音乐模型？</h2>
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

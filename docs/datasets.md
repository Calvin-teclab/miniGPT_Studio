# 可选数据集与原始数据格式

本文档说明训练流程第 1 步“准备训练数据”里当前可选的数据类型、数据来源、原始数据样例，以及 tokenizer 拆分方式。

## 1. 数据类型总览

当前前端支持两类训练数据：

| 数据类型 | UI 显示 | 用途 | 缓存目录 | 当前状态 |
| --- | --- | --- | --- | --- |
| 文字 / 通用文本 | `文字 / 通用文本` | 训练通用语言模型，学习自然语言续写能力 | `~/.cache/nanochat` | 已接入 |
| 音乐 / 和弦数据 | `音乐 / 和弦数据` | 训练能理解和生成和弦、风格、情绪、调性的模型 | `~/.cache/nanochat_music` | 已接入 |

两类数据使用不同缓存目录，因此音乐数据不会覆盖原有通用文本训练数据。

## 2. 文字 / 通用文本

### 2.1 内置数据集：FineWeb-Edu

UI 选项：

```text
数据类型：文字 / 通用文本
数据来源：选择内置/推荐数据集
可选文字数据集：FineWeb-Edu 教育文本（默认）
```

用途：

- 用于训练通用语言模型。
- 内容偏教育文本，适合做小型 GPT 的基础预训练。
- 后端会下载或复用 FineWeb-Edu 的 parquet 分片。

后端执行方式：

```bash
python -m nanochat_mlx.dataset -n 4
```

其中 `-n 4` 对应 UI 中的“数据分片数”。

缓存结果：

```text
~/.cache/nanochat/base_data/shard_00000.parquet
~/.cache/nanochat/base_data/shard_00001.parquet
...
```

parquet 内部核心字段：

```text
text
```

示意原始内容：

```text
The photosynthesis process allows plants to convert sunlight into chemical energy...
```

说明：

- FineWeb-Edu 原始数据由下载脚本拉取，项目内不直接保存完整原始语料。
- 页面会显示本地缓存检测状态，避免用户误以为刷新页面后自动重新下载。

### 2.2 自定义文字数据

UI 选项：

```text
数据类型：文字 / 通用文本
数据来源：填写本地数据集路径 / 从浏览器上传数据集
```

支持格式：

| 格式 | 说明 |
| --- | --- |
| `.txt` | 每个非空行作为一个训练样本 |
| `.md` | 每个非空行作为一个训练样本 |
| `.jsonl` | 每行一个 JSON 对象，支持 `text` 或 `prompt` + `completion` |
| `.csv` | 支持 `text` 列，或 `prompt` + `completion` 列 |

TXT 示例：

```text
机器学习是一种让计算机从数据中学习规律的方法。
大语言模型通过预测下一个 token 来学习文本分布。
训练数据质量会直接影响模型生成质量。
```

JSONL 示例：

```jsonl
{"text":"机器学习是一种让计算机从数据中学习规律的方法。"}
{"prompt":"解释什么是 token。","completion":"token 是模型处理文本时使用的最小文本单元，可以是字符、子词或符号片段。"}
```

CSV 示例：

```csv
prompt,completion
解释什么是预训练,预训练是让模型在大量通用文本上学习语言规律的阶段。
解释什么是分词器,分词器负责把文本转换成模型可处理的 token id。
```

上传后的本地保存位置：

```text
~/.cache/nanochat/uploaded_datasets/<filename>
```

转换后的训练分片：

```text
~/.cache/nanochat/base_data/shard_00000.parquet
~/.cache/nanochat/base_data/shard_00001.parquet
```

## 3. 音乐 / 和弦数据

### 3.1 内置数据集：和弦示例集

UI 选项：

```text
数据类型：音乐 / 和弦数据
数据来源：选择内置/推荐数据集
可选音乐数据集：内置和弦示例集（快速验证）
```

用途：

- 用于快速验证音乐数据链路是否跑通。
- 内置样例覆盖 pop、jazz、neo-soul、bossa、rock、mandopop、blues、cinematic 等常见风格。
- 后端会自动把可解析的绝对和弦转换成罗马数字，降低调性差异，帮助小模型学习和声结构。
- 正式训练仍建议上传更大规模的和弦/曲式/风格数据。

当前内置原始 JSONL：

```jsonl
{"key":"C major","style":"dream pop","mood":"warm","progression":"Cmaj7 - Em7 - Fmaj7 - G13"}
{"key":"A minor","style":"jazz","mood":"smooth","progression":"Am9 - D13 - Gmaj9 - Cmaj7 - F#dim7 - B7b9 - Em9"}
{"key":"E minor","style":"cinematic","mood":"hopeful","progression":"Emadd9 - Cmaj7 - G - D/F#"}
{"prompt":"Generate a beautiful neo-soul chord loop in D major.","completion":"Dmaj9 - F#m7 - Gmaj9 - A13sus - Bm9 - E13 - Amaj9"}
```

后端会转换为统一文本。可解析和弦会额外生成 `Roman` 与 `Detected Key`：

```text
Style: dream pop | Mood: warm | Key: C major | Roman: I iii IV V7 | Detected Key: C major | Progression: Cmaj7 - Em7 - Fmaj7 - G13
Style: jazz | Mood: smooth | Key: A minor | Roman: i iv VII III b6 V7 v | Detected Key: A minor | Progression: Am9 - D13 - Gmaj9 - Cmaj7 - F#dim7 - B7b9 - Em9
Style: cinematic | Mood: hopeful | Key: E minor | Roman: i VI III VII | Detected Key: E minor | Progression: Emadd9 - Cmaj7 - G - D/F#
Generate a beautiful neo-soul chord loop in D major.
Dmaj9 - F#m7 - Gmaj9 - A13sus - Bm9 - E13 - Amaj9
```

缓存位置：

```text
~/.cache/nanochat_music/uploaded_datasets/starter_chords.jsonl
~/.cache/nanochat_music/base_data/shard_00000.parquet
~/.cache/nanochat_music/base_data/shard_00001.parquet
```

### 3.2 自定义音乐 / 和弦数据

UI 选项：

```text
数据类型：音乐 / 和弦数据
数据来源：填写本地数据集路径 / 从浏览器上传数据集
```

支持格式：

| 格式 | 说明 |
| --- | --- |
| `.jsonl` | 推荐格式，每行一个和弦样本 |
| `.csv` | 表格化和弦样本 |
| `.txt` | 每行一个和弦/曲式文本，也支持 ChordPro 风格文件 |
| `.md` | 每行一个和弦/曲式文本，也支持 ChordPro 风格文件 |

推荐 JSONL 字段：

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| `key` | 调性 | `C major` |
| `tonality` | 调性别名 | `A minor` |
| `scale` | 音阶 | `D dorian` |
| `style` | 风格 | `jazz` |
| `genre` | 风格别名 | `neo-soul` |
| `mood` | 情绪 | `warm` |
| `emotion` | 情绪别名 | `hopeful` |
| `progression` | 和弦进行 | `Cmaj7 - Am7 - Dm7 - G13` |
| `chords` | 和弦进行别名 | `Em - C - G - D` |
| `chord_progression` | 和弦进行别名 | `ii - V - I` |
| `prompt` | 指令输入 | `Write a jazz progression in A minor.` |
| `completion` | 目标输出 | `Am9 - D13 - Gmaj9 - Cmaj7` |
| `text` | 已整理好的完整训练文本 | `Key: C major | Style: pop | Progression: C - G - Am - F` |

推荐 JSONL 示例：

```jsonl
{"key":"C major","style":"pop","mood":"bright","progression":"C - G - Am - F"}
{"key":"A minor","style":"jazz","mood":"late night","progression":"Am9 - D13 - Gmaj9 - Cmaj7"}
{"prompt":"Write a beautiful chord progression for a cinematic chorus in E minor.","completion":"Emadd9 - Cmaj7 - G - D/F# - Am9 - B7b9 - Em9"}
{"text":"Key: D major | Style: neo-soul | Mood: warm | Progression: Dmaj9 - F#m7 - Gmaj9 - A13sus"}
```

CSV 示例：

```csv
key,style,mood,progression
C major,pop,bright,C - G - Am - F
A minor,jazz,late night,Am9 - D13 - Gmaj9 - Cmaj7
E minor,cinematic,hopeful,Emadd9 - Cmaj7 - G - D/F#
```

TXT 示例：

```text
Key: C major | Style: dream pop | Mood: warm | Progression: Cmaj7 - Em7 - Fmaj7 - G13
Key: A minor | Style: jazz | Mood: smooth | Progression: Am9 - D13 - Gmaj9 - Cmaj7
Key: E minor | Style: cinematic | Mood: hopeful | Progression: Emadd9 - Cmaj7 - G - D/F#
```

ChordPro 示例：

```text
{verse}
[C]Let it be [G]let it be [Am]let it be [F]let it be
{chorus}
[C]Whisper words of [G]wisdom [F]let it [C]be
```

上传后的本地保存位置：

```text
~/.cache/nanochat_music/uploaded_datasets/<filename>
```

转换后的训练分片：

```text
~/.cache/nanochat_music/base_data/shard_00000.parquet
~/.cache/nanochat_music/base_data/shard_00001.parquet
```

元数据文件：

```text
~/.cache/nanochat_music/music_dataset_meta.json
```

示例元数据：

```json
{
  "source": "/Users/you/data/chords.jsonl",
  "examples": 1000,
  "train_rows": 900,
  "val_rows": 100
}
```

## 4. 数据转换规则

自定义数据最终都会转成 parquet，并包含统一字段：

```text
text
```

转换优先级：

1. 如果记录中有 `text` 字段，直接使用 `text`。
2. 如果记录中有 `prompt` + `completion`，拼接为：

```text
<prompt>
<completion>
```

3. 如果记录中有音乐字段，先尝试解析和弦并转换为罗马数字，再拼接为：

```text
Style: <style> | Mood: <mood> | Key: <key> | Roman: <roman progression> | Detected Key: <detected key> | Progression: <progression>
```

4. `.txt` / `.md` 默认每个非空行作为一个训练样本；如果检测到 ChordPro 方括号或段落标记，会把整个文件作为一首歌处理。
5. 识别到 `|` 会保留小节标记，识别到 `{verse}`、`{chorus}`、`//` 等会保留段落标记。
6. 转换时会做简单去重。
7. 至少会生成训练和验证两个 parquet shard。

## 5. Token 拆分方式

UI 当前显示：

```text
Token 拆分方式：BPE 字节对编码（当前可训练）
```

当前后端真实支持：

| Tokenizer | UI 状态 | 后端状态 | 说明 |
| --- | --- | --- | --- |
| BPE 字节对编码 | 可选 | 已实现 | 当前实际训练 tokenizer 的方式 |
| Byte-level 字节级 | 禁用展示 | 暂未实现 | 后续可作为扩展 |
| Char-level 字符级 | 禁用展示 | 暂未实现 | 后续可作为扩展 |

BPE 的作用：

- 把文本拆成模型能处理的 token id。
- 高频片段会被合并成更短 token，例如常见英文词、中文片段、和弦符号片段。
- 对音乐数据来说，`Cmaj7`、`Am9`、`D/F#`、`G13` 这类符号会参与 BPE 统计，数据量足够时会学到更适合和弦文本的拆分方式。

可配置参数：

| 参数 | UI 名称 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `tokenizer_method` | Token 拆分方式 | `bpe` | 当前只支持 BPE |
| `vocab_size` | 词表大小 | `32768` | token 词表容量 |
| `doc_cap` | 单文档最大字符数 | `10000` | 训练 tokenizer 前对超长文档截断 |

后端执行方式：

```bash
python -m scripts.tok_train --vocab-size=32768 --doc-cap=10000
```

输出文件：

```text
~/.cache/nanochat/tokenizer/tokenizer.pkl
~/.cache/nanochat/tokenizer/token_bytes.npy
```

音乐模式下输出文件：

```text
~/.cache/nanochat_music/tokenizer/tokenizer.pkl
~/.cache/nanochat_music/tokenizer/token_bytes.npy
```

## 6. 当前推荐使用方式

文字训练：

```text
数据类型：文字 / 通用文本
数据来源：选择内置/推荐数据集
可选文字数据集：FineWeb-Edu 教育文本（默认）
Token 拆分方式：BPE 字节对编码
```

音乐链路快速验证：

```text
数据类型：音乐 / 和弦数据
数据来源：选择内置/推荐数据集
可选音乐数据集：内置和弦示例集（快速验证）
Token 拆分方式：BPE 字节对编码
```

音乐正式训练：

```text
数据类型：音乐 / 和弦数据
数据来源：从浏览器上传数据集 或 填写本地数据集路径
数据格式：优先 JSONL
Token 拆分方式：BPE 字节对编码
```

推荐正式音乐 JSONL 单条格式：

```jsonl
{"key":"C major","style":"neo-soul","mood":"warm","progression":"Cmaj9 - E7#9 - Am9 - D13 - Gmaj9"}
```

## 7. 后续可扩展数据集

当前 UI 已经预留“可选数据集”的位置，后续可以继续补充：

| 数据类型 | 可扩展数据集 | 需要补充 |
| --- | --- | --- |
| 文字 / 通用文本 | TinyStories | 下载/转换脚本 |
| 文字 / 通用文本 | WikiText | 下载/转换脚本 |
| 音乐 / 和弦数据 | Jazz chord progressions | 数据源或内置样例 |
| 音乐 / 和弦数据 | Pop chord progressions | 数据源或内置样例 |
| 音乐 / 和弦数据 | MIDI 转和弦文本 | MIDI 解析与和弦抽取脚本 |

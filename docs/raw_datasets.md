# 待处理原始数据

下面内容是当前前端可选数据类型对应的原始数据，可直接复制保存为文件后上传到“准备训练数据”步骤。

## 文字 / 通用文本：TXT 原始数据

保存为：

```text
text_general_raw.txt
```

原始内容：

```text
机器学习是一种让计算机从数据中学习规律的方法。
大语言模型通过预测下一个 token 来学习文本分布。
训练数据质量会直接影响模型的生成质量。
分词器负责把人类文本拆成模型可以处理的 token id。
预训练阶段让模型大量阅读文本，并学习语言中的统计规律。
验证集用于观察模型是否只是记住训练数据，还是具备一定泛化能力。
```

## 文字 / 指令文本：JSONL 原始数据

保存为：

```text
text_instruction_raw.jsonl
```

原始内容：

```jsonl
{"prompt":"解释什么是 token。","completion":"token 是模型处理文本时使用的最小文本单元，可以是字符、子词、词或符号片段。"}
{"prompt":"解释什么是预训练。","completion":"预训练是让模型在大量通用文本上学习语言规律的阶段，通常目标是预测下一个 token。"}
{"prompt":"解释什么是验证集。","completion":"验证集是不参与参数更新的数据，用来观察模型是否具备泛化能力。"}
{"text":"训练一个模型就像培养学生：先准备教材，再教它识字，然后让它反复练习。"}
```

## 音乐 / 和弦：JSONL 原始数据

内置文件已经扩充为 27 条样例，覆盖 pop、jazz、neo-soul、bossa、rock、mandopop、blues、cinematic 等常见风格。下面只展示前几条，完整内容见 `sample_datasets/music_starter_chords_raw.jsonl`。

保存为：

```text
music_starter_chords_raw.jsonl
```

原始内容：

```jsonl
{"key":"C major","style":"dream pop","mood":"warm","progression":"Cmaj7 - Em7 - Fmaj7 - G13"}
{"key":"A minor","style":"jazz","mood":"smooth","progression":"Am9 - D13 - Gmaj9 - Cmaj7 - F#dim7 - B7b9 - Em9"}
{"key":"E minor","style":"cinematic","mood":"hopeful","progression":"Emadd9 - Cmaj7 - G - D/F#"}
{"prompt":"Generate a beautiful neo-soul chord loop in D major.","completion":"Dmaj9 - F#m7 - Gmaj9 - A13sus - Bm9 - E13 - Amaj9"}
{"key":"C major","style":"pop","mood":"anthemic","progression":"C - G - Am - F | C - G - F - C"}
{"key":"A minor","style":"mandopop ballad","mood":"nostalgic","progression":"Am - F - C - G | Am - F - G - C"}
```

后端准备数据时会自动解析可识别和弦，并生成 `Roman`、`Detected Key`、`Progression` 等统一训练文本。

## 音乐 / 和弦：CSV 原始数据

保存为：

```text
music_chords_raw.csv
```

原始内容：

```csv
key,style,mood,progression
C major,pop,bright,C - G - Am - F
A minor,jazz,late night,Am9 - D13 - Gmaj9 - Cmaj7
E minor,cinematic,hopeful,Emadd9 - Cmaj7 - G - D/F#
D major,neo-soul,warm,Dmaj9 - F#m7 - Gmaj9 - A13sus
G major,folk,gentle,G - D/F# - Em - Cadd9
F major,bossa nova,romantic,Fmaj7 - Gm7 - C13 - Fmaj7
```

## 音乐 / 和弦：TXT 原始数据

保存为：

```text
music_chords_raw.txt
```

原始内容：

```text
Key: C major | Style: dream pop | Mood: warm | Progression: Cmaj7 - Em7 - Fmaj7 - G13
Key: A minor | Style: jazz | Mood: smooth | Progression: Am9 - D13 - Gmaj9 - Cmaj7 - F#dim7 - B7b9 - Em9
Key: E minor | Style: cinematic | Mood: hopeful | Progression: Emadd9 - Cmaj7 - G - D/F#
Key: D major | Style: neo-soul | Mood: warm | Progression: Dmaj9 - F#m7 - Gmaj9 - A13sus
Key: G major | Style: folk | Mood: gentle | Progression: G - D/F# - Em - Cadd9
```

也支持 ChordPro 风格文件：

```text
{verse}
[C]Let it be [G]let it be [Am]let it be [F]let it be
{chorus}
[C]Whisper words of [G]wisdom [F]let it [C]be
```

## 已生成的本地原始数据文件

这些文件已经放在项目里，可直接在前端“从浏览器上传数据集”里选择：

```text
sample_datasets/text_general_raw.txt
sample_datasets/text_instruction_raw.jsonl
sample_datasets/music_starter_chords_raw.jsonl
sample_datasets/music_chords_raw.csv
sample_datasets/music_chords_raw.txt
```

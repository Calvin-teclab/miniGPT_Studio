"""Prepare custom text or chord/music data for miniGPT Studio pretraining.

Supported inputs:
- JSONL: {"prompt": "...", "completion": "..."} or {"text": "..."} or chord fields
- CSV: columns such as key, style, mood, progression, prompt, completion, text
- TXT/MD: one training example per non-empty line, or ChordPro-style chord files

Music records are normalized into a compact chord-aware text format. When a
progression can be parsed, the output includes roman numerals so a small model
can learn harmonic structure without memorizing every key separately.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanochat_mlx.common import get_base_dir


NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}

PC_TO_NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]
MAJOR_ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
MINOR_ROMAN = ["i", "ii°", "III", "iv", "v", "VI", "VII"]

MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

CHORD_RE = re.compile(
    r"^([A-G][#b]?)"
    r"(maj|min|m|M|dim|aug|sus|add)?"
    r"(\d+)?"
    r"([#b][0-9]+)?"
    r"(/[A-G][#b]?)?$"
)
CHORDPRO_BRACKET = re.compile(r"\[([^\]]+)\]")
SECTION_MARKER = re.compile(
    r"^\s*\{?(verse|chorus|bridge|intro|outro|pre[- ]?chorus|hook)",
    re.IGNORECASE,
)
TOKEN_SPLIT_RE = re.compile(r"(//|\||[-–—])|\s+")


def parse_key_hint(value):
    """Parse key strings like 'C major', 'Am', 'A minor', or 'D dorian'."""
    if not value:
        return None
    text = str(value).strip()
    match = re.match(r"^([A-G][#b]?)(?:\s*(maj(?:or)?|min(?:or)?|m|major|minor|dorian|aeolian))?", text, re.IGNORECASE)
    if not match:
        return None
    note, mode_text = match.groups()
    note = note[0].upper() + note[1:]
    if note not in NOTE_TO_PC:
        return None
    mode_text = (mode_text or "").lower()
    mode = "min" if mode_text in {"m", "min", "minor", "aeolian"} else "maj"
    return NOTE_TO_PC[note], mode


def parse_chord(symbol):
    """Return (root_pc, quality) for a common chord symbol, or None."""
    symbol = symbol.strip().rstrip("*.,;:")
    if not symbol:
        return None
    match = CHORD_RE.match(symbol)
    if not match:
        return None
    root, quality_raw, extension, _, _ = match.groups()
    root_pc = NOTE_TO_PC.get(root)
    if root_pc is None:
        return None

    quality_norm = (quality_raw or "").lower()
    if quality_norm in {"m", "min"}:
        quality = "min"
    elif quality_norm == "dim":
        quality = "dim"
    elif quality_norm == "aug":
        quality = "aug"
    elif quality_norm == "sus":
        quality = "sus"
    elif extension == "7" and quality_norm == "":
        quality = "dom7"
    else:
        quality = "maj"
    return root_pc, quality


def detect_key(chords):
    """Detect a major/minor key from parsed chord roots."""
    roots = [root for root, _ in chords]
    if not roots:
        return 0, "maj"

    histogram = [0.0] * 12
    for root in roots:
        histogram[root] += 1.0

    def correlate(profile, shift):
        rotated = profile[-shift:] + profile[:-shift] if shift else profile[:]
        return sum(h * p for h, p in zip(histogram, rotated))

    best = (-1.0, 0, "maj")
    for tonic in range(12):
        major_score = correlate(MAJOR_PROFILE, tonic)
        minor_score = correlate(MINOR_PROFILE, tonic)
        if major_score > best[0]:
            best = (major_score, tonic, "maj")
        if minor_score > best[0]:
            best = (minor_score, tonic, "min")
    return best[1], best[2]


def chord_to_roman(root_pc, quality, tonic_pc, mode):
    interval = (root_pc - tonic_pc) % 12
    steps = MAJOR_STEPS if mode == "maj" else MINOR_STEPS
    template = MAJOR_ROMAN if mode == "maj" else MINOR_ROMAN
    if interval not in steps:
        accidental = root_pc - tonic_pc
        while accidental < 0:
            accidental += 12
        return f"b{accidental}" if accidental else None

    base = template[steps.index(interval)]
    if quality == "dom7":
        return f"{base}7"
    if quality == "dim" and not base.endswith("°"):
        return f"{base.lower()}°"
    return base


def iter_chord_events(text):
    """Yield ('section',), ('bar',), and ('chord', symbol) events from raw text."""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if SECTION_MARKER.search(line):
            yield ("section",)
            continue

        bracketed = CHORDPRO_BRACKET.findall(line)
        if bracketed:
            for chord in bracketed:
                yield ("chord", chord)
            continue

        normalized = line.replace("–", "-").replace("—", "-")
        for token in TOKEN_SPLIT_RE.split(normalized):
            if not token:
                continue
            token = token.strip()
            if token == "//":
                yield ("section",)
            elif token == "|":
                yield ("bar",)
            elif token == "-":
                continue
            elif parse_chord(token):
                yield ("chord", token)


def convert_to_roman(progression, key_hint=None):
    """Convert chord text into roman numerals, preserving section and bar markers."""
    events = list(iter_chord_events(progression))
    parsed = [parse_chord(event[1]) for event in events if event[0] == "chord"]
    parsed = [item for item in parsed if item]
    if not parsed:
        return None

    tonic_pc, mode = parse_key_hint(key_hint) or detect_key(parsed)
    out = []
    for event in events:
        kind = event[0]
        if kind == "section":
            if out and out[-1] != "//":
                out.append("//")
            continue
        if kind == "bar":
            if out and out[-1] != "|":
                out.append("|")
            continue

        parsed_chord = parse_chord(event[1])
        if not parsed_chord:
            continue
        roman = chord_to_roman(parsed_chord[0], parsed_chord[1], tonic_pc, mode)
        if roman:
            out.append(roman)

    while out and out[0] in {"|", "//"}:
        out.pop(0)
    while out and out[-1] in {"|", "//"}:
        out.pop()
    if not out:
        return None
    key_name = f"{PC_TO_NOTE[tonic_pc]} {'minor' if mode == 'min' else 'major'}"
    return " ".join(out), key_name


def _looks_like_music_text(text):
    events = list(iter_chord_events(text))
    chord_count = sum(1 for event in events if event[0] == "chord")
    return chord_count >= 2


def _format_music_text(progression, key=None, style=None, mood=None):
    converted = convert_to_roman(progression, key)
    parts = []
    if style:
        parts.append(f"Style: {style}")
    if mood:
        parts.append(f"Mood: {mood}")
    if key:
        parts.append(f"Key: {key}")
    if converted:
        roman, detected_key = converted
        parts.append(f"Roman: {roman}")
        parts.append(f"Detected Key: {detected_key}")
    parts.append(f"Progression: {progression.strip()}")
    return " | ".join(parts).strip()


def _format_record(record):
    if "text" in record and record["text"]:
        text = str(record["text"]).strip()
        if _looks_like_music_text(text):
            return _format_music_text(text)
        return text

    if "prompt" in record and "completion" in record:
        return f"{record['prompt'].strip()}\n{record['completion'].strip()}"

    key = record.get("key") or record.get("tonality") or record.get("scale")
    style = record.get("style") or record.get("genre")
    mood = record.get("mood") or record.get("emotion")
    progression = record.get("progression") or record.get("chords") or record.get("chord_progression")

    if progression:
        return _format_music_text(str(progression), key=key, style=style, mood=mood)

    parts = []
    if key:
        parts.append(f"Key: {key}")
    if style:
        parts.append(f"Style: {style}")
    if mood:
        parts.append(f"Mood: {mood}")
    return " | ".join(parts).strip()


def _read_jsonl(path):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            text = _format_record(record)
            if text:
                texts.append(text)
    return texts


def _read_csv(path):
    texts = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for record in reader:
            text = _format_record({k: (v or "").strip() for k, v in record.items()})
            if text:
                texts.append(text)
    return texts


def _read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return load_music_texts_from_content(content, suffix=Path(path).suffix.lower())


def load_music_texts_from_content(content, suffix=".txt"):
    """Load music texts from in-memory content; useful for tests and future uploads."""
    if CHORDPRO_BRACKET.search(content) or SECTION_MARKER.search(content):
        text = _format_music_text(content)
        return [text] if text else []
    texts = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        texts.append(_format_music_text(line) if _looks_like_music_text(line) else line)
    return texts


def load_music_texts(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        texts = _read_jsonl(path)
    elif suffix == ".csv":
        texts = _read_csv(path)
    elif suffix in {".txt", ".md"}:
        texts = _read_txt(path)
    else:
        raise ValueError("Unsupported music data format. Use .jsonl, .csv, .txt, or .md")

    seen = set()
    deduped = []
    for text in texts:
        normalized = " ".join(text.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(text)
    return deduped


def summarize_music_texts(texts):
    roman_counter = Counter()
    roman_examples = 0
    for text in texts:
        match = re.search(r"Roman:\s*([^|]+)", text)
        if not match:
            continue
        roman_examples += 1
        roman_counter.update(match.group(1).strip().split())
    return {
        "roman_examples": roman_examples,
        "roman_vocab": sorted(roman_counter),
    }


def write_parquet_shards(texts, output_dir, val_ratio=0.1):
    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name.endswith(".parquet"):
            os.remove(os.path.join(output_dir, name))

    if not texts:
        raise ValueError("No usable music/chord examples found in input file")

    if len(texts) == 1:
        train_texts = texts
        val_texts = texts
    else:
        val_count = max(1, int(len(texts) * val_ratio))
        train_texts = texts[:-val_count]
        val_texts = texts[-val_count:]
        if not train_texts:
            train_texts = texts[:1]

    for idx, shard_texts in enumerate([train_texts, val_texts]):
        table = pa.table({"text": shard_texts})
        path = os.path.join(output_dir, f"shard_{idx:05d}.parquet")
        pq.write_table(table, path)
        print(f"Wrote {len(shard_texts):,} rows to {path}")

    return {"train_rows": len(train_texts), "val_rows": len(val_texts)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare custom text or chord/music data for miniGPT Studio")
    parser.add_argument("--input", required=True, help="Path to .jsonl, .csv, .txt, or .md data")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args(argv)

    input_path = os.path.expanduser(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Music data file not found: {input_path}")

    base_dir = get_base_dir()
    output_dir = os.path.join(base_dir, "base_data")
    print(f"Preparing custom training data from: {input_path}")
    print(f"Target miniGPT Studio base dir: {base_dir}")

    texts = load_music_texts(input_path)
    stats = write_parquet_shards(texts, output_dir, args.val_ratio)
    music_summary = summarize_music_texts(texts)

    meta_path = os.path.join(base_dir, "music_dataset_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": input_path,
            "examples": len(texts),
            **music_summary,
            **stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"Prepared {len(texts):,} custom examples")
    print(f"Saved metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

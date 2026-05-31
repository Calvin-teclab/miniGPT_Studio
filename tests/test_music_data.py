import json

import pyarrow.parquet as pq

from scripts import music_data


def test_convert_to_roman_uses_explicit_key():
    roman, detected_key = music_data.convert_to_roman("C - G - Am - F | C - G - F - C", "C major")

    assert roman == "I V vi IV | I V IV I"
    assert detected_key == "C major"


def test_chordpro_text_is_normalized():
    text = """
    {verse}
    [Am]Words [F]over [C]chords [G]
    {chorus}
    [Am]More [F]music [G]now [C]
    """

    rows = music_data.load_music_texts_from_content(text, suffix=".txt")

    assert len(rows) == 1
    assert "Roman:" in rows[0]
    assert "//" in rows[0]
    assert "Progression:" in rows[0]


def test_jsonl_music_records_include_roman(tmp_path):
    path = tmp_path / "music.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"key": "A minor", "style": "pop", "progression": "Am - F - C - G"}),
            json.dumps({"text": "C - G - Am - F"}),
        ]),
        encoding="utf-8",
    )

    rows = music_data.load_music_texts(path)

    assert rows[0].startswith("Style: pop")
    assert "Roman: i VI III VII" in rows[0]
    assert "Roman:" in rows[1]


def test_write_parquet_shards_and_summary(tmp_path):
    texts = [
        "Style: pop | Roman: I V vi IV | Progression: C - G - Am - F",
        "Style: jazz | Roman: ii V I I | Progression: Dm7 - G7 - Cmaj7 - Cmaj7",
    ]

    stats = music_data.write_parquet_shards(texts, tmp_path, val_ratio=0.5)
    summary = music_data.summarize_music_texts(texts)

    assert stats == {"train_rows": 1, "val_rows": 1}
    assert summary["roman_examples"] == 2
    assert "I" in summary["roman_vocab"]
    assert pq.read_table(tmp_path / "shard_00000.parquet").num_rows == 1

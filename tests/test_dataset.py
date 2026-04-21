"""Dataset loader tests."""

import json
from pathlib import Path

import pytest

from omegaprompt.dataset import Dataset, DatasetItem


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_dataset_item_requires_id_and_input():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DatasetItem(input="x")
    with pytest.raises(ValidationError):
        DatasetItem(id="t1")


def test_dataset_from_jsonl_happy_path(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "t1", "input": "hello"},
            {"id": "t2", "input": "world", "reference": "ref"},
        ],
    )
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2
    assert ds.items[0].id == "t1"
    assert ds.items[1].reference == "ref"


def test_dataset_from_jsonl_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id":"t1","input":"a"}\n\n   \n{"id":"t2","input":"b"}\n',
        encoding="utf-8",
    )
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2


def test_dataset_from_jsonl_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Dataset.from_jsonl(tmp_path / "nope.jsonl")


def test_dataset_from_jsonl_raises_on_bad_json(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text("{not valid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        Dataset.from_jsonl(p)


def test_dataset_from_jsonl_raises_on_schema_error(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"input":"no id"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema invalid"):
        Dataset.from_jsonl(p)


def test_dataset_from_jsonl_rejects_empty(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zero items"):
        Dataset.from_jsonl(p)


def test_dataset_from_jsonl_detects_duplicate_ids(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    _write_jsonl(
        p,
        [
            {"id": "dup", "input": "a"},
            {"id": "dup", "input": "b"},
        ],
    )
    with pytest.raises(ValueError, match="duplicate ids"):
        Dataset.from_jsonl(p)


def test_dataset_from_items_dicts_and_objects():
    ds = Dataset.from_items(
        [
            {"id": "t1", "input": "a"},
            DatasetItem(id="t2", input="b"),
        ]
    )
    assert len(ds) == 2
    assert ds.items[0].id == "t1"

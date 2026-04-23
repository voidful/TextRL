import json
from pathlib import Path

import pytest
from datasets import Dataset

from textrl.data import from_jsonl, from_list


def test_from_list_strings_are_wrapped():
    ds = from_list(["a", "b"])
    assert isinstance(ds, Dataset)
    assert ds[0] == {"prompt": "a"}


def test_from_list_dicts_pass_through():
    ds = from_list([{"prompt": "q", "extra": 1}])
    assert ds[0] == {"prompt": "q", "extra": 1}


def test_from_list_rejects_bad_types():
    with pytest.raises(TypeError):
        from_list([1, 2])


def test_from_list_rejects_empty():
    with pytest.raises(ValueError):
        from_list([])


def test_from_jsonl(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"prompt": f"q{i}"}) for i in range(3)))
    ds = from_jsonl(str(p))
    assert len(ds) == 3
    assert ds[0]["prompt"] == "q0"

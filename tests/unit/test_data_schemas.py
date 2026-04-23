import pytest
from datasets import Dataset

from textrl.data.schemas import (
    validate_kto_dataset,
    validate_preference_dataset,
    validate_prompt_dataset,
    validate_reward_model_dataset,
)


def test_prompt_dataset_accepts_prompt_column():
    ds = Dataset.from_list([{"prompt": "hi"}])
    assert validate_prompt_dataset(ds) is ds


def test_prompt_dataset_accepts_messages_column():
    ds = Dataset.from_list([{"messages": [{"role": "user", "content": "hi"}]}])
    assert validate_prompt_dataset(ds) is ds


def test_prompt_dataset_rejects_missing_columns():
    ds = Dataset.from_list([{"other": "hi"}])
    with pytest.raises(ValueError, match="prompt.+messages"):
        validate_prompt_dataset(ds)


def test_prompt_dataset_rejects_wrong_type():
    with pytest.raises(TypeError):
        validate_prompt_dataset([{"prompt": "hi"}])


def test_preference_dataset_requires_triple():
    ok = Dataset.from_list([{"prompt": "p", "chosen": "c", "rejected": "r"}])
    assert validate_preference_dataset(ok) is ok

    bad = Dataset.from_list([{"prompt": "p", "chosen": "c"}])
    with pytest.raises(ValueError, match="rejected"):
        validate_preference_dataset(bad)


def test_kto_dataset_requires_label():
    ok = Dataset.from_list([{"prompt": "p", "completion": "c", "label": True}])
    assert validate_kto_dataset(ok) is ok

    bad = Dataset.from_list([{"prompt": "p", "completion": "c"}])
    with pytest.raises(ValueError, match="label"):
        validate_kto_dataset(bad)


def test_reward_model_dataset_requires_chosen_rejected():
    ok = Dataset.from_list([{"chosen": "c", "rejected": "r"}])
    assert validate_reward_model_dataset(ok) is ok

    bad = Dataset.from_list([{"chosen": "c"}])
    with pytest.raises(ValueError, match="rejected"):
        validate_reward_model_dataset(bad)

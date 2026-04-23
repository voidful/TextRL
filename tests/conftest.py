import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: short end-to-end trainer tests")
    config.addinivalue_line("markers", "gpu: requires a GPU to run")


@pytest.fixture(scope="session")
def tiny_causal_model_name() -> str:
    return "trl-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture
def prompt_dataset():
    from textrl.data import from_list
    return from_list([{"prompt": f"sample {i}"} for i in range(8)])


@pytest.fixture
def preference_dataset():
    from datasets import Dataset
    return Dataset.from_list(
        [
            {"prompt": f"q {i}", "chosen": f"good answer {i}", "rejected": f"bad {i}"}
            for i in range(8)
        ]
    )


@pytest.fixture
def kto_dataset():
    from datasets import Dataset
    return Dataset.from_list(
        [
            {"prompt": f"q {i}", "completion": f"a {i}", "label": bool(i % 2)}
            for i in range(8)
        ]
    )

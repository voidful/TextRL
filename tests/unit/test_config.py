import pytest

from textrl.config import ALL_ALGOS, ONLINE_ALGOS, PREFERENCE_ALGOS, TextRLConfig


def test_algo_partition_is_consistent():
    assert ONLINE_ALGOS.isdisjoint(PREFERENCE_ALGOS)
    assert ONLINE_ALGOS | PREFERENCE_ALGOS | {"reward_model"} == ALL_ALGOS


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="Unknown algo"):
        TextRLConfig(algo="not_real", output_dir="/tmp/x")


def test_from_dict_round_trip():
    data = {
        "algo": "grpo",
        "output_dir": "/tmp/x",
        "beta": 0.1,
        "num_generations": 4,
        "custom_field_that_trl_config_accepts": 7,
    }
    cfg = TextRLConfig.from_dict(data)
    assert cfg.algo == "grpo"
    assert cfg.beta == 0.1
    assert cfg.extra["custom_field_that_trl_config_accepts"] == 7


def test_base_training_kwargs_populated():
    cfg = TextRLConfig(algo="grpo", output_dir="out", learning_rate=2e-5, bf16=True)
    kw = cfg._base_training_kwargs()
    assert kw["learning_rate"] == 2e-5
    assert kw["bf16"] is True
    assert kw["output_dir"] == "out"


@pytest.mark.parametrize("algo,hint", sorted({
    "ppo": "rloo",
    "orpo": "dpo",
    "cpo": "dpo",
    "simpo": "dpo",
    "bco": "bco_pair",
    "online_dpo": "grpo",
}.items()))
def test_removed_algos_raise_with_migration_hint(algo, hint):
    with pytest.raises(ValueError, match="(removed|Use algo)"):
        TextRLConfig(algo=algo, output_dir="/tmp/x")


@pytest.mark.parametrize("algo", sorted(ALL_ALGOS))
def test_to_trl_config_builds(algo):
    """Every supported algo must be able to produce its TRL config without error."""
    pytest.importorskip("trl")
    cfg = TextRLConfig(
        algo=algo,
        output_dir="out",
        beta=0.1,
        num_generations=2,
        max_steps=1,
    )
    obj = cfg.to_trl_config()
    assert obj is not None

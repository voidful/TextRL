import pytest

from textrl.rewards import (
    BaseReward,
    RewardFn,
    compose,
    length_penalty,
    repetition_penalty,
    reward_fn,
    to_reward_fn,
)


def test_reward_fn_decorator_satisfies_protocol():
    @reward_fn
    def my_reward(prompts, completions, **_):
        return [len(c) for c in completions]

    assert isinstance(my_reward, RewardFn)
    assert isinstance(my_reward, BaseReward)
    assert my_reward(["a"], ["abcd"]) == [4.0]


def test_reward_fn_keeps_name():
    @reward_fn(name="custom")
    def r(prompts, completions, **_):
        return [0.0 for _ in completions]
    assert r.name == "custom"


def test_to_reward_fn_coerces_plain_callable():
    r = to_reward_fn(lambda prompts, completions, **_: [1.0 for _ in completions])
    assert isinstance(r, RewardFn)
    assert r(["p"], ["c"]) == [1.0]


def test_to_reward_fn_rejects_non_callable():
    with pytest.raises(TypeError):
        to_reward_fn(42)


def test_compose_sums_scores():
    @reward_fn
    def a(prompts, completions, **_):
        return [1.0 for _ in completions]

    @reward_fn
    def b(prompts, completions, **_):
        return [2.0 for _ in completions]

    r = compose(a, b)
    assert r(["x"], ["y"]) == [3.0]


def test_compose_respects_weights():
    @reward_fn
    def a(prompts, completions, **_):
        return [1.0 for _ in completions]

    @reward_fn
    def b(prompts, completions, **_):
        return [1.0 for _ in completions]

    r = compose(a, b, weights=[2.0, 0.5])
    assert r(["x"], ["y"]) == [2.5]


def test_compose_detects_length_mismatch():
    @reward_fn
    def bad(prompts, completions, **_):
        return [0.0]  # wrong length

    with pytest.raises(ValueError, match="expected"):
        compose(bad)(["p1", "p2"], ["c1", "c2"])


def test_length_penalty_monotone():
    scores = length_penalty(["p"] * 3, ["", "a" * 64, "a" * 200], target_length=64, scale=1.0)
    assert scores[1] > scores[0] and scores[1] > scores[2]


def test_repetition_penalty_detects_repeats():
    scores = repetition_penalty(["p", "p"], ["the the the the the the", "alpha beta gamma delta"], n=2)
    assert scores[0] < scores[1]

import pytest
import jax.random
import jax.numpy as jnp
from . import *


def test_simulate_with_no_choices():
    @Gen
    def empty(x):
        return jnp.square(x - 3.0)

    key = jax.random.key(314159)
    fn = jax.jit(empty(jnp.ones(4)).simulate)
    key, sub_key = jax.random.split(key)
    tr = fn(sub_key)
    assert to_score(tr) == 0.0


def test_simple_normal_simulate():
    @Gen
    def simple_normal():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1 + y2

    key = jax.random.key(314159)
    fn = jax.jit(simple_normal().simulate)
    key, sub_key = jax.random.split(key)
    tr = fn(sub_key)
    choices = to_constraint(tr)
    score1 = Normal(0.0, 1.0).logpdf(choices["y1"])
    score2 = Normal(0.0, 1.0).logpdf(choices["y2"])
    test_score = score1 + score2
    assert to_score(tr) == pytest.approx(test_score)


def test_simple_normal_multiple_returns():
    @Gen
    def simple_normal_multiple_returns():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1, y2

    key = jax.random.key(314159)
    key, sub_key = jax.random.split(key)
    fn = jax.jit(simple_normal_multiple_returns().simulate)
    tr = fn(sub_key)
    choices = to_constraint(tr)
    y1_ = choices["y1"]
    y2_ = choices["y2"]
    y1, y2 = tr["retval"]
    assert y1 == y1_
    assert y2 == y2_
    score1 = Normal(0.0, 1.0).logpdf(y1)
    score2 = Normal(0.0, 1.0).logpdf(y2)
    test_score = score1 + score2
    assert to_score(tr) == pytest.approx(test_score)


def test_hierarchical_simple_normal_multiple_returns():
    @Gen
    def _submodel():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1, y2

    @Gen
    def hierarchical_simple_normal_multiple_returns():
        y1, y2 = _submodel() @ "y1"
        return y1, y2

    key = jax.random.key(314159)
    key, sub_key = jax.random.split(key)
    fn = jax.jit(hierarchical_simple_normal_multiple_returns().simulate)
    tr = fn(sub_key)
    choices = to_constraint(tr)
    y1_ = choices["y1"]["y1"]
    y2_ = choices["y1"]["y2"]
    y1, y2 = tr["retval"]
    assert y1 == y1_
    assert y2 == y2_
    score1 = Normal(0.0, 1.0).logpdf(y1)
    score2 = Normal(0.0, 1.0).logpdf(y2)
    test_score = score1 + score2
    assert to_score(tr) == pytest.approx(test_score)


def test_assess_with_no_choices():
    @Gen
    def empty(x):
        return jnp.square(x - 3.0)

    key = jax.random.key(314159)
    key, sub_key = jax.random.split(key)
    model = empty(jnp.ones(4))
    tr = jax.jit(model.simulate)(sub_key)
    jitted = jax.jit(model.assess)
    choices = to_constraint(tr)
    (score, _retval) = jitted(choices)
    assert score == to_score(tr)


def test_simple_normal_assess():
    @Gen
    def simple_normal():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1 + y2

    key = jax.random.key(314159)
    key, sub_key = jax.random.split(key)
    tr = jax.jit(simple_normal().simulate)(sub_key)
    jitted = jax.jit(simple_normal().assess)
    choice = to_constraint(tr)
    (score, _retval) = jitted(choice)
    assert score == to_score(tr)


def test_assess_missing_address():
    @Gen
    def model():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1 + y2

    with pytest.raises(MissingConstraint) as exc:
        _ = model().assess({"y1": 1.0})
    assert exc.value.args == (("y2",),)

    with pytest.raises(MissingConstraint) as exc:
        _ = model().assess({"y2": 1.0})
    assert exc.value.args == (("y1",),)

    score_retval = model().assess({"y1": 1.0, "y2": -1.0})
    assert score_retval == (-2.837877, 0.0)


def test_simple_normal_update():
    @Gen
    def simple_normal():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1 + y2

    key = jax.random.key(314159)
    key, sub_key = jax.random.split(key)
    tr = jax.jit(simple_normal().simulate)(sub_key)
    jitted = jax.jit(simple_normal().update)

    new = {"y1": 2.0}
    original_score = to_score(tr)
    key, sub_key = jax.random.split(key)
    # TODO: we don't have the discard map yet, update this test when we do
    updated = jitted(sub_key, new, tr)
    w = to_weight(updated)
    updated_choice = to_constraint(updated)
    _y1 = updated_choice["y1"]
    _y2 = updated_choice["y2"]
    score1 = Normal(0.0, 1.0).logpdf(_y1)
    score2 = Normal(0.0, 1.0).logpdf(_y2)
    test_score = score1 + score2
    # TODO: restore
    # original_choice = to_constraint(tr)
    # assert original_choice["y1",] == discard["y1",]
    updated_score = to_score(updated)
    assert updated_score == original_score + w
    assert updated_score == pytest.approx(test_score)

    new = {"y1": 2.0, "y2": 3.0}
    key, sub_key = jax.random.split(key)
    updated = jitted(sub_key, new, tr)
    w = to_weight(updated)
    updated_choice = to_constraint(updated)
    y1 = updated_choice["y1"]
    y2 = updated_choice["y2"]
    score1 = Normal(0.0, 1.0).logpdf(y1)
    score2 = Normal(0.0, 1.0).logpdf(y2)
    test_score = score1 + score2
    updated_score = to_score(updated)
    assert updated_score == original_score + w
    assert updated_score == pytest.approx(test_score)

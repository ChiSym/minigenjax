import pytest
import jax.random
import jax.numpy as jnp
from jaxtyping import Array

from . import *


def test_simulate_with_no_choices():
    @gen
    def empty(x):
        return jnp.square(x - 3.0)

    key = jax.random.key(314159)
    fn = jax.jit(empty(jnp.ones(4)).simulate)
    key, sub_key = jax.random.split(key)
    tr = fn(sub_key)
    assert to_score(tr) == 0.0


def test_simple_normal_simulate():
    @gen
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
    @gen
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
    @gen
    def _submodel():
        y1 = Normal(0.0, 1.0) @ "y1"
        y2 = Normal(0.0, 1.0) @ "y2"
        return y1, y2

    @gen
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
    @gen
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
    @gen
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
    @gen
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


class TestStaticGenFnUpdate:
    def test_simple_normal_update(self):
        @gen
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

    def test_simple_linked_normal_update(self):
        @gen
        def simple_linked_normal():
            y1 = Normal(0.0, 1.0) @ "y1"
            y2 = Normal(y1, 1.0) @ "y2"
            y3 = Normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal().simulate)(sub_key)
        jitted = jax.jit(simple_linked_normal().update)

        new = {"y1": 2.0}
        original_score = to_score(tr)
        key, sub_key = jax.random.split(key)
        updated = jitted(sub_key, new, tr)
        w = to_weight(updated)
        updated_choice = to_constraint(updated)
        updated_score = to_score(updated)
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2"]
        y3 = updated_choice["y3"]
        score1, _ = Normal(0.0, 1.0).assess(y1)
        score2, _ = Normal(y1, 1.0).assess(y2)
        score3, _ = Normal(y1 + y2, 1.0).assess(y3)
        test_score = score1 + score2 + score3
        # TODO restore
        # assert original_choice["y1"] == discard["y1"]
        assert updated_score == pytest.approx(original_score + w)
        assert updated_score == pytest.approx(test_score)

    def test_simple_hierarchical_normal(self):
        @gen
        def _inner(x):
            y1 = Normal(x, 1.0) @ "y1"
            return y1

        @gen
        def simple_hierarchical_normal():
            y1 = Normal(0.0, 1.0) @ "y1"
            y2 = _inner(y1) @ "y2"
            y3 = _inner(y1 + y2) @ "y3"
            return y1 + y2 + y3

        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_hierarchical_normal().simulate)(sub_key)
        jitted = jax.jit(simple_hierarchical_normal().update)

        new = {"y1": 2.0}
        original_choice = to_constraint(tr)
        original_score = to_score(tr)
        key, sub_key = jax.random.split(key)

        updated = jitted(sub_key, new, tr)
        w = to_weight(updated)

        updated_choice = to_constraint(updated)
        y1 = updated_choice["y1"]
        y2 = updated_choice["y2"]["y1"]
        y3 = updated_choice["y3"]["y1"]
        assert y1 == new["y1"]
        assert y2 == original_choice["y2"]["y1"]
        assert y3 == original_choice["y3"]["y1"]
        score1, _ = Normal(0.0, 1.0).assess(y1)
        score2, _ = Normal(y1, 1.0).assess(y2)
        score3, _ = Normal(y1 + y2, 1.0).assess(y3)
        test_score = score1 + score2 + score3
        # TODO : restore
        # assert original_choice["y1"] == discard["y1"]
        updated_score = to_score(updated)
        assert updated_score == original_score + w
        assert updated_score == pytest.approx(test_score)

    def update_weight_correctness_general_assertions(self, simple_linked_normal):
        key = jax.random.key(314159)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal.simulate)(sub_key)
        jitted = jax.jit(simple_linked_normal.update)
        choices = to_constraint(tr)

        old_y1 = choices["y1"]
        old_y2 = choices["y2"]
        old_y3 = choices["y3"]
        new_y1 = 2.0
        new = {"y1": new_y1}
        key, sub_key = jax.random.split(key)
        updated = jitted(sub_key, new, tr)
        w = to_weight(updated)
        # (_, w_edit, _, _) = tr.edit(sub_key, Update(new))
        # assert w_edit == w

        # TestStaticGenFn weight correctness.
        updated_sample = to_constraint(updated)
        assert updated_sample["y1"] == new_y1

        δ_y3 = Normal(new_y1 + old_y2, 1.0).logpdf(old_y3) - Normal(
            old_y1 + old_y2, 1.0
        ).logpdf(old_y3)
        δ_y2 = Normal(new_y1, 1.0).logpdf(old_y2) - Normal(old_y1, 1.0).logpdf(old_y2)
        δ_y1 = Normal(0.0, 1.0).logpdf(new_y1) - Normal(0.0, 1.0).logpdf(old_y1)
        assert w == pytest.approx((δ_y3 + δ_y2 + δ_y1))

        # TestStaticGenFn composition of update calls.
        new_y3 = 2.0
        new = {"y3": new_y3}
        key, sub_key = jax.random.split(key)
        # TODO: our argument order disagrees with GenJAX which is unfortunate
        updated = jitted(sub_key, new, updated)
        w = to_weight(updated)
        assert updated["subtraces"]["y3"]["retval"] == 2.0
        correct_w = Normal(new_y1 + old_y2, 1.0).logpdf(new_y3) - Normal(
            new_y1 + old_y2, 1.0
        ).logpdf(old_y3)
        assert w == pytest.approx(correct_w, 0.0001)

    def test_update_weight_correctness(self):
        @gen
        def simple_linked_normal():
            y1 = Normal(0.0, 1.0) @ "y1"
            y2 = Normal(y1, 1.0) @ "y2"
            y3 = Normal(y1 + y2, 1.0) @ "y3"
            return y1 + y2 + y3

        # easy case
        self.update_weight_correctness_general_assertions(simple_linked_normal())

        @gen
        def curried_linked_normal(v1, v2, v3):
            y1 = Normal(0.0, v1) @ "y1"
            y2 = Normal(y1, v2) @ "y2"
            y3 = Normal(y1 + y2, v3) @ "y3"
            return y1 + y2 + y3

        # curry
        self.update_weight_correctness_general_assertions(
            curried_linked_normal.partial(1.0, 1.0, 1.0)()
        )

        # double-curry
        self.update_weight_correctness_general_assertions(
            curried_linked_normal.partial(1.0).partial(1.0, 1.0)()
        )

        @pytree
        class Model:
            v1: Array
            v2: Array

            @gen
            def run(self, v3):
                y1 = Normal(0.0, self.v1) @ "y1"
                y2 = Normal(y1, self.v2) @ "y2"
                y3 = Normal(y1 + y2, v3) @ "y3"
                return y1 + y2 + y3

        # model method
        m = Model(jnp.array(1.0), jnp.array(1.0))
        # TODO: this works if we write m.run(m, 1.0), but we want
        # the method to operate generatively and so m.run(1.0) is
        # the correct thing to write.
        # update_weight_correctness_general_assertions(m.run(1.0))

        @gen
        def m_linked(m: Model, v2, v3):
            y1 = Normal(0.0, m.v1) @ "y1"
            y2 = Normal(y1, v2) @ "y2"
            y3 = Normal(y1 + y2, v3) @ "y3"
            return y1 + y2 + y3

        self.update_weight_correctness_general_assertions(m_linked.partial(m)(1.0, 1.0))

        @gen
        def m_created_internally(scale: Array):
            m_internal = Model(scale, scale)
            return m_internal.run.inline(scale)

        # TODO: minigenjax does not currently support `inline`
        # update_weight_correctness_general_assertions(
        #     m_created_internally(jnp.array(1.0))
        # )

    def test_update_pytree_argument(self):
        @pytree
        class SomePytree:
            x: Array
            y: Array

        @gen
        def simple_linked_normal_with_tree_argument(tree):
            y1 = Normal(tree.x, tree.y) @ "y1"
            return y1

        key = jax.random.key(314159)
        init_tree = SomePytree(0.0, 1.0)
        key, sub_key = jax.random.split(key)
        tr = jax.jit(simple_linked_normal_with_tree_argument(init_tree).simulate)(
            sub_key
        )
        jitted = jax.jit(simple_linked_normal_with_tree_argument(init_tree).update)
        new_y1 = 2.0
        constraints = {"y1": new_y1}
        key, sub_key = jax.random.split(key)
        updated = jitted(sub_key, constraints, tr)
        updated_choices = to_constraint(updated)
        assert updated_choices["y1"] == new_y1
        key, sub_key = jax.random.split(key)
        # TODO: our `update` doesn't allow for new arguments or argdiffs
        # the original version of this test passed new_ree tot he update below
        # new_tree = SomePytree(1.0, 2.0)
        updated = jitted(sub_key, constraints, tr)
        updated_choices = to_constraint(updated)
        assert updated_choices["y1"] == new_y1


class TestScanUpdate:
    @pytest.fixture
    def key(self):
        return jax.random.key(314159)

    @pytest.mark.skip(reason="not ready yet: pause to refactor")
    def test_scan_update(self, key):
        @pytree
        class A:
            x: Array

        @gen
        def step(b, a):
            return Normal(b + a.x, 1e-6) @ "b", None

        @gen
        def model(k):
            return step.scan()(k, A(jnp.array([1.0, 2.0, 3.0]))) @ "steps"

        k1, k2 = jax.random.split(key)
        tr = model(1.0).simulate(k1)
        choices = to_constraint(tr)
        new_ch = {"steps": {"b": choices["steps"]["b"].at[1].set(99.0)}}
        u = model(1.0).update(k2, new_ch, tr)
        new_choices = to_constraint(u)
        assert jnp.allclose(new_choices["steps"]["b"], jnp.array([2.0, 99.0, 7.0]))
        assert to_weight(u) < -100.0

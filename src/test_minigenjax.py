# %%
import dataclasses
import math
import jax
import jax.numpy as jnp
import pytest
from minigenjax import *


@Gen
def model1(b):
    y = Normal(b, 0.1) @ "x"
    return y


@Gen
def model2(b):
    return Uniform(b, b + 2.0) @ "x"


@Gen
def model3(x):
    a = model1(x) @ "a"
    b = model2(x / 2.0) @ "b"
    return a + b


@Gen
def cond_model(b):
    flip = Flip(0.5) @ "flip"
    y = Cond(model1(b), model2(b / 2.0))(flip) @ "s"
    return y


@Gen
@staticmethod
def inlier_model(y, sigma_inlier):
    return Normal(y, sigma_inlier) @ "value"


@Gen
@staticmethod
def outlier_model(y):
    return Uniform(y - 1.0, y + 1.0) @ "value"


@Gen
@staticmethod
def curve_model(f, x, p_outlier, sigma_inlier):
    outlier = Flip(p_outlier) @ "outlier"
    y = f(x)
    fork = Cond(outlier_model(y), inlier_model(y, sigma_inlier))
    return fork(outlier) @ "y"


@Gen
@staticmethod
def coefficient():
    return Normal(0.0, 1.0) @ "c"


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Poly:
    coefficients: jax.Array

    def __call__(self, x):
        if not self.coefficients.shape:
            return 0.0
        powers = jnp.pow(x, jnp.arange(len(self.coefficients)))
        return jnp.dot(powers, self.coefficients)

    def tree_flatten(self):
        return ((self.coefficients,), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


key0 = jax.random.key(0)


def test_pytree():
    poly = coefficient().repeat(3).map(Poly)
    tr = poly.simulate(key0)
    p = tr["retval"]
    assert jnp.allclose(p.coefficients, jnp.array([-0.37148237, 1.1890742, -0.6553323]))

    @Gen
    def noisy_eval(f, x):
        return f(x) + Normal(0.0, 0.01) @ "noise"

    tr = noisy_eval(p, 0.0).simulate(jax.random.key(1))
    assert tr["retval"] == -0.38295257
    assert tr["retval"] == p(0.0) + tr["subtraces"]["noise"]["retval"]

    tr = noisy_eval.vmap(in_axes=(None, 0))(p, jnp.arange(-2.0, 2.0)).simulate(
        jax.random.key(2)
    )
    assert jnp.allclose(
        tr["retval"], jnp.array([-5.361393, -2.220848, -0.36542255, 0.1720068])
    )


# %%
def test_normal_model():
    tr = model1(10.0).simulate(key0)
    expected = {
        "retval": 9.874846,
        "subtraces": {
            "x": {
                "retval": jnp.array(9.874846),
                "score": jnp.array(0.60047853),
            }
        },
    }
    assert tr == expected


def test_uniform_model():
    tr = model2(20.0).simulate(key0)
    assert tr == {
        "retval": 20.210737,
        "subtraces": {
            "x": {
                "retval": 20.210737,
                "score": -0.6931472,
            }
        },
    }


def test_model_vmap():
    tr = jax.vmap(model3(50.0).simulate)(jax.random.split(key0, 5))
    assert jnp.allclose(
        tr["retval"], jnp.array([75.292915, 76.52893, 75.79739, 76.22211, 76.13692])
    )
    assert jnp.allclose(
        tr["subtraces"]["a"]["subtraces"]["x"]["retval"],
        jnp.array([49.96525, 50.06475, 50.01337, 50.10502, 50.147545]),
    )
    assert jnp.allclose(
        tr["subtraces"]["b"]["subtraces"]["x"]["retval"],
        jnp.array([25.327665, 26.464176, 25.784023, 26.117092, 25.989374]),
    )
    assert jnp.allclose(
        tr["subtraces"]["a"]["subtraces"]["x"]["score"],
        jnp.array([1.3232566, 1.174024, 1.3747091, 0.83221716, 0.29519486]),
    )
    assert jnp.allclose(
        tr["subtraces"]["b"]["subtraces"]["x"]["score"],
        jnp.array([-0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472]),
    )


def test_distribution_as_sampler():
    def vmap(n):
        return lambda f: jax.vmap(f)(jax.random.split(key0, n))

    assert jnp.allclose(
        vmap(10)(Normal(0.0, 0.01)),
        jnp.array(
            [
                -0.00449334,
                -0.00115321,
                -0.005181,
                0.00307154,
                -0.02684483,
                -0.01266131,
                0.00193166,
                -0.01589755,
                -0.00339408,
                0.01838289,
            ]
        ),
    )
    assert jnp.allclose(
        vmap(10)(Uniform(5.0, 6.0)),
        jnp.array(
            [
                5.3265953,
                5.454095,
                5.302194,
                5.620637,
                5.003632,
                5.102733,
                5.576586,
                5.055945,
                5.3671513,
                5.96699,
            ]
        ),
    )
    assert jnp.allclose(
        vmap(10)(Flip(0.5)),
        jnp.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
    )
    assert jnp.allclose(
        vmap(10)(Categorical(logits=jnp.array([1.1, -1.0, 0.9]))),
        jnp.array([0, 0, 2, 2, 0, 1, 0, 0, 0, 0]),
    )
    assert jnp.allclose(
        vmap(10)(MvNormalDiag(jnp.array([1.0, 10.0, 100.0]), 0.1 * jnp.ones(3))),
        jnp.array(
            [
                [1.0678201, 9.900013, 100.02386],
                [1.0617225, 10.046055, 99.96553],
                [0.97022116, 10.116703, 100.034744],
                [0.8233146, 10.12513, 100.1254],
                [0.9622561, 9.9325695, 99.90703],
                [0.827492, 10.113411, 99.91007],
                [1.1142846, 10.202795, 99.900986],
                [0.9468851, 9.946305, 99.93078],
                [0.9302361, 9.9734125, 99.82939],
                [1.0136702, 9.983562, 99.88976],
            ]
        ),
    )


def test_cond_model():
    b = 100.0
    tr = model1(b).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(99.87485))
    tr = model2(b / 2).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(50.21074))
    c = Cond(model1(b), model2(b / 2.0))
    tr = c(0).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(51.74507))
    tr = c(1).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(100.113846))
    tr = jax.vmap(lambda i, k: c(i).simulate(k))(
        jnp.mod(jnp.arange(10.0), 2), jax.random.split(key0, 10)
    )
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                50.250393,
                100.00631,
                50.567764,
                100.01083,
                50.59677,
                100.11719,
                50.55182,
                100.09514,
                51.49389,
                99.99336,
            ]
        ),
    )

    @Gen
    def cond_model(b):
        flip = Flip(0.5) @ "flip"
        return c(flip) @ "c"

    tr = cond_model(b).simulate(key0)
    assert jnp.allclose(
        tr["subtraces"]["c"]["subtraces"]["x"]["retval"], jnp.array(100.138466)
    )
    assert jnp.allclose(tr["subtraces"]["flip"]["retval"], jnp.array(1))


def test_vmap_over_cond():
    tr = jax.vmap(cond_model(100.0).simulate)(jax.random.split(key0, 5))

    assert jnp.allclose(
        tr["retval"], jnp.array([99.82281, 50.112656, 50.439762, 51.793697, 100.29166])
    )
    assert jnp.allclose(tr["subtraces"]["flip"]["retval"], jnp.array([1, 0, 0, 0, 1]))
    assert jnp.allclose(
        tr["subtraces"]["s"]["retval"],
        jnp.array([99.82281, 50.112656, 50.439762, 51.793697, 100.29166]),
    )


def test_ordinary_cond():
    @Gen
    def f():
        n = Normal(0.0, 1.0) @ "n"
        return jax.lax.cond(n > 0, lambda: n, lambda: 10 * n)

    tr = f().simulate(key0)
    assert tr["retval"] == -12.5153885
    assert tr["subtraces"]["n"]["retval"] == -1.2515389


def test_intervening_functions():
    @Gen
    def h():
        return Normal(0.0, 1.0) @ "n"

    def g():
        return h()

    @Gen
    def f():
        return g() @ "g"

    tr = f().simulate(key0)
    assert tr["retval"] == 1.2767134
    assert tr["subtraces"]["g"]["retval"] == tr["retval"]


def test_scan_model():
    @Gen
    def update(state, delta):
        drift = Normal(delta, 0.01) @ "drift"
        new_position = state + drift
        return new_position, new_position

    @Gen
    def scan_update():
        return Scan(update)(10.0, jnp.arange(0.1, 0.6, 0.1)) @ "S"

    tr = scan_update().simulate(key0)
    assert jnp.allclose(tr["retval"][0], 11.497978)
    assert jnp.allclose(
        tr["retval"][1],
        jnp.array([10.101994, 10.284705, 10.613012, 10.999251, 11.497978]),
    )
    assert jnp.allclose(
        tr["subtraces"]["S"]["subtraces"]["drift"]["retval"],
        jnp.array([0.10199323, 0.18271188, 0.32830706, 0.386239, 0.4987273]),
    )
    assert jnp.allclose(
        tr["subtraces"]["S"]["subtraces"]["drift"]["score"],
        jnp.array([3.6663668, 2.1918368, -0.3202114, 2.739405, 3.6781328]),
    )


def test_plain_scan():
    @Gen
    def model(x):
        init = Normal(x, 0.01) @ "init"
        return jax.lax.scan(lambda a, b: (a + b, a + b), init, jnp.arange(5.0))

    tr = model(10.0).simulate(key0)
    assert tr["retval"][0] == 19.987484
    assert jnp.allclose(
        tr["retval"][1],
        jnp.array([9.987485, 10.987485, 12.987485, 15.987485, 19.987484]),
    )
    assert tr["subtraces"]["init"]["retval"] == 9.987485


class TestCurve:
    def test_curve_model(self):
        f = Poly(jnp.array([1.0, -1.0, 2.0]))  # x**2.0 - x + 1.0

        assert f(0.0) == 1.0

        tr = curve_model(f, 0.0, 0.0, 0.0).simulate(key0)
        assert tr["retval"] == 1.0
        assert tr["subtraces"]["outlier"]["retval"] == 0
        assert tr["subtraces"]["y"]["subtraces"]["value"]["retval"] == 1.0

        tr = curve_model.vmap(in_axes=(None, 0, None, None))(
            f, jnp.arange(-3.0, 3.0), 0.01, 0.01
        ).simulate(key0)
        assert jnp.allclose(
            tr["subtraces"]["__vmap"]["outlier"]["retval"],
            jnp.array([1, 0, 0, 0, 0, 0]),
        )
        assert jnp.allclose(
            tr["retval"],
            jnp.array(
                [21.718445, 11.011163, 3.9904368, 1.0180144, 1.991071, 6.9903884]
            ),
        )
        tr = curve_model.vmap(in_axes=(None, None, 0, None))(
            f, 0.0, jnp.array([0.001, 0.01, 0.9]), 0.3
        ).simulate(key0)
        assert jnp.allclose(
            tr["subtraces"]["__vmap"]["outlier"]["retval"], jnp.array([0, 0, 1])
        )
        assert jnp.allclose(
            tr["retval"], jnp.array([1.1806593, 0.97604936, 0.02672911])
        )

    def test_curve_generation(self):
        quadratic = coefficient().repeat(3).map(Poly)
        points = jnp.arange(-3.0, 3.0)

        tr = jax.vmap(quadratic.simulate)(jax.random.split(key0, 10))
        assert tr["retval"].coefficients.shape == (10, 3)
        graphs = jax.vmap(lambda p: jax.vmap(lambda x: p(x))(points))(tr["retval"])
        assert graphs.shape == (10, 6)


def test_map():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    def plus5(x):
        return x + 5.0

    noisy_plus5 = noisy(10).map(plus5)
    tr = jax.vmap(noisy_plus5.simulate)(jax.random.split(key0, 3))
    assert jnp.allclose(tr["retval"], jnp.array([14.998601, 14.99248, 14.996802]))


def test_repeat():
    @Gen
    def coefficient():
        return Normal(0.0, 1.0) @ "c"

    def Poly(coefficient_gf, n):
        @Gen
        def poly():
            return coefficient_gf().repeat(n) @ "cs"

        return poly

    poly4 = Poly(coefficient, 4)
    tr = poly4().simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([-2.2505789, 0.47611082, 0.5935723, 1.174374])
    )


def test_repeat_in_model():
    @Gen
    def x(y):
        return Normal(2.0 * y, 1.0) @ "x"

    @Gen
    def xs():
        return x(10.0).repeat(4) @ "xs"

    tr = xs().simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([17.74942, 20.47611, 20.593573, 21.174374])
    )


def test_map_in_model():
    @Gen
    def x(y):
        return Normal(y, 0.1) @ "x"

    @Gen
    def mx():
        return x(7.0).map(lambda t: t + 13.0) @ "mx"

    tr = jax.vmap(mx().simulate)(jax.random.split(key0, 5))
    assert jnp.allclose(
        tr["retval"], jnp.array([19.965248, 20.06475, 20.013372, 20.105017, 20.147545])
    )


def test_map_of_repeat():
    @Gen
    def coefficient():
        return Normal(0.0, 1.0) @ "c"

    pg = coefficient().repeat(3).map(Poly)

    tr = pg.simulate(key0)
    assert jnp.allclose(
        tr["retval"].coefficients, jnp.array([-0.37148237, 1.1890742, -0.6553323])
    )
    assert jnp.allclose(tr["retval"](1.0), jnp.array(0.16225946))
    assert jnp.allclose(tr["retval"](2.0), jnp.array(-0.61466336))

    kg = coefficient().repeat(3).map(jnp.sum)
    tr = kg.simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array([0.16225946]))


def test_repeat_of_map():
    @Gen
    def y(x):
        return Normal(x, 0.1) @ "y"

    mr = y(7.0).map(lambda x: x + 13.0).repeat(5)

    tr = mr.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([20.071547, 20.081505, 19.934692, 19.985865, 20.004135])
    )


def test_repeat_of_cond():
    repeated_model = cond_model(60.0).repeat(5)
    tr = repeated_model.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([30.571875, 30.681627, 59.978813, 59.962864, 30.644602])
    )


def test_vmap():
    @Gen
    def model(x, y):
        return x + Normal(y, 0.01) @ "a"

    tr = model(5.0, 1.0).simulate(key0)
    assert tr["retval"] == 5.9874845

    gf = model.vmap(in_axes=(0, None))(jnp.arange(5.0), 1.0)
    # vector_jaxpr = jax.make_jaxpr(gf.simulate)(key0)
    # print(f'vector_jaxpr {vector_jaxpr}')
    tr0 = gf.simulate(key0)

    assert jnp.allclose(
        tr0["retval"],
        jnp.array([1.0071547, 2.0081506, 2.9934692, 3.9985867, 5.0004134]),
    )
    tr1 = model.vmap(in_axes=(None, 0))(5.0, jnp.arange(0.1, 0.4, 0.1)).simulate(key0)
    assert jnp.allclose(
        tr1["retval"], jnp.array([5.077494, 5.204761, 5.305936, 5.4117436])
    )
    tr2 = model.vmap(in_axes=(0, 0))(
        jnp.arange(5.0), 0.1 * (1.0 + jnp.arange(5.0))
    ).simulate(key0)
    assert jnp.allclose(
        tr2["retval"],
        jnp.array([0.10715472, 1.2081504, 2.2934692, 3.3985865, 4.5004134]),
    )
    # try the above without enumerating axis/arguments in in_axes
    tr3 = model.vmap()(jnp.arange(5.0), 0.1 * (1.0 + jnp.arange(5.0))).simulate(key0)
    assert jnp.allclose(tr3["retval"], tr2["retval"])


def test_assess():
    @Gen
    def p():
        x = Normal(0.0, 1.0) @ "x"
        y = Normal(0.0, 1.0) @ "y"
        return x, y

    @Gen
    def q():
        return p() @ "p"

    constraints = {"x": 2.0, "y": 2.1}
    w = p().assess(constraints)
    assert w == -6.0428767
    w = q().assess({"p": constraints})
    assert w == -6.0428767


def test_bernoulli():
    @Gen
    def p():
        b = Bernoulli(probs=0.01) @ "b"
        c = Bernoulli(logits=-1) @ "c"
        return b, c

    tr = p().simulate(key0)
    assert tr["retval"] == [0, 0]
    assert tr["subtraces"]["b"]["score"] == math.log(1 - 0.01)
    assert tr["subtraces"]["c"]["score"] == math.log(
        1 - math.exp(-1) / (1 + math.exp(-1))
    )

    with pytest.raises(ValueError):
        Bernoulli()

    with pytest.raises(ValueError):
        Bernoulli(logits=-1, probs=0.5)


# %%

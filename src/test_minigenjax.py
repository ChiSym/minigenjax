# %%
import dataclasses
import jax
import jax.numpy as jnp
from .minigenjax import *
import pytest


@Gen
def model1(b):
    y = Normal(b, 0.1) @ "x"
    return y


@Gen
def model2(b):
    return Uniform(b, b + 2.0) @ "x"


@Gen
def model(x):
    a = model1(x) @ "a"
    b = model2(x / 2.0) @ "b"
    return a + b

@Gen
def cond_model(b):
    flip = Flip(0.5) @ "flip"
    y = Cond(model1(b), model2(b / 2.0))(flip) @ "s"
    return y

key0 = jax.random.PRNGKey(0)


# %%
def test_normal_model():
    tr = model1(10.0).simulate(key0)
    assert tr == {
        "retval": 9.874846,
        "subtraces": {
            "x": {
                "retval": 9.874846,
                "score": 0.60047853,
            }
        },
    }


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
    tr = jax.vmap(model(50.0).simulate)(jax.random.split(key0, 5))
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
        vmap(10)(Categorical(jnp.array([1.1, -1.0, 0.9]))),
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


def test_scan_model():
    @Gen
    def update(state, delta):
        drift = Normal(delta, 0.01) @ "drift"
        new_position = state + drift
        return new_position, new_position

    @Gen
    def scan_update():
        return Scan(update)(10.0, jnp.arange(0.1, 0.6, 0.1)) @ "S"

    # jax.make_jaxpr(scan_update().simulate)(key0)

    tr = scan_update().simulate(key0)
    assert jnp.allclose(tr["retval"][0], 11.494305)
    assert jnp.allclose(
        tr["retval"][1],
        jnp.array([10.111384, 10.305588, 10.607706, 11.00557, 11.494305]),
    )
    assert jnp.allclose(
        tr["subtraces"]["S"]["subtraces"]["drift"]["retval"],
        jnp.array([0.11138456, 0.1942033, 0.30211872, 0.39786434, 0.48873404]),
    )
    assert jnp.allclose(
        tr["subtraces"]["S"]["subtraces"]["drift"]["score"],
        jnp.array([3.0381901, 3.518223, 3.6637871, 3.6634264, 3.051622]),
    )


def test_curve_model():
    # %%
    @Gen
    def inlier_model(y, sigma_inlier):
        return Normal(y, sigma_inlier) @ "value"

    @Gen
    def outlier_model(y):
        return Uniform(y - 1.0, y + 1.0) @ "value"

    @Gen
    def curve_model(x, p_outlier):
        outlier = Flip(p_outlier) @ "outlier"
        y0 = x**2 - x + 1.0
        fork = Cond(outlier_model(y0), inlier_model(y0, 0.1))
        return fork(outlier) @ "y"

    jax.vmap(curve_model(1.0, 0.2).simulate)(jax.random.split(key0, 10))

    tr = jax.vmap(lambda x: curve_model(x, 0.2).simulate(key0))(jnp.arange(-3.0, 3.0))
    assert jnp.allclose(tr["subtraces"]["outlier"]["retval"], jnp.array([1.0]))
    assert jnp.allclose(
        tr["retval"],
        jnp.array([13.833855, 7.8338547, 3.8338544, 1.8338544, 1.8338544, 3.8338544]),
    )
    # we didn't change the seed in the vmap, so we got "the same curve" at different x values. Looking at the
    # function in curve_model, we indeed expect that f(-1) == f(2), f(0) == f(1)

    # jax.vmap(lambda x: curve_model(x, 0.2).simulate(key0))(jnp.arange(-3., 3.))

    curve_model(4, 0.2).simulate(key0)
    jax.make_jaxpr(lambda x, k: curve_model(x, 0.2).simulate(k))(1.0, key0)
    # %%
    # Not working yet: outlier_model doesn't get batched
    # jax.vmap(
    #     lambda k: jax.vmap(
    #         lambda x: curve_model(x, 0.2).simulate(k)
    #     )(jnp.arange(-2, 3.)))(jax.random.split(key0, 10))


def test_map():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    def plus5(x):
        return x + 5.0

    nplus5 = noisy(10).map(plus5)
    tr = jax.vmap(nplus5.simulate)(jax.random.split(key0, 3))
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

    @jax.tree_util.register_dataclass
    @dataclasses.dataclass
    class Poly:
        coefs: jax.Array

        def __call__(self, x):
            if not self.coefs.shape:
                return 0.0
            powers = jnp.pow(x, jnp.arange(len(self.coefs)))
            return jnp.dot(powers, self.coefs)

    pg = coefficient().repeat(3).map(Poly)

    tr = pg.simulate(key0)
    assert jnp.allclose(
        tr["retval"].coefs, jnp.array([-0.13994414, -0.7519509, -0.31980208])
    )
    assert jnp.allclose(tr["retval"](1.0), jnp.array(-1.2116971))
    assert jnp.allclose(tr["retval"](2.0), jnp.array(-2.9230543))


def test_repeat_of_map():
    @Gen
    def y(x):
        return Normal(x, 0.1) @ "y"

    mr = y(7.0).map(lambda x: x + 13.0).repeat(5)

    tr = mr.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([19.966385, 20.016861, 20.021904, 20.22247, 19.879295])
    )

@pytest.mark.skip(reason="possible bug in JAX preventing this from working")
def test_repeat_of_cond():
    repeated_model = cond_model(60.0).repeat(6)
    tr = repeated_model.simulate(key0)
    assert jnp.allclose(
        tr['retval'], jnp.array([1.0])
    )

def test_vmap():
    @Gen
    def model(x, y):
        return x + Normal(y, 0.01) @ "a"

    tr = model(5.0, 1.0).simulate(key0)
    assert tr["retval"] == 5.9874845

    tr0 = Vmap(model, in_axes=(0, None))(jnp.arange(5.0), 1.0).simulate(key0)
    assert jnp.allclose(
        tr0["retval"],
        jnp.array([0.98748463, 1.9874847, 2.9874847, 3.9874847, 4.9874845]),
    )
    tr1 = Vmap(model, in_axes=(None, 0))(5.0, jnp.arange(0.1, 0.4, 0.1)).simulate(key0)
    assert jnp.allclose(
        tr1["retval"], jnp.array([5.0838757, 5.183876, 5.283876, 5.383876])
    )

    # yikes! too much similarity: the vmapping isn't gathering as much randomness as it should


# %%

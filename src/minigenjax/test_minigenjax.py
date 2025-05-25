# %%
# pyright: reportWildcardImportFromLibrary=false
import math
import jax
import jax.numpy as jnp
import pytest
from . import *
from .trace import to_weight


@Gen
def model1(b):
    y = Normal(b, 0.1) @ "x"
    return y


@Gen
def void_model(b):
    _ = Normal(b, 9.1) @ "b"


@Gen
def model2(b):
    return Uniform(b, b + 2.0) @ "x"


@Gen
def model3(x):
    a = model1(x) @ "a"
    b = model2(x / 2.0) @ "b"
    return a, b


@Gen
def cond_model(b):
    flip = Flip(0.5) @ "flip"
    y = Cond(model1(b), model2(b / 2.0))(flip) @ "s"
    return y


@Gen
def inlier_model(y, sigma_inlier):
    return Normal(y, sigma_inlier) @ "value"


@Gen
def outlier_model(y):
    return Uniform(y - 1.0, y + 1.0) @ "value"


@Gen
def curve_model(f, x, p_outlier, sigma_inlier):
    outlier = Flip(p_outlier) @ "outlier"
    y = f(x)
    fork = Cond(outlier_model(y), inlier_model(y, sigma_inlier))
    return fork(outlier) @ "y"


@Gen
def coefficient():
    return Normal(0.0, 1.0) @ "c"


@pytree
class Poly:
    coefficients: jax.Array

    def __call__(self, x):
        if not self.coefficients.shape:
            return 0.0
        powers = jnp.pow(
            jnp.array(x)[jnp.newaxis], jnp.arange(self.coefficients.shape[0])
        )
        return self.coefficients.T @ powers


key0 = jax.random.key(0)


def test_pytree():
    poly = coefficient.repeat(3).map(Poly)
    tr = poly().simulate(key0)
    p = tr["retval"]
    assert jnp.allclose(p.coefficients, jnp.array([1.1188384, 0.5781488, 0.8535516]))

    @Gen
    def noisy_eval(f, x):
        return f(x) + Normal(0.0, 0.01) @ "noise"

    tr = jax.jit(noisy_eval(p, 0.0).simulate)(jax.random.key(1))
    assert tr["retval"] == 1.1069956
    assert tr["retval"] == p(0.0) + tr["subtraces"]["noise"]["retval"]

    tr = jax.jit(noisy_eval.vmap(in_axes=(None, 0))(p, jnp.arange(-2.0, 2.0)).simulate)(
        jax.random.key(2)
    )
    assert jnp.allclose(
        tr["retval"], jnp.array([3.37815, 1.3831037, 1.1251557, 2.533188])
    )

    @Gen
    def wrap_poly():
        p = poly() @ "p"
        return p

    tr = jax.jit(wrap_poly().simulate)(key0)
    assert isinstance(tr["retval"], Poly)

    @Gen
    def wrap_poly2():
        cs = coefficient.repeat(3)() @ "cs"
        return Poly(cs)

    tr = jax.jit(wrap_poly2().simulate)(key0)
    assert isinstance(tr["retval"], Poly)


def test_pytree_iteration():
    poly = coefficient.repeat(3).map(Poly)
    tr = jax.vmap(poly().simulate)(jax.random.split(key0, 100))
    ps = tr["retval"]
    # switch from plural Poly to list of Polys
    list_of_p = [p for p in ps]
    assert len(list_of_p) == 100
    assert jnp.allclose(list_of_p[0].coefficients, ps[0].coefficients)
    assert jnp.allclose(list_of_p[99].coefficients, ps[99].coefficients)


# %%
def test_normal_model():
    tr = model1(10.0).simulate(key0)
    expected = {
        "retval": 9.979416,
        "subtraces": {
            "x": {
                "retval": jnp.array(9.979416),
                "score": jnp.array(1.3624613),
            }
        },
    }
    assert tr == expected
    c, score, retval = model1(10.0).propose(key0)
    assert c == {"x": tr["subtraces"]["x"]["retval"]}
    assert score == tr["subtraces"]["x"]["score"]

    tr = void_model(10.0).simulate(key0)


def test_uniform_model():
    tr = model2(20.0).simulate(key0)
    assert tr == {
        "retval": 20.836914,
        "subtraces": {
            "x": {
                "retval": 20.836914,
                "score": -0.6931472,
            }
        },
    }


def test_multiple_results():
    tr = model3(50.0).simulate(key0)
    assert tr["retval"] == (49.874847, 26.114412)


def test_logit_vs_probs():
    def sigmoid(g):
        return math.exp(g) / (1.0 + math.exp(g))

    @Gen
    def model():
        g = Bernoulli(logits=-0.3) @ "l"
        p = Bernoulli(probs=0.3) @ "p"
        return g, p

    tr = model().simulate(key0)
    assert tr["subtraces"]["l"]["retval"] == 1.0
    assert (
        tr["subtraces"]["l"]["score"]
        == -0.8543553
        == pytest.approx(math.log(sigmoid(-0.3)))
    )
    assert tr["subtraces"]["p"]["retval"] == 0.0
    assert (
        tr["subtraces"]["p"]["score"] == -0.35667497 == pytest.approx(math.log(1 - 0.3))
    )


def test_model_vmap():
    tr = jax.vmap(model3.map(sum)(50.0).simulate)(jax.random.split(key0, 5))
    assert jnp.allclose(
        tr["retval"], jnp.array([76.4031, 76.777, 75.255844, 76.623726, 76.145515])
    )
    assert jnp.allclose(
        tr["subtraces"]["a"]["subtraces"]["x"]["retval"],
        jnp.array([49.966385, 50.01686, 50.021904, 50.22247, 49.879295]),
    )
    assert jnp.allclose(
        tr["subtraces"]["b"]["subtraces"]["x"]["retval"],
        jnp.array([26.436712, 26.760138, 25.233942, 26.401255, 26.26622]),
    )
    assert jnp.allclose(
        tr["subtraces"]["a"]["subtraces"]["x"]["score"],
        jnp.array([1.3271478, 1.369432, 1.3596607, -1.0910004, 0.65514755]),
    )
    assert jnp.allclose(
        tr["subtraces"]["b"]["subtraces"]["x"]["score"],
        jnp.array([-0.6931472, -0.6931472, -0.6931472, -0.6931472, -0.6931472]),
    )
    assert to_score(tr) == jnp.sum(
        tr["subtraces"]["b"]["subtraces"]["x"]["score"]
    ) + jnp.sum(tr["subtraces"]["a"]["subtraces"]["x"]["score"])


def test_distribution_as_sampler():
    def vmap(n):
        return lambda f: jax.vmap(f)(jax.random.split(key0, n))

    assert jnp.allclose(
        vmap(10)(Normal(0.0, 0.01).sample),
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
        vmap(10)(Uniform(5.0, 6.0).sample),
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
        vmap(10)(Flip(0.5).sample),
        jnp.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0]),
    )
    assert jnp.allclose(
        vmap(10)(Categorical(logits=jnp.array([1.1, -1.0, 0.9])).sample),
        jnp.array([0, 0, 2, 2, 0, 1, 0, 0, 0, 0]),
    )
    assert jnp.allclose(
        vmap(10)(MvNormalDiag(jnp.array([1.0, 10.0, 100.0]), 0.1 * jnp.ones(3)).sample),
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


def test_mixture():
    m = Mixture(Categorical(probs=[0.3, 0.7]), [Normal(0.0, 1.0), Normal(10.0, 1.0)])
    ys = jax.vmap(m.sample)(jax.random.split(key0, 10))
    assert jnp.allclose(
        ys,
        jnp.array(
            [
                12.664838,
                0.14686993,
                -1.2831976,
                10.281769,
                -0.5806916,
                9.012323,
                10.005001,
                10.695513,
                0.5217198,
                9.937291,
            ]
        ),
    )


def test_cond_model():
    b = 100.0
    tr = model1(b).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(99.979416))
    tr = model2(b / 2).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(50.836914))
    c = Cond(model1(b), model2(b / 2.0))
    tr = c(0).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(50.836914))
    tr = c(1).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(99.979416))
    tr = jax.vmap(lambda i, k: c(i).simulate(k))(
        jnp.mod(jnp.arange(10), 2), jax.random.split(key0, 10)
    )
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                50.65319,
                99.988464,
                50.60439,
                100.030716,
                50.007263,
                99.87339,
                51.15317,
                99.84103,
                50.734303,
                100.18383,
            ]
        ),
    )

    @Gen
    def cond_model(b):
        flip = Flip(0.5) @ "flip"
        return c(flip) @ "c"

    tr = cond_model(b).simulate(key0)
    assert jnp.allclose(
        tr["subtraces"]["c"]["subtraces"]["x"]["retval"], jnp.array(100.01439)
    )
    assert jnp.allclose(tr["subtraces"]["flip"]["retval"], jnp.array(1))


def test_vmap_over_cond():
    tr = jax.vmap(cond_model(100.0).simulate)(jax.random.split(key0, 5))

    assert jnp.allclose(
        tr["retval"], jnp.array([100.0578, 51.76014, 50.23394, 51.401253, 100.03401])
    )
    assert jnp.allclose(tr["subtraces"]["flip"]["retval"], jnp.array([1, 0, 0, 0, 1]))
    assert jnp.allclose(
        tr["subtraces"]["s"]["retval"],
        jnp.array([100.0578, 51.76014, 50.23394, 51.401253, 100.03401]),
    )


def test_ordinary_cond():
    @Gen
    def f():
        n = Normal(0.0, 1.0) @ "n"
        return jax.lax.cond(n > 0, lambda: n, lambda: 10 * n)

    tr = f().simulate(key0)
    assert tr["retval"] == -12.5153885
    assert tr["subtraces"]["n"]["retval"] == -1.2515389


def test_cond_of_two_distributions():
    @Gen
    def m():
        f = Flip(0.5) @ "f"
        p = Cond(Normal(10.0, 0.1), Normal(1.0, 0.1))(f) @ "p"
        return f, p

    tr = m().simulate(key0)
    assert tr["subtraces"]["f"]["retval"] == 1.0
    assert tr["subtraces"]["p"]["retval"] == 10.014389


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
    assert tr["retval"] == -0.20584226
    assert tr["subtraces"]["g"]["retval"] == tr["retval"]


class TestScan:
    @Gen
    def update(state, delta):
        drift = Normal(delta, 0.01) @ "drift"
        new_position = state + drift
        return new_position, new_position

    @Gen
    def scan_update():
        return TestScan.update.scan()(10.0, jnp.arange(0.1, 0.6, 0.1)) @ "S"

    def test_scan_model(self):
        tr = TestScan.update.scan()(10.0, jnp.arange(0.1, 0.6, 0.1)).simulate(key0)
        assert jnp.allclose(tr["retval"][0], 11.482168)
        assert jnp.allclose(
            tr["retval"][1],
            jnp.array([10.087484, 10.281618, 10.586483, 10.988654, 11.482168]),
        )
        assert jnp.allclose(
            tr["subtraces"]["drift"]["retval"],
            jnp.array([0.08748461, 0.19413349, 0.30486485, 0.4021714, 0.49351367]),
        )
        assert jnp.allclose(
            tr["subtraces"]["drift"]["score"],
            jnp.array([2.903057, 3.514152, 3.5678985, 3.6626565, 3.4758694]),
        )
        tr = TestScan.scan_update().simulate(key0)
        assert jnp.allclose(
            tr["retval"][1],
            jnp.array([10.087484, 10.281618, 10.586483, 10.988654, 11.482168]),
        )
        assert jnp.allclose(
            tr["subtraces"]["S"]["subtraces"]["drift"]["retval"],
            jnp.array([0.08748461, 0.19413349, 0.30486485, 0.4021714, 0.49351367]),
        )
        assert jnp.allclose(
            tr["subtraces"]["S"]["subtraces"]["drift"]["score"],
            jnp.array([2.903057, 3.514152, 3.5678985, 3.6626565, 3.4758694]),
        )

    # @pytest.mark.skip(reason="after we fix the base case")
    def test_scan_update(self):
        model = TestScan.update.scan()(10.0, jnp.arange(0.1, 0.6, 0.1))
        tr = model.simulate(key0)
        assert jnp.allclose(tr["retval"][0], 11.482168)
        assert jnp.allclose(
            tr["retval"][1],
            jnp.array([10.087484, 10.281618, 10.586483, 10.988654, 11.482168]),
        )
        assert jnp.allclose(
            tr["subtraces"]["drift"]["retval"],
            jnp.array([0.08748461, 0.19413349, 0.30486485, 0.4021714, 0.49351367]),
        )
        key, sub_key = jax.random.split(key0)
        choices = to_constraint(tr)
        tru = model.update(sub_key, choices, tr)
        assert jnp.allclose(tru["retval"][0], 11.482168)
        assert jnp.allclose(to_weight(tru), 0.0)
        new_choices = {"drift": choices["drift"].at[1].set(0.2)}
        trv = model.update(sub_key, new_choices, tr)
        assert jnp.allclose(to_weight(trv), 0.17207956)
        assert jnp.allclose(
            trv["subtraces"]["drift"]["retval"],
            jnp.array([0.08748461, 0.2, 0.30486485, 0.4021714, 0.49351367]),
        )
        assert jnp.allclose(
            trv["retval"][1],
            jnp.array([10.087484, 10.287484, 10.592349, 10.99452, 11.488034]),
        )


def test_plain_scan():
    @Gen
    def model(x):
        init = Normal(x, 0.01) @ "init"
        return jax.lax.scan(lambda a, b: (a + b, a + b), init, jnp.arange(5.0))

    tr = model(10.0).simulate(key0)
    assert tr["retval"][0] == 19.997942
    assert jnp.allclose(
        tr["retval"][1],
        jnp.array([9.997942, 10.997942, 12.997942, 15.997942, 19.997942]),
    )
    assert tr["subtraces"]["init"]["retval"] == 9.997942


def test_scan_map():
    @Gen
    def model(step, update):
        return step + Normal(0.0, update) @ "s"

    def diag(r):
        return r + 1.0, r

    tr = model.map(diag)(10.0, 0.01).simulate(key0)
    assert tr["retval"] == (10.997942, 9.997942)

    tr = model.map(diag).scan()(1.0, jnp.ones(3) * 0.01).simulate(key0)
    assert tr["retval"][0] == 3.986483
    assert jnp.allclose(tr["retval"][1], jnp.array([1.9874847, 2.9816182, 3.986483]))


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
            tr["subtraces"]["outlier"]["retval"],
            jnp.array([0, 0, 0, 0, 0, 0]),
        )
        assert jnp.allclose(
            tr["retval"],
            jnp.array([22.00005, 10.993716, 3.9880881, 0.99697554, 1.99467, 7.027452]),
        )
        tr = curve_model.vmap(in_axes=(None, None, 0, None))(
            f, 0.0, jnp.array([0.001, 0.01, 0.9]), 0.3
        ).simulate(key0)
        assert jnp.allclose(tr["subtraces"]["outlier"]["retval"], jnp.array([0, 0, 1]))
        assert jnp.allclose(tr["retval"], jnp.array([0.9980389, 0.91635126, 1.0424924]))

    def test_curve_generation(self):
        quadratic = coefficient.repeat(3).map(Poly)
        points = jnp.arange(-3, 4) / 10.0

        tr = quadratic().simulate(key0)
        assert isinstance(tr["retval"], Poly)

        tr = quadratic.repeat(n=3)().simulate(key0)
        assert isinstance(tr["retval"], Poly)

        @Gen
        def one_model(x):
            poly = quadratic() @ "p"
            return poly(x)

        tr = one_model(0.0).simulate(key0)
        assert tr["retval"] == 1.1188384
        assert isinstance(tr["subtraces"]["p"]["retval"], Poly)

        @Gen
        def model(xs):
            poly = quadratic() @ "p"
            p_outlier = Uniform(0.0, 1.0) @ "p_outlier"
            sigma_inlier = Uniform(0.0, 0.3) @ "sigma_inlier"
            return (
                curve_model.vmap(in_axes=(None, 0, None, None))(
                    poly, xs, p_outlier, sigma_inlier
                )
                @ "y"
            )

        jit_model = jax.jit(model(points).simulate)

        tr = jit_model(key0)
        assert jnp.allclose(
            tr["subtraces"]["p"]["subtraces"]["c"]["retval"],
            jnp.array([0.785558, 2.3734226, 0.07902155]),
        )
        assert jnp.allclose(
            tr["retval"],
            jnp.array(
                [
                    -0.4003628,
                    0.13468444,
                    0.43138668,
                    1.0624609,
                    0.82224107,
                    1.1899176,
                    1.8036526,
                ]
            ),
        )
        assert jnp.allclose(
            tr["subtraces"]["y"]["subtraces"]["outlier"]["retval"],
            jnp.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        )
        assert jnp.allclose(
            tr["retval"],
            jnp.array(
                [
                    -0.4003628,
                    0.13468444,
                    0.43138668,
                    1.0624609,
                    0.82224107,
                    1.1899176,
                    1.8036526,
                ]
            ),
        )

        tr = jax.vmap(jit_model)(jax.random.split(key0, 10))
        assert tr["subtraces"]["p"]["subtraces"]["c"]["retval"].shape == (10, 3)
        assert tr["retval"].shape == (10, 7)


def test_map_map():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    m = noisy.map(lambda x: 2.0 * x).map(lambda x: 10.0 + x)

    tr = m(1.0).simulate(key0)
    assert tr["retval"] == 11.995883

    m = noisy.map(lambda x: 2.0 * x).repeat(4).map(lambda x: 10.0 + x)
    tr = m(1.0).simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([12.006197, 11.972714, 12.045722, 12.013428])
    )


def test_map_vmap():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    def plus5(x):
        return x + 5.0

    noisy_plus5 = noisy.map(plus5)(10.0)
    tr = jax.vmap(noisy_plus5.simulate)(jax.random.split(key0, 3))
    assert jnp.allclose(tr["retval"], jnp.array([15.0111885, 15.005781, 15.008535]))


def test_map():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    def plus5(x):
        return x + 5.0

    noisy_plus5 = noisy.map(plus5)
    tr = noisy_plus5(10.0).simulate(key0)
    assert tr["retval"] == 14.997942


def test_simple_repeat():
    def make_poly_gf(coefficient_gf, n):
        @Gen
        def poly():
            return coefficient_gf.repeat(n)() @ "cs"

        return poly

    poly4 = make_poly_gf(coefficient, 4)
    tr = poly4().simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([0.30984825, -1.3642794, 2.2861156, 0.6714109])
    )


def test_repeat_in_model():
    @Gen
    def x(y):
        return Normal(2.0 * y, 1.0) @ "x"

    @Gen
    def xs():
        return x.repeat(4)(10.0) @ "xs"

    tr = xs().simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([20.309849, 18.635721, 22.286116, 20.671412])
    )


def test_repeat_of_repeat():
    @Gen
    def y(x):
        return Normal(2.0 * x + 1, 0.1) @ "y"

    tr = y.repeat(4).repeat(3)(5.0).simulate(key0)
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                [10.956121, 11.274088, 10.94439, 10.894012],
                [11.076275, 11.013761, 11.081145, 10.860387],
                [11.015622, 11.002483, 10.953617, 10.820805],
            ]
        ),
    )


def test_shaped_distribution():
    @Gen
    def f(x):
        lows = x + jnp.arange(4.0)
        highs = lows + 1
        y = Uniform(lows, highs) @ "y"
        print(f"y {y}")
        return y

    tr = jax.jit(f(2.0).simulate)(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([2.9653215, 3.225159, 4.63303, 5.296382])
    )
    tr = jax.jit(f.repeat(3)(2.0).simulate)(key0)
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                [2.8321762, 3.5617104, 4.3968754, 5.8156433],
                [2.356292, 3.640267, 4.5045667, 5.450263],
                [2.544363, 3.2582088, 4.394433, 5.1704683],
            ]
        ),
    )
    tr = f.vmap()(jnp.arange(2.0, 5.0)).simulate(key0)
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                [2.8321762, 3.5617104, 4.3968754, 5.8156433],
                [3.356292, 4.640267, 5.5045667, 6.450263],
                [4.544363, 5.2582088, 6.394433, 7.1704683],
            ]
        ),
    )


def test_map_in_model():
    @Gen
    def x(y):
        return Normal(y, 0.1) @ "x"

    @Gen
    def mx():
        return x.map(lambda t: t + 13.0)(7.0) @ "mx"

    tr = jax.vmap(mx().simulate)(jax.random.split(key0, 5))
    assert jnp.allclose(
        tr["retval"], jnp.array([19.907951, 19.926113, 20.161331, 19.842987, 19.94448])
    )


def test_map_of_repeat():
    @Gen
    def coefficient():
        return Normal(0.0, 1.0) @ "c"

    pg = coefficient.repeat(3).map(Poly)

    tr = pg().simulate(key0)
    assert jnp.allclose(
        tr["retval"].coefficients, jnp.array([1.1188384, 0.5781488, 0.8535516])
    )
    assert jnp.allclose(tr["retval"](1.0), jnp.array(2.5505388))
    assert jnp.allclose(tr["retval"](2.0), jnp.array(5.6893425))

    kg = coefficient.repeat(3).map(jnp.sum)
    tr = kg().simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(2.5505388))


def test_repeat_of_map():
    @Gen
    def y(x):
        return Normal(x, 0.1) @ "y"

    mr = y.map(lambda x: x + 13.0).repeat(5)(7.0)

    tr = mr.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([19.907951, 19.926113, 20.161331, 19.842987, 19.94448])
    )


def test_repeat_of_cond():
    repeated_model = cond_model.repeat(5)(60.0)
    tr = repeated_model.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([60.057796, 31.760138, 30.233942, 31.401255, 60.03401])
    )


def test_vmap():
    @Gen
    def model(x, y):
        return x + Normal(y, 0.01) @ "a"

    tr = model(5.0, 1.0).simulate(key0)
    assert tr["retval"] == 5.9979415

    gf = model.vmap(in_axes=(0, None))(jnp.arange(5.0), 1.0)
    tr0 = gf.simulate(key0)

    assert jnp.allclose(
        tr0["retval"],
        jnp.array([0.99079514, 1.9926113, 3.016133, 3.9842987, 4.994448]),
    )
    tr1 = model.vmap(in_axes=(None, 0))(5.0, jnp.arange(0.1, 0.4, 0.1)).simulate(key0)
    assert jnp.allclose(
        tr1["retval"], jnp.array([5.1030984, 5.186357, 5.322861, 5.406714])
    )
    tr2 = jax.jit(
        model.vmap(in_axes=(0, 0))(
            jnp.arange(5.0), 0.1 * (1.0 + jnp.arange(5.0))
        ).simulate
    )(key0)
    assert jnp.allclose(
        tr2["retval"],
        jnp.array([0.09079514, 1.1926112, 2.316133, 3.3842988, 4.494448]),
    )
    # try the above without enumerating axis/arguments in in_axes
    tr3 = model.vmap()(jnp.arange(5.0), 0.1 * (1.0 + jnp.arange(5.0))).simulate(key0)
    assert jnp.allclose(tr3["retval"], tr2["retval"])


def test_vmap_of_vmap():
    @Gen
    def model(x, y):
        return Normal(x, y) @ "n"

    tr = (
        model.vmap(in_axes=(0, None))
        .vmap(in_axes=(None, 0))(jnp.arange(10.0, 15.0), jnp.arange(0.01, 1.6, 0.2))
        .simulate(key0)
    )
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                [9.980059, 10.992979, 11.998503, 13.001312, 13.9893465],
                [9.772663, 10.905136, 12.166004, 13.0292635, 14.077281],
                [9.753119, 11.061454, 11.409804, 12.195795, 13.536638],
                [9.595552, 10.138563, 12.190451, 13.225649, 13.833795],
                [9.990936, 11.381924, 12.071567, 12.975254, 14.015008],
                [10.9114685, 10.054609, 14.262349, 13.079898, 14.60372],
                [9.475578, 12.222159, 10.996534, 12.424293, 14.845172],
                [7.6874547, 12.22144, 14.473383, 13.540619, 13.886542],
            ]
        ),
    )


def test_repeat_of_vmap_of_vmap():
    @Gen
    def model(x, y):
        return Normal(x, y) @ "n"

    tr = (
        model.vmap(in_axes=(0, None))
        .vmap(in_axes=(None, 0))
        .repeat(2)(jnp.arange(10.0, 15.0), jnp.arange(0.01, 1.6, 0.2))
        .simulate(key0)
    )
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                [
                    [9.98233, 11.012119, 11.995529, 12.987698, 13.984681],
                    [9.880095, 10.422894, 11.8534565, 13.146841, 14.150014],
                    [10.230905, 10.994424, 11.013716, 13.473481, 14.141039],
                    [10.71579, 12.493758, 11.768768, 13.410502, 14.459529],
                    [10.525383, 10.150476, 11.152355, 13.285344, 15.705538],
                    [12.124693, 11.984819, 11.852717, 11.719085, 14.450966],
                    [10.691346, 11.318585, 14.612149, 13.587466, 14.294254],
                    [7.8196483, 10.1949415, 9.634014, 13.361837, 16.955917],
                ],
                [
                    [9.9943495, 11.002772, 11.99024, 13.001721, 14.030841],
                    [10.308016, 10.665509, 11.690561, 12.951824, 13.819061],
                    [9.571521, 10.517977, 11.905054, 13.208078, 14.319961],
                    [9.619272, 11.235063, 11.404709, 12.237741, 14.299757],
                    [11.264292, 12.461842, 11.243113, 13.063782, 13.424688],
                    [12.247608, 11.796663, 10.529431, 13.871386, 12.75831],
                    [12.851632, 9.468134, 11.794028, 14.386365, 15.522833],
                    [6.649953, 10.184404, 11.6551485, 12.685906, 14.721641],
                ],
            ]
        ),
    )


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
    w, retval = p().assess(constraints)
    assert w == -6.0428767
    assert retval == (2.0, 2.1)
    w, retval = q().assess({"p": constraints})
    assert w == -6.0428767

    with pytest.raises(MissingConstraint) as e:
        _ = p().assess({"x": 2.0})
    assert e.value.args == (("y",),)

    with pytest.raises(MissingConstraint) as e:
        _ = p().assess({"y": 2.0})
    assert e.value.args == (("x",),)


def test_assess_vmap1():
    @Gen
    def p(a):
        return Normal(a, 0.01) @ "x"

    w, retval = p.vmap()(jnp.arange(5.0)).assess({"x": jnp.arange(5.0) + 0.2})
    assert w == pytest.approx(-981.56934)


def test_assess_vmap():
    @Gen
    def p(a, b):
        x = Normal(a, 1.0) @ "x"
        y = Normal(b, 1.0) @ "y"
        return x, y

    model = p.vmap()(jnp.arange(5.0), 10.0 + jnp.arange(5.0))
    w, retval = model.assess(
        {"x": jnp.arange(5.0) + 0.1, "y": 10.0 + jnp.arange(5.0) + 0.2}
    )
    assert w == -9.314385


def test_assess_repeat():
    @Gen
    def m(a):
        return Normal(a, 1.0) @ "x"

    w, _ = m.repeat(4)(10.0).assess({"x": 10.0 + 0.1 * jnp.ones(4)})
    assert w == -3.6957543


def test_bernoulli():
    @Gen
    def p():
        b = Bernoulli(probs=0.01) @ "b"
        c = Bernoulli(logits=-1.0) @ "c"
        return b, c

    tr = p().simulate(key0)
    assert tr["retval"] == (0, 0)
    assert tr["subtraces"]["b"]["score"] == math.log(1 - 0.01)
    assert tr["subtraces"]["c"]["score"] == math.log(
        1 - math.exp(-1) / (1 + math.exp(-1))
    )

    with pytest.raises(ValueError):
        Bernoulli()

    with pytest.raises(TypeError):
        Bernoulli(0.1)  # type: ignore

    with pytest.raises(ValueError):
        Bernoulli(logits=-1, probs=0.5)


def test_importance():
    @Gen
    def model():
        a = Normal(0.0, 1.0) @ "a"
        b = Normal(0.0, 0.1) @ "b"
        return a, b

    @Gen
    def outer():
        c = Normal(0.0, 1.0) @ "c"
        a, b = model() @ "d"
        return a + b + c

    model_imp = jax.jit(model().importance)
    outer_imp = jax.jit(outer().importance)

    tr1, w1 = model_imp(key0, {"a": 1.0})
    assert w1 == -1.4189385
    tr2, w2 = model_imp(key0, {"b": 1.0})
    assert w2 == -48.616352
    tr3, w3 = model_imp(key0, {"a": 1.0, "b": 1.0})
    assert w3 == w1 + w2

    tr4, w4 = outer_imp(key0, {"c": 0.5, "d": {"b": 0.3}})
    assert w4 == -4.160292


def test_repeat_importance():
    @Gen
    def model(z):
        a = Normal(z, 0.1) @ "a"
        b = Normal(z, 1.0) @ "b"
        return a + b

    mr = model.repeat(4)(1.0)
    mr_imp = jax.jit(mr.importance)
    values = jnp.arange(4) / 10.0
    tr, w = mr_imp(key0, {"a": values})
    assert jnp.allclose(tr["subtraces"]["a"]["retval"], values)
    assert w == -141.46541
    assert w == jnp.sum(tr["subtraces"]["a"]["w"])


def test_vmap_importance():
    @Gen
    def model(x, y):
        a = Normal(x, 0.1) @ "a"
        b = Normal(y, 0.2) @ "b"
        return a + b

    values = jnp.arange(5.0)
    mv1 = model.vmap(in_axes=(0, None))(values, 10.0)
    mv1_imp = jax.jit(mv1.importance)
    observed_values = values + 0.2
    tr, w1 = mv1_imp(key0, {"a": observed_values})
    assert w1 == -3.0817661
    mv2 = model.vmap(in_axes=(None, 0))(10.0, values)
    tr, w2 = jax.jit(mv2.importance)(key0, {"b": observed_values})
    assert w2 == 0.95249736
    mv3 = model.vmap()(values, values)
    tr, w3 = jax.jit(mv3.importance)(key0, {"a": observed_values, "b": observed_values})
    assert w3 == w2 + w1


def test_partial():
    @Gen
    def model(x, y):
        return Normal(x, y) @ "x"

    tr = model.partial(10.0)(0.01).simulate(key0)
    assert tr["retval"] == 9.997942
    tr1 = model.partial(10.0, 0.01)().simulate(key0)
    assert tr1["retval"] == tr["retval"]


def test_categorial_jaxpr():
    N = 10

    @Gen
    def model(key):
        logits = Normal(jnp.zeros(5), jnp.ones(5)) @ "logits"
        return jax.vmap(Categorical(logits=logits).sample)(jax.random.split(key, N))

    key, k1, k2 = jax.random.split(key0, 3)
    m = model(k1).simulate(k2)
    print(m)
    # this is basically a stub, since we may want to experiment with
    # a new kind of primitive for sampling outside of a generative function
    # that does not expand to machine code under vmap. The problem is that
    # if someone does their own vmap over Categorical, vmap passes over the
    # primitive boundary, since the distribution isn't being used generatively.


def test_gen_paper_update():
    @Gen
    def inner1(val):
        return jnp.logical_and(Bernoulli(probs=0.6) @ "c", val)

    @Gen
    def inner2(val):
        # was d, but the restrictions around Cond are strict.
        # the branches have to generate data with the same
        return jnp.logical_and(Bernoulli(probs=0.1) @ "c", val)

    @Gen
    def foo():
        val = Bernoulli(probs=0.3) @ "a"
        val = Cond(inner1(val), inner2(val))(Bernoulli(probs=0.4) @ "b") @ "x"
        val = jnp.logical_and(Bernoulli(probs=0.7) @ "e", val)
        return val

    # roll the dice until we get the configuration illustrated in figure 3 of the paper
    key = key0
    while True:
        key, sub_key = jax.random.split(key)
        tr = foo().simulate(sub_key)
        ch = to_constraint(tr)
        if ch["a"] == 0 and ch["b"] == 1 and ch["x"]["c"] == 0 and ch["e"] == 1:
            break

    assert jnp.exp(to_score(tr)) == pytest.approx(0.0784)
    u = {"b": 0, "x": {"c": 1}}
    tr2 = foo().update(sub_key, u, tr)
    assert jnp.exp(to_weight(tr2)) == pytest.approx(0.375)

# %%
# pyright: reportWildcardImportFromLibrary=false
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass
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
    poly = coefficient().repeat(3).map(Poly)
    tr = poly.simulate(key0)
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
    print(jax.make_jaxpr(model().simulate)(key0))
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
    tr = jax.vmap(model3(50.0).map(lambda t: t[0] + t[1]).simulate)(
        jax.random.split(key0, 5)
    )
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
    assert jnp.allclose(tr["retval"], jnp.array(99.979416))
    tr = model2(b / 2).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(50.836914))
    c = Cond(model1(b), model2(b / 2.0))
    tr = c(0).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(50.836914))
    tr = c(1).simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(99.979416))
    tr = jax.vmap(lambda i, k: c(i).simulate(k))(
        jnp.mod(jnp.arange(10.0), 2), jax.random.split(key0, 10)
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
    assert jnp.allclose(tr["retval"][0], 11.482168)
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
        quadratic = coefficient().repeat(3).map(Poly)
        points = jnp.arange(-3, 4) / 10.0

        # curve_model(f, x, p_outlier, sigma_inlier)
        @Gen
        def model(xs):
            poly = quadratic @ "p"
            p_outlier = Uniform(0.0, 1.0) @ "p_outlier"
            sigma_inlier = Uniform(0.0, 0.3) @ "sigma_inlier"
            return (
                curve_model.vmap(in_axes=(None, 0, None, None))(
                    poly, xs, p_outlier, sigma_inlier
                )
                @ "y"
            )

        print(jax.make_jaxpr(model(points).simulate)(key0))
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


def test_map():
    @Gen
    def noisy(x):
        return Normal(x, 0.01) @ "x"

    def plus5(x):
        return x + 5.0

    noisy_plus5 = noisy(10).map(plus5)
    tr = jax.vmap(noisy_plus5.simulate)(jax.random.split(key0, 3))
    assert jnp.allclose(tr["retval"], jnp.array([15.0111885, 15.005781, 15.008535]))


def test_simple_repeat():
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
        tr["retval"], jnp.array([0.30984825, -1.3642794, 2.2861156, 0.6714109])
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
        tr["retval"], jnp.array([20.309849, 18.635721, 22.286116, 20.671412])
    )


def test_repeat_of_repeat():
    @Gen
    def y(x):
        return Normal(2.0 * x + 1, 0.1) @ "y"

    tr = y(5.0).repeat(4).repeat(3).simulate(key0)
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


def test_map_in_model():
    @Gen
    def x(y):
        return Normal(y, 0.1) @ "x"

    @Gen
    def mx():
        return x(7.0).map(lambda t: t + 13.0) @ "mx"

    tr = jax.vmap(mx().simulate)(jax.random.split(key0, 5))
    assert jnp.allclose(
        tr["retval"], jnp.array([19.907951, 19.926113, 20.161331, 19.842987, 19.94448])
    )


def test_map_of_repeat():
    @Gen
    def coefficient():
        return Normal(0.0, 1.0) @ "c"

    pg = coefficient().repeat(3).map(Poly)

    tr = pg.simulate(key0)
    assert jnp.allclose(
        tr["retval"].coefficients, jnp.array([1.1188384, 0.5781488, 0.8535516])
    )
    assert jnp.allclose(tr["retval"](1.0), jnp.array(2.5505388))
    assert jnp.allclose(tr["retval"](2.0), jnp.array(5.6893425))

    kg = coefficient().repeat(3).map(jnp.sum)
    tr = kg.simulate(key0)
    assert jnp.allclose(tr["retval"], jnp.array(2.5505388))


def test_repeat_of_map():
    @Gen
    def y(x):
        return Normal(x, 0.1) @ "y"

    mr = y(7.0).map(lambda x: x + 13.0).repeat(5)

    tr = mr.simulate(key0)
    assert jnp.allclose(
        tr["retval"], jnp.array([19.907951, 19.926113, 20.161331, 19.842987, 19.94448])
    )


def test_repeat_of_cond():
    repeated_model = cond_model(60.0).repeat(5)
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


@pytest.mark.skip(reason="nested vmap not working yet")
def test_vmap_of_vmap():
    @Gen
    def model(x, y):
        return Normal(x, y) @ "n"

    tr = (
        model.vmap(in_axes=(0, None))
        # .vmap(in_axes=(None, 0))
        (jnp.arange(10.0, 15.0), jnp.arange(0.0, 1.6, 0.2)).simulate(key0)
    )
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            # this looked promising but it was cheating by using broadcast semantics on the inner loop
            [
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [9.807053, 11.150885, 11.985554, 12.91451, 13.865575],
                [9.949703, 11.00537, 12.350649, 13.016957, 14.176276],
                [9.51422, 10.44197, 10.95027, 12.562811, 14.896659],
                [9.411664, 11.082625, 11.907326, 14.153446, 13.8836355],
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
    assert tr["retval"] == (0, 0)
    assert tr["subtraces"]["b"]["score"] == math.log(1 - 0.01)
    assert tr["subtraces"]["c"]["score"] == math.log(
        1 - math.exp(-1) / (1 + math.exp(-1))
    )

    with pytest.raises(ValueError):
        Bernoulli()

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

    tr1 = model_imp(key0, {"a": 1.0})
    assert tr1["w"] == -1.4189385
    tr2 = model_imp(key0, {"b": 1.0})
    assert tr2["w"] == -48.616352
    tr3 = model_imp(key0, {"a": 1.0, "b": 1.0})
    assert tr3["w"] == tr1["w"] + tr2["w"]

    tr4 = outer_imp(key0, {"c": 0.5, "d": {"b": 0.3}})

    assert tr4["w"] == -4.160292


def test_repeat_importance():
    @Gen
    def model(a):
        return Normal(a, 0.1) @ "a"

    mr = model(1.0).repeat(4)
    mr_imp = jax.jit(mr.importance)
    tr = mr_imp(key0, {"b": 0.2})
    print(tr)


# %%

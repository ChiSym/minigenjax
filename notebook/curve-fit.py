# %%
# pyright: reportUnusedExpression=false
import jax
import genstudio.plot as Plot
import jax.numpy as jnp
import dataclasses
import minigenjax as mg
# %%


@mg.Gen
def inlier_model(y, sigma_inlier):
    return mg.Normal(y, sigma_inlier) @ "value"


@mg.Gen
def outlier_model(y):
    return mg.Uniform(y - 1.0, y + 1.0) @ "value"


@mg.Gen
def curve_model(f, x, p_outlier, sigma_inlier):
    outlier = mg.Flip(p_outlier) @ "outlier"
    y = f(x)
    fork = mg.Cond(outlier_model(y), inlier_model(y, sigma_inlier))
    return fork(outlier) @ "y"


@mg.Gen
def coefficient():
    return mg.Normal(0.0, 1.0) @ "c"


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


@mg.Gen
def model(xs):
    poly = quadratic @ "p"
    p_outlier = mg.Uniform(0.0, 1.0) @ "p_outlier"
    sigma_inlier = mg.Uniform(0.0, 0.3) @ "sigma_inlier"
    return (
        curve_model.vmap(in_axes=(None, 0, None, None))(
            poly, xs, p_outlier, sigma_inlier
        )
        @ "y"
    )


quadratic = coefficient().repeat(3).map(Poly)
xs = jnp.arange(-3, 4) / 10.0
key = jax.random.key(0)
prior = jax.jit(model(xs).simulate)
importance = jax.jit(model(xs).importance)


# %%
def goal(x):
    return -0.4 * x**2 + 0.8 * x + 0.7


ys = jax.vmap(goal)(xs)

(
    Plot.line([(x, goal(x)) for x in jnp.arange(-1.0, 1.0, 0.1)])
    + Plot.dot(zip(xs, ys))
    + Plot.domain([-1, 1])
)


# %%
key, sub_key = jax.random.split(key)
tr = prior(sub_key)
# %%
# this is the polynomial (but it has lost the type, which is a bug)
tr["subtraces"]["p"]["c"]["retval"]
# %%
# These are the y values
tr["subtraces"]["y"]["y"]["retval"]
# %%
key, sub_key = jax.random.split(key)
model(xs).importance(sub_key, {"y": {"y": {"value": ys}}})
# %%
print(jax.make_jaxpr(lambda k: model(xs).simulate(k)['retval'])(sub_key))

# %%

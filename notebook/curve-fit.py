# %%
# pyright: reportUnusedExpression=false
import jax
import genstudio.plot as Plot
import jax.numpy as jnp
import minigenjax as mg


# from pprint import pprint as pp
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


@mg.pytree
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
    poly = quadratic() @ "p"
    p_outlier = mg.Uniform(0.0, 0.3) @ "p_outlier"
    sigma_inlier = mg.Uniform(0.0, 0.3) @ "sigma_inlier"
    return (
        curve_model.vmap(in_axes=(None, 0, None, None))(
            poly, xs, p_outlier, sigma_inlier
        )
        @ "ys"
    )


quadratic = coefficient.repeat(3).map(Poly)
xs = jnp.arange(-3, 4) / 10.0
key = jax.random.key(0)
prior = jax.jit(model(xs).simulate)
importance = jax.jit(model(xs).importance)


# %%
def goal(x):
    return -0.4 * x**2 + 0.8 * x + 0.7


ys = jax.vmap(goal)(xs)
# %%
prior(key)
# %%
key, sub_key = jax.random.split(key)
prior_ps = jax.vmap(prior)(jax.random.split(sub_key, 100))["subtraces"]["p"]["retval"]

# %%
prior_ps


# %%
def plot_curves(curves):
    return (
        Plot.line([(x, goal(x)) for x in jnp.arange(-1.0, 1.0, 0.1)])
        + [
            Plot.line(
                [(x, p(x)) for x in jnp.arange(-1.0, 1.0, 0.1)],
                stroke="#00f",
                opacity=0.2,
            )
            for p in curves
        ]
        + Plot.dot(zip(xs, ys))
        + Plot.domain([-1, 1])
    )


plot_curves(prior_ps)


# %%
key, sub_key = jax.random.split(key)
tr = prior(sub_key)

mg.to_constraint(tr)
# %%
# this is the polynomial (but it has lost the type, which is a bug)
tr["subtraces"]["p"]["subtraces"]["c"]["retval"]
# %%
# These are the y values
tr["subtraces"]["ys"]["retval"]
# %%
key, sub_key = jax.random.split(key)
imp = jax.jit(model(xs).importance)
observations: mg.Constraint = {"ys": {"y": {"value": ys}}}
imp(sub_key, observations)
# %%
print(jax.make_jaxpr(lambda k: model(xs).importance(k, observations))(sub_key))

# %%
key, sub_key = jax.random.split(key)
tr, ws = jax.vmap(lambda k: imp(k, observations))(jax.random.split(sub_key, 500000))
key, sub_key = jax.random.split(sub_key)
winners = jax.vmap(mg.Categorical(logits=ws).sample)(jax.random.split(sub_key, 100))
posterior_ps = tr["subtraces"]["p"]["retval"][winners]
plot_curves(posterior_ps)
# %%
mg.to_score(tr)
# %%

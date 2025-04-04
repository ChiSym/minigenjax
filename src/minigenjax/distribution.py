import functools
from . import minigenjax as mg
import jax
import jax.core
from jax.interpreters import mlir
from jaxtyping import Array, PRNGKeyArray, Float, DTypeLike
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp


class Distribution(mg.GenPrimitive):
    def __init__(self, name, tfd_ctor, dtype: DTypeLike = jnp.dtype("float32")):
        super().__init__(name)
        self.tfd_ctor = tfd_ctor
        self.dtype = dtype
        mlir.register_lowering(self, mlir.lower_fun(self.impl, False))

    def batch(self, vector_args, batch_axes, **kwargs):
        if axes := kwargs.pop("axes", None):
            axes = axes + [batch_axes]
        else:
            axes = [batch_axes]
        return self.bind(*vector_args, axes=axes, **kwargs), 0

    def operation(self, arg_tuple, op):
        match op:
            case "Sample":
                return self.tfd_ctor(*arg_tuple[1:]).sample(seed=arg_tuple[0])
            case "Score":
                return self.tfd_ctor(*arg_tuple[1:]).log_prob(arg_tuple[0])

    def abstract(self, *args, **kwargs):
        return jax.core.get_aval(
            jax.eval_shape(lambda args: self.concrete(*args, **kwargs), args)
        )

    def concrete(self, *args, **kwargs):
        axeses = kwargs.get("axes")
        op = kwargs["op"]
        if axeses:
            return functools.reduce(
                lambda f, axes: jax.vmap(f, in_axes=axes),
                axeses,
                lambda *args: self.operation(args, op),
            )(*args)
        else:
            return self.operation(args, op)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: mg.Address,
        constraint: mg.Constraint | None,
    ):
        if constraint is not None:  # TODO: fishy
            score = self.bind(constraint, *arg_tuple[1:], op="Score")
            ans = {"w": score, "retval": constraint}
        else:
            retval = self.bind(key, *arg_tuple[1:], op="Sample")
            score = self.bind(retval, *arg_tuple[1:], op="Score")
            ans = {"retval": retval, "score": score}
        return ans

    def assess_p(
        self,
        arg_tuple: tuple,
        constraint: mg.Constraint | Float,
        address: tuple[str, ...],
    ) -> tuple[Array, Array]:
        assert not isinstance(constraint, dict)
        score = self.bind(constraint, *arg_tuple[1:], op="Score")
        return score, constraint

    def __call__(self, *args):
        this = self

        class Binder:
            def __matmul__(self, address: str):
                return this.bind(mg.PHANTOM_KEY, *args, op="Sample", at=address)

            def to_tfp(self):
                return this.tfd_ctor(*args)

            def sample(self, key: PRNGKeyArray):
                return self.to_tfp().sample(seed=key)

            # TODO: from here, you can't `map` a distribution.

        return Binder()


BernoulliL = Distribution(
    "Bernoulli:L",
    lambda logits: tfp.distributions.Bernoulli(logits=logits),
    dtype=jnp.dtype("int32"),
)
BernoulliP = Distribution(
    "Bernoulli:P",
    lambda probs: tfp.distributions.Bernoulli(probs=probs),
    dtype=jnp.dtype("int32"),
)
Normal = Distribution("Normal", tfp.distributions.Normal)
MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution(
    "Flip", lambda p: tfp.distributions.Bernoulli(probs=p), dtype=jnp.dtype("int32")
)
CategoricalL = Distribution(
    "Categorical:L",
    lambda logits: tfp.distributions.Categorical(logits=logits),
)
CategoricalP = Distribution(
    "Categorical:P",
    lambda probs: tfp.distributions.Categorical(probs=probs),
)

# TODO: This can't be stored as a JAX eqn, since it involves a list, sub-constructors,
# etc., so it's useless for now. Doing this in our current system would involve
# flattening the parameters, and storing the underlying distribution names in the
# bind parameters, which is probably not worth it
Mixture = Distribution(
    "Mixture",
    lambda cat, components: tfp.distributions.Mixture(
        cat=cat.to_tfp(), components=list(map(lambda c: c.to_tfp(), components))
    ),
)


def choose_scale(logits, probs, logit_dist, prob_dist):
    if (logits is None) == (probs is None):
        raise ValueError("Supply exactly one of logits=, probs=")
    return logit_dist(logits) if logits is not None else prob_dist(probs)


def Bernoulli(*, logits=None, probs=None):
    return choose_scale(logits, probs, BernoulliL, BernoulliP)


def Categorical(*, logits=None, probs=None):
    return choose_scale(logits, probs, CategoricalL, CategoricalP)

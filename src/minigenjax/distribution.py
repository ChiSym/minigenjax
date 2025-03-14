from . import minigenjax as mg
from jax.interpreters import mlir
from jaxtyping import Array, PRNGKeyArray, Float
import tensorflow_probability.substrates.jax as tfp
import jax.numpy as jnp


class Distribution(mg.GenPrimitive):
    def __init__(self, name, tfd_ctor):
        super().__init__(name)
        self.tfd_ctor = tfd_ctor
        mlir.register_lowering(self, mlir.lower_fun(self.impl, False))

    def abstract(self, *args, **kwargs):
        return args[1]

    def concrete(self, *args, **kwargs):
        match kwargs.get("op", "Sample"):
            case "Sample":
                # we convert to float here because Bernoulli/Flip will
                # normally return an int, and that confuses XLA, since
                # our abstract implementation says the return types are
                # floats. TODO: consider allowing the marking of integer-
                # returning distributions as ints are sometimes nice to
                # work with.
                return jnp.asarray(
                    self.tfd_ctor(*args[1:]).sample(seed=args[0]), dtype=float
                )
            case "Score":
                return self.tfd_ctor(*args[1:]).log_prob(args[0])
            case _:
                raise NotImplementedError(f"{self.name}.{kwargs['op']}")

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
                return this.bind(mg.PHANTOM_KEY, *args, at=address)

            def to_tfp(self):
                return this.tfd_ctor(*args)

            def sample(self, key: PRNGKeyArray):
                return self.to_tfp().sample(seed=key)

            # TODO: from here, you can't `map` a distribution.

        return Binder()


BernoulliL = Distribution(
    "Bernoulli:L",
    lambda logits: tfp.distributions.Bernoulli(logits=logits),
)
BernoulliP = Distribution(
    "Bernoulli:P",
    lambda probs: tfp.distributions.Bernoulli(probs=probs),
)
Normal = Distribution("Normal", tfp.distributions.Normal)
MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution("Flip", lambda p: tfp.distributions.Bernoulli(probs=p))
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

# %%
from typing import Any, Callable
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.tree
import jax.numpy as jnp
from jax.interpreters import batching, mlir
import jax.core
from jaxtyping import PRNGKeyArray

# %%
KEY_TYPE = jax.core.ShapedArray((2,), jnp.uint32)
SCORE_TYPE = jax.core.ShapedArray((), jnp.float32)
PHANTOM_KEY = jax.random.key(987654321)
SCALAR_STRUCTURE = jax.tree.structure({"retval": 0})


class GenPrimitive(jax.core.Primitive):
    def __init__(self, name):
        super().__init__(name)
        self.def_abstract_eval(self.abstract)
        self.def_impl(self.concrete)
        mlir.register_lowering(self, mlir.lower_fun(self.impl, self.multiple_results))
        batching.primitive_batchers[self] = self.batch

    def abstract(self, *args, **kwargs):
        raise NotImplementedError()

    def concrete(self, *args, **kwargs):
        raise NotImplementedError()

    def simulate_p(self, sub_key: PRNGKeyArray, arg_tuple: tuple):
        raise NotImplementedError()

    def batch(self, vector_args, batch_axes, **kwargs):
        # TODO assert all axes equal
        result = jax.vmap(lambda *args: self.impl(*args, **kwargs), in_axes=batch_axes)(
            *vector_args
        )
        batched_axes = (
            (batch_axes[0],) * len(result) if self.multiple_results else batch_axes[0]
        )
        return result, batched_axes

    def __call__(self, *args):
        this = self

        class Binder:
            def __matmul__(self, address: str):
                return this.bind(PHANTOM_KEY, *args, at=address)

        return Binder()


class Distribution(GenPrimitive):
    def __init__(self, name, tfd_ctor):
        super().__init__(name)
        self.tfd_ctor = tfd_ctor

    def abstract(self, *args, **kwargs):
        return args[1]

    def concrete(self, *args, **kwargs):
        match kwargs.get("op", "Sample"):
            case "Sample":
                return self.tfd_ctor(*args[1:]).sample(seed=args[0])
            case "Score":
                return self.tfd_ctor(*args[1:]).log_prob(args[0])
            case _:
                raise NotImplementedError(f'{self.name}.{kwargs['op']}')

    def simulate_p(self, sub_key: PRNGKeyArray, arg_tuple: tuple):
        print(f"distribution simulate_p sub_key {sub_key} arg_tuple {arg_tuple}")
        v = self.bind(sub_key, *arg_tuple[1:], op="Sample")
        return {"retval": v, "score": self.bind(v, *arg_tuple[1:], op="Score")}


Normal = Distribution("Normal", tfp.distributions.Normal)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution("Bernoulli", lambda p: tfp.distributions.Bernoulli(probs=p))


class KeySplitP(jax.core.Primitive):
    def __init__(self):
        super().__init__("KeySplit")
        self.def_impl(lambda k: jax.random.split(k, 2))
        self.multiple_results = True
        self.def_abstract_eval(lambda _: [KEY_TYPE, KEY_TYPE])

        mlir.register_lowering(self, mlir.lower_fun(self.impl, self.multiple_results))

        batching.primitive_batchers[self] = self.batch

    def batch(self, vector_args, batch_axes):
        key_pair_vector = jax.vmap(self.impl, in_axes=batch_axes)(*vector_args)
        # Transpose key_pair_vector into a pair of key vectors
        return [key_pair_vector[:, :, 0], key_pair_vector[:, :, 1]], (
            batch_axes[0],
            batch_axes[0],
        )


KeySplit = KeySplitP()
GenSymT = Callable[[jax.core.AbstractValue], jax.core.Var]


# %%
class Gen:
    def __init__(self, f: Callable[..., Any]):
        self.f = f

    def __call__(self, *args) -> "GF":
        return GF(self.f, args)


class GF(GenPrimitive):
    def __init__(self, f: Callable[..., Any], args: tuple):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f
        self.args = args
        self.jaxpr = jax.make_jaxpr(f)(*args)
        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        self.abstract_value = a_vals if self.multiple_results else a_vals[0]

    def abstract(self, *args, at: str):
        return self.abstract_value

    def concrete(self, *args, at: str):
        v = jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *args)
        return v if self.multiple_results else v[0]

    # TODO: see if we can unify these two: if we're given any args, use them, else use the stored args?
    def simulate(self, key: PRNGKeyArray):
        return Simulate.run_gf(self, key, *self.args)

    def simulate_p(self, sub_key: PRNGKeyArray, arg_tuple: tuple):
        return Simulate.run(self.jaxpr, sub_key, arg_tuple)

    def __matmul__(self, address: str):
        print(f"GF bind {self.f} {self.args} @ {address}")
        return self.bind(*self.args, at=address)


# %%
key0 = jax.random.PRNGKey(0)


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


# %%
class Simulate:
    score_gensym: GenSymT = jax.core.gensym("_score")

    @staticmethod
    def run_gf(gf: GF, key: PRNGKeyArray, *args):
        return Simulate.run(gf.jaxpr, key, args)

    @staticmethod
    def run(closed_jaxpr: jax.core.ClosedJaxpr, key: PRNGKeyArray, arg_tuple):
        jaxpr = closed_jaxpr.jaxpr
        trace = {}
        env: dict[jax.core.Var, Any] = {}

        def read(v: jax.core.Atom) -> Any:
            return v.val if isinstance(v, jax.core.Literal) else env[v]

        def write(v: jax.core.Var, val: Any) -> None:
            # if config.enable_checks.value and not config.dynamic_shapes.value:
            #   assert typecheck(v.aval, val), (v.aval, val)
            env[v] = val

        jax.util.safe_map(write, jaxpr.constvars, closed_jaxpr.consts)
        print(f"invars {jaxpr.invars} arg_tuple {arg_tuple}")
        jax.util.safe_map(write, jaxpr.invars, arg_tuple)

        for eqn in jaxpr.eqns:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            # name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            # traceback = eqn.source_info.traceback if propagate_source_info else None
            # with source_info_util.user_context(
            #    traceback, name_stack=name_stack), eqn.ctx.manager:
            params = tuple(jax.util.safe_map(read, eqn.invars))
            if isinstance(eqn.primitive, GenPrimitive):
                key, sub_key = KeySplit.bind(key)
                print(
                    f"GF eqn {eqn.primitive}{(sub_key,) + params} (bp {bind_params}) (sf {subfuns})"
                )
                ans = eqn.primitive.simulate_p(sub_key, params)
                # ans = Simulate.run_gf(eqn.primitive, sub_key, *params)
                print(f"ans = {ans}")
                trace[bind_params["at"]] = ans
                ans = ans["retval"]
            elif eqn.primitive is jax.lax.cond_p:
                key, sub_key = KeySplit.bind(key)
                branches = bind_params["branches"]

                def simify(jaxpr):
                    """Apply simulate to jaxpr and return the transformed jaxpr together
                    with its return shape."""
                    return jax.make_jaxpr(
                        lambda key, *args: Simulate.run(jaxpr, key, *args),
                        return_shape=True,
                    )(sub_key, params[1:])

                # simified is a list of pairs (jaxpr, shape)
                simified = list(map(simify, branches))
                simmed_branches = tuple(s[0] for s in simified)
                shapes = [s[1] for s in simified]
                ans = eqn.primitive.bind(
                    *subfuns, params[0], sub_key, *params[1:], branches=simmed_branches
                )

                # tricky: we want to find the address at which the cond result should be traced,
                # but this is inside the cond. If, in fact, both branches are GF invocations traced
                # to the same address (the Cond combinator arranges for this), we will use that address.
                def address_from_branch(b: jax.core.ClosedJaxpr):
                    if len(b.jaxpr.eqns) == 1 and isinstance(
                        b.jaxpr.eqns[0].primitive, GF
                    ):
                        return b.jaxpr.eqns[0].params.get("at")

                branch_addresses = tuple(map(address_from_branch, branches))
                if branch_addresses[0] and all(
                    b == branch_addresses[0] for b in branch_addresses[1:]
                ):
                    address = branch_addresses[0]
                    u = jax.tree.unflatten(jax.tree_structure(shapes[0]), ans)
                    # flatten out an extra layer of subtraces
                    trace[address] = u["subtraces"][address]
                    ans = [u["retval"]]
                print(f"cond outvars {eqn.outvars} {eqn.primitive.multiple_results}")
            else:
                ans = eqn.primitive.bind(*subfuns, *params, **bind_params)
            if eqn.primitive.multiple_results:
                jax.util.safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
            # clean_up_dead_vars(eqn, env, lu)
        retvals = jax.util.safe_map(read, jaxpr.outvars)

        return {
            "retval": retvals if len(jaxpr.outvars) > 1 else retvals[0],
            "subtraces": trace,
        }


# %%


def Cond(tf, ff):
    def ctor(pred):
        class Binder:
            def __matmul__(self, address: str):
                return jax.lax.cond(
                    jnp.int32(pred), lambda: tf @ address, lambda: ff @ address
                )

        return Binder()

    return ctor


@Gen
def switch_prototype(b):
    flip = Flip(0.5) @ "flip"
    ma = model1(b)
    mb = model2(b / 2)
    a = jax.lax.cond(
        jnp.int32(flip),
        lambda: ma @ "s",
        lambda: mb @ "s",
    )
    return a


@Gen
def sp2(b):
    flip = Flip(0.5) @ "flip"
    y = Cond(model1(b), model2(b / 2.0))(flip) @ "s"
    return y


# switch_prototype(100.0).simulate(key0)
sp2(100.0).simulate(key0)
# %%
jax.vmap(sp2(100.0).simulate)(jax.random.split(key0, 20))


# %%
model(20.0).simulate(key0)
# %%
jax.vmap(model(40.0).simulate)(jax.random.split(key0, 1000))
# %%
model1(25.0).simulate(key0)


# %%
@Gen
def f():
    return Flip(0.5) @ "f"


# %%
jax.vmap(f().simulate)(jax.random.split(key0, 100))
# %%

switch_prototype(100.0).jaxpr

# %%
switch_prototype(100.0) @ "x"
# %%
jax.vmap(switch_prototype(100.0).simulate)(jax.random.split(key0, 100))


# %%
@Gen
def inlier_model(y, sigma_inlier):
    return Normal(y, sigma_inlier) @ "value"


@Gen
def outlier_model():
    return Uniform(-1.0, 1.0) @ "value"


@Gen
def curve_model(x, p_outlier):
    outlier = Flip(p_outlier) @ "outlier"
    y0 = x**2 - x + 1.0
    fork = Cond(outlier_model(), inlier_model(y0, 0.1))
    return fork(outlier) @ "y"


# %%
curve_model(4, 0.2).simulate(key0)
# %%
jax.vmap(curve_model(1.0, 0.2).simulate)(jax.random.split(key0, 10))
# %%
jax.make_jaxpr(lambda x, k: curve_model(x, 0.2).simulate(k))(1.0, key0)
# %%
# Not working yet: outlier_model doesn't get batched
# jax.vmap(
#     lambda k: jax.vmap(
#         lambda x: curve_model(x, 0.2).simulate(k)
#     )(jnp.arange(-2, 3.)))(jax.random.split(key0, 10))

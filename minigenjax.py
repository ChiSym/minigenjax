# %%
from typing import Any, Callable, Sequence
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.tree
import jax.numpy as jnp
from jax.interpreters import batching, mlir
import jax.core
from jaxtyping import PRNGKeyArray

# %%
PHANTOM_KEY = jax.random.key(987654321)


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
                raise NotImplementedError(f'{self.name}.{kwargs["op"]}')

    def simulate_p(self, sub_key: PRNGKeyArray, arg_tuple: tuple):
        v = self.bind(sub_key, *arg_tuple[1:], op="Sample")
        return {"retval": v, "score": self.bind(v, *arg_tuple[1:], op="Score")}

    def __call__(self, *args):
        this = self

        class Binder:
            def __matmul__(self, address: str):
                return this.bind(PHANTOM_KEY, *args, at=address)

            def __call__(self, key: PRNGKeyArray):
                return this.tfd_ctor(*args).sample(seed=key)

        return Binder()


Normal = Distribution("Normal", tfp.distributions.Normal)
MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution("Bernoulli", lambda p: tfp.distributions.Bernoulli(probs=p))
# MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
# bug with numpy 2.0
Categorical = Distribution(
    "Categorical", lambda ls: tfp.distributions.Categorical(logits=ls)
)


class KeySplitP(jax.core.Primitive):
    KEY_TYPE = jax.core.ShapedArray((2,), jnp.uint32)

    def __init__(self):
        super().__init__("KeySplit")
        self.def_impl(lambda k: jax.random.split(k, 2))
        self.multiple_results = True
        self.def_abstract_eval(lambda _: [self.KEY_TYPE, self.KEY_TYPE])

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
class Gen[R]:
    def __init__(self, f: Callable[..., R]):
        self.f = f

    def __call__(self, *args) -> "GF[R]":
        return GF(self.f, args)


class GF[R](GenPrimitive):
    def __init__(self, f: Callable[..., R], args: tuple):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f
        self.args = args
        self.jaxpr, self.shape = jax.make_jaxpr(f, return_shape=True)(*args)
        self.multiple_results = isinstance(self.shape, tuple)
        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        self.abstract_value = a_vals if self.multiple_results else a_vals[0]

    def abstract(self, *args, at: str):
        return self.abstract_value

    def concrete(self, *args, at: str):
        v = jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *args)
        return v if self.multiple_results else v[0]

    # TODO: see if we can unify these two: if we're given any args, use them, else use the stored args?
    def simulate(self, key: PRNGKeyArray) -> dict:
        return Simulate(key).run(self.jaxpr, self.args)

    def simulate_p(self, sub_key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        return Simulate(sub_key).run(self.jaxpr, arg_tuple)

    def __matmul__(self, address: str):
        return self.bind(*self.args, at=address)


# %%


class Transformation[R]:
    def __init__(self):
        pass

    def handle_eqn(self, eqn, params, bind_params):
        return eqn.primitive.bind(*params, **bind_params)

    def run(self, closed_jaxpr: jax.core.ClosedJaxpr, arg_tuple):
        jaxpr = closed_jaxpr.jaxpr
        env: dict[jax.core.Var, Any] = {}

        def read(v: jax.core.Atom) -> Any:
            return v.val if isinstance(v, jax.core.Literal) else env[v]

        def write(v: jax.core.Var, val: Any) -> None:
            # if config.enable_checks.value and not config.dynamic_shapes.value:
            #   assert typecheck(v.aval, val), (v.aval, val)
            env[v] = val

        jax.util.safe_map(write, jaxpr.constvars, closed_jaxpr.consts)
        jax.util.safe_map(write, jaxpr.invars, arg_tuple)

        for eqn in jaxpr.eqns:
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            if subfuns:
                raise NotImplementedError("nonempty subfuns")
            # name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
            # traceback = eqn.source_info.traceback if propagate_source_info else None
            # with source_info_util.user_context(
            #    traceback, name_stack=name_stack), eqn.ctx.manager:
            params = tuple(jax.util.safe_map(read, eqn.invars))
            ans = self.handle_eqn(eqn, params, bind_params)
            if eqn.primitive.multiple_results:
                jax.util.safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
            # clean_up_dead_vars(eqn, env, lu)
        retvals = jax.util.safe_map(read, jaxpr.outvars)
        retval = retvals if len(jaxpr.outvars) > 1 else retvals[0]

        return self.construct_retval(retval)

    def construct_retval(self, retval) -> R:
        return retval


class Simulate(Transformation[dict]):
    def __init__(self, key: PRNGKeyArray):
        self.key = key
        self.trace = {}

    def address_from_branch(self, b: jax.core.ClosedJaxpr):
        """Look at the given JAXPR and find out if it is a single-instruction
        call to a GF traced to an address. If so, return that address. This is
        used to detect when certain JAX primitives (e.g., `scan_p`, `cond_p`)
        have been applied directly to traced generative functions, in which case
        the current transformation should be propagated to the jaxpr within."""
        if len(b.jaxpr.eqns) == 1 and isinstance(b.jaxpr.eqns[0].primitive, GF):
            return b.jaxpr.eqns[0].params.get("at")

    def transform_inner(self, jaxpr, sub_key, arg_tuple):
        """Apply simulate to jaxpr and return the transformed jaxpr together
        with its return shape."""
        # TODO: this function feels sort of redundant here. Maybe there's a
        # way to move it up a level.
        return jax.make_jaxpr(
            lambda key, arg_tuple: Simulate(key).run(jaxpr, arg_tuple),
            return_shape=True,
        )(sub_key, arg_tuple)

    def handle_eqn(self, eqn, params, bind_params):
        if isinstance(eqn.primitive, GenPrimitive):
            self.key, sub_key = KeySplit.bind(self.key)
            ans = eqn.primitive.simulate_p(sub_key, params)
            self.trace[bind_params["at"]] = ans
            return ans["retval"]

        if eqn.primitive is jax.lax.cond_p:
            self.key, sub_key = KeySplit.bind(self.key)
            branches = bind_params["branches"]

            transformed = list(
                map(
                    lambda branch: self.transform_inner(branch, sub_key, params[1:]),
                    branches,
                )
            )
            transformed_branches = tuple(t[0] for t in transformed)
            shapes = [s[1] for s in transformed]
            new_bind_params = bind_params | {"branches": transformed_branches}
            ans = eqn.primitive.bind(params[0], sub_key, *params[1:], **new_bind_params)

            branch_addresses = tuple(map(self.address_from_branch, branches))
            if branch_addresses[0] and all(
                b == branch_addresses[0] for b in branch_addresses[1:]
            ):
                address = branch_addresses[0]
                u = jax.tree.unflatten(jax.tree.structure(shapes[0]), ans)
                # flatten out an extra layer of subtraces
                self.trace[address] = u["subtraces"][address]
                ans = [u["retval"]]
            return ans

        if eqn.primitive is jax.lax.scan_p:
            self.key, sub_key = KeySplit.bind(self.key)
            inner = bind_params["jaxpr"]
            # at this point params contains (init, xs). We want to simify with
            # (carry, x) i.e. (init, xs[0])
            scan_params = [params[0], params[1][0]]
            transformed_inner, shape = self.transform_inner(inner, sub_key, scan_params)
            inner_jaxpr = transformed_inner.jaxpr
            # simmed_inner is nice but in the scan context we need to return the
            # new root key among the return values
            transformed_inner = transformed_inner.replace(
                jaxpr=inner_jaxpr.replace(
                    outvars=inner_jaxpr.eqns[0].outvars[:1] + inner_jaxpr.outvars,
                    debug_info=None,
                )
            )
            new_bind_params = bind_params | {
                "jaxpr": transformed_inner,
                "linear": (False,) + bind_params["linear"],
                "num_carry": bind_params["num_carry"] + 1,
            }
            ans = eqn.primitive.bind(sub_key, *params, **new_bind_params)
            if address := self.address_from_branch(inner):
                # drop returned key from the unflattening part
                u = jax.tree.unflatten(jax.tree.structure(shape), ans[1:])
                self.trace[address] = u["subtraces"][address]
                ans = u["retval"]

            return ans

        return super().handle_eqn(eqn, params, bind_params)

    def construct_retval(self, retval):
        return {
            "retval": retval,
            "subtraces": self.trace,
        }


# %%
def Cond(tf, ff):
    """Cond combinator. Turns (tf, ff) into a function of a boolean
    argument which will switch between the true and false branches."""

    def ctor(pred):
        class Binder:
            def __matmul__(self, address: str):
                return jax.lax.cond(
                    jnp.int32(pred), lambda: tf @ address, lambda: ff @ address
                )

            def simulate(self, key: PRNGKeyArray):
                return GF(lambda: self @ "__cond", ()).simulate(key)

        return Binder()

    return ctor


def Scan(gf: Gen):
    """Scan combinator. Turns a GF of two parameters `(state, update)`
    returning a pair of updated state and step data to record into
    a generative function of an initial state and an array of updates."""

    def ctor(init, steps):
        class Binder:
            def __matmul__(self, address: str):
                def inner(carry, step):
                    c, s = gf(carry, step) @ address
                    return c, s

                return jax.lax.scan(inner, init, steps)

        return Binder()

    return ctor


InAxesT = int | Sequence[Any] | None


def Vmap(g: Gen, in_axes: InAxesT = 0):
    """Note: this Vmap, while it looks like it might sort-of work, is not
    doing the right thing. Randomness is not propagated along with the
    batch, for one thing. Idea for now is to work at GFI time, so
    that we store the information we need, and then adjust simulate
    be a batched simulate."""

    def ctor(*args):
        class Binder:
            def __matmul__(self, address: str):
                return jax.vmap(lambda *args: g(*args) @ address, in_axes=in_axes)(
                    *args
                )

            def simulate(self, key: PRNGKeyArray):
                @Gen
                def inner():
                    return self @ "__vmap"

                return inner().simulate(key)

        return Binder()

    return ctor


# %%

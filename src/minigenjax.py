# %%
import functools
from typing import Any, Callable, Sequence
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.tree
import jax.numpy as jnp
from jax.interpreters import batching, mlir
import jax.extend as jx
import jax.core
from jaxtyping import Array, PRNGKeyArray


# %%
class GenPrimitive(jx.core.Primitive):
    def __init__(self, name):
        super().__init__(name)
        self.def_abstract_eval(self.abstract)
        self.def_impl(self.concrete)
        mlir.register_lowering(self, mlir.lower_fun(self.impl, self.multiple_results))
        batching.primitive_batchers[self] = self.batch

    def abstract(self, *args, **kwargs):
        raise NotImplementedError(f"abstract: {self}")

    def concrete(self, *args, **kwargs):
        raise NotImplementedError(f"concrete: {self}")

    def batch(self, vector_args, batch_axes, **kwargs):
        # TODO assert all axes equal
        result = jax.vmap(lambda *args: self.impl(*args, **kwargs), in_axes=batch_axes)(
            *vector_args
        )
        batched_axes = (
            (batch_axes[0],) * len(result) if self.multiple_results else batch_axes[0]
        )
        return result, batched_axes

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        raise NotImplementedError(f"simulate_p: {self}")

    # TODO: type of constraint?
    def assess_p(
        self, arg_tuple: tuple, constraint, base_address: tuple[str, ...]
    ) -> tuple[Array, Array]:
        raise NotImplementedError(f"assess_p: {self}")


InAxesT = int | Sequence[Any] | None


class GFI[R](GenPrimitive):
    def __init__(self, name):
        super().__init__(name)

    def simulate(self, key: PRNGKeyArray) -> dict:
        return self.simulate_p(key, self.get_args())

    def assess(self, constraint) -> Array:
        return self.assess_p(self.get_args(), constraint, ())[1]

    def __matmul__(self, address: str) -> R:
        raise NotImplementedError(f"{self} @ {address}")

    def get_args(self) -> tuple:
        raise NotImplementedError(f"get_args: {self}")

    def map[S](self, f: Callable[[R], S]) -> "MapGF[R,S]":
        return MapGF(self, f)

    def repeat(self, n: int) -> "RepeatGF[R]":
        return RepeatGF(self, n)

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        raise NotImplementedError(f"get_jaxpr: {self}")


class Distribution(GenPrimitive):
    PHANTOM_KEY = jax.random.key(987654321)

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

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple):
        v = self.bind(key, *arg_tuple[1:], op="Sample")
        return {"retval": v, "score": self.bind(v, *arg_tuple[1:], op="Score")}

    def __call__(self, *args):
        this = self

        class Binder:
            def __matmul__(self, address: str):
                return this.bind(this.PHANTOM_KEY, *args, at=address)

            def __call__(self, key: PRNGKeyArray):
                return this.tfd_ctor(*args).sample(seed=key)

        return Binder()


Normal = Distribution("Normal", tfp.distributions.Normal)
MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution("Bernoulli", lambda p: tfp.distributions.Bernoulli(probs=p))
Categorical = Distribution(
    "Categorical", lambda ls: tfp.distributions.Categorical(logits=ls)
)


class KeySplitP(jx.core.Primitive):
    KEY_TYPE = jax.core.ShapedArray((2,), jnp.uint32)

    def __init__(self):
        super().__init__("KeySplit")

        def impl(k, a=None):
            r = jax.random.split(k, 2)
            return r[0], r[1]

        self.def_impl(impl)
        self.multiple_results = True
        self.def_abstract_eval(lambda _, a=None: [self.KEY_TYPE, self.KEY_TYPE])

        mlir.register_lowering(self, mlir.lower_fun(self.impl, self.multiple_results))

        batching.primitive_batchers[self] = self.batch

    def batch(self, vector_args, batch_axes, a=None):
        # key_pair_vector = jax.vmap(self.impl, in_axes=batch_axes)(*vector_args)
        v0, v1 = jax.vmap(self.impl, in_axes=batch_axes)(*vector_args)
        return [v0, v1], (batch_axes[0], batch_axes[0])


KeySplit = KeySplitP()
GenSymT = Callable[[jax.core.AbstractValue], jx.core.Var]


# %%
class Gen[R]:
    def __init__(self, f: Callable[..., R]):
        self.f = f

    def __call__(self, *args) -> "GF[R]":
        return GF(self.f, args)

    def vmap(self, in_axes: InAxesT = 0) -> Callable[..., "VmapGF[R]"]:
        return lambda *args: VmapGF(self, args, in_axes)


# TODO: we need to separate the GFI From GF here.
# We need a way for combinators to represent the GFI without
# necessarily holding a bare python function that computes it.


class GF[R](GFI[R]):
    def __init__(self, f: Callable[..., R], args: tuple):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f
        self.args = args
        self.jaxpr, self.shape = jax.make_jaxpr(f, return_shape=True)(*args)
        self.multiple_results = isinstance(self.shape, tuple)
        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        self.abstract_value = a_vals if self.multiple_results else a_vals[0]

    def abstract(self, *args, at: str, **_kwargs):
        return self.abstract_value

    def concrete(self, *args):
        v = jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *args)
        return v if self.multiple_results else v[0]

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        return Simulate(key).run(self.jaxpr, arg_tuple)

    def assess_p(
        self, arg_tuple: tuple, constraint, base_address
    ) -> tuple[Array, Array]:
        a = Assess(constraint, base_address)
        value = a.run(self.jaxpr, arg_tuple)
        return value, a.score

    def __matmul__(self, address: str):
        return self.bind(*self.args, at=address)

    def get_args(self) -> tuple:
        return self.args

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr


class Transformation[R]:
    def __init__(self):
        pass

    def handle_eqn(self, eqn, params, bind_params):
        return eqn.primitive.bind(*params, **bind_params)

    def run(self, closed_jaxpr: jx.core.ClosedJaxpr, arg_tuple):
        jaxpr = closed_jaxpr.jaxpr
        env: dict[jx.core.Var, Any] = {}

        def read(v: jax.core.Atom) -> Any:
            return v.val if isinstance(v, jx.core.Literal) else env[v]

        def write(v: jx.core.Var, val: Any) -> None:
            # if config.enable_checks.value and not config.dynamic_shapes.value:
            #   assert typecheck(v.aval, val), (v.aval, val)
            env[v] = val

        jax.util.safe_map(write, jaxpr.constvars, closed_jaxpr.consts)
        jax.util.safe_map(write, jaxpr.invars, arg_tuple)

        for eqn in jaxpr.eqns:
            sub_fns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            if sub_fns:
                raise NotImplementedError("nonempty sub_fns")
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
        retval = jax.util.safe_map(read, jaxpr.outvars)
        retval = retval if len(jaxpr.outvars) > 1 else retval[0]
        return self.construct_retval(retval)

    def construct_retval(self, retval) -> R:
        return retval


class Assess(Transformation[Array]):
    def __init__(
        self, constraint: dict[tuple[str, ...], Array], address: tuple[str, ...]
    ):
        self.constraint = constraint
        self.address = address
        self.score = jnp.array(0.0)

    def apply_constraint(self, addr):
        if isinstance(self.constraint, dict):
            return functools.reduce(lambda d, a: d[a], addr, self.constraint)
        else:
            return self.constraint(addr)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        addr = self.address + (bind_params["at"],)
        if isinstance(eqn.primitive, Distribution):
            v = self.apply_constraint(addr)
            self.score += eqn.primitive.bind(v, *params[1:], op="Score")
            return v

        if isinstance(eqn.primitive, GenPrimitive):
            ans, score = eqn.primitive.assess_p(params, self.constraint, addr)
            self.score += score
            return ans

        return super().handle_eqn(eqn, params, bind_params)

    def construct_retval(self, retval) -> Array:
        return self.score


class Simulate(Transformation[dict]):
    S_KEY = jax.random.PRNGKey(99999)

    def __init__(self, key: PRNGKeyArray):
        self.key = key
        self.trace = {}

    def address_from_branch(self, b: jx.core.ClosedJaxpr):
        """Look at the given JAXPR and find out if it is a single-instruction
        call to a GF traced to an address. If so, return that address. This is
        used to detect when certain JAX primitives (e.g., `scan_p`, `cond_p`)
        have been applied directly to traced generative functions, in which case
        the current transformation should be propagated to the jaxpr within."""
        if len(b.jaxpr.eqns) == 1 and isinstance(b.jaxpr.eqns[0].primitive, GF):
            return b.jaxpr.eqns[0].params.get("at")

    def transform_inner(self, jaxpr, in_avals):
        """Apply simulate to jaxpr and return the transformed jaxpr together
        with its return shape."""

        return jax.make_jaxpr(
            lambda key, in_avals: Simulate(key).run(jaxpr, in_avals),
            return_shape=True,
        )(self.S_KEY, in_avals)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        # TODO: This case (repeat) and the following (vmap) are quite similar.
        # We ought to do something to unify them so they can share code. Note
        # that the scan combinator doesn't use the primitive technique,
        if isinstance(eqn.primitive, RepeatGF):
            self.key, sub_key = KeySplit.bind(self.key, a="repeat")
            transformed, shape = self.transform_inner(bind_params["inner"], params)
            new_params = bind_params | {"inner": transformed}
            ans = eqn.primitive.Simulate(eqn.primitive, shape).bind(
                sub_key, *params, **new_params
            )
            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            self.trace[bind_params["at"]] = u["subtraces"]
            return u["retval"]

        if isinstance(eqn.primitive, VmapGF):
            self.key, sub_key = KeySplit.bind(self.key, a="vmap")
            transformed, shape = self.transform_inner(
                bind_params["inner"], eqn.primitive.reduced_avals(params)
            )
            new_params = bind_params | {"inner": transformed}
            ans = eqn.primitive.Simulate(eqn.primitive, shape).bind(
                sub_key, *params, **new_params
            )
            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            self.trace[bind_params["at"]] = u["subtraces"]
            return u["retval"]

        if isinstance(eqn.primitive, GenPrimitive):
            self.key, sub_key = KeySplit.bind(self.key, a="gen_p")
            ans = eqn.primitive.simulate_p(sub_key, params)
            self.trace[bind_params["at"]] = ans
            return ans["retval"]

        if eqn.primitive is jax.lax.cond_p:
            self.key, sub_key = KeySplit.bind(self.key, a="cond_p")
            branches = bind_params["branches"]
            avals = [jax.core.get_aval(p) for p in params[1:]]

            transformed = list(
                map(
                    lambda branch: self.transform_inner(branch, avals),
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
            else:
                raise Exception(f"missing address in {eqn}")
            return ans

        if eqn.primitive is jax.lax.scan_p:
            self.key, sub_key = KeySplit.bind(self.key, a="scan_p")
            inner = bind_params["jaxpr"]
            # at this point params contains (init, xs). We want to simulate with
            # (carry, x) i.e. (init, xs[0])
            xs_aval = eqn.invars[1].aval
            assert isinstance(xs_aval, jax.core.ShapedArray)
            x_aval = xs_aval.update(shape=xs_aval.shape[1:])
            scan_avals = [eqn.invars[0].aval, x_aval]
            transformed_inner, shape = self.transform_inner(inner, scan_avals)
            inner_jaxpr = transformed_inner.jaxpr
            # transformed_inner is nice but in the scan context we need to return the
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
                # drop returned key from what we unflatten
                u = jax.tree.unflatten(jax.tree.structure(shape), ans[1:])
                self.trace[address] = u["subtraces"][address]
                ans = u["retval"]
            else:
                raise Exception(f"missing address in {eqn}")

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
        pred_as_int = jnp.int32(pred)

        class Binder:
            def __matmul__(self, address: str):
                return jax.lax.switch(
                    pred_as_int, [lambda: ff @ address, lambda: tf @ address]
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


class RepeatGF[R](GFI[R]):
    def __init__(self, gfi: GFI[R], n: int):
        super().__init__(f"Repeat[{gfi.name}, {n}]")
        self.gfi = gfi
        self.n = n
        self.multiple_results = self.gfi.multiple_results

    def abstract(self, *args, **kwargs):
        a = self.gfi.abstract(*args, **kwargs)
        # just because we aren't sure how to deal with structure yet
        assert isinstance(a, jax.core.ShapedArray)
        return jax.core.ShapedArray((self.n,) + a.shape, a.dtype)

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        return GF(lambda: self @ "__repeat", ()).simulate(key)

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.gfi.get_args(), at=address, n=self.n, inner=self.gfi.get_jaxpr()
        )

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    class Simulate[S](GFI[S]):
        def __init__(self, r: "RepeatGF[S]", shape: Any):
            super().__init__("Repeat.Simulate")
            self.r = r
            self.shape = shape
            self.multiple_results = True

        def abstract(self, *args, **kwargs):
            return [jax.core.get_aval(s) for s in jax.tree.flatten(self.shape)[0]]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            return jax.vmap(
                lambda k: jax.core.eval_jaxpr(j.jaxpr, j.consts, k, *args[1:]),
                in_axes=(0,),
            )(jax.random.split(args[0], kwargs["n"]))


class VmapGF[R](GFI[R]):
    def __init__(self, g: Gen, arg_tuple: tuple, in_axes: InAxesT):
        super().__init__(f"Vmap[{g.f.__name__}]")
        if in_axes is None or in_axes == ():
            raise NotImplementedError(
                "must specify at least one argument/axis for Vmap"
            )
        self.arg_tuple = arg_tuple
        self.in_axes = in_axes
        # find one pair of (parameter number, axis) to use to determine size of vmap
        if isinstance(self.in_axes, tuple):
            self.p_index, self.an_axis = next(
                filter(lambda b: b[1] is not None, enumerate(self.in_axes))
            )
        else:
            self.p_index, self.an_axis = 0, self.in_axes
        # Compute the "scalar" jaxpr by feeding the un-v-mapped arguments to make_jaxpr
        self.jaxpr, self.shape = jax.make_jaxpr(g.f, return_shape=True)(
            *self.reduced_avals(arg_tuple)
        )

    def reduced_avals(self, arg_tuple):
        # if in_axes is not an tuple, lift it to the same shape as arg tuple
        if isinstance(self.in_axes, int):
            ia = jax.tree.map(lambda _: self.in_axes, arg_tuple)
        else:
            ia = self.in_axes

        # Now produce an abstract arg tuple in which the shape of the
        # arrays is contracted in those position where vmap would expect
        # to find a mapping axis
        def deflate(array, axis):
            if axis is None:
                return array
            aval = jax.core.get_aval(array)
            # delete the indicated axes from teh shape tuple
            assert isinstance(aval, jax.core.ShapedArray)
            return aval.update(shape=aval.shape[:axis] + aval.shape[axis + 1 :])

        return jax.tree.map(deflate, arg_tuple, ia)

    def abstract(self, *args, **kwargs):
        return self.shape

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        return GF(lambda: self @ "__vmap", ()).simulate(key)

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.arg_tuple, at=address, in_axes=self.in_axes, inner=self.jaxpr
        )

    def get_args(self) -> tuple:
        return self.arg_tuple

    class Simulate[S](GFI[S]):
        def __init__(self, r: "VmapGF[S]", shape: Any):  # TODO: PyTreeDef?
            super().__init__("Vmap.Simulate")
            self.r = r
            self.shape = shape
            self.multiple_results = True

        def abstract(self, *args, **kwargs):
            return [jax.core.get_aval(s) for s in jax.tree.flatten(self.shape)[0]]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            n = self.r.arg_tuple[self.r.p_index].shape[self.r.an_axis]
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            return jax.vmap(
                lambda k, arg_tuple: jax.core.eval_jaxpr(
                    j.jaxpr, j.consts, k, *arg_tuple
                ),
                in_axes=(0, self.r.in_axes),
            )(jax.random.split(args[0], n), args[1:])


class MapGF[R, S](GFI[S]):
    def __init__(self, gfi: GFI[R], f: Callable[[R], S]):
        super().__init__(f"Map[{gfi.name}, {f.__name__}]")
        self.gfi = gfi
        self.f = f

    def abstract(self, *args, **kwargs):
        # this can't be right: what about the effect of f? Why isn't it
        # sufficient to apply f to a tracer to compose the abstraction?
        return self.gfi.abstract(*args, **kwargs)

    def simulate_p(self, key: PRNGKeyArray, arg_tuple: tuple) -> dict:
        v = self.gfi.simulate_p(key, arg_tuple)
        v["retval"] = self.f(v["retval"])
        return v

    def __matmul__(self, address: str) -> S:
        return self.f(self.gfi @ address)

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        ij = self.gfi.get_jaxpr()
        return jax.make_jaxpr(
            lambda args: self.f(*jax.core.eval_jaxpr(ij.jaxpr, ij.consts, *args))
        )(self.gfi.get_args())


# %%

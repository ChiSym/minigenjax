# %%
from typing import Any, Callable, Sequence
from jax._src.core import ClosedJaxpr as ClosedJaxpr
import jax
import jax.tree
import jax.api_util
import jax.numpy as jnp
from jax.interpreters import batching, mlir

import jax.extend as jx
import jax.core
from jaxtyping import Array, ArrayLike, PRNGKeyArray, Float


# %%
Address = tuple[str, ...]
Constraint = dict[str, "ArrayLike|Constraint"]
PHANTOM_KEY = jax.random.key(987654321)

WrappedFunWithAux = tuple[jx.linear_util.WrappedFun, Callable[[], Any]]

# Wrapper to assign a correct type.
flatten_fun_nokwargs: Callable[[jx.linear_util.WrappedFun, Any], WrappedFunWithAux] = (
    jax.api_util.flatten_fun_nokwargs  # pyright: ignore[reportAssignmentType]
)


class GenPrimitive(jx.core.Primitive):
    def __init__(self, name):
        super().__init__(name)
        self.def_abstract_eval(self.abstract)
        self.def_impl(self.concrete)
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

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        raise NotImplementedError(f"simulate_p: {self}")

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint | Float, address: tuple[str, ...]
    ) -> tuple[Array, Array]:
        raise NotImplementedError(f"assess_p: {self}")

    def inflate(self, v: Any, n: int):
        def inflate_one(v):
            return v.update(shape=(n,) + v.shape)

        return jax.tree.map(inflate_one, v)


InAxesT = int | Sequence[Any] | None


class GFI[R](GenPrimitive):
    def __init__(self, name):
        super().__init__(name)

    def simulate(self, key: PRNGKeyArray) -> dict:
        return self.simulate_p(key, self.get_args(), (), {})

    def propose(self, key: PRNGKeyArray) -> tuple[Constraint, Float, R]:
        tr = self.simulate(key)
        return to_constraint(tr), to_score(tr), tr["retval"]

    def importance(
        self, key: PRNGKeyArray, constraint: Constraint
    ) -> tuple[dict, Float]:
        tr = self.simulate_p(key, self.get_args(), (), constraint)
        return tr, to_weight(tr)

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

    def get_structure(self) -> jax.tree_util.PyTreeDef:
        raise NotImplementedError(f"get_structure: {self}")

    @staticmethod
    def make_jaxpr(f, arg_tuple):
        flat_args, in_tree = jax.tree.flatten(arg_tuple)
        flat_f, out_tree = flatten_fun_nokwargs(jx.linear_util.wrap_init(f), in_tree)
        # TODO: consider whether we need shape here
        jaxpr, shape = jax.make_jaxpr(flat_f.call_wrapped, return_shape=True)(
            *flat_args
        )
        structure = out_tree()
        return jaxpr, flat_args, structure


class KeySplitP(jx.core.Primitive):
    KEY_TYPE = jax.core.get_aval(PHANTOM_KEY)  # jax.core.ShapedArray((2,), jnp.uint32)

    def __init__(self):
        super().__init__("KeySplit")

        def impl(k, a=None):
            r = jax.random.split(k, 2)
            return r[0], r[1]

        self.def_impl(impl)
        self.multiple_results = True
        self.def_abstract_eval(
            lambda _, a=None: [KeySplitP.KEY_TYPE, KeySplitP.KEY_TYPE]
        )

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


class GF[R](GFI[R]):
    def __init__(self, f: Callable[..., R], args: tuple):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f

        self.args = args
        self.jaxpr, self.flat_args, self.structure = self.make_jaxpr(self.f, self.args)
        self.multiple_results = self.structure.num_leaves > 1

        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        self.abstract_value = a_vals if self.multiple_results else a_vals[0]

    def abstract(self, *args, **_kwargs):
        return self.abstract_value

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return Simulate(key, address, constraint).run(
            self.jaxpr, arg_tuple, self.structure
        )

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Array, Array]:
        a = Assess(address, constraint)
        value = a.run(self.jaxpr, arg_tuple)
        return value, a.score

    def __matmul__(self, address: str):
        return self.bind(*self.flat_args, at=address)

    def get_args(self) -> tuple:
        return self.args

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    def get_structure(self) -> jax.tree_util.PyTreeDef:
        return self.structure


class Transformation[R]:
    def __init__(self, key: PRNGKeyArray, address: Address, constraint: Constraint):
        self.key = key
        self.address = address
        self.constraint = constraint

    def handle_eqn(self, eqn, params, bind_params):
        return eqn.primitive.bind(*params, **bind_params)

    def get_sub_key(self):
        if self.key_consumer_count > 1:
            self.key, sub_key = KeySplit.bind(self.key)
        elif self.key_consumer_count == 1:
            sub_key = self.key
        else:
            raise Exception("more sub_key requests than expected")
        self.key_consumer_count -= 1
        return sub_key

    def run(
        self,
        closed_jaxpr: jx.core.ClosedJaxpr,
        arg_tuple,
        structure: jax.tree_util.PyTreeDef | None = None,
    ):
        jaxpr = closed_jaxpr.jaxpr
        flat_args, in_tree = jax.tree.flatten(arg_tuple)
        env: dict[jx.core.Var, Any] = {}

        def read(v: jax.core.Atom) -> Any:
            return v.val if isinstance(v, jx.core.Literal) else env[v]

        def write(v: jx.core.Var, val: Any) -> None:
            # if config.enable_checks.value and not config.dynamic_shapes.value:
            #   assert typecheck(v.aval, val), (v.aval, val)
            env[v] = val

        jax.util.safe_map(write, jaxpr.constvars, closed_jaxpr.consts)
        jax.util.safe_map(write, jaxpr.invars, flat_args)

        # count the number of PRNG keys that will be consumed during the
        # evaluation of this JAXPR alone. We assume that the key that we
        # were provided with is good for one random number generation. If
        # there's only one key consumer in this JAXPR, then there's no need
        # to split it.
        self.key_consumer_count = sum(
            isinstance(eqn.primitive, GenPrimitive)
            or eqn.primitive is jax.lax.cond_p
            or eqn.primitive is jax.lax.scan_p
            for eqn in jaxpr.eqns
        )

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
            # the return value is currently not flat. (is that the right choice?)
            if eqn.primitive.multiple_results:
                jax.util.safe_map(write, eqn.outvars, jax.tree.flatten(ans)[0])
            else:
                write(eqn.outvars[0], ans)
            # clean_up_dead_vars(eqn, env, lu)
        retval = jax.util.safe_map(read, jaxpr.outvars)
        if structure is not None:
            retval = jax.tree.unflatten(structure, retval)
        else:
            retval = retval if len(jaxpr.outvars) > 1 else retval[0]
        return self.construct_retval(retval)

    def address_from_branch(self, b: jx.core.ClosedJaxpr):
        """Look at the given JAXPR and find out if it is a single-instruction
        call to a GF traced to an address. If so, return that address. This is
        used to detect when certain JAX primitives (e.g., `scan_p`, `cond_p`)
        have been applied directly to traced generative functions, in which case
        the current transformation should be propagated to the jaxpr within."""
        if len(b.jaxpr.eqns) == 1 and isinstance(b.jaxpr.eqns[0].primitive, GF):
            return b.jaxpr.eqns[0].params.get("at")

    def apply_constraint(self, addr: str) -> ArrayLike | None:
        if isinstance(self.constraint, dict):
            v = self.constraint.get(addr)
            if isinstance(v, dict):
                raise Exception(
                    "broken constraint: {addr} in {self.address} is not a leaf"
                )
            return v
        else:
            return self.constraint

    def get_sub_constraint(self, a: str) -> Constraint | Float:
        if self.constraint is None:
            return None
        return self.constraint.get(a)

    def construct_retval(self, retval) -> R:
        return retval


class Assess(Transformation[Array]):
    def __init__(self, address: Address, constraint: Constraint):
        super().__init__(PHANTOM_KEY, address, constraint)
        self.score = jnp.array(0.0)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            addr = self.address + (at,)
            ans, score = eqn.primitive.assess_p(
                params, self.get_sub_constraint(at), addr
            )
            self.score += score
            return ans

        return super().handle_eqn(eqn, params, bind_params)


class Simulate(Transformation[dict]):
    def __init__(
        self,
        key: PRNGKeyArray,
        address: Address,
        constraint: Constraint,
    ):
        super().__init__(key, address, constraint)
        self.trace = {}
        self.w = jnp.array(0.0)

    def transform_inner(
        self, jaxpr, in_avals, addr: str, constraint: Constraint | None = None
    ):
        """Apply simulate to jaxpr and return the transformed jaxpr together
        with its return shape."""

        if constraint:
            reduced_constraint = jax.tree.map(lambda v: v[0], constraint)
            return jax.make_jaxpr(
                lambda key, co, in_avals: Simulate(key, self.address + (addr,), co).run(
                    jaxpr, in_avals
                ),
                return_shape=True,
            )(PHANTOM_KEY, reduced_constraint, in_avals)
        else:
            return jax.make_jaxpr(
                lambda key, in_avals: Simulate(key, self.address + (addr,), {}).run(
                    jaxpr, in_avals
                ),
                return_shape=True,
            )(PHANTOM_KEY, in_avals)

    def record(self, subtrace, at):
        if (
            (inner_trace := subtrace.get("subtraces"))
            and len(keys := inner_trace.keys()) == 1
            and (key := next(iter(keys))).startswith("__")
        ):
            # absorb interstitial trace points like __repeat, __vmap that may
            # occur when combinators are stacked.
            subtrace["subtraces"] = inner_trace[key]["subtraces"]
        if at:
            self.trace[at] = subtrace
        self.w += jnp.sum(subtrace.get("w", 0.0))
        return subtrace["retval"]

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, RepeatGF):
            at = bind_params["at"]
            sub_constraint = self.get_sub_constraint(at)
            transformed, shape = self.transform_inner(
                bind_params["inner"], params, at, sub_constraint
            )
            new_params = bind_params | {"inner": transformed}
            has_constraint = bool(sub_constraint)
            if has_constraint:
                flat_constraint, _ = jax.tree.flatten(sub_constraint)
                ans = eqn.primitive.Simulate(eqn.primitive, shape, has_constraint).bind(
                    self.get_sub_key(), *flat_constraint, *params, **new_params
                )
            else:
                ans = eqn.primitive.Simulate(eqn.primitive, shape, has_constraint).bind(
                    self.get_sub_key(), *params, **new_params
                )

            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            return self.record(u, at)

        if isinstance(eqn.primitive, VmapGF):
            at = bind_params["at"]
            sub_constraint = self.get_sub_constraint(at)
            transformed, shape = self.transform_inner(
                bind_params["inner"],
                eqn.primitive.reduced_avals,
                at,
                sub_constraint,
            )
            new_params = bind_params | {"inner": transformed}
            has_constraint = bool(sub_constraint)
            # TODO: the below is sus. Aren't the params already flat here?
            flat_params, _ = jax.tree.flatten(params)
            if has_constraint:
                flat_constraint, _ = jax.tree.flatten(sub_constraint)
                ans = eqn.primitive.Simulate(eqn.primitive, shape, has_constraint).bind(
                    self.get_sub_key(), *flat_constraint, *flat_params, **new_params
                )
            else:
                ans = eqn.primitive.Simulate(eqn.primitive, shape, has_constraint).bind(
                    self.get_sub_key(), *flat_params, **new_params
                )
            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            return self.record(u, at)

        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            addr = self.address + (at,)
            ans = eqn.primitive.simulate_p(
                self.get_sub_key(), params, addr, self.get_sub_constraint(at)
            )
            return self.record(ans, at)

        if eqn.primitive is jax.lax.cond_p:
            branches = bind_params["branches"]

            branch_addresses = tuple(map(self.address_from_branch, branches))
            if branch_addresses[0] and all(
                b == branch_addresses[0] for b in branch_addresses[1:]
            ):
                sub_address = branch_addresses[0]
            else:
                sub_address = None

            # TODO: is it OK to pass the same sub_key to both sides?
            # NB! branches[0] is the false branch, [1] is the true branch,
            sub_key = self.get_sub_key()
            ans = jax.lax.cond(
                params[0],
                lambda: Simulate(sub_key, self.address, self.constraint).run(
                    branches[1], params[1:]
                ),
                lambda: Simulate(sub_key, self.address, self.constraint).run(
                    branches[0], params[1:]
                ),
            )
            if sub_address:
                self.trace[sub_address] = ans["subtraces"][sub_address]

            self.w += jnp.sum(ans.get("w", 0))
            # The reasons why this result has to be a sequence is obscure to me,
            # but cond_p as a primitive requires "multiple results."
            return (ans["retval"],)

        if eqn.primitive is jax.lax.scan_p:
            inner = bind_params["jaxpr"]

            if at := self.address_from_branch(inner):
                address = self.address + (at,)
                sub_address = at
            else:
                address = self.address
                sub_address = None

            def step(carry_key, s):
                carry, key = carry_key
                key, k1 = KeySplit.bind(key, a="scan_step")
                v = Simulate(k1, address, self.constraint).run(inner, (carry, s))
                return ((v["retval"][0], key), v)

            ans = jax.lax.scan(step, (params[0], self.get_sub_key()), params[1])
            if sub_address:
                self.trace[sub_address] = ans[1]["subtraces"][sub_address]

            self.w += jnp.sum(ans[1].get("w", 0))
            # we extended the carry with the key; now drop it
            return (ans[0][0], ans[1]["retval"][1])

        return super().handle_eqn(eqn, params, bind_params)

    def construct_retval(self, retval):
        r = {"retval": retval, "subtraces": self.trace}
        if self.constraint:
            r["w"] = self.w
        return r


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
    SUB_TRACE = "__repeat"

    def __init__(self, gfi: GFI[R], n: int):
        super().__init__(f"Repeat[{gfi.name}, {n}]")
        self.gfi = gfi
        self.n = n
        self.multiple_results = self.gfi.multiple_results
        # TODO: try to reuse self.__matmul__ here, if possible
        self.jaxpr, self.flat_args, self.structure = self.make_jaxpr(
            lambda *args: self.bind(
                *args, at=RepeatGF.SUB_TRACE, n=self.n, inner=self.gfi.get_jaxpr()
            ),
            self.get_args(),
        )

    def abstract(self, *args, **kwargs):
        return self.inflate(self.gfi.abstract(*args, **kwargs), self.n)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        tr = Simulate(key, address, {RepeatGF.SUB_TRACE: constraint}).run(
            self.jaxpr, arg_tuple, self.structure
        )["subtraces"][RepeatGF.SUB_TRACE]
        # if "w" in tr:
        #     tr["w"] = jnp.sum(tr["w"])
        return tr

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.gfi.get_args(), at=address, n=self.n, inner=self.gfi.get_jaxpr()
        )

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    def get_structure(self):
        return self.structure

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    class Simulate[S](GFI[S]):
        def __init__(self, r: "RepeatGF[S]", shape: Any, has_constraint: bool):
            super().__init__("Repeat.Simulate")
            self.r = r
            self.shape = shape
            self.has_constraint = has_constraint
            self.multiple_results = True
            mlir.register_lowering(
                self, mlir.lower_fun(self.impl, self.multiple_results)
            )

        def abstract(self, *args, **kwargs):
            return [
                self.inflate(jax.core.get_aval(s), self.r.n)
                for s in jax.tree.flatten(self.shape)[0]
            ]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            if self.has_constraint:
                # TODO: there could be several constraint arguments; have a param count how many
                return jax.vmap(
                    lambda k, constraint, arg_tuple: jax.core.eval_jaxpr(
                        j.jaxpr, j.consts, k, constraint, *arg_tuple
                    ),
                    in_axes=(0, 0, None),
                )(jax.random.split(args[0], kwargs["n"]), args[1], args[2:])
            return jax.vmap(
                lambda k: jax.core.eval_jaxpr(j.jaxpr, j.consts, k, *args[1:]),
                in_axes=(0,),
            )(jax.random.split(args[0], kwargs["n"]))


class VmapGF[R](GFI[R]):
    SUB_TRACE = "__vmap"

    def __init__(self, g: Gen, arg_tuple: tuple, in_axes: InAxesT):
        super().__init__(f"Vmap[{g.f.__name__}]")
        if in_axes is None or in_axes == ():
            raise NotImplementedError(
                "must specify at least one argument/axis for Vmap"
            )
        # TODO: consider if we want to make this
        self.arg_tuple = arg_tuple
        self.flat_args, self.in_tree = jax.tree.flatten(self.arg_tuple)
        self.flat_args = tuple(self.flat_args)
        self.in_axes = in_axes
        # find one pair of (parameter number, axis) to use to determine size of vmap
        if isinstance(self.in_axes, tuple):
            self.p_index, self.an_axis = next(
                filter(lambda b: b[1] is not None, enumerate(self.in_axes))
            )
        else:
            self.p_index, self.an_axis = 0, self.in_axes
        # Compute the "scalar" jaxpr by feeding the un-v-mapped arguments to make_jaxpr
        # TODO: does `self` need to remember these?
        self.reduced_avals = self.get_reduced_avals(self.arg_tuple)
        self.inner_jaxpr, self.inner_shape = jax.make_jaxpr(g.f, return_shape=True)(
            *self.reduced_avals
        )
        n = self.arg_tuple[self.p_index].shape[self.an_axis]
        # the shape of the v-mapped function will increase every axis of the result
        self.shape = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct((n,) + s.shape, s.dtype), self.inner_shape
        )
        self.multiple_results = isinstance(self.shape, tuple)
        self.jaxpr, self.shape = jax.make_jaxpr(
            lambda *args: self.bind(
                *args, in_axes=self.in_axes, at=VmapGF.SUB_TRACE, inner=self.inner_jaxpr
            ),
            return_shape=True,
        )(*self.flat_args)

    def get_reduced_avals(self, arg_tuple):
        # Produce an abstract arg tuple in which the shape of the
        # arrays is contracted in those position where vmap would expect
        # to find a mapping axis
        def deflate(axis, aval):
            def deflate_one(axis, aval):
                assert isinstance(aval, jax.core.ShapedArray)
                return aval.update(shape=aval.shape[:axis] + aval.shape[axis + 1 :])

            if axis is None:
                return aval
            if isinstance(aval, tuple):
                return jax.tree.map(lambda a: deflate_one(axis, a), aval)
            return deflate_one(axis, aval)

        aval_tree = jax.tree.map(jax.core.get_aval, arg_tuple)
        return jax.tree.map(
            deflate, self.in_axes, aval_tree, is_leaf=lambda x: x is None
        )

    def abstract(self, *args, **kwargs):
        return self.shape

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        tr = Simulate(key, address, {VmapGF.SUB_TRACE: constraint}).run(
            self.jaxpr, arg_tuple
        )["subtraces"][VmapGF.SUB_TRACE]
        # if "w" in tr:
        #     tr["w"] = jnp.sum(tr["w"])
        return tr

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.flat_args, at=address, in_axes=self.in_axes, inner=self.inner_jaxpr
        )

    def get_args(self) -> tuple:
        return self.arg_tuple

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    class Simulate[S](GFI[S]):
        def __init__(
            self, r: "VmapGF[S]", shape: Any, has_constraint: bool
        ):  # TODO: PyTreeDef?
            super().__init__("Vmap.Simulate")
            self.r = r
            self.n = self.r.arg_tuple[self.r.p_index].shape[self.r.an_axis]
            self.shape = shape
            self.has_constraint = has_constraint
            self.multiple_results = True
            mlir.register_lowering(
                self, mlir.lower_fun(self.impl, self.multiple_results)
            )

        def abstract(self, *args, **kwargs):
            return [
                self.inflate(jax.core.get_aval(s), self.n)
                for s in jax.tree.flatten(self.shape)[0]
            ]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            if self.has_constraint:
                # TODO: there could be several constraint arguments; have a param count how many
                return jax.vmap(
                    lambda k, constraint, arg_tuple: jax.core.eval_jaxpr(
                        j.jaxpr,
                        j.consts,
                        k,
                        constraint,
                        *jax.tree.flatten(arg_tuple)[0],
                    ),
                    in_axes=(0, 0, self.r.in_axes),
                )(
                    jax.random.split(args[0], self.n),
                    args[1],
                    jax.tree.unflatten(self.r.in_tree, args[2:]),
                )

            return jax.vmap(
                lambda k, arg_tuple: jax.core.eval_jaxpr(
                    j.jaxpr, j.consts, k, *jax.tree.flatten(arg_tuple)[0]
                ),
                in_axes=(0, self.r.in_axes),
            )(
                jax.random.split(args[0], self.n),
                jax.tree.unflatten(self.r.in_tree, args[1:]),
            )


class MapGF[R, S](GFI[S]):
    def __init__(self, gfi: GFI[R], f: Callable[[R], S]):
        super().__init__(f"Map[{gfi.name}, {f.__name__}]")
        self.gfi = gfi
        self.f = f

        inner = self.gfi.get_jaxpr()
        self.jaxpr, self.flat_args, self.structure = self.make_jaxpr(
            lambda *args: self.f(
                jax.tree.unflatten(
                    self.gfi.get_structure(),
                    jax.core.eval_jaxpr(inner.jaxpr, inner.consts, *args),
                )
            ),
            self.gfi.get_args(),
        )

    def abstract(self, *args, **kwargs):
        # this can't be right: what about the effect of f? Why isn't it
        # sufficient to apply f to a tracer to compose the abstraction?
        return self.gfi.abstract(*args, **kwargs)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return Simulate(key, address, constraint).run(
            self.jaxpr, arg_tuple, self.structure
        )

    def __matmul__(self, address: str) -> S:
        # TODO (Q): can we declare multiple returns and drop the brackets?
        return jax.tree.unflatten(
            self.structure, [self.bind(*self.flat_args, at=address)]
        )

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr


def to_constraint(trace: dict) -> Constraint:
    if "subtraces" in trace:
        return {k: to_constraint(v) for k, v in trace["subtraces"].items()}
    return trace["retval"]


def trace_sum(trace: dict, key: str) -> Float:
    if "subtraces" in trace:
        return sum(jnp.sum(trace_sum(v, key)) for v in trace["subtraces"].values())
    return jnp.sum(trace.get(key, 0.0))


def to_score(trace: dict) -> Float:
    return trace_sum(trace, "score")


def to_weight(trace: dict) -> Float:
    return trace_sum(trace, "w")

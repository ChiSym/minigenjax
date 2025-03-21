# %%
from typing import Any, Callable, Sequence
from jax._src.core import ClosedJaxpr as ClosedJaxpr
import jax
import jax.tree
import jax.api_util
import jax.numpy as jnp
from jax.interpreters import batching, mlir
from .key import KeySplit
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


class MissingConstraint(Exception):
    pass


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
    ) -> tuple[Array, Any]:
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

    def assess(self, constraint) -> tuple[Array, Array]:
        return self.assess_p(self.get_args(), constraint, ())

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

        self.wf = jx.linear_util.wrap_init(f)
        self.f = f
        self.args = args
        self.jaxpr, self.flat_args, self.structure = self.make_jaxpr(self.f, self.args)
        self.multiple_results = self.structure.num_leaves > 1

        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        if a_vals:
            self.abstract_value = a_vals if self.multiple_results else a_vals[0]
        else:
            self.abstract_value = None

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
    ) -> tuple[Float, R]:
        a = Assess[R](address, constraint)
        retval = a.run(self.jaxpr, arg_tuple, self.structure)
        return a.score, retval

    def __matmul__(self, address: str):
        r = self.bind(*self.flat_args, at=address)
        return jax.tree.unflatten(self.structure, r if self.multiple_results else [r])

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
        if len(b.jaxpr.eqns) == 1 and isinstance(
            b.jaxpr.eqns[0].primitive, GenPrimitive
        ):
            return b.jaxpr.eqns[0].params.get("at")

    def get_sub_constraint(self, a: str, required: bool = False) -> Constraint | Float:
        if self.constraint is None:
            c = None
        else:
            c = self.constraint.get(a)
        if required and c is None:
            raise MissingConstraint(self.address + (a,))
        return c

    def construct_retval(self, retval) -> R:
        return retval


class Assess[R](Transformation[R]):
    def __init__(self, address: Address, constraint: Constraint):
        super().__init__(PHANTOM_KEY, address, constraint)
        self.score = jnp.array(0.0)

    def transform_inner(self, jaxpr, in_avals, addr: str, constraint: Constraint):
        """Apply simulate to jaxpr and return the transformed jaxpr together
        with its return shape."""

        def inner(in_avals):
            a = Assess(self.address + (addr,), constraint)
            retval = a.run(jaxpr, in_avals)
            return a.score, retval

        return jax.make_jaxpr(
            inner,
            return_shape=True,
        )(in_avals)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, VmapGF):
            at = bind_params["at"]

            def vmap_inner(constraint, params):
                a = Assess(self.address + (at,), constraint)
                retval = a.run(bind_params["inner"], params)
                return a.score, retval

            if at != VmapGF.SUB_TRACE:
                cons = self.get_sub_constraint(at, required=True)
            else:
                cons = self.constraint

            score, ans = jax.vmap(vmap_inner, in_axes=(0, eqn.primitive.in_axes))(
                cons, jax.tree.unflatten(eqn.primitive.in_tree, params)
            )
            self.score += jnp.sum(score)
            return ans

        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            addr = self.address + (at,)
            score, ans = eqn.primitive.assess_p(
                params, self.get_sub_constraint(at, required=True), addr
            )
            self.score += jnp.sum(score)
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

    def record(self, sub_trace, at):
        if (
            (inner_trace := sub_trace.get("subtraces"))
            and len(keys := inner_trace.keys()) == 1
            and (key := next(iter(keys))).startswith("__")
        ):
            # absorb interstitial trace points like __repeat, __vmap that may
            # occur when combinators are stacked.
            sub_trace["subtraces"] = inner_trace[key]["subtraces"]
        if at:
            self.trace[at] = sub_trace
        self.w += jnp.sum(sub_trace.get("w", 0.0))
        return sub_trace["retval"]

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
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
            return ans["retval"]

        if eqn.primitive is jax.lax.scan_p:
            inner = bind_params["jaxpr"]

            if at := self.address_from_branch(inner):
                address = self.address + (at,)
                sub_address = at
            else:
                address = self.address
                sub_address = None

            num_carry = bind_params['num_carry']

            def step(carry_key, s):
                print(f'ck in {carry_key}')
                carry, key = carry_key
                key, k1 = KeySplit.bind(key, a="scan_step")
                v = Simulate(k1, address, self.constraint).run(inner, (carry, s))
                print(f'v.retval = {v['retval']}')
                # this computation below is bogus. We need a way to store the argument
                # structure which can be retrieved here the arguments can be packed and
                # unpacked correctly.
                return ((tuple(jax.tree.flatten(v["retval"][:num_carry])[0]), key), v)

            # Where we left off: we need the in_tree into which to decompose
            # the parameters

            ans = jax.lax.scan(step, (params[:num_carry], self.get_sub_key()), params[num_carry:])
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
        class Binder:
            def __matmul__(self, address: str):
                return jax.lax.switch(
                    pred, [lambda: ff @ address, lambda: tf @ address]
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
        # TODO: get this from self.make_jaxpr below
        self.flat_args, self.in_tree = jax.tree.flatten(self.get_args())

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
        def repeat_inner(key, constraint, params):
            s = Simulate(key, address, constraint)
            return s.run(self.gfi.get_jaxpr(), params)

        return jax.vmap(repeat_inner, in_axes=(0, 0, None))(
            jax.random.split(key, self.n),
            constraint,
            jax.tree.unflatten(self.in_tree, arg_tuple),
        )

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
        self.n = self.arg_tuple[self.p_index].shape[self.an_axis]
        # the shape of the v-mapped function will increase every axis of the result
        self.shape = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct((self.n,) + s.shape, s.dtype),
            self.inner_shape,
        )
        self.multiple_results = isinstance(self.shape, tuple)
        self.jaxpr = jax.make_jaxpr(
            lambda *args: self.bind(
                *args, in_axes=self.in_axes, at=VmapGF.SUB_TRACE, inner=self.inner_jaxpr
            ),
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
        def vmap_inner(key, constraint, params):
            s = Simulate(key, address, constraint)
            return s.run(self.inner_jaxpr, params)

        return jax.vmap(vmap_inner, in_axes=(0, 0, self.in_axes))(
            jax.random.split(key, self.n),
            constraint,
            jax.tree.unflatten(self.in_tree, arg_tuple),
        )

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address: tuple[str, ...]
    ) -> Float:
        a = Assess(address, constraint)
        retval = a.run(self.jaxpr, arg_tuple)
        return a.score, retval

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.flat_args, at=address, in_axes=self.in_axes, inner=self.inner_jaxpr
        )

    def get_args(self) -> tuple:
        return self.arg_tuple

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    class Simulate(GenPrimitive):
        def __init__(self, r: "VmapGF", shape: Any):  # TODO: PyTreeDef?
            super().__init__("Vmap.Simulate")
            self.r = r
            self.n = self.r.arg_tuple[self.r.p_index].shape[self.r.an_axis]
            self.shape = shape
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
            return jax.vmap(
                lambda k, arg_tuple: jax.core.eval_jaxpr(
                    j.jaxpr,
                    j.consts,
                    k,
                    *jax.tree.flatten(arg_tuple)[0],
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

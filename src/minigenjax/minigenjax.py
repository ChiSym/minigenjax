# %%
from collections import namedtuple
from typing import Any, Callable, Sequence
from jax._src.core import ClosedJaxpr as ClosedJaxpr
import jax
from jax._src.tree_util import PyTreeDef as PyTreeDef
import jax.tree
import jax.api_util
import jax.numpy as jnp
from jax.interpreters import batching
from .key import KeySplit
from .trace import to_constraint, to_score, to_weight
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

    def get_args(self) -> tuple:
        raise NotImplementedError(f"get_args: {self}")

    def inflate(self, v: Any, n: int):
        def inflate_one(v):
            return v.update(shape=(n,) + v.shape)

        return jax.tree.map(inflate_one, v)

    def unflatten(self, flat_args):
        raise NotImplementedError(f"unflatten: {self}")


InAxesT = int | Sequence[Any] | None


class GFI[R]:
    def __init__(self, name):
        self.name = name

    def simulate(self, key: PRNGKeyArray) -> dict:
        raise NotImplementedError(f"simulate: {self}")

    def propose(self, key: PRNGKeyArray) -> tuple[Constraint, Float, R]:
        tr = self.simulate(key)
        return to_constraint(tr), to_score(tr), tr["retval"]

    def importance(self, key: PRNGKeyArray, constraint: Constraint) -> dict:
        raise NotImplementedError(f"importance: {self}")

    def assess(self, constraint) -> tuple[Array, Array]:
        raise NotImplementedError(f"assess: {self}")


# %%
class Gen[R]:
    def __init__(self, f: Callable[..., R], partial_args=()):
        self.f = f
        self.partial_args = partial_args

    def get_name(self) -> str:
        return self.f.__name__

    def __call__(self, *args):
        return GFA(self.f, args, partial_args=self.partial_args)

    def repeat(self, n):
        this = self

        class ToRepeat(Gen):
            def __init__(self, n):
                self.n = n

            def __call__(self, *args):
                return RepeatA(n, args, this)

        return ToRepeat(n)

    def vmap(self, in_axes: InAxesT = 0):
        this = self
        class ToVmap(Gen):
            def __init__(self, in_axes):
                self.in_axes = in_axes

            def __call__(self, *args):
                return  VmapA(self.in_axes, args, this)

        return ToVmap(in_axes)

    def map(self, g):
        this = self

        class ToMap(Gen):
            def __init__(self, g):
                self.g = g

            def __call__(self, *args):
                return MapA(self.g, args, this)

        return ToMap(g)

    def scan(self):
        this = self

        class ToScan(Gen):
            def __init__(self):
                pass

            def __call__(self, *args):
                return ScanA(args, this)
            
        return ToScan()

    def partial(self, *args):
        this = self

        class ToPartial(Gen):
            def __init__(self, partial_args):
                self.partial_args = partial_args

            def __call__(self, *args):
                return PartialA(self.partial_args, args, this)

        return ToPartial(args)


# gfb is sort of the "final form"


class GFA[R]:
    def __init__(self, f, args, partial_args):
        self.f = f
        self.name = self.f.__name__
        self.args = args
        self.partial_args = partial_args

        shape = jax.eval_shape(f, *args)
        self.multiple_results = isinstance(shape, (list, tuple))
        self.structure = jax.tree.structure(shape)

    def get_impl(self):
        return GFB(self.f, self.multiple_results)

    def simulate(self, key: PRNGKeyArray):
        return self.get_impl().simulate_p(key, self.get_args(), (), {})

    def assess(self, constraint) -> tuple[Array, Array]:
        return self.get_impl().assess_p(self.get_args(), constraint, ())

    def propose(self, key: PRNGKeyArray) -> tuple[Constraint, Float, R]:
        tr = self.simulate(key)
        return to_constraint(tr), to_score(tr), tr["retval"]

    def importance(self, key: PRNGKeyArray, constraint: Constraint):
        tr = self.get_impl().simulate_p(key, self.get_args(), (), constraint)
        return tr, to_weight(tr)

    def __matmul__(self, address: str):
        flat_args, in_tree = jax.tree.flatten(self.get_args())
        return self.get_impl().bind(*flat_args, at=address, in_tree=in_tree)

    def abstract(self, *args, **kwargs):
        return self.get_impl().abstract(*args, **kwargs)

    def get_args(self):
        return self.partial_args + self.args


class PartialA(GFA):
    def __init__(self, partial_args, args, next):
        self.partial_args = partial_args
        self.args = args
        self.gfa = next(*self.partial_args, *self.args)

    def get_impl(self):
        return self.gfa.get_impl()
    
    def get_args(self):
        return self.partial_args + self.args

class ScanA(GFA):
    def __init__(self, args, nxt):
        assert len(args) == 2
        self.args = args
        # fixed_args = (arg_tuple[0], jax.tree.map(lambda v: v[0], arg_tuple[1]))
        self.gfa = nxt(args[0], jax.tree.map(lambda v: v[0], args[1]))

    def get_impl(self):
        return ScanGF(self.gfa)

    def get_args(self):
        return self.args

class RepeatA[R](GFA):
    def __init__(self, n, arg_tuple, next: Gen):
        self.n = n
        self.arg_tuple = arg_tuple
        self.gfa = next(*arg_tuple)
        self.name = f"Repeat[{n}, {self.gfa.name}]"

    def get_impl(self):
        return RepeatGF(self.gfa, self.n)

    # def simulate(self, key: PRNGKeyArray):
    #     return self.get_impl().simulate_p(key, self.arg_tuple, (), {})

    def __matmul__(self, address: str):
        return self.get_impl().bind(*self.gfa.partial_args + self.gfa.args, at=address)

    def get_args(self):
        return self.arg_tuple


class MapA[R](GFA):
    def __init__(self, g, arg_tuple, next):
        self.g = g
        self.arg_tuple = arg_tuple
        self.inner = next(*arg_tuple)
        self.name = f"Map[{g.__name__}, {self.inner.name}]"

    def get_impl(self):
        return MapGF(self.inner, self.g)

    # def simulate(self, key: PRNGKeyArray):
    #     return self.get_impl().simulate_p(key, self.arg_tuple, (), {})

    def get_args(self):
        return self.arg_tuple


class VmapA[R](GFA):
    def __init__(self, in_axes: InAxesT, args: tuple, nxt):
        self.in_axes = in_axes
        self.arg_tuple = args

        # find one pair of (parameter number, axis) to use to determine size of vmap
        if isinstance(self.in_axes, tuple):
            self.p_index, self.an_axis = next(
                filter(lambda b: b[1] is not None, enumerate(self.in_axes))
            )
        else:
            self.p_index, self.an_axis = 0, self.in_axes
        self.n = args[self.p_index].shape[self.an_axis]
        self.inner = nxt(*self.un_vmap_arguments(args))
        self.name = f"VMap[in_axes={in_axes}, {self.inner.name}]"

    def un_vmap_arguments(self, arg_tuple):
        def un_vmap(axis, arg):
            if axis is None:
                return arg
            if isinstance(arg, tuple):
                return jax.tree.map(lambda a: un_vmap(axis, a), arg)
            if axis == 0:
                return arg[0]
            raise Exception(f"unimplemented vmap axis :( {axis} ")

        return jax.tree.map(
            un_vmap, self.in_axes, arg_tuple, is_leaf=lambda x: x is None
        )

    def get_impl(self):
        return VmapGF(self)

    # def simulate(self, key: PRNGKeyArray):
    #     return self.get_impl().simulate_p(key, self.arg_tuple, (), {})

    def __matmul__(self, address: str):
        return self.get_impl().bind(*self.inner.get_args(), at=address)

    def get_args(self):
        return self.arg_tuple

    # def simulate(self, key: PRNGKeyArray):
    #     gfa = self.gen(*self.args)
    #     return VMapB(gfa, self.n, self.in_axes).simulate_p(
    #         key, gfa.partial_args + gfa.args, (), {}
    #     )

    # def __matmul__(self, address: str):
    #     return VmapB(self.gfa, n).bind(
    #         *self.gfa.partial_args + self.gfa.args, at=address
    #     )


class VMapB:
    def __init__(self, inner, n, in_axes):
        self.inner = inner
        self.n = n
        self.in_axes = in_axes

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return jax.vmap(
            lambda key, constraint, arg_tuple: self.inner.get_impl().simulate_p(
                key, arg_tuple, address, constraint
            ),
            in_axes=(0, 0, self.in_axes),
        )(jax.random.split(key, self.n), constraint, arg_tuple)

    def abstract(self, *args, **kwargs):
        ia = self.inner.abstract(*args, **kwargs)
        return ia.update(shape=(self.n,) + ia.shape)


class GFB[R](GenPrimitive):
    def __init__(self, f, multiple_results):
        super().__init__(f"GFB[{f.__name__}]")
        self.f = f
        self.multiple_results = multiple_results

    def concrete(self, *args, **kwargs):
        return self.f(*args)

    def abstract(self, *args, **kwargs):
        return jax.tree.map(jax.core.get_aval, jax.eval_shape(self.f, *args))
        return jax.core.get_aval(
            jax.eval_shape(self.f, *args)
        )

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        j, shape = jax.make_jaxpr(self.f, return_shape=True)(*arg_tuple)
        structure = jax.tree.structure(shape)
        return Simulate(key, address, constraint).run(j, arg_tuple, structure)

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        j, shape = jax.make_jaxpr(self.f, return_shape=True)(*arg_tuple)
        structure = jax.tree.structure(shape)
        a = Assess[R](address, constraint)
        retval = a.run(j, arg_tuple, structure)
        return a.score, retval


class RepeatGF[R](GenPrimitive):
    SUB_TRACE = "__repeat"

    def __init__(self, inner: GFA[R], n: int):
        name = f"RepeatGF[{n}, {inner.name}]"
        super().__init__(name)
        self.n = n
        self.inner = inner

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return jax.vmap(
            lambda key, constraint: self.inner.get_impl().simulate_p(
                key, arg_tuple, address, constraint
            )
        )(jax.random.split(key, self.n), constraint)

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        score, retval = jax.vmap(
            lambda constraint: self.inner.get_impl().assess_p(
                arg_tuple, constraint, address
            )
        )(constraint)
        return jnp.sum(score), retval

    def abstract(self, *args, **kwargs):
        ia = self.inner.abstract(*args, **kwargs)
        return ia.update(shape=(self.n,) + ia.shape)


class GF[R](GFI[R]):
    def __init__(self, g: Gen, args: tuple, partial_args=()):
        super().__init__(f"GF[{g.get_name()}]")

        _, self.in_tree = jax.tree.flatten(args)

        # TODO: make_jaxpr already computes this, reuse it
        self.f = g.f
        self.args = args
        self.partial_args = partial_args
        abstract_args = jax.tree.map(jax.core.get_aval, partial_args + args)
        with jax.check_tracer_leaks():
            pass
        self.shape = jax.eval_shape(self.f, *abstract_args)
        self.structure = jax.tree.structure(self.shape)
        self.abstract_value = list(
            map(jax.core.get_aval, jax.tree.flatten(self.shape)[0])
        )

        # with jax.check_tracer_leaks():
        self.jaxpr = self.make_jaxpr(g.f, partial_args + args)
        self.multiple_results = self.structure.num_leaves > 1

    def abstract(self, *args, **_kwargs):
        return self.abstract_value if self.multiple_results else self.abstract_value[0]

    def concrete(self, *args, **kwargs):
        return self.f(*jax.tree.unflatten(self.in_tree, args))

    def repeat(self, n):
        return RepeatGF(self, n)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return Simulate(key, address, constraint).run(
            self.jaxpr, self.partial_args + arg_tuple, self.structure
        )

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        a = Assess[R](address, constraint)
        retval = a.run(self.jaxpr, self.partial_args + arg_tuple, self.structure)
        return a.score, retval

    def get_args(self) -> tuple:
        return self.args

    def get_structure(self) -> jax.tree_util.PyTreeDef:
        return self.structure

    def unflatten(self, flat_args) -> jax.tree_util.PyTreeDef:
        return jax.tree.unflatten(self.in_tree, flat_args)


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
            isinstance(eqn.primitive, GenPrimitive) or eqn.primitive is jax.lax.cond_p
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

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
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
            raise Exception("foo")
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
            if in_tree := bind_params.get("in_tree"):
                params = jax.tree.unflatten(in_tree, params)
            addr = self.address + (at,)
            ans = eqn.primitive.simulate_p(
                self.get_sub_key(),
                params,
                addr,
                self.get_sub_constraint(at),
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
            return ans["retval"]

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
                return GF(Gen(lambda: self @ "__cond"), ()).simulate(key)

        return Binder()

    return ctor


# I'm running out of good ideas.
# Maybe we want to move the creation of the
# GFI to a lower level than here?


class ScanGF[R](GenPrimitive):
    def __init__(self, inner: ScanA):
        super().__init__(f"ScanGF[{inner.name}]")
        self.inner = inner
        self.multiple_results = inner.multiple_results

    def abstract(self, *args, **kwargs):
        return self.inner.abstract(*args, **kwargs)

    def simulate_p(
        self,
        key: Array,
        arg_tuple: tuple,
        address: tuple[str, ...],
        constraint: Constraint,
    ) -> dict:
        #fixed_args = (arg_tuple[0], jax.tree.map(lambda v: v[0], arg_tuple[1]))
        #gfi = self.g(*fixed_args)

        def step(carry_key, s):
            carry, key = carry_key
            key, k1 = KeySplit.bind(key, a="scan_step")
            v = self.inner.get_impl().simulate_p(k1, (carry, s), address, constraint)
            return (v["retval"][0], key), v

        print(f"simulate_p with arg_tuple {arg_tuple}")
        ans = jax.lax.scan(step, (arg_tuple[0], key), arg_tuple[1])
        # Fix the return values to report the things an ordinary use of
        # scan would produce.
        ans[1]["retval"] = (ans[0][0], ans[1]["retval"][0])
        return ans[1]

    def get_args(self) -> tuple:
        return self.arg_tuple

    def get_structure(self) -> PyTreeDef:
        return self.gfi.get_structure()

    def unflatten(self, flat_args):
        return self.gfi.unflatten(flat_args)


class VmapGF[R](GenPrimitive):
    def __init__(self, inner: VmapA):
        self.inner = inner
        self.name = inner.name
        self.in_axes = inner.in_axes
        if isinstance(self.in_axes, tuple):
            self.p_index, self.an_axis = next(
                filter(lambda b: b[1] is not None, enumerate(self.in_axes))
            )
        else:
            self.p_index, self.an_axis = 0, self.in_axes
        super().__init__(self.name)

    def abstract(self, *args, **kwargs):
        # TODO: probably wrong, these are flat args, and the indices are vs. tree args
        n = args[self.p_index].shape[self.an_axis]
        ia = self.inner.abstract(*args, **kwargs)
        return ia.update(shape=(n,) + ia.shape)


    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        n = arg_tuple[self.p_index].shape[self.an_axis]
        return jax.vmap(
            lambda key, constraint, arg_tuple: self.inner.inner.get_impl().simulate_p(
                key, arg_tuple, address, constraint
            ),
            in_axes=(0, 0, self.inner.in_axes)
        )(jax.random.split(key, n), constraint, arg_tuple)
    
    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address: tuple[str, ...]
    ) -> Float:
        def vmap_inner(constraint, params):
            return self.inner.inner.get_impl().assess_p(params, constraint, address)

        score, retval = jax.vmap(vmap_inner, in_axes=(0, self.in_axes))(
            constraint, arg_tuple
        )
        return jnp.sum(score), retval


    # def concrete(self, *args, **kwargs):
    #     return jax.vmap(
    #         lambda *args: self.gfi.concrete(*jax.tree.flatten(args)[0], **kwargs),
    #         in_axes=self.in_axes,
    #     )(*jax.tree.unflatten(self.in_tree, args))

    # def get_structure(self):
    #     return self.gfi   .get_structure()

    # def assess_p(
    #     self, arg_tuple: tuple, constraint: Constraint, address: tuple[str, ...]
    # ) -> Float:
    #     def vmap_inner(constraint, params):
    #         return self.gfi.assess_p(params, constraint, address)

    #     score, retval = jax.vmap(vmap_inner, in_axes=(0, self.in_axes))(
    #         constraint, jax.tree.unflatten(self.in_tree, arg_tuple)
    #     )
    #     return jnp.sum(score), retval

    # def unflatten(self, flat_args):
    #     return jax.tree.unflatten(self.in_tree, flat_args)

    # def get_args(self) -> tuple:
    #     return self.arg_tuple


class MapGF[R, S](GenPrimitive):
    def __init__(self, inner: GFI[R], f: Callable[[R], S]):
        super().__init__(f"Map[{inner.name}, {f.__name__}]")
        self.inner = inner
        self.f = f
        self.multiple_results = False
        # self.structure = jax.tree.structure(
        #     jax.eval_shape(
        #         lambda *args: self.f(self.gfi.concrete(*args)), *self.gfi.get_args()
        #     )
        # )

    def abstract(self, *args, **kwargs):
        ia = self.inner.abstract(*args, **kwargs)
        return ia

    # def abstract(self, *args, **kwargs):
    #     # this can't be right: what about the effect of f? Why isn't it
    #     # sufficient to apply f to a tracer to compose the abstraction?
    #     return self.gfi.abstract(*args, **kwargs)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        out = self.inner.get_impl().simulate_p(key, arg_tuple, address, constraint)
        out["retval"] = self.f(out["retval"])
        return out


# %%

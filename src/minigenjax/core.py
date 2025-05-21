# %%
from typing import Callable
import jax
import jax.tree
import jax.api_util
import jax.numpy as jnp
from .key import KeySplit
from .types import InAxesT
from .primitive import GenPrimitive
from .simulate import Simulate
from .update import Update
from .assess import Assess
from .trace import to_constraint, to_score, to_weight
from minigenjax.types import Address, Constraint
import jax.core
from jaxtyping import Array, PRNGKeyArray, Float, PyTreeDef


# %%


# %%
class Gen[R]:
    def __init__(self, f: Callable[..., R] | None = None):
        self.f = f

    def get_name(self) -> str:
        return self.f.__name__ if self.f else "?"

    def __call__(self, *args) -> "GFI":
        assert self.f is not None
        # TODO: fix this so that the base class isn't holding the reference to f.
        # Maybe there should be a constructor function and a class, and we're
        # working too hard to overload this
        return GFA(self.f, args)

    def repeat(self, n):
        this = self

        class ToRepeat(Gen):
            def __init__(self, n):
                super().__init__()
                self.n = n

            def __call__(self, *args):
                return RepeatA(n, args, this)

        return ToRepeat(n)

    def vmap(self, in_axes: InAxesT = 0):
        this = self

        class ToVmap(Gen):
            def __init__(self, in_axes):
                super().__init__()
                self.in_axes = in_axes

            def __call__(self, *args):
                return VmapA(self.in_axes, args, this)

        return ToVmap(in_axes)

    def map(self, g):
        this = self

        class ToMap(Gen):
            def __init__(self, g):
                super().__init__()
                self.g = g

            def __call__(self, *args):
                return MapA(self.g, args, this)

        return ToMap(g)

    def scan(self):
        this = self

        class ToScan(Gen):
            def __init__(self):
                super().__init__()

            def __call__(self, *args):
                return ScanA(args, this)

        return ToScan()

    def partial(self, *args):
        this = self

        class ToPartial(Gen):
            def __init__(self, partial_args):
                super().__init__()
                self.partial_args = partial_args

            def __call__(self, *args):
                return PartialA(self.partial_args, args, this)

        return ToPartial(args)


class GFI[R]:
    def __init__(self, name):
        self.name = name

    def simulate(self, key: PRNGKeyArray) -> dict:
        return self.get_impl().simulate_p(key, self.get_args(), (), {})

    def update(
        self, key: PRNGKeyArray, constraint: Constraint, previous_trace: dict
    ) -> dict:
        return self.get_impl().update_p(
            key, self.get_args(), (), constraint, previous_trace
        )

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
        return jax.tree.unflatten(
            self.get_structure(),
            self.get_impl().bind(*flat_args, at=address, in_tree=in_tree),
        )

    def get_impl(self) -> "GFImpl":
        raise NotImplementedError(f"get_impl: {self}")

    def get_args(self) -> tuple:
        raise NotImplementedError(f"get_args: {self}")

    def get_structure(self) -> PyTreeDef:
        raise NotImplementedError(f"get_structure: {self}")

    def abstract(self, *args, **kwargs):
        return self.get_impl().abstract(*args, **kwargs)


class GFA[R](GFI[R]):
    def __init__(self, f: Callable[..., R], args):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f
        self.args = args

        with jax.check_tracer_leaks():
            self.shape = jax.eval_shape(f, *jax.tree.map(jax.core.get_aval, args))
        self.structure = jax.tree.structure(self.shape)

    def get_impl(self):
        return GFB(self.f, self.shape)

    def get_args(self):
        return self.args

    def get_structure(self):
        return self.structure


class PartialA(GFI):
    def __init__(self, partial_args, args, next):
        # TODO: make the initialization sane by extracting a base class
        # so that GFA is not the root
        self.partial_args = partial_args
        self.args = args
        self.gfa = next(*self.partial_args, *self.args)
        super().__init__(f"Partial[{self.gfa.name}]")

    def get_impl(self):
        return PartialGF(self.partial_args, self.gfa)

    def get_args(self):
        return self.args

    def get_structure(self):
        return self.gfa.get_structure()


class GFImpl[R](GenPrimitive):
    def __init__(self, name):
        super().__init__(name)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        raise NotImplementedError(f"simulate_p: {self}")

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        raise NotImplementedError(f"assess_p: {self}")


class PartialGF(GFImpl):
    def __init__(self, partial_args, inner: GFI):
        super().__init__(f"Partial[{inner.name}]")
        self.inner_impl = inner.get_impl()
        self.abstract_value = inner.abstract(
            *jax.tree.flatten(partial_args + inner.get_args())
        )
        self.partial_args = partial_args
        self.multiple_results = True

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return self.inner_impl.simulate_p(
            key, self.partial_args + arg_tuple, address, constraint
        )

    def abstract(self, *args, **kwargs):
        return self.abstract_value


class ScanA(GFI):
    def __init__(self, args, nxt):
        assert len(args) == 2
        self.args = args
        self.multiple_results = True
        # fixed_args = (arg_tuple[0], jax.tree.map(lambda v: v[0], arg_tuple[1]))
        self.gfa = nxt(args[0], jax.tree.map(lambda v: v[0], args[1]))
        super().__init__(f"Scan[{self.gfa.name}]")

    def get_impl(self):
        return ScanGF(self.gfa)

    def get_args(self):
        return self.args

    def get_structure(self):
        # return self.structure
        return self.gfa.get_structure()


class RepeatA[R](GFI):
    def __init__(self, n, arg_tuple, next: Gen):
        self.repeated = next(*arg_tuple)
        super().__init__(f"Repeat[{n}, {self.repeated.name}]")
        self.n = n
        self.arg_tuple = arg_tuple

    def get_impl(self):
        return RepeatGF(self)

    def get_structure(self):
        return self.repeated.get_structure()

    def get_args(self):
        return self.arg_tuple


class MapA[R](GFI):
    def __init__(self, g, arg_tuple, next):
        self.g = g
        self.arg_tuple = arg_tuple
        _, self.in_tree = jax.tree.flatten(self.arg_tuple)
        self.inner = next(*arg_tuple)
        super().__init__(f"Map[{g.__name__}, {self.inner.name}]")
        self.shape = jax.eval_shape(lambda: self.g(self.inner @ "a"))

        self.abstract_value = jax.tree.map(
            jax.core.get_aval, jax.tree.flatten(self.shape)[0]
        )

        self.structure = jax.tree.structure(self.shape)

    def get_impl(self):
        return MapGF(self, self.g)

    def get_args(self):
        return self.arg_tuple

    def get_structure(self):
        return self.structure


class VmapA[R](GFI):
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
        super().__init__(f"VMap[in_axes={in_axes}, {self.inner.name}]")

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

    def get_args(self):
        return self.arg_tuple

    def get_structure(self):
        return self.inner.get_structure()


class GFB[R](GFImpl):
    def __init__(self, f: Callable[..., R], shape):
        super().__init__(f"GFB[{f.__name__}]")
        self.f = f
        self.abstract_value = jax.tree.map(
            jax.core.get_aval, jax.tree.flatten(shape)[0]
        )
        self.multiple_results = True

    def concrete(self, *args, **kwargs):
        return self.f(*args)

    def abstract(self, *args, **kwargs):
        return self.abstract_value

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        with jax.check_tracer_leaks():
            j, shape = jax.make_jaxpr(self.f, return_shape=True)(*arg_tuple)
        structure = jax.tree.structure(shape)
        return Simulate(key, address, constraint).run(j, arg_tuple, structure)

    def update_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
        previous_trace: dict,
    ):
        # TODO: move to ctor
        with jax.check_tracer_leaks():
            j, shape = jax.make_jaxpr(self.f, return_shape=True)(*arg_tuple)
        structure = jax.tree.structure(shape)
        return Update(key, address, constraint, previous_trace).run(
            j, arg_tuple, structure
        )

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        with jax.check_tracer_leaks():
            j, shape = jax.make_jaxpr(self.f, return_shape=True)(*arg_tuple)
        structure = jax.tree.structure(shape)
        a = Assess[R](address, constraint)
        retval = a.run(j, arg_tuple, structure)
        return a.score, retval


class RepeatGF[R](GFImpl):
    SUB_TRACE = "__repeat"

    def __init__(self, inner: RepeatA[R]):
        super().__init__(inner.name)
        self.repeated = inner.repeated
        self.n = inner.n
        self.multiple_results = True

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return jax.vmap(
            lambda key, constraint: self.repeated.get_impl().simulate_p(
                key, arg_tuple, address, constraint
            )
        )(jax.random.split(key, self.n), constraint)

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Float, R]:
        score, retval = jax.vmap(
            lambda constraint: self.repeated.get_impl().assess_p(
                arg_tuple, constraint, address
            )
        )(constraint)
        return jnp.sum(score), retval

    def abstract(self, *args, **kwargs):
        ia = self.repeated.abstract(*args, **kwargs)
        return jax.tree.map(lambda a: a.update(shape=(self.n,) + a.shape), ia)


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
                return Gen(lambda: self @ "__cond")().simulate(key)

        return Binder()

    return ctor


class ScanGF[R](GFImpl):
    def __init__(self, inner: ScanA):
        super().__init__(f"ScanGF[{inner.name}]")
        self.inner_impl = inner.get_impl()
        # TODO: could we just compute abstract_value up front, forever?
        self.abstract_value = inner.abstract(*jax.tree.flatten(inner.get_args())[0])
        self.multiple_results = True

    def abstract(self, *args, **kwargs):
        return self.abstract_value

    def simulate_p(
        self,
        key: Array,
        arg_tuple: tuple,
        address: tuple[str, ...],
        constraint: Constraint,
    ) -> dict:
        # fixed_args = (arg_tuple[0], jax.tree.map(lambda v: v[0], arg_tuple[1]))
        # gfi = self.g(*fixed_args)

        def step(carry_key, step_constraint):
            carry, key = carry_key
            step, constraint = step_constraint
            key, k1 = KeySplit.bind(key, a="scan_step")
            v = self.inner_impl.simulate_p(k1, (carry, step), address, constraint)
            return (v["retval"][0], key), v

        ans = jax.lax.scan(step, (arg_tuple[0], key), (arg_tuple[1], constraint))
        # Fix the return values to report the things an ordinary use of
        # scan would produce.
        ans[1]["retval"] = (ans[0][0], ans[1]["retval"][0])
        return ans[1]


class VmapGF[R](GFImpl):
    def __init__(self, inner: VmapA):
        super().__init__(inner.name)
        self.in_axes = inner.in_axes
        self.n = inner.n
        self.p_index = inner.p_index
        self.an_axis = inner.an_axis
        av = inner.inner.abstract(*jax.tree.flatten(inner.get_args()))
        self.abstract_value = jax.tree.map(
            lambda a: a.update(shape=(self.n,) + a.shape), av
        )
        self.inner_impl = inner.inner.get_impl()
        self.multiple_results = True

    def abstract(self, *args, **kwargs):
        return self.abstract_value

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        n = arg_tuple[self.p_index].shape[self.an_axis]
        return jax.vmap(
            lambda key, constraint, arg_tuple: self.inner_impl.simulate_p(
                key, arg_tuple, address, constraint
            ),
            in_axes=(0, 0, self.in_axes),
        )(jax.random.split(key, n), constraint, arg_tuple)

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address: tuple[str, ...]
    ) -> Float:
        def vmap_inner(constraint, params):
            return self.inner_impl.assess_p(params, constraint, address)

        score, retval = jax.vmap(vmap_inner, in_axes=(0, self.in_axes))(
            constraint, arg_tuple
        )
        return jnp.sum(score), retval


class MapGF[R, S](GFImpl):
    def __init__(self, inner: MapA[R], g: Callable[[R], S]):
        super().__init__(inner.name)
        self.inner_impl = inner.inner.get_impl()
        self.g = g
        self.multiple_results = True
        self.abstract_value = inner.abstract_value

    def abstract(self, *args, **kwargs):
        return self.abstract_value

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        out = self.inner_impl.simulate_p(key, arg_tuple, address, constraint)
        out["retval"] = self.g(out["retval"])
        return out


# %%

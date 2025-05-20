from jax.interpreters import batching
from minigenjax.types import Address, Constraint, InAxesT
from jaxtyping import Array, PRNGKeyArray, Float
from typing import Any
import jax
import jax.extend as jx


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

    def batch(
        self,
        vector_args,
        batch_axes: InAxesT,
        **kwargs,
    ) -> tuple[Any, InAxesT]:
        # TODO assert all axes equal
        result = jax.vmap(lambda *args: self.impl(*args, **kwargs), in_axes=batch_axes)(
            *vector_args
        )
        return result, batch_axes

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

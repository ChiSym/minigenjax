from jax.interpreters import batching
from minigenjax.types import Address, InAxesT
from jaxtyping import Array, PRNGKeyArray, Float
from typing import Any
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
        raise NotImplementedError(f"batch: {self}")

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: dict,
    ) -> dict:
        raise NotImplementedError(f"simulate_p: {self}")

    def update_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: dict,
        previous_trace: dict,
    ) -> dict:
        raise NotImplementedError(f"update_p: {self}")

    def assess_p(
        self, arg_tuple: tuple, constraint: dict | Float, address: tuple[str, ...]
    ) -> tuple[Array, Any]:
        raise NotImplementedError(f"assess_p: {self}")

    def get_args(self) -> tuple:
        raise NotImplementedError(f"get_args: {self}")

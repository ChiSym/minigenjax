from jaxtyping import PRNGKeyArray
import jax
import jax.extend as jx
from .primitive import GenPrimitive
from .transform import TracingTransform
from .types import Address, Constraint


class Simulate(TracingTransform):
    def __init__(
        self,
        key: PRNGKeyArray,
        address: Address,
        constraint: Constraint,
    ):
        super().__init__(key, address, constraint)

    def make_inner(self, key: PRNGKeyArray):
        return Simulate(key, self.address, self.constraint)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            if in_tree := bind_params.get("in_tree"):
                params = jax.tree.unflatten(in_tree, params)
            addr = self.address + (at,)
            ans = eqn.primitive.simulate_p(
                self.get_sub_key(), params, addr, self.get_sub_constraint(at)
            )
            return self.record(ans, at)

        return super().handle_eqn(eqn, params, bind_params)

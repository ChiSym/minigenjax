from .primitive import GenPrimitive
from .transform import TracingTransform
from .types import Address, Constraint
from jaxtyping import PRNGKeyArray
import jax.extend as jx
import jax


class Update(TracingTransform):
    def __init__(
        self,
        key: PRNGKeyArray,
        address: Address,
        constraint: Constraint,
        previous_trace: dict,
    ):
        super().__init__(key, address, constraint)
        self.previous_trace = previous_trace

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            if in_tree := bind_params.get("in_tree"):
                params = jax.tree.unflatten(in_tree, params)
            addr = self.address + (at,)
            ans = eqn.primitive.update_p(
                self.get_sub_key(),
                params,
                addr,
                self.get_sub_constraint(at),
                self.previous_trace["subtraces"][at],
            )
            # TODO: in update we have to keep track of discarded things
            return self.record(ans, at)

        return super().handle_eqn(eqn, params, bind_params)

    def make_inner(self, key: PRNGKeyArray):
        return Update(key, self.address, self.constraint, self.previous_trace)

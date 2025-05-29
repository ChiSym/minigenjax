from .transform import Transformation
from .types import Address, PHANTOM_KEY
from .primitive import GenPrimitive
import jax.extend as jx
import jax.numpy as jnp
import jax


class Assess[R](Transformation[R]):
    def __init__(self, address: Address, constraint: dict):
        super().__init__(PHANTOM_KEY, address, constraint)
        self.score = jnp.array(0.0)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, GenPrimitive):
            at = bind_params["at"]
            addr = self.address + (at,)
            if in_tree := bind_params.get("in_tree"):
                params = jax.tree.unflatten(in_tree, params)
            score, ans = eqn.primitive.assess_p(
                params, self.get_sub_constraint(at, required=True), addr
            )
            self.score += jnp.sum(score)
            return ans

        return super().handle_eqn(eqn, params, bind_params)

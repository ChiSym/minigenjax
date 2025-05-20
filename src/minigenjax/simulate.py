from jaxtyping import PRNGKeyArray
import jax
import jax.numpy as jnp
import jax.extend as jx
from .primitive import GenPrimitive
from .transform import Transformation
from .types import Address, Constraint


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

from .primitive import GenPrimitive
from .transform import Transformation
from .types import Address, Constraint
from jaxtyping import PRNGKeyArray
import jax.numpy as jnp
import jax.extend as jx
import jax


class Update(Transformation[dict]):
    def __init__(
        self,
        key: PRNGKeyArray,
        address: Address,
        constraint: Constraint,
        previous_trace: dict,
    ):
        super().__init__(key, address, constraint)
        self.trace = {}
        self.previous_trace = previous_trace
        self.w = jnp.array(0.0)

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
                lambda: Update(
                    sub_key, self.address, self.constraint, self.previous_trace
                ).run(branches[1], params[1:]),
                lambda: Update(
                    sub_key, self.address, self.constraint, self.previous_trace
                ).run(branches[0], params[1:]),
            )
            if sub_address:
                self.trace[sub_address] = ans["subtraces"][sub_address]

            self.w += jnp.sum(ans.get("w", 0))
            return ans["retval"]

        return super().handle_eqn(eqn, params, bind_params)

    def record(self, sub_trace, at):
        # TODO: this is copied from Simulate
        self.trace[at] = sub_trace
        self.w += jnp.sum(sub_trace.get("w", 0.0))
        return sub_trace["retval"]

    def construct_retval(self, retval):
        r = {"retval": retval, "subtraces": self.trace}
        if self.constraint:
            r["w"] = self.w
        return r

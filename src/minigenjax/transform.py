from jaxtyping import PRNGKeyArray, Float
from .types import Address, Constraint
from .key import KeySplit
from .primitive import GenPrimitive
import jax
import jax.extend as jx
from typing import Any


class MissingConstraint(Exception):
    pass


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

        def read(v: jx.core.Var) -> Any:
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
            jax.util.safe_map(write, eqn.outvars, jax.tree.flatten(ans)[0])
            # clean_up_dead_vars(eqn, env, lu)
        retval = jax.util.safe_map(read, jaxpr.outvars)
        if structure is not None:
            retval = jax.tree.unflatten(structure, retval)
        else:
            pass
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
        return None

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

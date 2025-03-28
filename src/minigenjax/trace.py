import jax.numpy as jnp
from jaxtyping import Float


def to_constraint(trace: dict) -> dict:
    if "subtraces" in trace:
        return {k: to_constraint(v) for k, v in trace["subtraces"].items()}
    return trace["retval"]


def trace_sum(trace: dict, key: str) -> Float:
    if "subtraces" in trace:
        return sum(jnp.sum(trace_sum(v, key)) for v in trace["subtraces"].values())
    return jnp.sum(trace.get(key, 0.0))


def to_score(trace: dict) -> Float:
    return trace_sum(trace, "score")


def to_weight(trace: dict) -> Float:
    return trace_sum(trace, "w")

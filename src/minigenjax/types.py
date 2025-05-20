import jax.random
from jaxtyping import ArrayLike

Address = tuple[str, ...]
Constraint = dict[str, "ArrayLike | Constraint"]
PHANTOM_KEY = jax.random.key(987654321)
InAxesT = int | tuple[int | None, ...] | None

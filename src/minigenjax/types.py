import jax.random

Address = tuple[str, ...]
PHANTOM_KEY = jax.random.key(987654321)
InAxesT = int | tuple[int | None, ...] | None

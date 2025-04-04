import dataclasses
import jax.tree_util
import jax.numpy as jnp


class IterablePytree:
    def __getitem__(self, i):
        return jax.tree.map(lambda v: v[i], self)

    def __len__(self):
        return len(jax.tree.leaves(self)[0])

    def __iter__(self):
        return (self.__getitem__(i) for i in range(self.__len__()))

    def prepend(self, child):
        """Prepends a scalar element to the front of each leaf in a Pytree."""
        return jax.tree.map(lambda x: x[jnp.newaxis], child) + self

    def __add__(self, other):
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b]), self, other)


def pytree(cls: type) -> type:
    T = type(
        cls.__name__,
        (IterablePytree, dataclasses.dataclass(frozen=True)(cls)),
        {
            "attributes_dict": lambda self: {
                field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)
            }
        },
    )
    return jax.tree_util.register_dataclass(T)

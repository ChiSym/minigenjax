import dataclasses
import jax.tree_util


class IterablePytree:
    def __getitem__(self, i):
        return jax.tree.map(lambda v: v[i], self)

    def __size__(self):
        return len(jax.tree.leaves(self)[0])

    def __iter__(self):
        return (self.__getitem__(i) for i in range(self.__size__()))


def pytree[R](cls: type) -> type:
    T = type(
        cls.__name__,
        (dataclasses.dataclass(cls), IterablePytree),
        {},
    )
    return jax.tree_util.register_dataclass(T)

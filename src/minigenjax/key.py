import jax.extend as jx
import jax.interpreters.mlir as mlir
import jax.interpreters.batching as batching
import jax.core


class KeySplitP(jx.core.Primitive):
    KEY_TYPE = jax.core.get_aval(jax.random.key(0))

    def __init__(self):
        super().__init__("KeySplit")

        def impl(k, a=None):
            r = jax.random.split(k, 2)
            return r[0], r[1]

        self.def_impl(impl)
        self.multiple_results = True
        self.def_abstract_eval(
            lambda _, a=None: [KeySplitP.KEY_TYPE, KeySplitP.KEY_TYPE]
        )

        mlir.register_lowering(self, mlir.lower_fun(self.impl, self.multiple_results))

        batching.primitive_batchers[self] = self.batch

    def batch(self, vector_args, batch_axes, a=None):
        # key_pair_vector = jax.vmap(self.impl, in_axes=batch_axes)(*vector_args)
        v0, v1 = jax.vmap(self.impl, in_axes=batch_axes)(*vector_args)
        return [v0, v1], (batch_axes[0], batch_axes[0])


KeySplit = KeySplitP()

# %%
from typing import Any, Callable, Sequence
import tensorflow_probability.substrates.jax as tfp
import jax
import jax.tree
import jax.numpy as jnp
from jax.interpreters import batching, mlir
import jax.extend as jx
import jax.core
from jaxtyping import Array, ArrayLike, PRNGKeyArray


# %%
Address = tuple[str, ...]
Constraint = dict[str, "ArrayLike|Constraint"] | Callable[[Address], Array | None]
PHANTOM_KEY = jax.random.key(987654321)


class GenPrimitive(jx.core.Primitive):
    def __init__(self, name):
        super().__init__(name)
        self.def_abstract_eval(self.abstract)
        self.def_impl(self.concrete)
        batching.primitive_batchers[self] = self.batch

    def abstract(self, *args, **kwargs):
        raise NotImplementedError(f"abstract: {self}")

    def concrete(self, *args, **kwargs):
        raise NotImplementedError(f"concrete: {self}")

    def batch(self, vector_args, batch_axes, **kwargs):
        # TODO assert all axes equal
        result = jax.vmap(lambda *args: self.impl(*args, **kwargs), in_axes=batch_axes)(
            *vector_args
        )
        batched_axes = (
            (batch_axes[0],) * len(result) if self.multiple_results else batch_axes[0]
        )
        return result, batched_axes

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        raise NotImplementedError(f"simulate_p: {self}")

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address: tuple[str, ...]
    ) -> tuple[Array, Array]:
        raise NotImplementedError(f"assess_p: {self}")

    def inflate(self, v: Any, n: int):
        def inflate_one(v):
            if isinstance(v, jax.core.ShapedArray):
                return v.update(shape=(n,) + v.shape)
            elif isinstance(v, jax.ShapeDtypeStruct):
                return jax.ShapeDtypeStruct(shape=(n,) + v.shape, dtype=v.dtype)

        if isinstance(v, tuple):
            return tuple(map(inflate_one, v))
        return inflate_one(v)


InAxesT = int | Sequence[Any] | None


class GFI[R](GenPrimitive):
    def __init__(self, name):
        super().__init__(name)

    def simulate(self, key: PRNGKeyArray) -> dict:
        return self.simulate_p(key, self.get_args(), (), {})

    def importance(self, key: PRNGKeyArray, constraint: Constraint) -> dict:
        return Simulate(key, (), constraint).run(self.get_jaxpr(), self.get_args())

    def assess(self, constraint) -> Array:
        return self.assess_p(self.get_args(), constraint, ())[1]

    def __matmul__(self, address: str) -> R:
        raise NotImplementedError(f"{self} @ {address}")

    def get_args(self) -> tuple:
        raise NotImplementedError(f"get_args: {self}")

    def map[S](self, f: Callable[[R], S]) -> "MapGF[R,S]":
        return MapGF(self, f)

    def repeat(self, n: int) -> "RepeatGF[R]":
        return RepeatGF(self, n)

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        raise NotImplementedError(f"get_jaxpr: {self}")


class Distribution(GenPrimitive):
    def __init__(self, name, tfd_ctor, scale: str | None = None):
        super().__init__(name)
        self.tfd_ctor = tfd_ctor
        self.scale = scale
        mlir.register_lowering(self, mlir.lower_fun(self.impl, False))

    def abstract(self, *args, **kwargs):
        return args[1]

    def concrete(self, *args, **kwargs):
        match kwargs.get("op", "Sample"):
            case "Sample":
                # we convert to float here because Bernoulli/Flip will
                # normally return an int, and that confuses XLA, since
                # our abstract implementation says the return types are
                # floats. TODO: consider allowing the marking of integer-
                # returning distributions as ints are sometimes nice to
                # work with.
                return jnp.asarray(
                    self.tfd_ctor(*args[1:]).sample(seed=args[0]), dtype=float
                )
            case "Score":
                return self.tfd_ctor(*args[1:]).log_prob(args[0])
            case _:
                raise NotImplementedError(f'{self.name}.{kwargs["op"]}')

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ):
        # TODO: should we delegate the constraint lookup here, or have an assess_p method?
        # TODO: are we sure we are propagating `scale` through the transformations?
        keys = {}
        if self.scale:
            keys["scale"] = self.scale
        v = self.bind(key, *arg_tuple[1:], **(keys | {"op": "Sample"}))
        return {
            "retval": v,
            "score": self.bind(v, *arg_tuple[1:], **(keys | {"op": "Score"})),
        }

    def __call__(self, *args):
        this = self

        class Binder:
            def __matmul__(self, address: str):
                return this.bind(PHANTOM_KEY, *args, at=address)

            def __call__(self, key: PRNGKeyArray):
                return this.tfd_ctor(*args).sample(seed=key)

        return Binder()


BernoulliL = Distribution(
    "Bernoulli",
    lambda logits: tfp.distributions.Bernoulli(logits=logits),
    scale="Logit",
)
BernoulliP = Distribution(
    "Bernoulli", lambda probs: tfp.distributions.Bernoulli(probs=probs), scale="Prob"
)
Normal = Distribution("Normal", tfp.distributions.Normal)
MvNormalDiag = Distribution("MvNormalDiag", tfp.distributions.MultivariateNormalDiag)
Uniform = Distribution("Uniform", tfp.distributions.Uniform)
Flip = Distribution("Flip", lambda p: tfp.distributions.Bernoulli(probs=p))
CategoricalL = Distribution(
    "Categorical",
    lambda logits: tfp.distributions.Categorical(logits=logits),
    scale="Logit",
)
CategoricalP = Distribution(
    "Categorical",
    lambda probs: tfp.distributions.Categorical(probs=probs),
    scale="Prob",
)


def choose_scale(logits, probs, logit_dist, prob_dist):
    if (logits is None) == (probs is None):
        raise ValueError("Supply exactly one of logits=, probs=")
    return logit_dist(logits) if logits is not None else prob_dist(probs)


def Bernoulli(*, logits=None, probs=None):
    return choose_scale(logits, probs, BernoulliL, BernoulliP)


def Categorical(*, logits=None, probs=None):
    return choose_scale(logits, probs, CategoricalL, CategoricalP)


class KeySplitP(jx.core.Primitive):
    KEY_TYPE = jax.core.get_aval(PHANTOM_KEY)  # jax.core.ShapedArray((2,), jnp.uint32)

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
GenSymT = Callable[[jax.core.AbstractValue], jx.core.Var]


# %%
class Gen[R]:
    def __init__(self, f: Callable[..., R]):
        self.f = f

    def __call__(self, *args) -> "GF[R]":
        return GF(self.f, args)

    def vmap(self, in_axes: InAxesT = 0) -> Callable[..., "VmapGF[R]"]:
        return lambda *args: VmapGF(self, args, in_axes)


class GF[R](GFI[R]):
    def __init__(self, f: Callable[..., R], args: tuple):
        super().__init__(f"GF[{f.__name__}]")
        self.f = f
        self.args = args
        self.flat_args, self.in_tree = jax.tree.flatten(self.args)
        self.jaxpr, self.shape = jax.make_jaxpr(f, return_shape=True)(*args)
        self.structure = jax.tree.structure(self.shape)
        self.multiple_results = isinstance(self.shape, tuple)
        a_vals = [ov.aval for ov in self.jaxpr.jaxpr.outvars]
        self.abstract_value = a_vals if self.multiple_results else a_vals[0]

    def abstract(self, *args, at: str, **_kwargs):
        return self.abstract_value

    def concrete(self, *args):
        v = jax.core.eval_jaxpr(self.jaxpr.jaxpr, self.jaxpr.consts, *args)
        return v if self.multiple_results else v[0]

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        v = Simulate(key, address, constraint).run(self.jaxpr, arg_tuple)
        if self.multiple_results:
            v["retval"] = jax.tree.unflatten(self.structure, v["retval"])
        return v

    def assess_p(
        self, arg_tuple: tuple, constraint: Constraint, address
    ) -> tuple[Array, Array]:
        a = Assess(address, constraint)
        value = a.run(self.jaxpr, arg_tuple)
        return value, a.score

    def __matmul__(self, address: str):
        return self.bind(*self.flat_args, at=address)

    def get_args(self) -> tuple:
        return self.args

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr


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

    def run(self, closed_jaxpr: jx.core.ClosedJaxpr, arg_tuple):
        jaxpr = closed_jaxpr.jaxpr
        flat_args, in_tree = jax.tree.flatten(arg_tuple)
        env: dict[jx.core.Var, Any] = {}

        def read(v: jax.core.Atom) -> Any:
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
            isinstance(eqn.primitive, GenPrimitive)
            or eqn.primitive is jax.lax.cond_p
            or eqn.primitive is jax.lax.scan_p
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
            if eqn.primitive.multiple_results:
                jax.util.safe_map(write, eqn.outvars, ans)
            else:
                write(eqn.outvars[0], ans)
            # clean_up_dead_vars(eqn, env, lu)
        retval = jax.util.safe_map(read, jaxpr.outvars)
        retval = retval if len(jaxpr.outvars) > 1 else retval[0]
        return self.construct_retval(retval)

    def apply_constraint(self, addr):
        if isinstance(self.constraint, dict):
            c = self.constraint
            for a in addr:
                if not isinstance(c, dict):
                    raise Exception("broken constraint: {a} is not a leaf")
                c = c.get(a)
                if c is None:
                    return None
            return c
        else:
            return self.constraint

    def construct_retval(self, retval) -> R:
        return retval


class Assess(Transformation[Array]):
    def __init__(self, address: Address, constraint: Constraint):
        super().__init__(PHANTOM_KEY, address, constraint)
        self.score = jnp.array(0.0)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, Distribution):
            addr = self.address + (bind_params["at"],)
            v = self.apply_constraint(addr)
            self.score += eqn.primitive.bind(v, *params[1:], op="Score")
            return v

        if isinstance(eqn.primitive, GenPrimitive):
            addr = self.address + (bind_params["at"],)
            ans, score = eqn.primitive.assess_p(params, self.constraint, addr)
            self.score += score
            return ans

        return super().handle_eqn(eqn, params, bind_params)


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

    def address_from_branch(self, b: jx.core.ClosedJaxpr):
        """Look at the given JAXPR and find out if it is a single-instruction
        call to a GF traced to an address. If so, return that address. This is
        used to detect when certain JAX primitives (e.g., `scan_p`, `cond_p`)
        have been applied directly to traced generative functions, in which case
        the current transformation should be propagated to the jaxpr within."""
        if len(b.jaxpr.eqns) == 1 and isinstance(b.jaxpr.eqns[0].primitive, GF):
            return b.jaxpr.eqns[0].params.get("at")

    def transform_inner(self, jaxpr, in_avals, address):
        """Apply simulate to jaxpr and return the transformed jaxpr together
        with its return shape."""

        return jax.make_jaxpr(
            lambda key, in_avals: Simulate(key, address, self.constraint).run(
                jaxpr, in_avals
            ),
            return_shape=True,
        )(PHANTOM_KEY, in_avals)

    def handle_eqn(self, eqn: jx.core.JaxprEqn, params, bind_params):
        if isinstance(eqn.primitive, RepeatGF):
            sub_key = self.get_sub_key()
            addr = self.address + (bind_params["at"],)
            transformed, shape = self.transform_inner(
                bind_params["inner"], params, addr
            )
            new_params = bind_params | {"inner": transformed}
            ans = RepeatGF.Simulate(eqn.primitive, shape).bind(
                sub_key, *params, **new_params
            )
            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            self.trace[bind_params["at"]] = u["subtraces"]
            self.w += jnp.sum(u.get("w", 0))
            return u["retval"]

        if isinstance(eqn.primitive, VmapGF):
            sub_key = self.get_sub_key()
            addr = self.address + (bind_params["at"],)
            transformed, shape = self.transform_inner(
                bind_params["inner"], eqn.primitive.reduced_avals(params), addr
            )
            new_params = bind_params | {"inner": transformed}
            ans = eqn.primitive.Simulate(eqn.primitive, shape).bind(
                sub_key, *params, **new_params
            )
            u = jax.tree.unflatten(jax.tree.structure(shape), ans)
            self.trace[bind_params["at"]] = u["subtraces"]
            self.w += jnp.sum(u.get("w", 0))
            return u["retval"]

        # we regard the presence of constraints as enabling importance mode,
        # so you can't do an importance over empty constraints to get a zero
        # score; instead, that will be interpreted as a regular simulate.
        if isinstance(eqn.primitive, Distribution) and self.constraint:
            addr = self.address + (bind_params["at"],)
            if (retval := self.apply_constraint(addr)) is not None:
                score = eqn.primitive.bind(retval, *params[1:], op="Score")
                self.w += jnp.sum(score)
                ans = {"score": score, "retval": retval}
            else:
                sub_key = self.get_sub_key()
                retval = eqn.primitive.bind(sub_key, *params[1:], op="Sample")
                ans = {"retval": retval}
            self.trace[bind_params["at"]] = ans
            return ans["retval"]

        if isinstance(eqn.primitive, GenPrimitive):
            sub_key = self.get_sub_key()
            addr = self.address + (bind_params["at"],)
            ans = eqn.primitive.simulate_p(sub_key, params, addr, self.constraint)
            self.trace[bind_params["at"]] = ans
            # TODO: this `get` comes about because if we are a Distribution, but
            # are not running in importance mode, there will be no 'w' sub-entry.
            # There are better ways to do this: (1) move this logic up so that one
            # case handles all of distribution, eliminating distribution's simulate_p
            # or move the logics of these to the containing classes (probably a better
            # idea).
            self.w += jnp.sum(ans.get("w", 0.0))
            return ans["retval"]

        if eqn.primitive is jax.lax.cond_p:
            sub_key = self.get_sub_key()
            branches = bind_params["branches"]

            branch_addresses = tuple(map(self.address_from_branch, branches))
            if branch_addresses[0] and all(
                b == branch_addresses[0] for b in branch_addresses[1:]
            ):
                # address = self.address + (branch_addresses[0],)
                address = self.address
                sub_address = branch_addresses[0]
            else:
                address = self.address
                sub_address = None

            # TODO: is it OK to pass the same sub_key to both sides?
            # NB! branches[0] is the false branch, [1] is the true branch,
            ans = jax.lax.cond(
                params[0],
                lambda: Simulate(sub_key, address, self.constraint).run(
                    branches[1], params[1:]
                ),
                lambda: Simulate(sub_key, address, self.constraint).run(
                    branches[0], params[1:]
                ),
            )

            if sub_address:
                self.trace[sub_address] = ans["subtraces"][sub_address]
            self.w += jnp.sum(ans.get("w", 0))
            # The reasons why this result has to be a sequence is obscure to me,
            # but cond_p as a primitive requires "multiple results."
            return (ans["retval"],)

        if eqn.primitive is jax.lax.scan_p:
            self.key, sub_key = KeySplit.bind(self.key, a="scan_p")
            inner = bind_params["jaxpr"]

            if a := self.address_from_branch(inner):
                address = self.address + (a,)
                sub_address = a
            else:
                address = self.address
                sub_address = None

            def step(carry_key, s):
                carry, key = carry_key
                key, k1 = KeySplit.bind(key, a="scan_step")
                v = Simulate(k1, address, self.constraint).run(inner, (carry, s))
                return ((v["retval"][0], key), v)

            ans = jax.lax.scan(step, (params[0], sub_key), params[1])
            if sub_address:
                self.trace[sub_address] = ans[1]["subtraces"][sub_address]

            # we extended the carry with the key; now drop it
            self.w += jnp.sum(ans[1].get("w", 0))
            return (ans[0][0], ans[1]["retval"][1])

        return super().handle_eqn(eqn, params, bind_params)

    def construct_retval(self, retval):
        r = {"retval": retval, "subtraces": self.trace}
        if self.constraint:
            r["w"] = self.w
        return r


# %%
def Cond(tf, ff):
    """Cond combinator. Turns (tf, ff) into a function of a boolean
    argument which will switch between the true and false branches."""

    def ctor(pred):
        pred_as_int = jnp.int32(pred)

        class Binder:
            def __matmul__(self, address: str):
                return jax.lax.switch(
                    pred_as_int, [lambda: ff @ address, lambda: tf @ address]
                )

            def simulate(self, key: PRNGKeyArray):
                return GF(lambda: self @ "__cond", ()).simulate(key)

        return Binder()

    return ctor


def Scan(gf: Gen):
    """Scan combinator. Turns a GF of two parameters `(state, update)`
    returning a pair of updated state and step data to record into
    a generative function of an initial state and an array of updates."""

    def ctor(init, steps):
        class Binder:
            def __matmul__(self, address: str):
                def inner(carry, step):
                    c, s = gf(carry, step) @ address
                    return c, s

                return jax.lax.scan(inner, init, steps)

        return Binder()

    return ctor


class RepeatGF[R](GFI[R]):
    def __init__(self, gfi: GFI[R], n: int):
        super().__init__(f"Repeat[{gfi.name}, {n}]")
        self.gfi = gfi
        self.n = n
        self.multiple_results = self.gfi.multiple_results
        # TODO: try to reuse self.__matmul__ here, if possible
        self.jaxpr, self.shape = jax.make_jaxpr(
            lambda *args: self.bind(
                *args, n=self.n, at="__r_gf", inner=self.gfi.get_jaxpr()
            ),
            return_shape=True,
        )(*self.get_args())

    def abstract(self, *args, **kwargs):
        return self.inflate(self.gfi.abstract(*args, **kwargs), self.n)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        return Simulate(key, address, constraint).run(self.get_jaxpr(), arg_tuple)

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.gfi.get_args(), at=address, n=self.n, inner=self.gfi.get_jaxpr()
        )

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    class Simulate[S](GFI[S]):
        def __init__(self, r: "RepeatGF[S]", shape: Any):
            super().__init__("Repeat.Simulate")
            self.r = r
            self.shape = shape
            self.multiple_results = True
            mlir.register_lowering(
                self, mlir.lower_fun(self.impl, self.multiple_results)
            )

        def abstract(self, *args, **kwargs):
            return [
                self.inflate(jax.core.get_aval(s), self.r.n)
                for s in jax.tree.flatten(self.shape)[0]
            ]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            return jax.vmap(
                lambda k: jax.core.eval_jaxpr(j.jaxpr, j.consts, k, *args[1:]),
                in_axes=(0,),
            )(jax.random.split(args[0], kwargs["n"]))


class VmapGF[R](GFI[R]):
    def __init__(self, g: Gen, arg_tuple: tuple, in_axes: InAxesT):
        super().__init__(f"Vmap[{g.f.__name__}]")
        if in_axes is None or in_axes == ():
            raise NotImplementedError(
                "must specify at least one argument/axis for Vmap"
            )
        # TODO: consider if we want to make this
        self.arg_tuple = arg_tuple
        self.flat_args, self.in_tree = jax.tree.flatten(self.arg_tuple)
        self.flat_args = tuple(self.flat_args)
        self.in_axes = in_axes
        # find one pair of (parameter number, axis) to use to determine size of vmap
        if isinstance(self.in_axes, tuple):
            self.p_index, self.an_axis = next(
                filter(lambda b: b[1] is not None, enumerate(self.in_axes))
            )
        else:
            self.p_index, self.an_axis = 0, self.in_axes
        # Compute the "scalar" jaxpr by feeding the un-v-mapped arguments to make_jaxpr
        # TODO: does `self` need to remember these?
        self.inner_jaxpr, self.inner_shape = jax.make_jaxpr(g.f, return_shape=True)(
            *jax.tree.unflatten(self.in_tree, self.reduced_avals(self.arg_tuple))
        )
        n = self.arg_tuple[self.p_index].shape[self.an_axis]
        # the shape of the v-mapped function will increase every axis of the result
        self.shape = jax.ShapeDtypeStruct(
            (n,) + self.inner_shape.shape, self.inner_shape.dtype
        )
        self.jaxpr, self.shape = jax.make_jaxpr(
            lambda *args: self.bind(
                *args, in_axes=self.in_axes, at="__vmap_gf", inner=self.inner_jaxpr
            ),
            return_shape=True,
        )(*self.flat_args)

    def reduced_avals(self, arg_tuple):
        # if in_axes is not an tuple, lift it to the same shape as arg tuple
        if isinstance(self.in_axes, int):
            ia = jax.tree.map(lambda _: self.in_axes, arg_tuple)
        else:
            ia = self.in_axes

        # Now produce an abstract arg tuple in which the shape of the
        # arrays is contracted in those position where vmap would expect
        # to find a mapping axis
        def deflate(array, axis):
            if axis is None:
                return array
            aval = jax.core.get_aval(array)
            # delete the indicated axes from teh shape tuple
            assert isinstance(aval, jax.core.ShapedArray)
            return aval.update(shape=aval.shape[:axis] + aval.shape[axis + 1 :])

        return jax.tree.map(deflate, self.flat_args, ia)

    def abstract(self, *args, **kwargs):
        return self.shape

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        # TODO: fishy: can we fix this?
        return GF(lambda: self @ "__vmap", ()).simulate_p(key, (), address, constraint)

    def __matmul__(self, address: str) -> R:
        return self.bind(
            *self.flat_args, at=address, in_axes=self.in_axes, inner=self.inner_jaxpr
        )

    def get_args(self) -> tuple:
        return self.arg_tuple

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        return self.jaxpr

    class Simulate[S](GFI[S]):
        def __init__(self, r: "VmapGF[S]", shape: Any):  # TODO: PyTreeDef?
            super().__init__("Vmap.Simulate")
            self.r = r
            self.n = self.r.arg_tuple[self.r.p_index].shape[self.r.an_axis]
            self.shape = shape
            self.multiple_results = True
            mlir.register_lowering(
                self, mlir.lower_fun(self.impl, self.multiple_results)
            )

        def abstract(self, *args, **kwargs):
            return [
                self.inflate(jax.core.get_aval(s), self.n)
                for s in jax.tree.flatten(self.shape)[0]
            ]

        def concrete(self, *args, **kwargs):
            # this is called after the simulate transformation so the key is the first argument
            j: jx.core.ClosedJaxpr = kwargs["inner"]
            return jax.vmap(
                lambda k, arg_tuple: jax.core.eval_jaxpr(
                    j.jaxpr, j.consts, k, *arg_tuple
                ),
                in_axes=(0, self.r.in_axes),
            )(jax.random.split(args[0], self.n), args[1:])


class MapGF[R, S](GFI[S]):
    def __init__(self, gfi: GFI[R], f: Callable[[R], S]):
        super().__init__(f"Map[{gfi.name}, {f.__name__}]")
        self.gfi = gfi
        self.f = f

    def abstract(self, *args, **kwargs):
        # this can't be right: what about the effect of f? Why isn't it
        # sufficient to apply f to a tracer to compose the abstraction?
        return self.gfi.abstract(*args, **kwargs)

    def simulate_p(
        self,
        key: PRNGKeyArray,
        arg_tuple: tuple,
        address: Address,
        constraint: Constraint,
    ) -> dict:
        v = self.gfi.simulate_p(key, arg_tuple, address, constraint)
        v["retval"] = self.f(v["retval"])
        return v

    def __matmul__(self, address: str) -> S:
        return self.f(self.gfi @ address)

    def get_args(self) -> tuple:
        return self.gfi.get_args()

    def get_jaxpr(self) -> jx.core.ClosedJaxpr:
        ij = self.gfi.get_jaxpr()
        return jax.make_jaxpr(
            lambda args: self.f(*jax.core.eval_jaxpr(ij.jaxpr, ij.consts, *args))
        )(self.gfi.get_args())

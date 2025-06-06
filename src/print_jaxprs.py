import jax
import jax.numpy as jnp
import minigenjax
from minigenjax.test_minigenjax import model1, model2, model3, cond_model


@minigenjax.Gen
def model2p(x, y):
    return minigenjax.Normal(x, y) @ "n"


def print_model_jaxprs():
    def print_sim_jaxpr(m, *args):
        key0 = jax.random.key(0)
        j = jax.make_jaxpr(m(*args).simulate)(key0)
        print("/* simulate */\n", j)
        print("-" * 72)
        try:
            tr = m(*args).simulate(key0)
            cm = minigenjax.to_constraint(tr)
            ja = jax.make_jaxpr(m(*args).assess)(cm)
            print("/* assess */\n", ja)
            print("-" * 72)
        except Exception:
            print("----- no assess yet ")
            pass

    print_sim_jaxpr(model1, 11.0)
    print_sim_jaxpr(model2, 12.0)
    print_sim_jaxpr(model3, 13.0)
    print_sim_jaxpr(cond_model, 14.0)
    print_sim_jaxpr(model3.vmap(), jnp.arange(15.0, 20.0))
    print_sim_jaxpr(model3.repeat(5), 21.0)
    print_sim_jaxpr(model3.repeat(3).repeat(2), 22.0)
    print_sim_jaxpr(model2p, 23.0, 24.0)
    print_sim_jaxpr(model2p.vmap(in_axes=(0, None)), jnp.arange(25.0, 36.0), 26.0)
    print_sim_jaxpr(
        model2p.vmap(in_axes=(0, None)).vmap(in_axes=(None, 0)),
        jnp.arange(25.0, 36.0),
        jnp.arange(0.0, 1.0, 0.2),
    )


if __name__ == "__main__":
    print_model_jaxprs()

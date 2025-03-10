import jax
import jax.numpy as jnp
import minigenjax
from minigenjax.test_minigenjax import model1, model2, model3, cond_model


@minigenjax.Gen
def model2p(x, y):
    return minigenjax.Normal(x, y) @ "n"


def print_model_jaxprs():
    def print_sim_jaxpr(m, *args):
        j = jax.make_jaxpr(m(*args).simulate)(jax.random.key(0))
        print("-" * 72)
        print(j)

    print_sim_jaxpr(model1, 11.0)
    print_sim_jaxpr(model2, 12.0)
    print_sim_jaxpr(model3, 13.0)
    print_sim_jaxpr(cond_model, 14.0)
    print_sim_jaxpr(model3.vmap(), jnp.arange(15.0, 20.0))
    print_sim_jaxpr(model3(21.0).repeat, 5)
    print_sim_jaxpr(model3(22.0).repeat(3).repeat, 4)
    print_sim_jaxpr(model2p, 23.0, 24.0)
    print_sim_jaxpr(model2p.vmap(in_axes=(0, None)), jnp.arange(25.0, 36.0), 26.0)


#    print_sim_jaxpr(model2p.vmap(in_axes=(0, None)).vmap(in_axes=(None, 0)), jnp.arange(25., 36.), jnp.arange(0.0, 1.0, 0.2))

if __name__ == "__main__":
    print_model_jaxprs()

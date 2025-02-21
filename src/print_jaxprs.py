import jax
import jax.numpy as jnp
import minigenjax
from test_minigenjax import model1, model2, model3, cond_model


def print_model_jaxprs():
    def print_sim_jaxpr(m, *args):
        j = jax.make_jaxpr(m(*args).simulate)(jax.random.key(0))
        print('-' * 72)
        print(j)

    print_sim_jaxpr(model1, 11.)
    print_sim_jaxpr(model2, 12.)
    print_sim_jaxpr(model3, 13.)
    print_sim_jaxpr(cond_model, 14.)
    print_sim_jaxpr(model3.vmap(), jnp.arange(15., 20.))
    print_sim_jaxpr(model3(21.).repeat, 5)



if __name__ == '__main__':
    print_model_jaxprs()

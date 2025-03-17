import pytest
from minigenjax import *
import jax.numpy as jnp
from jaxtyping import Array
import jax


@pytree
class Pose:
    p: Array
    hd: Array

    def rotate(self, a):
        return Pose(self.p, self.hd + a)

    def dp(self):
        return jnp.array([jnp.cos(self.hd), jnp.sin(self.hd)])

    def as_array(self):
        return jnp.append(self.p, self.hd)


def distance(p: Pose, seg: Array, PARALLEL_TOL=1.0e-6):
    """
    Computes the distance from a pose to a segment, considering the pose's direction.

    Args:
    - p: The Pose object.
    - seg: The segment [p1, p2].

    Returns:
    - float: The distance to the segment. Returns infinity if no valid intersection is found.
    """
    pdp = p.dp()
    segdp = seg[1] - seg[0]
    # Compute unique s, t such that p.p + s * pdp == seg[0] + t * segdp
    pq = p.p - seg[0]
    det = pdp[0] * segdp[1] - pdp[1] * segdp[0]
    st = jnp.where(
        jnp.abs(det) < PARALLEL_TOL,
        jnp.array([jnp.nan, jnp.nan]),
        jnp.array(
            [segdp[0] * pq[1] - segdp[1] * pq[0], pdp[0] * pq[1] - pdp[1] * pq[0]]
        )
        / det,
    )
    return jnp.where((st[0] >= 0) & (st[1] >= 0) & (st[1] <= 1), st[0], jnp.inf)


def sensor_distance(pose, walls, box_size):
    d = jnp.min(jax.vmap(distance, in_axes=(None, 0))(pose, walls))
    # Capping to a finite value avoids issues below.
    return jnp.where(jnp.isinf(d), 2 * box_size, d)


walls = jnp.array(
    [
        [[-1.0, -1.0], [-1.0, 1.0]],
        [[-1.0, 1.0], [1.0, 1.0]],
        [[1.0, 1.0], [1.0, -1.0]],
        [[1.0, -1.0], [-1.0, -1.0]],
    ]
)


def test_localization():
    sensor_fov = jnp.pi
    sensor_angles = jnp.arange(-10, 11) / sensor_fov
    sensor_range = 2.0
    sensor_noise = 0.1
    key = jax.random.key(0)

    @Gen
    def sensor_model_one(pose, angle, sensor_noise):
        return (
            Normal(
                sensor_distance(pose.rotate(angle), walls, sensor_range),
                sensor_noise,
            )
            @ "distance"
        )

    @Gen
    def uniform_pose(mins, maxes):
        p_array = Uniform(mins, maxes) @ "p_array"
        return Pose(p_array[0:2], p_array[2])

    whole_map_prior = uniform_pose(
        jnp.array([-1.0, -1.0, -jnp.pi]), jnp.array([1.0, 1.0, jnp.pi])
    )

    sensor_model = sensor_model_one.vmap(in_axes=(None, 0, None))

    def readings_at_pose(pose):
        return jax.vmap(
            lambda angle: sensor_distance(pose.rotate(angle), walls, sensor_range)
        )(sensor_angles)

    @Gen
    def joint_model():
        pose = whole_map_prior @ "pose"
        sensor = sensor_model(pose, sensor_angles, sensor_noise) @ "sensor"

    key, sub_key = jax.random.split(key)
    target = joint_model().simulate(sub_key)
    target_pose = target["subtraces"]["pose"]["retval"]
    target_readings = target["subtraces"]["sensor"]["subtraces"]["distance"]["retval"]

    print(f"chm {to_constraint(target)}")
    print(f"target_pose {target_pose}")
    print(f"target_readings {target_readings}")

    key, sub_key = jax.random.split(key)
    poses = jax.vmap(whole_map_prior.simulate)(jax.random.split(sub_key, 10))
    print(f'poses {poses["retval"]}')

    print(jax.make_jaxpr(joint_model().assess)(to_constraint(target)))

    def trial(pose, target_readings):
        cm = {
            "pose": {"p_array": pose.as_array()},
            "sensor": {"distance": target_readings},
        }
        print(cm)
        return joint_model().assess(cm)

    j_trial = jax.jit(trial)

    print("t1", j_trial(Pose(jnp.array([0.0, 0.0]), 0.0), target_readings))
    print("t2", j_trial(Pose(jnp.array([0.5, 0.5]), 0.2), target_readings))
    print("t3", j_trial(Pose(jnp.array([0.12, 0.9]), -1.98), target_readings))
    print(poses["retval"])
    # print(jax.vmap(trial)(poses["retval"]))
    print("tp0", trial(poses["retval"][0], target_readings))
    print("tp1", trial(poses["retval"][1], target_readings))
    print(
        "tpx",
        jax.vmap(lambda pose: j_trial(pose, target_readings))(poses["retval"][:2]),
    )
    return

    scores = jax.vmap(trial)(poses["retval"])
    print(scores)

    # target_pose = whole_map_prior.simulate(sub_key)["retval"]
    # target_readings = readings_at_pose(target_pose)

    # observations = {"pose": {"p_array": target_pose.as_array()}, "sensor": {"distance": target_readings}}
    # print(f'observations {observations}')
    # key, sub_key = jax.random.split(key)
    # tr = joint_model().simulate(sub_key)
    # print(to_constraint(tr))
    # w, retval = joint_model().assess(observations)
    # print(w)
    # print(retval)
    # w, retval = joint_model().assess()

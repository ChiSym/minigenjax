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

    key, sub_key = jax.random.split(key)
    some_pose = whole_map_prior.simulate(sub_key)["retval"]
    tr = jax.jit(whole_map_prior.simulate)(sub_key)
    assert jnp.allclose(some_pose.hd, tr["retval"].hd)
    assert jnp.allclose(some_pose.p, tr["retval"].p)
    sensor_model = sensor_model_one.vmap(in_axes=(None, 0, None))

    key, sub_key = jax.random.split(key)
    tr = sensor_model(some_pose, sensor_angles, sensor_noise).simulate(sub_key)
    assert jnp.allclose(
        tr["retval"],
        jnp.array(
            [
                1.8319645,
                1.8249748,
                2.3729553,
                2.1042943,
                1.7308967,
                1.3865273,
                0.5385731,
                0.15848322,
                0.23405387,
                0.3614612,
                0.26093096,
                0.38295925,
                0.36778474,
                0.06698091,
                0.25884473,
                0.24397819,
                0.15396227,
                0.4171125,
                0.643867,
                1.6671362,
                1.8732764,
            ],
        ),
    )

    def readings_at_pose(pose):
        return jax.vmap(
            lambda angle: sensor_distance(pose.rotate(angle), walls, sensor_range)
        )(sensor_angles)

    @Gen
    def joint_model():
        pose = whole_map_prior @ "pose"
        _ = sensor_model(pose, sensor_angles, sensor_noise) @ "sensor"

    key, sub_key = jax.random.split(key)
    target = joint_model().simulate(sub_key)
    target_pose = target["subtraces"]["pose"]["retval"]
    target_readings = target["subtraces"]["sensor"]["subtraces"]["distance"]["retval"]

    key, sub_key = jax.random.split(key)
    poses = jax.vmap(whole_map_prior.simulate)(jax.random.split(sub_key, 50000))[
        "retval"
    ]

    def trial(pose, target_readings):
        cm = {
            "pose": {"p_array": pose.as_array()},
            "sensor": {"distance": target_readings},
        }
        return joint_model().assess(cm)[0]

    scores = jax.vmap(trial, in_axes=(0, None))(poses, target_readings)
    key, sub_key = jax.random.split(key)
    winners = jax.vmap(Categorical(logits=scores).sample)(jax.random.split(sub_key, 10))
    winning_poses = jax.tree.map(lambda v: v[winners], poses)
    print(target_pose, winning_poses)

    @Gen
    def step_model(motion_settings, start, control):
        p = (
            MvNormalDiag(
                start.p + control.ds * start.dp(),
                motion_settings["p_noise"] * jnp.ones(2),
            )
            @ "p"
        )
        hd = Normal(start.hd + control.dhd, motion_settings["hd_noise"]) @ "hd"
        return Pose(p, hd)

    @pytree
    class Control:
        ds: Array
        dhd: Array

    robot_inputs = {
        "start": some_pose,
        "controls": Control(
            ds=jnp.array([0.1, 0.2, 0.2]), dhd=jnp.array([0.3, -0.4, 0.1])
        ),
    }

    motion_settings = {"p_noise": 0.05, "hd_noise": 0.06}

    @Gen
    def path_model(motion_settings):
        @Gen
        def step(motion_settings, start, control):
            s = step_model(motion_settings, start, control) @ "step"
            return s, s

        return (
            step.partial(motion_settings).scan()(
                robot_inputs["start"], robot_inputs["controls"]
            )
            @ "steps"
        )

    key, sub_key = jax.random.split(key)
    tr = path_model(motion_settings).simulate(sub_key)
    assert tr["retval"] == 999.0

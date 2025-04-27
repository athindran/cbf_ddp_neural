import jax
import jax.numpy as jnp
import mediapy as media
import sys
import functools

from brax.training.agents.ppo import train as ppo
from brax.io import model
from brax import envs
from brax_utils import WrappedBraxEnv, LinearPolicy, ReacherRegularizedGoalCost

# RL trained policy
def get_neural_policy():
    make_inference_fn, params, _ = train_fn(environment=brax_env.env)
    eval_env = envs.create(env_name=env_name, backend=backend)
    params = model.load_params(f'./brax_utils/trained_models/{env_name}_params')
    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)
    return jit_inference_fn

sys.path.append(".")

env_name = 'reacher'  # @param ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup', 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
backend = 'generalized'  # @param ['generalized', 'positional', 'spring']
brax_env = WrappedBraxEnv(env_name, backend)
rng = jax.random.PRNGKey(seed=1)
state = brax_env.reset(rng=rng)
print(f"State: {state}")

train_fn = {
  'reacher': functools.partial(ppo.train, num_timesteps=1, num_evals=1, reward_scaling=5, episode_length=1000, normalize_observations=True, action_repeat=4, unroll_length=50, num_minibatches=1, num_updates_per_batch=8, discounting=0.95, learning_rate=3e-4, entropy_cost=1e-3, num_envs=1, batch_size=256, max_devices_per_host=8, seed=1),
}[env_name]

policy = get_neural_policy()

rollout = []
for _ in range(500):
  rollout.append(state.pipeline_state)
  act_rng, rng = jax.random.split(rng)
  act, _ = policy(state.obs, act_rng)
  state = brax_env.step(state, act)
  gc_from_state_grad, gc_from_action_grad = brax_env.get_generalized_coordinates_grad(state, act)
  obs_from_gc_grad = brax_env.get_obs_grad(state.pipeline_state)
  #print(gc_from_state_grad, obs_from_gc_grad, gc_from_action_grad)

render_every = 2
media.write_video(f'./brax_videos/reacher_linear_policy.mp4',
    brax_env.env.render(rollout[::render_every]),
    fps=1.0 / brax_env.env.dt / render_every)

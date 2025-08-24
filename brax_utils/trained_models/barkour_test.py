#@title Import MuJoCo, MJX, and Brax

import functools
import jax
import numpy as np
import mujoco
import mediapy as media

from jax import numpy as jp
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model
from brax_utils import BarkourEnv
from matplotlib import pyplot as plt
from mujoco import mjx

envs.register_environment('barkour', BarkourEnv)

env_name = 'barkour'
eval_env = envs.get_environment(env_name)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
train_fn = functools.partial(
      ppo.train, num_timesteps=1, num_evals=1,
      reward_scaling=1, episode_length=1, normalize_observations=True,
      action_repeat=1, unroll_length=2, num_minibatches=1,
      num_updates_per_batch=4, discounting=0.97, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=1, batch_size=256,
      network_factory=make_networks_factory, seed=0)

make_inference_fn, _, _ = train_fn(environment=eval_env)

model_path = './mjx_brax_quadruped_policy'
params = model.load_params(model_path)
inference_fn = make_inference_fn(params, deterministic=True)
jit_inference_fn = jax.jit(inference_fn)

jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
     

# @markdown Commands **only used for Barkour Env**:
x_vel = 1.0  #@param {type: "number"}
y_vel = 0.0  #@param {type: "number"}
ang_vel = -0.5  #@param {type: "number"}

the_command = jp.array([x_vel, y_vel, ang_vel])

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info['command'] = the_command
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 600
render_every = 2

ctrls = np.zeros((12, n_steps))

for idx in range(n_steps):
  #act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, rng)
  ctrls[:, idx] = ctrl
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

fig, axes = plt.subplots(3, 4, figsize=(22, 12))
range_space = np.arange(n_steps)
axes = axes.ravel()
linewidth = 1.0
legend_fontsize = 8
for idx in range(12):
    axes[idx].plot(range_space, ctrls[idx, :], linewidth=linewidth)
    axes[idx].set_xlabel('Time (s)', 
                fontsize=legend_fontsize)
    axes[idx].set_ylabel('Control ' + str(idx), 
                fontsize=legend_fontsize)

fig.savefig('barkour_test.png', bbox_inches='tight', dpi=300)
media.write_video(f'./brax_utils/videos/{env_name}.mp4',
    eval_env.render(rollout[::render_every], camera='default'),
    fps=1.0 / eval_env.dt / render_every)
     

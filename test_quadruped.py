import jax
from jax import numpy as jnp
from jax import random
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from etils import epath

from simulators import BarkourEnv

env = BarkourEnv()
rng = jax.random.PRNGKey(0)
state = env.reset(rng)

for _ in range(10):
    ctrl = random.uniform(rng, shape=(12,))
    state = env.step(state, ctrl)
    print(state.pipeline_state)
    jac_state, jac_ctrl = env.get_jacobian(jnp.expand_dims(state.pipeline_state, 0), jnp.expand_dims(ctrl, 0))



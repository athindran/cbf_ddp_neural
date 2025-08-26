from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost

import numpy as np
import jax.numpy as jnp
import jax

class BarkourObstacleAvoidanceConstraintCost(BaseMargin):
    def __init__(self, config, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.dim_q_states = env.dim_q_states
        self.dim_qd_states = env.dim_qd_states
        self.obstacles = [[2.0, 2.0, 0.5], [-2.0, 2.0, 0.5], [2.0, -2.0, 0.5], [-2.0, -2.0, 0.5]]
        self.obs_margins = []
        for obs_idx in range(4):
            self.obs_margins.append(CircleObsMargin(circle_spec = np.asarray(self.obstacles[obs_idx]), buffer=0.0))

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        cost = jnp.inf

        for idx in range(4):
            cost = jnp.minimum(cost, self.obs_margins[idx].get_stage_margin(state, ctrl)**2 - 0.25)

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """

        return self.get_stage_margin(state, ctrl)


class BarkourHardConstraintCost(BaseMargin):
    def __init__(self, config, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.dim_q_states = env.dim_q_states
        self.dim_qd_states = env.dim_qd_states
        self.obstacles = [[2.0, 2.0, 0.5], [-2.0, 2.0, 0.5], [2.0, -2.0, 0.5], [-2.0, -2.0, 0.5]]
        self.obs_margins = []
        for obs_idx in range(4):
            self.obs_margins.append(CircleObsMargin(circle_spec = np.asarray(self.obstacles[obs_idx]), buffer=0.0))

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        cost = jnp.inf

        for idx in range(4):
            cost = jnp.minimum(cost, self.obs_margins[idx].get_stage_margin(state, ctrl)**2 - 0.25)

        cost = jnp.minimum(cost, jnp.floor(100*(state[2] - 0.05)))

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """
        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        # Slow down to a small squared velocity of 0.5 while staying up.
        cost = jnp.inf

        for idx in range(4):
            cost = jnp.minimum(cost, self.obs_margins[idx].get_stage_margin(state, ctrl)**2 - 0.25)
        cost = jnp.minimum(cost, 0.5 - state[18]**2 - state[19]**2)
        cost = jnp.minimum(cost, jnp.floor(100*(state[2] - 0.05)))

        return cost


class BarkourReachabilityMargin(BaseMargin):

    def __init__(self, config, env: WrappedBraxEnv, filter_type: str ='CBF'):
        super().__init__()
        if config.COST_TYPE=='Reachability':
            self.constraint = BarkourObstacleAvoidanceConstraintCost(config, env)
        elif config.COST_TYPE=='Reachavoid':
            self.constraint = BarkourHardConstraintCost(config, env)
        R = jnp.diag(jnp.array(config.W_ctrl))
        self.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(env.dim_u))
        self.constraint.ctrl_cost = QuadraticControlCost(
            R=R, r=jnp.zeros(env.dim_u))
        self.N = config.N

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        state_cost = self.constraint.get_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return state_cost + ctrl_cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        target_cost = self.constraint.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost

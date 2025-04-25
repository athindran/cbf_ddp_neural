from functools import partial

from jax import Array
import jax.numpy as jnp
import jax

from .base_margin import BaseMargin
from .obs_margin import CircleObsMargin
from .quadratic_penalty import QuadraticControlCost

class ReacherGoalCost(BaseMargin):
    def __init__(self, dim_u: int, goal_spec: jnp.ndarray):
        super().__init__()
        self.dim_u = dim_u
        self.goal_spec = goal_spec
        self.goal_reaching_cost = CircleObsMargin(circle_spec=goal_spec, buffer=0.0)

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, obs: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            obs (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        return -1*self.goal_reaching_cost(obs, ctrl)

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, obs: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            obs (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        return -1*self.goal_reaching_cost(obs, ctrl)

class ReacherRegularizedGoalCost(BaseMargin):
    def __init__(self, dim_u: int, dim_x: int, goal_spec: jnp.ndarray, ctrl_cost_matrix: jnp.ndarray):
        super().__init__()
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.goal_cost = ReacherGoalCost(dim_u, goal_spec)
        self.ctrl_cost = QuadraticControlCost(R = ctrl_cost_matrix, r = jnp.zeros(dim_u))

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
        state_cost = self.goal_cost.get_stage_margin(
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
        target_cost = self.goal_cost.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost

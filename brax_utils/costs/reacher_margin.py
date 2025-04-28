from functools import partial

from brax_utils import WrappedBraxEnv
from jax import Array
import jax.numpy as jnp
import jax

from simulators import BaseMargin, CircleObsMargin, QuadraticControlCost
from simulators import LowerHalfMargin, UpperHalfMargin

class ReacherGoalCost(BaseMargin):
    def __init__(self, center: Array, env: WrappedBraxEnv):
        super().__init__()
        self.center = center        
        self.env = env

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        fingertip = self.env.get_fingertip(state)
        cost = (self.center[0] - fingertip[0])**2 + (self.center[1] - fingertip[1])**2

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state, ctrl
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        return self.get_state_margin(state, ctrl)


class ReacherRegularizedGoalCost(BaseMargin):
    def __init__(self, center: Array, ctrl_cost_matrix: jnp.ndarray, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        self.goal_cost = ReacherGoalCost(center, env)
        self.ctrl_cost = QuadraticControlCost(R = ctrl_cost_matrix, r = jnp.zeros(self.dim_u))

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: Array, ctrl: Array
    ) -> Array:
        """

        Args:
            state aka generalized coordinates (Array, vector shape)
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
            state aka generalized coordinates (Array, vector shape)
            ctrl (Array, vector shape)

        Returns:
            Array: scalar.
        """
        target_cost = self.goal_cost.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost

class ReacherConstraintCost(BaseMargin):
    def __init__(self, config, env: WrappedBraxEnv):
        super().__init__()
        self.dim_x = env.dim_x
        self.dim_u = env.dim_u
        max_angular_velocity = config.MAX_ANGULAR_VELOCITY
        self.q0d_lower_cost = LowerHalfMargin(value=-1 * max_angular_velocity[0], buffer=0.0, dim=2)
        self.q1d_lower_cost = LowerHalfMargin(value=-1 * max_angular_velocity[1], buffer=0.0, dim=3)
        self.q0d_upper_cost = UpperHalfMargin(value=max_angular_velocity[0], buffer=0.0, dim=2)
        self.q1d_upper_cost = UpperHalfMargin(value=max_angular_velocity[1], buffer=0.0, dim=3)
        self.kappa = config.SMOOTHING_TEMP

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
        # cost = jnp.inf
        # cost = jnp.minimum(cost, state[2]**2 - self.max_angular_velocity_squared[0])
        # cost = jnp.minimum(cost, state[3]**2 - self.max_angular_velocity_squared[1])
        cost = 0.0
        cost += jnp.exp(-1 * self.kappa * self.q0d_lower_cost.get_stage_margin(
                state, ctrl
            ))

        cost += jnp.exp(-1 * self.kappa * self.q1d_lower_cost.get_stage_margin(
                state, ctrl
            ))

        cost += jnp.exp(-1 * self.kappa * self.q0d_upper_cost.get_stage_margin(
                state, ctrl
            ))

        cost += jnp.exp(-1 * self.kappa * self.q1d_upper_cost.get_stage_margin(
                state, ctrl
            ))

        cost = -jnp.log(cost)/self.kappa

        # cost = jnp.minimum(
        #     cost,
        #     self.q0d_lower_cost.get_stage_margin(
        #         state, ctrl
        #     )
        # )

        # cost = jnp.minimum(
        #     cost,
        #     self.q1d_lower_cost.get_stage_margin(
        #         state, ctrl
        #     )
        # )

        # cost = jnp.minimum(
        #     cost,
        #     self.q0d_upper_cost.get_stage_margin(
        #         state, ctrl
        #     )
        # )

        # cost = jnp.minimum(
        #     cost,
        #     self.q1d_upper_cost.get_stage_margin(
        #         state, ctrl
        #     )
        # )

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
        return self.stage_margin(state, ctrl)

class ReacherReachabilityMargin(BaseMargin):

    def __init__(self, config, env: WrappedBraxEnv, filter_type: str ='CBF'):
        super().__init__()
        # Removing the square
        # if filter_type == 'SoftCBF' or filter_type=='SoftLR':
        #     self.constraint = Bicycle5DSoftConstraintMargin(config, env)
        # else:
        self.constraint = ReacherConstraintCost(config, env)

        if env.dim_u == 2:
            R = jnp.array([[config.W_1, 0.0], [0.0, config.W_2]])
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

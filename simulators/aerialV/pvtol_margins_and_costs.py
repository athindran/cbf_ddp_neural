from typing import Dict
from functools import partial

from jax import Array as DeviceArray
import jax.numpy as jnp
import jax

from simulators.costs.base_margin import BaseMargin, SoftBarrierEnvelope
from simulators.costs.obs_margin import CircleObsMargin
from simulators.costs.quadratic_penalty import QuadraticControlCost, QuadraticStateCost
from simulators.costs.half_space_margin import LowerHalfMargin, UpperHalfMargin

class Pvtol6DCost(BaseMargin):

    def __init__(self, config, plan_dyn):
        super().__init__()
        self.R = jnp.array([[config.W_CTRL_X, 0.0], [0.0, config.W_CTRL_Y]])
        state_ref = jnp.zeros((plan_dyn.dim_x, ))
        control_ref = jnp.array([0.0, plan_dyn.mass*plan_dyn.g])
        self.ctrl_cost = QuadraticControlCost(ref=control_ref, R=self.R, r=jnp.zeros(plan_dyn.dim_u))
        self.Q =jnp.diag(jnp.array([config.W_STATE_X, config.W_STATE_Y, config.W_STATE_THETA, config.W_STATE_XD, config.W_STATE_YD, config.W_STATE_THETAD]))
        self.state_cost = QuadraticStateCost(ref=state_ref, Q=self.Q, q=jnp.zeros(plan_dyn.dim_x))
        #self.dyn = plan_dyn

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """

        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        cost = -1*self.state_cost.get_stage_margin(state, ctrl)
        cost += -1*self.ctrl_cost.get_stage_margin(state, ctrl)

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """

        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        return self.get_stage_margin(
            state, ctrl
        )

class Pvtol6DConstraintMargin(BaseMargin):
    def __init__(self, config, plan_dyn):
        super().__init__()
        # System parameters.
        self.ego_radius = config.EGO_RADIUS

        # Safety cost function parameters.
        self.width_right = config.WIDTH_RIGHT
        self.width_left = config.WIDTH_LEFT
        self.kappa = config.SMOOTHING_TEMP

        self.obs_spec = config.OBS_SPEC
        self.obsc_type = config.OBSC_TYPE
        self.plan_dyn = plan_dyn

        self.dim_x = plan_dyn.dim_x
        self.dim_u = plan_dyn.dim_u

        self.obs_constraint = []
        if self.obsc_type == 'circle':
            for circle_spec in self.obs_spec:
                self.obs_constraint.append(
                    CircleObsMargin(
                        circle_spec=circle_spec, buffer=config.EGO_RADIUS
                    )
                )

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        cost = jnp.inf

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            cost = jnp.minimum(
                cost, _obs_constraint.get_stage_margin(
                    state, ctrl))

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_constraint_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        cost = jnp.inf

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            cost = jnp.minimum(
                cost, _obs_constraint.get_stage_margin(
                    state, ctrl))

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        return self.get_stage_margin(
            state, ctrl
        )

    @partial(jax.jit, static_argnames='self')
    def get_safety_metric(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_constraint_margin(
            state, ctrl
        )

    @partial(jax.jit, static_argnames='self')
    def get_cost_dict(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> Dict:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        obs_cons = jnp.inf
        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            obs_cons = jnp.minimum(
                obs_cons, _obs_constraint.get_stage_margin(
                    state, ctrl))

        return dict(
            obs_cons=obs_cons
        )

class Pvtol6DSoftConstraintMargin(Pvtol6DConstraintMargin):
    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        cost = 0

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            cost += jnp.exp(-1 * self.kappa * _obs_constraint.get_stage_margin(state, ctrl))

        cost = -jnp.log(cost)/self.kappa

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        return self.get_stage_margin(
            state, ctrl
        )

    @partial(jax.jit, static_argnames='self')
    def get_cost_dict(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> Dict:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        obs_cons = jnp.inf
        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            obs_cons = jnp.minimum(
                obs_cons, _obs_constraint.get_stage_margin(
                    state, ctrl))

        return dict(
            obs_cons=obs_cons
        )

class PvtolReachAvoid6DMargin(BaseMargin):

    def __init__(self, config, plan_dyn, filter_type):
        super().__init__()
        # Removing the square
        if filter_type == 'SoftCBF' or filter_type=='SoftLR':
            self.constraint = Pvtol6DSoftConstraintMargin(config, plan_dyn)
        else:
            self.constraint = Pvtol6DConstraintMargin(config, plan_dyn)

        if plan_dyn.dim_u == 2:
            R = jnp.array([[config.W_1, 0.0], [0.0, config.W_2]])
        control_ref = jnp.array([0.0, plan_dyn.mass*plan_dyn.g])
        self.ctrl_cost = QuadraticControlCost(ref=control_ref, R=R, r=jnp.zeros(plan_dyn.dim_u))
        self.constraint.ctrl_cost = QuadraticControlCost(
            R=R, r=jnp.zeros(plan_dyn.dim_u))
        self.N = config.N

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """

        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        state_cost = self.constraint.get_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)
        return state_cost + ctrl_cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """

        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """
        target_cost = self.constraint.get_target_stage_margin(
            state, ctrl
        )
        ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)

        return target_cost + ctrl_cost

    # @partial(jax.jit, static_argnames='self')
    # def get_target_stage_margin_with_derivative(
    #     self, state: DeviceArray, ctrl: DeviceArray
    # ) -> DeviceArray:
    #     """

    #     Args:
    #         state (DeviceArray, vector shape)
    #         ctrl (DeviceArray, vector shape)

    #     Returns:
    #         DeviceArray: scalar.
    #     """
    #     target_cost, c_x_target, c_xx_target = self.constraint.get_target_stage_margin_with_derivatives(
    #         state, ctrl
    #     )
    #     ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)
    #     c_u_target = self.ctrl_cost.get_cu(
    #         state[:, jnp.newaxis], ctrl[:, jnp.newaxis])[:, -1]
    #     c_uu_target = self.ctrl_cost.get_cuu(
    #         state[:, jnp.newaxis], ctrl[:, jnp.newaxis])[:, :, -1]

    #     return target_cost + ctrl_cost, c_x_target, c_xx_target, c_u_target, c_uu_target

    # UNUSED FUNCTION
    @partial(jax.jit, static_argnames='self')
    def get_traj_cost(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> float:
        state_costs = self.constraint.get_stage_margin(
            state, ctrl
        )

        ctrl_costs = self.ctrl_cost.get_stage_margin(state, ctrl)
        # TODO: critical points version

        return (jnp.min(state_costs[1:]) + jnp.sum(ctrl_costs)).astype(float)

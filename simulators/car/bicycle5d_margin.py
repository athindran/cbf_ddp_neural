from typing import Dict
from functools import partial

from jax import Array as DeviceArray
import jax.numpy as jnp
import jax

from simulators.costs.base_margin import BaseMargin, SoftBarrierEnvelope
from simulators.costs.obs_margin import CircleObsMargin, BoxObsMargin, SoftBoxObsMargin, EllipseObsMargin
from simulators.costs.quadratic_penalty import QuadraticControlCost
from simulators.costs.half_space_margin import LowerHalfMargin, UpperHalfMargin


# Task policy ILQR cost. This is a random ad-hoc policy that is not meant to be principled.
class Bicycle5DCost(BaseMargin):

    def __init__(self, config, plan_dyn):
        super().__init__()

        # Lagrange cost parameters.
        self.v_ref = config.V_REF  # reference velocity.

        self.w_vel = config.W_VEL
        self.w_accel = config.WT_ACCEL
        self.w_omega = config.WT_OMEGA
        self.w_track = config.W_TRACK

        # Soft constraint parameters.
        self.q1_v = config.Q1_V
        self.q2_v = config.Q2_V
        self.q1_yaw = config.Q1_YAW
        self.q2_yaw = config.Q2_YAW

        self.v_min = config.V_MIN
        self.v_max = config.V_MAX
        self.yaw_min = config.YAWT_MIN
        self.yaw_max = config.YAWT_MAX

        self.barrier_clip_min = config.BARRIER_CLIP_MIN
        self.barrier_clip_max = config.BARRIER_CLIP_MAX
        self.buffer = getattr(config, "BUFFER", 0.)

        self.dim_x = plan_dyn.dim_x
        self.dim_u = plan_dyn.dim_u

        if self.dim_x < 7:
            self.vel_max_barrier_cost = SoftBarrierEnvelope(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
                UpperHalfMargin(value=self.v_max, buffer=0, dim=2)
            )
            self.vel_min_barrier_cost = SoftBarrierEnvelope(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
                LowerHalfMargin(value=self.v_min, buffer=0, dim=2)
            )
        else:
            self.vel_max_barrier_cost = SoftBarrierEnvelope(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
                UpperHalfMargin(value=self.v_max, buffer=0, dim=3)
            )
            self.vel_min_barrier_cost = SoftBarrierEnvelope(
                self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
                LowerHalfMargin(value=self.v_min, buffer=0, dim=3)
            )

        if self.dim_x < 7:
            self.yaw_min_cost = LowerHalfMargin(
                value=self.yaw_min, buffer=0, dim=3)
            self.yaw_max_cost = UpperHalfMargin(
                value=self.yaw_max, buffer=0, dim=3)
        else:
            self.yaw_min_cost = LowerHalfMargin(
                value=self.yaw_min, buffer=0, dim=4)
            self.yaw_max_cost = UpperHalfMargin(
                value=self.yaw_max, buffer=0, dim=4)

        self.yaw_min_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw, self.q2_yaw,
            self.yaw_min_cost
        )

        self.yaw_max_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_yaw, self.q2_yaw,
            self.yaw_max_cost
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
        # control cost

        if self.dim_u == 2:
            cost = self.w_accel * ctrl[0]**2 + self.w_omega * ctrl[1]**2
        elif self.dim_u == 3:
            cost = self.w_accel * \
                ctrl[0]**2 + self.w_omega * \
                ctrl[1]**2 + self.w_omega * ctrl[2]**2
        elif self.dim_u == 4:
            cost = self.w_accel * ctrl[0]**2 + self.w_accel * ctrl[1]**2 + \
                self.w_omega * ctrl[2]**2 + self.w_omega * ctrl[3]**2

        # state cost
        if self.dim_x < 7:
            cost += self.w_vel * (state[2] - self.v_ref)**2
        else:
            cost += self.w_vel * \
                (state[2] - self.v_ref)**2 + \
                self.w_vel * (state[3] - self.v_ref)**2

        cost += self.w_track * state[1]**2

        # soft constraint cost
        # cost += self.vel_max_barrier_cost.get_stage_margin(state, ctrl)
        # cost += self.vel_min_barrier_cost.get_stage_margin(state, ctrl)
        cost += self.yaw_max_barrier_cost.get_stage_margin(state, ctrl)
        cost += self.yaw_min_barrier_cost.get_stage_margin(state, ctrl)

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


# Hard constraint margin function.
class Bicycle5DConstraintMargin(BaseMargin):
    def __init__(self, config, plan_dyn):
        super().__init__()
        # System parameters.
        self.ego_radius = config.EGO_RADIUS

        # Racing cost parameters.
        self.w_accel = config.W_ACCEL
        self.w_omega = config.W_OMEGA
        self.track_width_right = config.TRACK_WIDTH_RIGHT
        self.track_width_left = config.TRACK_WIDTH_LEFT
        self.kappa = config.SMOOTHING_TEMP

        # Constraints toggling.
        self.use_yaw = getattr(config, 'USE_YAW', False)
        self.use_vel = getattr(config, 'USE_VEL', False)
        self.use_road = getattr(config, 'USE_ROAD', False)
        self.use_delta = getattr(config, 'USE_DELTA', False)
        self.use_track_exit = False
        self.yaw_min = getattr(config, 'YAW_MIN', -1.8)
        self.yaw_max = getattr(config, 'YAW_MAX', 1.8)
        self.delta_min = getattr(config, 'DELTA_MIN', -1.6)
        self.delta_max = getattr(config, 'DELTA_MAX', 1.6)

        self.obs_spec = config.OBS_SPEC
        self.obsc_type = config.OBSC_TYPE
        self.plan_dyn = plan_dyn

        self.dim_x = plan_dyn.dim_x
        self.dim_u = plan_dyn.dim_u

        # for temporary visualization
        self.target_constraint = CircleObsMargin(circle_spec=[4.1, 0.7, 0.4], buffer=0.0)

        self.obs_constraint = []
        if self.obsc_type == 'circle':
            for circle_spec in self.obs_spec:
                self.obs_constraint.append(
                    CircleObsMargin(
                        circle_spec=circle_spec, buffer=config.EGO_RADIUS
                    )
                )
        elif self.obsc_type == 'box':
            for box_spec in self.obs_spec:
                self.obs_constraint.append(
                    SoftBoxObsMargin(box_spec=box_spec, buffer=config.EGO_RADIUS)
                )
        elif self.obsc_type == 'ellipse':
            for ellipse_spec in self.obs_spec:
                self.obs_constraint.append(
                    EllipseObsMargin(ellipse_spec=ellipse_spec, buffer=config.EGO_RADIUS)
                )

        self.road_position_min_cost = LowerHalfMargin(
            value=-1 * config.TRACK_WIDTH_LEFT, buffer=config.EGO_RADIUS, dim=1)
        self.road_position_max_cost = UpperHalfMargin(
            value=config.TRACK_WIDTH_RIGHT, buffer=config.EGO_RADIUS, dim=1)

        if self.use_yaw:
            if plan_dyn.dim_x < 7:
                self.yaw_min_cost = LowerHalfMargin(
                    value=self.yaw_min, buffer=0, dim=3)
                self.yaw_max_cost = UpperHalfMargin(
                    value=self.yaw_max, buffer=0, dim=3)
            else:
                self.yaw_min_cost = LowerHalfMargin(
                    value=self.yaw_min, buffer=0, dim=4)
                self.yaw_max_cost = UpperHalfMargin(
                    value=self.yaw_max, buffer=0, dim=4)

        if self.use_delta:
            self.delta_min_cost = LowerHalfMargin(
                value=self.delta_min, buffer=0, dim=4)
            self.delta_max_cost = UpperHalfMargin(
                value=self.delta_max, buffer=0, dim=4)

        if self.use_vel:
            self.vel_min_cost = LowerHalfMargin(value=0.0, buffer=0, dim=2)

        if self.use_track_exit:
            self.track_exit_cost = LowerHalfMargin(value=0.0, buffer=0, dim=0)

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

        state_offset = self.plan_dyn.apply_rear_offset_correction(state)

        if self.use_road:
            cost = jnp.minimum(
                cost,
                self.road_position_min_cost.get_stage_margin(
                    state_offset, ctrl
                )
            )

            cost = jnp.minimum(
                cost,
                self.road_position_max_cost.get_stage_margin(
                    state_offset, ctrl
                )
            )

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            cost = jnp.minimum(
                cost, _obs_constraint.get_stage_margin(
                    state_offset, ctrl))

        if self.use_yaw:
            cost = jnp.minimum(cost, self.yaw_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            )

            cost = jnp.minimum(cost, self.yaw_max_cost.get_stage_margin(
                state_offset, ctrl
            )
            )

        if self.use_delta:
            cost = jnp.minimum(cost, self.delta_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            )

            cost = jnp.minimum(cost, self.delta_max_cost.get_stage_margin(
                state_offset, ctrl
            )
            )

        if self.use_vel:
            cost = jnp.minimum(cost, self.vel_min_cost.get_stage_margin(
                state_offset, ctrl)
            )

        if self.use_track_exit:
            cost = jnp.minimum(cost, self.track_exit_cost.get_stage_margin(
                state_offset, ctrl)
            )

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
        @jax.jit
        def roll_forward(args):
            current_state, stopping_ctrl, target_cost, v_min = args

            current_state_offset = self.plan_dyn.apply_rear_offset_correction(current_state)

            if self.use_road:
                target_cost = jnp.minimum(target_cost, self.road_position_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

                target_cost = jnp.minimum(target_cost, self.road_position_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            if self.use_yaw:
                target_cost = jnp.minimum(target_cost, self.yaw_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

                target_cost = jnp.minimum(target_cost, self.yaw_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            if self.use_delta:
                target_cost = jnp.minimum(target_cost, self.delta_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

                target_cost = jnp.minimum(target_cost, self.delta_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            for _obs_constraint in self.obs_constraint:
                _obs_constraint: BaseMargin
                target_cost = jnp.minimum(target_cost, _obs_constraint.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            current_state, _ = self.plan_dyn.integrate_forward_jax(
                current_state, stopping_ctrl)

            return current_state, stopping_ctrl, target_cost, v_min

        @jax.jit
        def check_stopped(args):
            current_state, stopping_ctrl, target_cost, v_min = args
            return current_state[2] > v_min

        target_cost = jnp.inf

        stopping_ctrl = jnp.array([self.plan_dyn.ctrl_space[0, 0], 0.])

        current_state = jnp.array(state)

        current_state, stopping_ctrl, target_cost, v_min = jax.lax.while_loop(
            check_stopped, roll_forward, (current_state, stopping_ctrl, target_cost, self.plan_dyn.v_min))

        _, _, target_cost, _ = roll_forward(
            (current_state, stopping_ctrl, target_cost, v_min))

        return target_cost

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin_with_derivatives(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        """
        Args:
            state (DeviceArray, vector shape)
            ctrl (DeviceArray, vector shape)

        Returns:
            DeviceArray: scalar.
        """

        @jax.jit
        def true_fn(args):
            (new_cost, target_cost, c_x_target, c_xx_target,
             c_x_new, c_xx_new, iters, pinch_point) = args
            pinch_point = iters
            return new_cost, c_x_new[:, -1], c_xx_new[:, :, -1], pinch_point

        @jax.jit
        def false_fn(args):
            (new_cost, target_cost, c_x_target, c_xx_target,
             c_x_new, c_xx_new, iters, pinch_point) = args
            return target_cost, c_x_target, c_xx_target, pinch_point

        @jax.jit
        def roll_forward(args):
            current_state, stopping_ctrl, target_cost, v_min, c_x_target, c_xx_target, iters, pinch_point, f_x_all = args

            current_state_offset = self.plan_dyn.apply_rear_offset_correction(current_state)
            #f_x_curr, f_u_curr = self.plan_dyn.get_jacobian(current_state[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
            #f_x_all = f_x_all.at[:, :, iters].set(f_x_curr[:, :, -1])
            f_x_all = f_x_all.at[:, :, iters].set(
                self.plan_dyn.get_jacobian_fx(current_state, stopping_ctrl))

            for _obs_constraint in self.obs_constraint:
                _obs_constraint: BaseMargin
                new_cost = _obs_constraint.get_stage_margin(
                    current_state_offset, stopping_ctrl)
                c_x_new = _obs_constraint.get_cx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                c_xx_new = _obs_constraint.get_cxx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                target_cost, c_x_target, c_xx_target, pinch_point = jax.lax.cond(new_cost < target_cost, true_fn, false_fn, (
                    new_cost, target_cost, c_x_target, c_xx_target, c_x_new, c_xx_new, iters, pinch_point))

            if self.use_road:
                new_cost = self.road_position_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl)
                c_x_new = self.road_position_min_cost.get_cx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                # Half space cost has no second derivative
                #c_xx_new = self.road_position_min_cost.get_cxx(current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                c_xx_new = jnp.zeros((self.dim_x, self.dim_x, 1))
                target_cost, c_x_target, c_xx_target, pinch_point = jax.lax.cond(new_cost < target_cost, true_fn, false_fn, (
                    new_cost, target_cost, c_x_target, c_xx_target, c_x_new, c_xx_new, iters, pinch_point))

                new_cost = self.road_position_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl)
                c_x_new = self.road_position_max_cost.get_cx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                # Half space cost has no second derivative
                c_xx_new = jnp.zeros((self.dim_x, self.dim_x, 1))
                #c_xx_new = self.road_position_max_cost.get_cxx(current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                target_cost, c_x_target, c_xx_target, pinch_point = jax.lax.cond(new_cost < target_cost, true_fn, false_fn, (
                    new_cost, target_cost, c_x_target, c_xx_target, c_x_new, c_xx_new, iters, pinch_point))

            if self.use_yaw:
                new_cost = self.yaw_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl)
                c_x_new = self.yaw_min_cost.get_cx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                # Half space cost has no second derivative
                c_xx_new = jnp.zeros((self.dim_x, self.dim_x, 1))
                target_cost, c_x_target, c_xx_target, pinch_point = jax.lax.cond(new_cost < target_cost, true_fn, false_fn, (
                    new_cost, target_cost, c_x_target, c_xx_target, c_x_new, c_xx_new, iters, pinch_point))

                new_cost = self.yaw_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl)
                c_x_new = self.yaw_max_cost.get_cx(
                    current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                # Half space cost has no second derivative
                #c_xx_new = self.yaw_max_cost.get_cxx(current_state_offset[:, jnp.newaxis], stopping_ctrl[:, jnp.newaxis])
                c_xx_new = jnp.zeros((self.dim_x, self.dim_x, 1))
                target_cost, c_x_target, c_xx_target, pinch_point = jax.lax.cond(new_cost < target_cost, true_fn, false_fn, (
                    new_cost, target_cost, c_x_target, c_xx_target, c_x_new, c_xx_new, iters, pinch_point))

            current_state, _ = self.plan_dyn.integrate_forward_jax(
                current_state, stopping_ctrl)
            iters = iters + 1

            return current_state, stopping_ctrl, target_cost, v_min, c_x_target, c_xx_target, iters, pinch_point, f_x_all

        @jax.jit
        def check_stopped(args):
            current_state, stopping_ctrl, target_cost, v_min, c_x_target, c_xx_target, iters, pinch_point, _ = args
            return current_state[2] > v_min

        @jax.jit
        def backprop_jacobian(idx, jacobian):
            jacobian = f_x_all[:, :, idx] @ jacobian
            return jacobian

        target_cost = jnp.inf
        stopping_ctrl = jnp.array([self.plan_dyn.ctrl_space[0, 0], 0.])

        current_state = jnp.array(state)

        c_x_target = jnp.zeros((self.plan_dyn.dim_x,))
        c_xx_target = jnp.zeros((self.plan_dyn.dim_x, self.plan_dyn.dim_x))

        f_x_all = jnp.zeros((self.plan_dyn.dim_x, self.plan_dyn.dim_x, 50))
        current_state, stopping_ctrl, target_cost, v_min, c_x_target, c_xx_target, iters, pinch_point, f_x_all = jax.lax.while_loop(
            check_stopped, roll_forward, (current_state, stopping_ctrl, target_cost, self.plan_dyn.v_min, c_x_target, c_xx_target, 0, 0, f_x_all))
        _, _, target_cost, _, c_x_target, c_xx_target, iters, pinch_point, f_x_all = roll_forward(
            (current_state, stopping_ctrl, target_cost, v_min, c_x_target, c_xx_target, iters, pinch_point, f_x_all))

        jacobian = jax.lax.fori_loop(
            0, pinch_point, backprop_jacobian, jnp.eye(
                self.plan_dyn.dim_x))

        # Backpropagate derivatives from pinch point
        c_x_target = jacobian.T @ c_x_target
        c_xx_target = jacobian.T @ c_xx_target @ jacobian

        return target_cost, c_x_target, c_xx_target

    @partial(jax.jit, static_argnames='self')
    def get_safety_metric(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_target_stage_margin(state, ctrl)

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
        state_offset = self.plan_dyn.apply_rear_offset_correction(state)

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            obs_cons = jnp.minimum(
                obs_cons, _obs_constraint.get_stage_margin(
                    state_offset, ctrl))

        cost_dict = dict(obs_cons=obs_cons)

        if self.use_road:
            road_min_cons = self.road_position_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            road_max_cons = self.road_position_max_cost.get_stage_margin(
                state_offset, ctrl
            )
            cost_dict.update(dict(road_min_cons=road_min_cons, road_max_cons=road_max_cons))

        if self.use_yaw:
            yaw_min_cons = self.yaw_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            yaw_max_cons = self.yaw_max_cost.get_stage_margin(
                state_offset, ctrl
            )
            cost_dict.update(dict(yaw_min_cons=yaw_min_cons, yaw_max_cons=yaw_max_cons))
        
        if self.use_delta:
            delta_min_cons = self.delta_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            delta_max_cons = self.delta_max_cost.get_stage_margin(
                state_offset, ctrl
            )
            cost_dict.update(dict(delta_min_cons=delta_min_cons, delta_max_cons=delta_max_cons))

        if self.use_vel:
            vel_min_cons = self.vel_min_cost.get_stage_margin(
                state_offset, ctrl
            )
            cost_dict.update(dict(vel_min_cons=vel_min_cons))

        return cost_dict


class Bicycle5DSoftConstraintMargin(Bicycle5DConstraintMargin):
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

        state_offset = self.plan_dyn.apply_rear_offset_correction(state)

        for _obs_constraint in self.obs_constraint:
            _obs_constraint: BaseMargin
            cost += jnp.exp(-1 * self.kappa * _obs_constraint.get_stage_margin(state_offset, ctrl))

        if self.use_road:
            cost += jnp.exp(-1 * self.kappa * self.road_position_min_cost.get_stage_margin(
                state_offset, ctrl
            ))

            cost += jnp.exp(-1 * self.kappa * self.road_position_max_cost.get_stage_margin(
                state_offset, ctrl
            ))

        if self.use_yaw:
            cost += jnp.exp(-1 * self.kappa * self.yaw_max_cost.get_stage_margin(
                state_offset, ctrl
            ))
            cost += jnp.exp(-1 * self.kappa * self.yaw_min_cost.get_stage_margin(
                state_offset, ctrl
            ))

        if self.use_delta:
            cost += jnp.exp(-1 * self.kappa * self.delta_max_cost.get_stage_margin(
                state_offset, ctrl
            ))
            cost += jnp.exp(-1 * self.kappa * self.delta_min_cost.get_stage_margin(
                state_offset, ctrl
            ))

        if self.use_vel:
            cost += jnp.exp(-1 * self.kappa * self.vel_min_cost.get_stage_margin(
                state_offset, ctrl
            ))
        
        if self.use_track_exit:
            cost += jnp.exp(-1 * self.kappa * self.track_exit_cost.get_stage_margin(
                state_offset, ctrl
            ))

        cost = -jnp.log(cost)/self.kappa

        return cost

    @partial(jax.jit, static_argnames='self')
    def get_safety_metric(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_target_stage_margin(state, ctrl)

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
        @jax.jit
        def roll_forward(args):
            current_state, stopping_ctrl, target_cost, v_min = args

            current_state_offset = self.plan_dyn.apply_rear_offset_correction(current_state)

            curr_target_cost = 0
            for _obs_constraint in self.obs_constraint:
                _obs_constraint: BaseMargin
                curr_target_cost += jnp.exp(-1 * self.kappa * _obs_constraint.get_stage_margin(current_state_offset, stopping_ctrl))
            
            if self.use_road:
                curr_target_cost += jnp.exp(-1 * self.kappa * self.road_position_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

                curr_target_cost += jnp.exp(-1 * self.kappa * self.road_position_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))
            
            if self.use_yaw:
                curr_target_cost += jnp.exp(-1 * self.kappa * self.yaw_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))
                curr_target_cost += jnp.exp(-1 * self.kappa * self.yaw_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            if self.use_delta:
                curr_target_cost += jnp.exp(-1 * self.kappa * self.delta_max_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))
                curr_target_cost += jnp.exp(-1 * self.kappa * self.delta_min_cost.get_stage_margin(
                    current_state_offset, stopping_ctrl
                ))

            curr_target_cost = -jnp.log(curr_target_cost)/self.kappa

            target_cost = jnp.minimum(target_cost, curr_target_cost)

            current_state, _ = self.plan_dyn.integrate_forward_jax(
                current_state, stopping_ctrl)

            return current_state, stopping_ctrl, target_cost, v_min

        @jax.jit
        def check_stopped(args):
            current_state, stopping_ctrl, target_cost, v_min = args
            return current_state[2] > v_min

        target_cost = jnp.inf

        stopping_ctrl = jnp.asarray([self.plan_dyn.ctrl_space[0, 0], 0.])

        current_state = jnp.asarray(state)

        current_state, stopping_ctrl, target_cost, v_min = jax.lax.while_loop(
            check_stopped, roll_forward, (current_state, stopping_ctrl, target_cost, self.plan_dyn.v_min))

        _, _, target_cost, _ = roll_forward(
            (current_state, stopping_ctrl, target_cost, v_min))

        return target_cost

class Bicycle5DTargetConstraintMargin(Bicycle5DSoftConstraintMargin):
    
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
        return -1*self.target_constraint.get_stage_margin(state, ctrl)

class BicycleReachAvoid5DMargin(BaseMargin):

    def __init__(self, config, plan_dyn, filter_type):
        super().__init__()
        # Removing the square
        if filter_type == 'SoftCBF' or filter_type=='SoftLR':
            self.constraint = Bicycle5DSoftConstraintMargin(config, plan_dyn)
        else:
            self.constraint = Bicycle5DConstraintMargin(config, plan_dyn)

        if plan_dyn.dim_u == 2:
            R = jnp.array([[config.W_ACCEL, 0.0], [0.0, config.W_OMEGA]])
        elif plan_dyn.dim_u == 3:
            R = jnp.array([[config.W_ACCEL, 0.0, 0.0], [
                          0.0, config.W_OMEGA, 0.0], [0.0, 0.0, config.W_OMEGA]])
        elif plan_dyn.dim_u == 4:
            R = jnp.array([[config.W_ACCEL, 0.0, 0.0, 0.0], [0.0, config.W_ACCEL, 0.0, 0.0], [
                          0.0, 0.0, config.W_OMEGA, 0.0], [0.0, 0.0, 0.0, config.W_OMEGA]])
        self.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(plan_dyn.dim_u))
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

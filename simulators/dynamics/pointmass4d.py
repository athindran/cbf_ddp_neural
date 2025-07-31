from typing import Tuple, Any
import numpy as np
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics import BaseDynamics


class PointMass4D(BaseDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        """
        Implements the 2D double integrator dynamics.
        Args:
            config (Any): an object specifies configuration.
            action_space (np.ndarray): action space.
        """
        super().__init__(config, action_space)
        self.dim_x = 4  # [x, y, vx, vy].

        # load parameters
        self.v_min = 0
        self.v_max = config.V_MAX

    @partial(jax.jit, static_argnames='self')
    def apply_rear_offset_correction(self, state: DeviceArray):
        return state

    @partial(jax.jit, static_argnames=['self'])
    def get_batched_rear_offset_correction(self, nominal_states):
        jac = jax.jit(jax.vmap(self.apply_rear_offset_correction, in_axes=(1), out_axes=(1)))
        return jac(nominal_states)

    @partial(jax.jit, static_argnames='self')
    def integrate_forward_jax(
        self, state: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """Clips the control and computes one-step time evolution of the system.
        Args:
            state (DeviceArray): [x, y, vx, vy].
            control (DeviceArray): [accel_x, accel_y].
        Returns:
            DeviceArray: next state.
            DeviceArray: clipped control.
        """
        # Clips the controller values between min and max accel and steer
        # values.
        ctrl_clip = jnp.clip(
            control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])

        state_nxt = self._integrate_forward(state, ctrl_clip)

        return state_nxt, ctrl_clip

    @partial(jax.jit, static_argnames='self')
    def disc_deriv(
        self, state: DeviceArray, control: DeviceArray
    ) -> DeviceArray:
        deriv = jnp.zeros((self.dim_x,))
        deriv = deriv.at[0].set(state[2])
        deriv = deriv.at[1].set(state[3])
        deriv = deriv.at[2].set(control[0])
        deriv = deriv.at[3].set(control[1])
        return deriv

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward(
        self, state: DeviceArray, control: DeviceArray
    ) -> DeviceArray:
        return self._integrate_forward_dt(state, control, self.dt)

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward_dt(
        self, state: DeviceArray, ctrl_clip: DeviceArray, dt: float
    ) -> DeviceArray:
        k1 = self.disc_deriv(state, ctrl_clip)
        k2 = self.disc_deriv(state + k1 * dt / 2, ctrl_clip)
        k3 = self.disc_deriv(state + k2 * dt / 2, ctrl_clip)
        k4 = self.disc_deriv(state + k3 * dt, ctrl_clip)

        state_nxt = state + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        return state_nxt

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx(
        self, obs: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        Ac = jnp.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt

        return Ad

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fu(
        self, obs: DeviceArray, control: DeviceArray
    ) -> DeviceArray:
        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1]])

        Bd = self.dt * Bc

        return Bd

    @partial(jax.jit, static_argnames='self')
    def get_jacobian(
        self, nominal_states: DeviceArray, nominal_controls: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        jac = jax.jit(
            jax.vmap(
                self.get_jacobian_fx_fu, in_axes=(
                    1, 1), out_axes=(
                    2, 2)))
        return jac(nominal_states, nominal_controls)

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx_fu(self, obs: DeviceArray,
                           control: DeviceArray) -> Tuple:
        Ac = jnp.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, 1]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt
        Bd = self.dt * Bc

        return Ad, Bd

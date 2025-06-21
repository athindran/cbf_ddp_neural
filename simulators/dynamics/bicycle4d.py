from typing import Tuple, Any
import numpy as np
from functools import partial
from jax import Array as DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics import BaseDynamics


class Bicycle4D(BaseDynamics):

    def __init__(self, config: Any, action_space: np.ndarray) -> None:
        """
        Implements the bicycle dynamics (for Princeton race car). The state is the
        center of the rear axis.
        Args:
            config (Any): an object specifies configuration.
            action_space (np.ndarray): action space.
        """
        super().__init__(config, action_space)
        self.dim_x = 4  # [x, y, v, psi, delta].

        # load parameters
        self.wheelbase: float = config.WHEELBASE  # vehicle chassis length
        self.v_min = 0
        self.v_max = config.V_MAX
        self.rear_wheel_offset = 0.4 * self.wheelbase

    @partial(jax.jit, static_argnames='self')
    def apply_rear_offset_correction(self, state: DeviceArray):
        """
        Correct for moving from the rear wheel to centroid.

        Args:
            state (DeviceArray): [x, y, v, psi, delta].
        Returns:
            state_offset 
        """
        state_offset = state.at[0].set(state[0] + self.rear_wheel_offset*jnp.cos(state[3]))
        state_offset = state_offset.at[1].set(state_offset[1] + self.rear_wheel_offset*jnp.sin(state[3]))

        return state_offset

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
            state (DeviceArray): [x, y, v, psi, delta].
            control (DeviceArray): [accel, omega].
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
        deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
        deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
        deriv = deriv.at[2].set(control[0])
        deriv = deriv.at[3].set(state[2] * jnp.tan(control[1]) / self.wheelbase)
        return deriv

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward(
        self, state: DeviceArray, control: DeviceArray
    ) -> DeviceArray:
        """ Computes one-step time evolution of the system: x_+ = f(x, u).
        The discrete-time dynamics is as below:
            x_k+1 = x_k + v_k cos(psi_k) dt
            y_k+1 = y_k + v_k sin(psi_k) dt
            v_k+1 = v_k + u0_k dt
            psi_k+1 = psi_k + v_k tan(u1_k) / L dt
        Args:
            state (DeviceArray): [x, y, v, psi, delta].
            control (DeviceArray): [accel, omega].
        Returns:
            DeviceArray: next state.
        """
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
        # state_nxt = state_nxt.at[2].set(
        #     jnp.clip(state_nxt[2], self.v_min, self.v_max)
        # )

        return state_nxt

    @partial(jax.jit, static_argnames='self')
    def get_jacobian_fx(
        self, obs: DeviceArray, control: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -obs[2] * jnp.sin(obs[3])],
                        [0, 0, jnp.sin(obs[3]), obs[2] * jnp.cos(obs[3])],
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
                       [0, obs[2] / (1e-6 + self.wheelbase * jnp.cos(control[1])**2)]])

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
        Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -1 * obs[2] * jnp.sin(obs[3])],
                        [0, 0, jnp.sin(obs[3]), obs[2] * jnp.cos(obs[3])],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])

        Bc = jnp.array([[0, 0],
                       [0, 0],
                       [1, 0],
                       [0, obs[2] / (1e-6 + self.wheelbase * jnp.cos(control[1])**2)]])

        Ad = jnp.eye(self.dim_x) + Ac * self.dt + \
            0.5 * Ac @ Ac * self.dt * self.dt
        Bd = self.dt * Bc

        return Ad, Bd

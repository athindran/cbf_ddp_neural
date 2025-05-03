import numpy as np
import jax
import jax.numpy as jnp
from jax import Array as DeviceArray
from functools import partial

from .base_margin import BaseMargin


class QuadraticCost(BaseMargin):

    def __init__(
        self, Q: np.ndarray, R: np.ndarray, S: np.ndarray, q: np.ndarray,
        r: np.ndarray
    ):
        super().__init__()
        self.Q = jnp.array(Q)  # (n, n)
        self.R = jnp.array(R)  # (m, m)
        self.S = jnp.array(S)  # (m, n)
        self.q = jnp.array(q)  # (n,)
        self.r = jnp.array(r)  # (m,)

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        Qx = jnp.einsum("i,ni->n", state, self.Q)
        xtQx = jnp.einsum("n,n", state, Qx)
        Sx = jnp.einsum("n,mn->m", state, self.S)
        utSx = jnp.einsum("m,m", ctrl, Sx)
        Ru = jnp.einsum("i,mi->m", ctrl, self.R)
        utRu = jnp.einsum("m,m", ctrl, Ru)
        qtx = jnp.einsum("n,n", state, self.q)
        rtu = jnp.einsum("m,m", ctrl, self.r)
        return -0.5 * (xtQx + utRu) - utSx - qtx - rtu

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)


class QuadraticControlCost(BaseMargin):

    def __init__(self, R: np.ndarray, r: np.ndarray, ref: np.ndarray = None):
        super().__init__()
        self.R = jnp.array(R)  # (m, m)
        self.r = jnp.array(r)  # (m,)
        if ref==None:
            self.ref = jnp.zeros_like(self.r)
        else:
            self.ref = jnp.array(ref)

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        ctrl_delta = ctrl - self.ref
        Ru = jnp.einsum("i,mi->m", ctrl_delta, self.R)
        utRu = jnp.einsum("m,m", ctrl_delta, Ru)
        rtu = jnp.einsum("m,m", ctrl_delta, self.r)
        return -0.5 * utRu - rtu

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)

class QuadraticStateCost(BaseMargin):

    def __init__(self, Q: np.ndarray, q: np.ndarray, ref: np.ndarray = None):
        super().__init__()
        self.Q = jnp.array(Q)  # (n, n)
        self.q = jnp.array(q)  # (n,)
        if ref==None:
            self.ref = jnp.zeros_like(self.q)
        else:
            self.ref = jnp.array(ref)

    @partial(jax.jit, static_argnames='self')
    def get_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        state_delta = state - self.ref
        Qx = jnp.einsum("i,ni->n", state_delta, self.Q)
        xtQx = jnp.einsum("n,n", state_delta, Qx)
        qtx = jnp.einsum("n,n", state_delta, self.q)
        return -0.5 * xtQx - qtx

    @partial(jax.jit, static_argnames='self')
    def get_target_stage_margin(
        self, state: DeviceArray, ctrl: DeviceArray
    ) -> DeviceArray:
        return self.get_stage_margin(state, ctrl)

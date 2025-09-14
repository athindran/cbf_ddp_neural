from typing import Tuple, Optional, Dict
import time
import copy
import jax
from jax import numpy as jp
from jax import Array as DeviceArray
from functools import partial

from simulators import BasePolicy
from brax_utils import WrappedBraxEnv
from simulators.costs.base_margin import BaseMargin


class iLQRBrax(BasePolicy):

    def __init__(
        self, id: str, config, brax_env: WrappedBraxEnv, cost: BaseMargin
    ) -> None:
        super().__init__(id, config)
        self.policy_type = "iLQR"
        self.brax_env = brax_env
        self.cost = cost

        # iLQR parameters
        self.dim_x = brax_env.dim_x
        self.dim_u = brax_env.dim_u
        self.N = config.N
        self.max_iter = config.MAX_ITER
        self.tol = config.TOLERANCE  # ILQR update tolerance.
        self.eps = getattr(config, "EPS", 1e-6)
        self.line_search = config.LINE_SEARCH
        # Stepsize scheduler.
        self.alphas = 0.5**(jp.arange(30))

    def get_action(
        self, initial_state, controls: Optional[DeviceArray] = None,
         **kwargs
    ) -> DeviceArray:
        status = 0

        # `controls` include control input at timestep N-1, which is a dummy
        # control of zeros.
        if controls is None:
            controls = jp.zeros((self.dim_u, self.N))
        else:
            assert controls.shape[1] == self.N
            controls = jp.array(controls)

        # Rolls out the nominal trajectory and gets the initial cost.
        gc_states, controls, pipeline_states = self.rollout_nominal(
            initial_state, controls
        )
        J = self.cost.get_traj_cost(gc_states, controls)

        converged = False
        time0 = time.time()
        
        for i in range(self.max_iter):
            # We need cost derivatives from 0 to N-1, but we only need brax_envamics
            # jacobian from 0 to N-2.
            c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
                gc_states, controls)
            fx, fu = self.brax_env.get_batched_generalized_coordinates_grad(pipeline_states, controls)

            K_closed_loop, k_open_loop = self.backward_pass(
                c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu
            )
            updated = False
            for alpha in self.alphas:
                X_new, U_new, pipeline_states_new, J_new = self.forward_pass(
                    initial_state, gc_states, controls, K_closed_loop, k_open_loop, alpha
                )

                if J_new <= J:  # Improved!
                    # Small improvement.
                    if jp.abs((J - J_new) / J) < self.tol:
                        converged = True

                    # Updates nominal trajectory and best cost.
                    J = J_new
                    gc_states = X_new
                    controls = U_new
                    pipeline_states = pipeline_states_new
                    updated = True
                    break

            # Terminates early if there is no update within alphas.
            if not updated:
                #print("Did not converge")
                status = 2
                break

            # Terminates early if the objective improvement is negligible.
            if converged:
                #print(f"Converged with cost {J}")
                status = 1
                break
        t_process = time.time() - time0

        gc_states = jp.array(gc_states)
        controls = jp.array(controls)
        K_closed_loop = jp.array(K_closed_loop)
        k_open_loop = jp.array(k_open_loop)
        solver_info = dict(
            gc_states=gc_states, controls=controls, K_closed_loop=K_closed_loop,
            k_open_loop=k_open_loop, t_process=t_process, status=status, J=J
        )
        return controls[:, 0], solver_info

    @partial(jax.jit, static_argnames='self')
    def forward_pass(
        self, initial_state, nominal_gc_states: DeviceArray, nominal_controls: DeviceArray,
        K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
    ) -> Tuple[DeviceArray, DeviceArray, float]:
        # We seperate the rollout and cost explicitly since get_cost might rely on
        # other information, such as env parameters (track), and is difficult for
        # jax to differentiate.
        X, U, pipeline_states = self.rollout(
            initial_state, nominal_gc_states, nominal_controls, K_closed_loop, k_open_loop, alpha
        )
        J = self.cost.get_traj_cost(X, U)
        return X, U, pipeline_states, J

    @partial(jax.jit, static_argnames='self')
    def rollout(
        self, initial_state, nominal_gc_states: DeviceArray, nominal_controls: DeviceArray,
        K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_step(i, args):
            X, U, state_prev, pipeline_states = args
            u_fb = jp.einsum(
                "ik,k->i", K_closed_loop[:, :,
                                         i], (X[:, i] - nominal_gc_states[:, i])
            )
            u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
            u_clip = jp.clip(u, min=self.brax_env.action_limits[0], max=self.brax_env.action_limits[1])
            state_nxt = self.brax_env.step(state_prev, u_clip)
            state_grad, action_grad = self.brax_env.get_generalized_coordinates_grad(state_prev, u)
            X = X.at[:, i + 1].set(self.brax_env.get_generalized_coordinates(state_nxt))
            U = U.at[:, i].set(u_clip)
            pipeline_states = jax.tree.map(lambda xs, ys: xs.at[..., i + 1].set(ys), pipeline_states, state_nxt)

            return X, U, state_nxt, pipeline_states

        X = jp.zeros((self.dim_x, self.N))
        fx = jp.zeros((self.dim_x, self.dim_x, self.N))
        fu = jp.zeros((self.dim_x, self.dim_u, self.N))
        U = jp.zeros((self.dim_u, self.N))  # Assumes the last ctrl are zeros.
        X = X.at[:, 0].set(nominal_gc_states[:, 0])
        state_expanded = jax.tree.map(lambda xs: jp.expand_dims(xs, axis=-1), initial_state)
        pipeline_states = jax.tree.map(lambda xs: jp.repeat(xs, self.N, axis=-1), state_expanded)

        X, U, _, pipeline_states = jax.lax.fori_loop(0, self.N - 1, _rollout_step, (X, U, initial_state, pipeline_states))
        return X, U, pipeline_states

    @partial(jax.jit, static_argnames='self')
    def rollout_nominal(
        self, state, controls: DeviceArray,
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_nominal_step(i, args):
            X, U, state_prev, pipeline_states = args
            u_clip = jp.clip(U[:, i], min=self.brax_env.action_limits[0], max=self.brax_env.action_limits[1])
            state_nxt = self.brax_env.step(state_prev, u_clip)
            state_grad, action_grad = self.brax_env.get_generalized_coordinates_grad(state_prev, U[:, i])
            X = X.at[:, i + 1].set(self.brax_env.get_generalized_coordinates(state_nxt))
            U = U.at[:, i].set(u_clip)
            pipeline_states = jax.tree.map(lambda xs, ys: xs.at[..., i + 1].set(ys), pipeline_states, state_nxt)

            return X, U, state_nxt, pipeline_states

        X = jp.zeros((self.dim_x, self.N))
        fx = jp.zeros((self.dim_x, self.dim_x, self.N))
        fu = jp.zeros((self.dim_x, self.dim_u, self.N))
        X = X.at[:, 0].set(self.brax_env.get_generalized_coordinates(state))
        state_expanded = jax.tree.map(lambda xs: jp.expand_dims(xs, axis=-1), state)
        pipeline_states = jax.tree.map(lambda xs: jp.repeat(xs, self.N, axis=-1), state_expanded)

        X, U, _, pipeline_states = jax.lax.fori_loop(
            0, self.N - 1, _rollout_nominal_step, (X, controls, state, pipeline_states)
        )
        return X, U, pipeline_states

    @partial(jax.jit, static_argnames='self')
    def backward_pass(
        self, c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
        c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray
    ) -> Tuple[DeviceArray, DeviceArray]:
        """
        Jitted backward pass looped computation.

        Args:
            c_x (DeviceArray): (dim_x, N)
            c_u (DeviceArray): (dim_u, N)
            c_xx (DeviceArray): (dim_x, dim_x, N)
            c_uu (DeviceArray): (dim_u, dim_u, N)
            c_ux (DeviceArray): (dim_u, dim_x, N)
            fx (DeviceArray): (dim_x, dim_x, N-1)
            fu (DeviceArray): (dim_x, dim_u, N-1)

        Returns:
            Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
            ks (DeviceArray): gain vectors (dim_u, N - 1)
        """

        @jax.jit
        def backward_pass_looper(i, _carry):
            V_x, V_xx, ks, Ks = _carry
            n = self.N - 2 - i

            Q_x = c_x[:, n] + fx[:, :, n].T @ V_x
            Q_u = c_u[:, n] + fu[:, :, n].T @ V_x
            Q_xx = c_xx[:, :, n] + fx[:, :, n].T @ V_xx @ fx[:, :, n]
            Q_ux = c_ux[:, :, n] + fu[:, :, n].T @ V_xx @ fx[:, :, n]
            Q_uu = c_uu[:, :, n] + fu[:, :, n].T @ V_xx @ fu[:, :, n]

            Q_uu_inv = jp.linalg.inv(Q_uu + reg_mat)

            Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
            ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

            V_x = Q_x + Ks[:, :, n].T @ Q_u + Q_ux.T @ ks[:, n] + Ks[:, :, n].T @ Q_uu @ ks[:, n]
            V_xx = (Q_xx + Ks[:, :, n].T @ Q_ux + Q_ux.T @ Ks[:, :, n]
                    + Ks[:, :, n].T @ Q_uu @ Ks[:, :, n])

            return V_x, V_xx, ks, Ks

        # Initializes.
        Ks = jp.zeros((self.dim_u, self.dim_x, self.N - 1))
        ks = jp.zeros((self.dim_u, self.N - 1))
        V_x = c_x[:, -1]
        V_xx = c_xx[:, :, -1]
        reg_mat = self.eps * jp.eye(self.dim_u)

        V_x, V_xx, ks, Ks = jax.lax.fori_loop(
            0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
        )
        return Ks, ks

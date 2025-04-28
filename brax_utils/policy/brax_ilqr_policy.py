from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import ArrayImpl as DeviceArray
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
        self.tol = 1e-3  # ILQR update tolerance.
        self.eps = getattr(config, "EPS", 1e-6)
        # Stepsize scheduler.
        self.alphas = 0.5**(np.arange(30))

    def get_action(
        self, initial_state, controls: Optional[np.ndarray] = None,
         **kwargs
    ) -> np.ndarray:
        status = 0

        # `controls` include control input at timestep N-1, which is a dummy
        # control of zeros.
        if controls is None:
            controls_np = np.random.rand(self.dim_u, self.N)
            controls = jnp.array(controls_np)
        else:
            assert controls.shape[1] == self.N
            controls = jnp.array(controls)

        # Rolls out the nominal trajectory and gets the initial cost.
        gc_states, controls, fx, fu = self.rollout_nominal(
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
            K_closed_loop, k_open_loop = self.backward_pass(
                c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu
            )
            updated = False
            for alpha in self.alphas:
                X_new, U_new, fx, fu, J_new = self.forward_pass(
                    initial_state, gc_states, controls, K_closed_loop, k_open_loop, alpha
                )

                if J_new <= J:  # Improved!
                    # Small improvement.
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True

                    # Updates nominal trajectory and best cost.
                    J = J_new
                    gc_states = X_new
                    controls = U_new
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

        gc_states = np.asarray(gc_states)
        controls = np.asarray(controls)
        K_closed_loop = np.asarray(K_closed_loop)
        k_open_loop = np.asarray(k_open_loop)
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
        X, U, fx, fu = self.rollout(
            initial_state, nominal_gc_states, nominal_controls, K_closed_loop, k_open_loop, alpha
        )
        J = self.cost.get_traj_cost(X, U)
        return X, U, fx, fu, J

    @partial(jax.jit, static_argnames='self')
    def rollout(
        self, initial_state, nominal_gc_states: DeviceArray, nominal_controls: DeviceArray,
        K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_step(i, args):
            X, U, fx, fu, state_prev = args
            u_fb = jnp.einsum(
                "ik,k->i", K_closed_loop[:, :,
                                         i], (X[:, i] - nominal_gc_states[:, i])
            )
            u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
            u_clip = jnp.clip(u, min=jnp.array([-1.0, -1.0]), max=jnp.array([1.0, 1.0]))
            state_nxt = self.brax_env.step(state_prev, u_clip)
            state_grad, action_grad = self.brax_env.get_generalized_coordinates_grad(state_prev, u)
            X = X.at[:, i + 1].set(self.brax_env.get_generalized_coordinates(state_nxt))
            U = U.at[:, i].set(u_clip)
            fx = fx.at[:, :, i].set(state_grad)
            fu = fu.at[:, :, i].set(action_grad)
            return X, U, fx, fu, state_nxt

        X = jnp.zeros((self.dim_x, self.N))
        fx = jnp.zeros((self.dim_x, self.dim_x, self.N))
        fu = jnp.zeros((self.dim_x, self.dim_u, self.N))
        U = jnp.zeros((self.dim_u, self.N))  # Assumes the last ctrl are zeros.
        X = X.at[:, 0].set(nominal_gc_states[:, 0])

        X, U, fx, fu, _ = jax.lax.fori_loop(0, self.N, _rollout_step, (X, U, fx, fu, initial_state))
        return X, U, fx, fu

    @partial(jax.jit, static_argnames='self')
    def rollout_nominal(
        self, state, controls: DeviceArray,
    ) -> Tuple[DeviceArray, DeviceArray]:

        @jax.jit
        def _rollout_nominal_step(i, args):
            X, U, fx, fu, state_prev = args
            u_clip = jnp.clip(U[:, i], min=jnp.array([-1.0, -1.0]), max=jnp.array([1.0, 1.0]))
            state_nxt = self.brax_env.step(state_prev, u_clip)
            state_grad, action_grad = self.brax_env.get_generalized_coordinates_grad(state_prev, U[:, i])
            X = X.at[:, i + 1].set(self.brax_env.get_generalized_coordinates(state_nxt))
            U = U.at[:, i].set(u_clip)
            fx = fx.at[:, :, i].set(state_grad)
            fu = fu.at[:, :, i].set(action_grad)
            return X, U, fx, fu, state_nxt

        X = jnp.zeros((self.dim_x, self.N))
        fx = jnp.zeros((self.dim_x, self.dim_x, self.N))
        fu = jnp.zeros((self.dim_x, self.dim_u, self.N))
        X = X.at[:, 0].set(self.brax_env.get_generalized_coordinates(state))
        X, U, fx, fu, _ = jax.lax.fori_loop(
            0, self.N - 1, _rollout_nominal_step, (X, controls, fx, fu, state)
        )
        return X, U, fx, fu

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

            Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)

            Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
            ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

            V_x = Q_x + Q_ux.T @ ks[:, n]
            V_xx = Q_xx + Q_ux.T @ Ks[:, :, n]

            return V_x, V_xx, ks, Ks

        # Initializes.
        Ks = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
        ks = jnp.zeros((self.dim_u, self.N - 1))
        V_x = c_x[:, -1]
        V_xx = c_xx[:, :, -1]
        reg_mat = self.eps * jnp.eye(self.dim_u)

        V_x, V_xx, ks, Ks = jax.lax.fori_loop(
            0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
        )
        return Ks, ks

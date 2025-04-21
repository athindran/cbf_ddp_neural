from typing import Tuple, Optional, Dict
import time
import copy
import numpy as np
import jax
from jax import numpy as jnp
from jax import Array
from simulators.brax.wrapper_env import WrappedBraxEnv
from functools import partial

from simulators.policy.base_policy import BasePolicy
from simulators.costs.base_margin import BaseMargin


class iLQR(BasePolicy):

    def __init__(
        self, id: str, config, dyn: WrappedBraxEnv, cost: BaseMargin
    ) -> None:
        super().__init__(id, config)
        self.policy_type = "iLQR"
        self.dyn = dyn
        self.cost = copy.deepcopy(cost)

        # iLQR parameters
        self.dim_u = cost.dim_u
        self.dim_x = cost.dim_x
        self.N = config.N
        self.max_iter = config.MAX_ITER
        self.tol = 1e-5  # ILQR update tolerance.
        self.eps = getattr(config, "EPS", 1e-6)
        # Stepsize scheduler.
        self.alphas = 0.5**(np.arange(30))

    def get_action(
        self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
        agents_action: Optional[Dict] = None, **kwargs
    ) -> np.ndarray:
        status = 0

        # `controls` include control input at timestep N-1, which is a dummy
        # control of zeros.
        if controls is None:
            controls = jnp.zeros((self.dim_u, self.N))
        else:
            assert controls.shape[1] == self.N

        # Rolls out the nominal trajectory and gets the initial cost.
        states, trimmed_states, controls = self.rollout_nominal(
            state, controls
        )
        J = self.cost.get_traj_cost(trimmed_states, controls)

        converged = False
        time0 = time.time()
        for _ in range(self.max_iter):
            # We need cost derivatives from 0 to N-1, but we only need dynamics
            # jacobian from 0 to N-2.
            c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
                trimmed_states, controls)
            fx, fu = self.dyn.get_jacobian(trimmed_states[:, :-1], controls[:, :-1])
            K_closed_loop, k_open_loop = self.backward_pass(
                c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu
            )
            updated = False
            for alpha in self.alphas:
                X_new, U_new, J_new = self.forward_pass(
                    trimmed_states, controls, K_closed_loop, k_open_loop, alpha
                )

                if J_new <= J:  # Improved!
                    # Small improvement.
                    if np.abs((J - J_new) / J) < self.tol:
                        converged = True

                    # Updates nominal trajectory and best cost.
                    J = J_new
                    states = X_new
                    controls = U_new
                    updated = True
                    break

            # Terminates early if there is no update within alphas.
            if not updated:
                status = 2
                break

            # Terminates early if the objective improvement is negligible.
            if converged:
                status = 1
                break
        t_process = time.time() - time0

        states = np.asarray(states)
        controls = np.asarray(controls)
        K_closed_loop = np.asarray(K_closed_loop)
        k_open_loop = np.asarray(k_open_loop)
        solver_info = dict(
            states=states, controls=controls, K_closed_loop=K_closed_loop,
            k_open_loop=k_open_loop, t_process=t_process, status=status, J=J
        )
        return controls[:, 0], solver_info

    @partial(jax.jit, static_argnames='self')
    def forward_pass(
        self, nominal_states: Array, nominal_controls: Array,
        K_closed_loop: Array, k_open_loop: Array, alpha: float
    ) -> Tuple[Array, Array, float]:
        # We seperate the rollout and cost explicitly since get_cost might rely on
        # other information, such as dyn parameters (track), and is difficult for
        # jax to differentiate.
        X, U = self.rollout(
            nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
        )
        J = self.cost.get_traj_cost(X, U)
        return X, U, J

    @partial(jax.jit, static_argnames='self')
    def rollout(
        self, nominal_states: Array, nominal_controls: Array,
        K_closed_loop: Array, k_open_loop: Array, alpha: float
    ) -> Tuple[Array, Array]:

        @jax.jit
        def _rollout_step(i, args):
            X, U = args
            u_fb = jnp.einsum(
                "ik,k->i", K_closed_loop[:, :,
                                         i], (X[:, i] - nominal_states[:, i])
            )
            u = nominal_controls[:, i] + alpha * k_open_loop[:, i] + u_fb
            u_clip = jnp.clip(u, min=-1, max=1)
            x_nxt = self.dyn.step(X[:, i], u_clip)
            X = X.at[:, i + 1].set(x_nxt)
            U = U.at[:, i].set(u_clip)
            return X, U

        X = jnp.zeros((self.dim_x, self.N))
        U = jnp.zeros((self.dim_u, self.N))  # Assumes the last ctrl are zeros.
        X = X.at[:, 0].set(nominal_states[:, 0])

        X, U = jax.lax.fori_loop(0, self.N, _rollout_step, (X, U))
        return X, U

    @partial(jax.jit, static_argnames='self')
    def rollout_nominal(
        self, initial_state: Array, controls: Array
    ) -> Tuple[Array, Array]:

        @jax.jit
        def _rollout_nominal_step(i, args):
            states, X, U = args
            u_clip = jnp.clip(U[:, i], min=-1, max=1)
            state_nxt = self.dyn.step(states[i], u_clip)
            X = X.at[:, i + 1].set(self.dyn.get_trimmed_state( state_nxt ))
            U = U.at[:, i].set(u_clip)
            states.append( state_nxt )
            return states, X, U
        
        states = []
        states.append( initial_state )
        X = jnp.zeros((self.dim_x, self.N))
        X = X.at[:, 0].set( self.dyn.get_trimmed_state( initial_state ) )
        states, X, U = jax.lax.fori_loop(
            0, self.N - 1, _rollout_nominal_step, (states, X, controls)
        )
        return states, X, U

    @partial(jax.jit, static_argnames='self')
    def backward_pass(
        self, c_x: Array, c_u: Array, c_xx: Array,
        c_uu: Array, c_ux: Array, fx: Array, fu: Array
    ) -> Tuple[Array, Array]:
        """
        Jitted backward pass looped computation.

        Args:
            c_x (Array): (dim_x, N)
            c_u (Array): (dim_u, N)
            c_xx (Array): (dim_x, dim_x, N)
            c_uu (Array): (dim_u, dim_u, N)
            c_ux (Array): (dim_u, dim_x, N)
            fx (Array): (dim_x, dim_x, N-1)
            fu (Array): (dim_x, dim_u, N-1)

        Returns:
            Ks (Array): gain matrices (dim_u, dim_x, N - 1)
            ks (Array): gain vectors (dim_u, N - 1)
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

from typing import Optional, Tuple, Dict, List
import copy
import numpy as np
import time

# Dynamics.
from .dynamics.bicycle5d import Bicycle5D
from .dynamics.bicycle4d import Bicycle4D
from .dynamics.pvtol6d import Pvtol6D
from .dynamics.pointmass4d import PointMass4D

from .costs.base_margin import BaseMargin

# Footprint.
from .footprint.circle import CircleFootprint
from .footprint.rectangle import RectangleFootprint

# Policy.
from .policy.base_policy import BasePolicy
from .policy.ilqr_policy import iLQR
from .policy.ilqr_filter_policy import iLQRSafetyFilter
from .policy.ilqr_reachavoid_policy import iLQRReachAvoid
from .policy.ilqr_reachability_policy import iLQRReachability
from .policy.manual_task_policies import bicycle_linear_task_policy, pvtol_linear_task_policy

class Agent:
    """A basic unit in our environments.

    Attributes:
        dyn (object): agent's dynamics.
        footprint (object): agent's shape.
        policy (object): agent's policy.
    """
    policy: Optional[BasePolicy]
    safety_policy: Optional[BasePolicy]
    task_policy: Optional[BasePolicy]
    ego_observable: Optional[List]
    agents_policy: Dict[str, BasePolicy]
    agents_order: Optional[List]

    def __init__(self, config, action_space: np.ndarray, env=None) -> None:
        if config.DYN == "Bicycle5D":
            self.dyn = Bicycle5D(config, action_space)
        elif config.DYN == "Bicycle4D":
            self.dyn = Bicycle4D(config, action_space)
        elif config.DYN == "PointMass4D":
            self.dyn = PointMass4D(config, action_space)
        elif config.DYN == "PVTOL6D":
            self.dyn = Pvtol6D(config, action_space)
        else:
            raise ValueError("Dynamics type not supported!")

        try:
            self.env = copy.deepcopy(env)  # imaginary environment
        except Exception as e:
            print("WARNING: Cannot copy env - {}".format(e))

        if config.FOOTPRINT == "Circle":
            self.footprint = CircleFootprint(ego_radius=config.EGO_RADIUS)
        elif config.FOOTPRINT == "Rectangle":
            self.footprint = RectangleFootprint(ego_radius=config.EGO_RADIUS)

        # Policy should be initialized by `init_policy()`.
        self.policy = None
        self.safety_policy = None
        self.task_policy = None
        self.id: str = config.AGENT_ID
        self.ego_observable = None
        self.agents_policy = {}
        self.agents_order = None
        self.is_task_ilqr = getattr(config, 'is_task_ilqr', False)

    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds the next state of the vehicle given the current state and
        control input.

        Args:
            state (np.ndarray): (dyn.dim_x, ) array.
            control (np.ndarray): (dyn.dim_u, ) array.

        Returns:
            np.ndarray: next state.
            np.ndarray: clipped control.
        """
        assert control is not None, (
            "You need to pass in a control!"
        )

        return self.dyn.integrate_forward(
            state=state, control=control
        )

    def get_dyn_jacobian(
        self, nominal_states: np.ndarray, nominal_controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the linearized 'A' and 'B' matrix of the ego vehicle around
        nominal states and controls.

        Args:
            nominal_states (np.ndarray): states along the nominal trajectory.
            nominal_controls (np.ndarray): controls along the trajectory.

        Returns:
            np.ndarray: the Jacobian of next state w.r.t. the current state.
            np.ndarray: the Jacobian of next state w.r.t. the current control.
        """
        A, B = self.dyn.get_jacobian(nominal_states, nominal_controls)
        return np.asarray(A), np.asarray(B)

    def get_action(
        self, obs: np.ndarray,
        agents_action: Optional[Dict[str, np.ndarray]] = None,
        warmup: bool = False,
        prev_sol: Optional[Dict] = None, 
        prev_ctrl:np.ndarray = np.array([0.0, 0.0]), 
        **kwargs
    ) -> Tuple[np.ndarray, dict]:
        """Gets the action to execute.

        Args:
            obs (np.ndarray): current observation.
            agents_action (Optional[Dict]): other agents' actions that are
                observable to the ego agent.

        Returns:
            np.ndarray: the action to be executed.
            dict: info for the solver, e.g., processing time, status, etc.
        """
        if self.ego_observable is not None:
            for agent_id in self.ego_observable:
                assert agent_id in agents_action

        if agents_action is not None:
            _action_dict = copy.deepcopy(agents_action)
        else:
            _action_dict = {}

        if self.policy_type == "iLQRSafetyFilter":
            # Execute task control
            if self.is_task_ilqr:
                task_ctrl, _ = self.task_policy.get_action(obs=obs, controls=None, state=kwargs['state'], warmup=warmup)
            elif self.dyn.id ==  "PVTOL6D":
                task_ctrl = self.task_policy(obs, self.dyn)
            else:
                task_ctrl = self.task_policy(obs)
            # Filter to safe control
            start_time = time.time()
            _action, _solver_info = self.safety_policy.get_action(  # Proposed action.
                state=kwargs['state'], obs=obs, task_ctrl=task_ctrl, warmup=warmup, 
                prev_sol=prev_sol, prev_ctrl=prev_ctrl, 
            )
            _solver_info['process_time'] = time.time() - start_time
        else:
            _action, _solver_info = self.policy.get_action(  # Proposed action.
                obs=obs, agents_action=agents_action, **kwargs
            )
        _action_dict[self.id] = _action

        return _action, _solver_info

    def init_policy(
        self, policy_type: str, config, cost: Optional[BaseMargin] = None, **kwargs
    ):
        self.policy_type = policy_type

        if policy_type == "iLQR":
            self.policy = iLQR(self.id, config, self.dyn, cost, **kwargs)
        elif policy_type == "iLQRReachAvoid":
            self.policy = iLQRReachAvoid(
                self.id, config, self.dyn, cost
            )
        elif policy_type == "iLQRReachability":
            self.policy = iLQRReachability(
                self.id, config, self.dyn, cost
            )
        elif policy_type == "iLQRSafetyFilter":            
            if self.is_task_ilqr:
                self.task_policy = iLQR(
                    self.id,
                    config,
                    self.dyn,
                    kwargs["task_cost"])
            elif self.dyn.id ==  "PVTOL6D":
                self.task_policy = pvtol_linear_task_policy
            else:
                self.task_policy = bicycle_linear_task_policy    
        
            self.safety_policy = iLQRSafetyFilter(
                self.id, config, self.dyn, cost
            )
        else:
            raise ValueError(
                "The policy type ({}) is not supported!".format(policy_type)
            )

    def report(self):
        print(self.id)
        if self.ego_observable is not None:
            print("  - The agent can observe:", end=' ')
            for i, k in enumerate(self.ego_observable):
                print(k, end='')
                if i == len(self.ego_observable) - 1:
                    print('.')
                else:
                    print(', ', end='')
        else:
            print("  - The agent can only access observation.")

        if self.agents_order is not None:
            print("  - The agent keeps agents order:", end=' ')
            for i, k in enumerate(self.agents_order):
                print(k, end='')
                if i == len(self.agents_order) - 1:
                    print('.')
                else:
                    print(' -> ', end='')

# Soft-DDPCBF

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. For public release, the code will go through another iteration of cleaning.

TODO:

- Rename 'jerk' to 'action_fluctuation' - Done
- The `ilqr_filter_policy.py` and `base_single_env.py` needs urgent refactoring - Done
- DDP - Done but not as stable as ILQR
- Box obstacles - Done with better results with soft box constraints.
- Line searches - Done -  more line searches can be explored.
- Reach-avoid can be improved by increasing control cost weighting leaving it in a weird situation where the reachability weight is higher than Lagrange ILQR weight - Done
- Velocity and delta constraints found to work with reachability and delta constraints with reach-avoid - Done
- Resolve inconsistency with rear-wheel bicycle model
- Look at other means of obtaining target functions such as using a learned target function for better reach-avoid results - 

### Brax testing

- Reacher produces meaningful results.
- Ant does something but not entirely meaningful.


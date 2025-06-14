# Soft-DDPCBF

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. For public release, the code will go through another iteration of cleaning.

TODO:

- Rename 'jerk' to 'action_fluctuation' - Done
- The `ilqr_filter_policy.py` and `base_single_env.py` needs urgent refactoring - Done
- DDP - Done but not as stable as ILQR
- Box obstacles - Done but the reach-avoid needs improvement.
- Line searches - Done -  more line searches can be explored.

### Brax testing

- Reacher produces meaningful results.
- Ant does something but not entirely meaningful.


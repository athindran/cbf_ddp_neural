# Soft-DDPCBF

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. For public release, the code will go through another iteration of cleaning.

TODO:

- Rename 'jerk' to 'action_fluctuation' - Done
- The `ilqr_filter_policy.py` and `base_single_env.py` needs urgent refactoring - Done
- DDP - Done but not as stable as ILQR
- Box obstacles - Done with better results with soft box constraints.
- Line searches - Done -  more line searches can be explored.
- Look at other means of obtaining target functions such as using a learned target function for better reach-avoid results - 

### Brax testing

- Reacher produces meaningful results.
- Ant does something but not entirely meaningful.


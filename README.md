# Soft-DDPCBF

This is a repository for using CBF-DDP with soft minimum and maximum operators in place of the hard operators that were originally used. All changes are contained in the main branch. For public release, the code will go through another iteration of cleaning.

TODO:

1) Rename 'jerk' to 'action_fluctuation' - DONE
2) The `ilqr_filter_policy.py` and `base_single_env.py` needs urgent refactoring - DONE
3) Box obstackes - TBD
4) Line searches - To be explored again.

### Test gradient extraction from brax.

In MJX, it is more complicated to retrieve a sufficient state

Too slow on CPU. Maybe, try on GPU.

More exploration to be done later.

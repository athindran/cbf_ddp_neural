# Test scripts

# Run each config independently for best performance in process time.
# The QCQP constraint solver has some known anomalies in solver time. Set the CONSTRAINT_TYPE to 'linear' in config if solver takes too long.

# Reachability
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 3.5 -ls 'baseline'

# # Reachavoid
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 1.5 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 1.5 -ls 'baseline'

# Reachability
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'armijo'

# Reachavoid
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 1.5 -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 1.5 -ls 'armijo'


# Reachability
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'trust_region_constant_margin'

# Reachavoid
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 1.5 -ls 'trust_region_constant_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 1.5 -ls 'trust_region_constant_margin'

# Reachability
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'trust_region_tune_margin'

# Reachavoid
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic4D.yaml -rb 3.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 2.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 2.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic4D.yaml -rb 1.5 -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_shorter_wheelbase_multiple_obs_1_bic5D.yaml -rb 1.5 -ls 'trust_region_tune_margin'

# Box obstacles 
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_box_config_obs_1_bic5D.yaml -rb 4.0 -ls 'baseline'
# Reach-avoid improves with soft box constraints.
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_box_config_obs_1_bic5D.yaml -rb 4.0 -ls 'baseline'

# Ellipse obstacles
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_ellipse_config_obs_1_bic5D.yaml -rb 4.0 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_ellipse_config_obs_1_bic5D.yaml -rb 4.0 -ls 'baseline'

# Velocity and delta constraints
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_ellipse_config_obs_vel_delta_constraint_bic5D.yaml -rb 4.0 -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_box_config_delta_constraint_bic5D.yaml -rb 4.0 -ls 'baseline'


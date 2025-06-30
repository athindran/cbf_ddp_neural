# Test scripts

# Run each config independently for best performance in process time.
# The QCQP constraint solver has some known anomalies in solver time. Set the CONSTRAINT_TYPE to 'linear' in config if solver takes too long.

python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'baseline'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'baseline'

python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'armijo'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'armijo'

python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_1_bic5D.yaml -rb 3.0 --naive_task  -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'trust_region_tune_margin'
python evaluate_soft_ddpcbf.py -cf ./test_configs/reachability/test_config_cbf_reachability_circle_config_multiple_obs_2_bic5D.yaml -rb 3.5 --naive_task  -ls 'trust_region_tune_margin'

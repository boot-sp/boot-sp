#!/bin/bash
# The sample sizes etc are way to small for good results; this is just to test
# NOTE: do not be alarmed by infeasibility message during the confidence interval calculations

echo "Running in serial, compute xhat within user_boot"
echo
python -m bootsp.user_boot multi_knapsack --max-count 121 --candidate-sample-size 1 --sample-size 75 --subsample-size 10 --nB 10 --alpha 0.1 --seed-offset 100  --solver-name cplex --boot-method Bagging_with_replacement --deterministic-data-json=test.json
echo
exit

echo "========================"
echo
echo "Running in serial, use an xhat that is computed elsewhere"
echo


#!/bin/bash
# run the farmer example in user mode

python user_boot.py farmer --max-count 121 --candidate-sample-size 1 --sample-size 75 --subsample-size 10 --nB 10 --alpha 0.05 --seed-offset 100  --solver-name cplex --boot-method Bagging_with_replacement  --crops-multiplier 1
exit
echo
echo "========================"
echo
python user_boot.py farmer --max-count 121 --candidate-sample-size 1 --sample-size 75 --subsample-size 10 --nB 10 --alpha 0.05 --seed-offset 100  --solver-name cplex --boot-method Bagging_with_replacement --xhat-fname xhat.npy

#--optimal-fname schultz_optimal.npy

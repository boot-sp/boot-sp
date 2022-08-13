# A driver for boot_sp to be used by researchers interested in doing simulations

import sys
import time
import numpy as np
import mpisppy.utils.config as config
import mpisppy.confidence_intervals.ciutils as ciutils
import bootsp.boot_utils as boot_utils
import bootsp.boot_sp as boot_sp

# TBD: we are using the mpi-sppy MPI wrapper to help windows users live without MPI.
import mpisppy.MPI as MPI
my_rank = MPI.COMM_WORLD.Get_rank()


def main_routine(cfg, module):
    """ The top level of simulate_boot; called by __main__
        and by drivers such as simulate_experiments.py
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers        
    Returns:
        coverage_rate (float): the coverage detected in the simulations
        average_length (float): the average width of the interval around z*
    """
    
    start_time = time.time()

    if my_rank == 0:
        opt_obj, opt_gap = boot_sp.process_optimal(cfg, module)
    else:
        opt_obj = None  # only rank 0 should use the opt_obj in analysis anyway 
        opt_gap = None

    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        xhat = boot_utils.compute_xhat(cfg, module)


    coverage_cnt = 0
    total_len = 0
    seed_list = range(cfg.coverage_replications)
    for seed in seed_list:
        cfg.seed_offset = seed
        
        if cfg.boot_method == "Extended":
            ci_optimal,ci_upper, ci_gap =  boot_sp.extended_bootstrap(cfg, module, xhat)
        elif cfg.boot_method == "Bagging_with_replacement":
            ci_optimal,ci_upper, ci_gap = boot_sp.bagging_bootstrap(cfg, module, xhat, replacement = True)
        elif cfg.boot_method == "Bagging_without_replacement":
            ci_optimal,ci_upper, ci_gap = boot_sp.bagging_bootstrap(cfg, module, xhat, replacement = False)
        elif cfg.boot_method == "Classical_quantile":
            ci_optimal,ci_upper, ci_gap =  boot_sp.classical_bootstrap(cfg, module, xhat, quantile = True)
        elif cfg.boot_method == "Classical_gaussian":
            ci_optimal,ci_upper, ci_gap =  boot_sp.classical_bootstrap(cfg, module, xhat, quantile = False)
        elif cfg.boot_method == "Subsampling":
            ci_optimal,ci_upper, ci_gap =  boot_sp.subsampling(cfg, module, xhat)
        else:
            raise ValueError(f"boot_method={cfg.boot_method} is not supported.")


        # the last candidate_sample_size samples are already used to compute the candidate solution, so when applying bootstrap we only use max_count-candidate_sample_size samples


        if my_rank == 0:
        # print result
            # file_name = cfg.module_name +"_sample_" + str(cfg.sample_size)+ "_nB_" + str(cfg.nB) + ".txt"
            if cfg.trace_fname is not None:
                with open(cfg.trace_fname,  "a+") as f:
                    f.write(f"seed: {seed}\n")
                    f.write(f"optimal function value z^*: {opt_obj}\n")
                    f.write(f"ci for optimal function value z^*: {ci_optimal}\n")
                    f.write(f"function value evaluated at xhat: {opt_obj + opt_gap} \n" )
                    f.write(f"ci for function value at xhat: {ci_upper}\n")
                    f.write(f"optimality gap: {opt_gap}\n")   
                    f.write(f"ci for optimality gap: {ci_gap}\n")
            # if (ci_gap[0] <= opt_gap) and (opt_gap<=ci_gap[1]):
            #     coverage_cnt += 1
            # total_len += ci_gap[1] - ci_gap[0]    
            if (ci_optimal[0] <= opt_obj) and (opt_obj<=ci_optimal[1]):
                coverage_cnt += 1
            total_len += ci_optimal[1] - ci_optimal[0]    

    # only rank 0 gets accumulated confidence interval
    if my_rank == 0:
        assert cfg.coverage_replications != 0
        return coverage_cnt / cfg.coverage_replications, total_len / cfg.coverage_replications
    else:
        return None, None

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("need json file")
        print("usage, e.g.: python -m bootsp.simulate_boot farmer.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)

    module = boot_utils.module_name_to_module(cfg.module_name)

    xhat_fname = cfg["xhat_fname"]

    coverage = main_routine(cfg, module)
    if my_rank == 0:
        print("Coverage", coverage)
    # print(f"--- {time.time()-start_time} seconds for rank {my_rank}")



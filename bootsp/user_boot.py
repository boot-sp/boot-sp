# A driver for boot_sp for command-line users

import sys
import numpy as np
import mpisppy.utils.config as config
import mpisppy.confidence_intervals.ciutils as ciutils
import bootsp.boot_utils as boot_utils
import bootsp.boot_sp as boot_sp
import bootsp.smoothed_boot_sp as smoothed_boot_sp

# TBD: we are using the mpi-sppy MPI wrapper to help windows users live without MPI.
import mpisppy.MPI as MPI
my_rank = MPI.COMM_WORLD.Get_rank()


def empirical_main_routine(cfg, module):
    """ The top level of user_boot; called by __main__
        and by drivers such as simulate_experiments.py
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers        
    Note:
        prints confidence interval results to the terminal
    """

    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        xhat = boot_utils.compute_xhat(cfg, module)
        
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

    if my_rank == 0:
        # print result    
        print(f"ci for optimal function value: {ci_optimal}")
        print(f"ci for function value at xhat: {ci_upper}")
        print(f"ci for optimality gap: {ci_gap}")


def smoothed_main_routine(cfg, module):
    cfg.add_to_config(name="use_fitted",
                    description="a boolean to control use of fitted distribution",
                    domain=bool,
                    default=None,
                    argparse=False)
    cfg.use_fitted = False
    cfg.add_to_config(name="fitted_distribution",
                    description="a fitted distribution from sample data",
                    domain=None,
                    default=None,
                    argparse=False)
    if "deterministic_data_json" in cfg:
        json_fname = cfg.deterministic_data_json
        try:
            with open(json_fname, "r") as read_file:
                detdata = json.load(read_file)
        except:
            print(f"Could not read the json file: {json_fname}")
            raise
        cfg.add_to_config("detdata",
                        description="determinstic data from json file",
                        domain=dict,
                        default=detdata)

    if cfg["xhat_fname"] is not None and cfg["xhat_fname"] != "None":
        xhat = ciutils.read_xhat(cfg["xhat_fname"])
    else:
        xhat = boot_utils.compute_xhat(cfg, module)

    if cfg.boot_method == "smoothed_boot_epi":
            ci_gap_two_sided = smoothed_boot_sp.smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline')
    elif cfg.boot_method == "smoothed_boot_kernel":
        ci_gap_two_sided = smoothed_boot_sp.smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-kernel')
    elif cfg.boot_method == "smoothed_boot_epi_quantile":
        ci_gap_two_sided = smoothed_boot_sp.smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline', quantile=True)
    elif cfg.boot_method == "smoothed_boot_kernel_quantile":
        ci_gap_two_sided = smoothed_boot_sp.smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-kernel', quantile=True)
    elif cfg.boot_method == "smoothed_bagging":
        ci_gap_two_sided = smoothed_boot_sp.smoothed_bagging(cfg, module, xhat, distr_type='univariate-kernel')
    else:
        raise ValueError(f"unrecognized method: {cfg.boot_method} ")
    
    if my_rank == 0:
        # print result    
        print(f"two-sided CI for optimality gap: {ci_gap_two_sided}")
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("need module name")
        print("usage python boot_sp.py module --options")
        print("usage (e.g.): python -m boosp.user_boot farmer.json  --help")
        print("   note: module name should not end it .py")
        quit()

    module_name = sys.argv[1]
    cfg = boot_utils.cfg_from_parse(module_name, name="user_boot")
    boot_utils.check_BFs(cfg)

    module = boot_utils.module_name_to_module(cfg["module_name"])

    xhat_fname = cfg["xhat_fname"]

    if "smoothed" not in cfg.boot_method:
        empirical_main_routine(cfg, module)
    else:
        smoothed_main_routine(cfg, module)


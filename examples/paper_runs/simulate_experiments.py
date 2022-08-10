import sys
import time
import numpy as np
import mpisppy.utils.config as config
import mpisppy.confidence_intervals.ciutils as ciutils
import boot_utils
import boot_sp
import datetime
import simulate_boot

# TBD: we are using the mpi-sppy MPI wrapper to help windows users live without MPI.
import mpisppy.MPI as MPI
my_rank = MPI.COMM_WORLD.Get_rank()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("need json file")
        print("usage, e.g.: python simulate_experiments.py schultz.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)
    module = boot_utils.module_name_to_module(cfg.module_name)
    
    sample_size_list = [40, 80]
    problem_sizelist = {"unique_schultz": [40, 80],
                        "nonunique_schultz": [40,80],
                        "cvar": [50, 100],
                        "farmer": [30, 60]
                    }
    nB_list = [100, 500]
    method_kfac = {"Classical_gaussian" : [0],
                   "Classical_quantile" : [0],
                   "Bagging_with_replacement": [0.4, 0.6, 0.8],
                   "Bagging_without_replacement": [0.4, 0.6, 0.8],
                   "Subsampling": [0.4, 0.6, 0.8],
                   "Extended": [0]
               }

    if my_rank == 0:
        append_name = cfg.module_name +"_table.tex"
        with open(append_name,  "w") as f:
            f.write("% "+cfg.module_name+"\n")
            f.write("\\begin{table}\n")
            f.write("\\begin{tabular}{||l|rrr|rr||}\n")
            f.write("\\hline\\hline\n")
            f.write("method & N & nB & k & avg len & coverage\\\\\n")

    for method, kmultlist in method_kfac.items():
        cfg.boot_method = method
        for nB in nB_list:
            if cfg.module_name in problem_sizelist:
                sample_size_list =problem_sizelist[cfg.module_name]
            for sample_size in sample_size_list:
                for kmult in kmultlist:
                    cfg.subsample_size = int(sample_size * kmult)
                    cfg.sample_size = sample_size
                    cfg.nB = nB
                    coverage, avg_len = simulate_boot.main_routine(cfg, module)
                    if my_rank == 0:
                        with open(append_name,  "a+") as f:
                            f.write(cfg.boot_method+" & " )
                            f.write(f"{cfg.sample_size}"+" & ")
                            f.write(f"{cfg.nB}"+" & ")
                            f.write(f"{cfg.subsample_size}"+" & ")
                            f.write(f"{avg_len:.2f}" +" & ")
                            f.write(f"{coverage:.3f}\\\\ \n")
    
    with open(append_name,  "a+") as f:
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write(f"\\caption{{Results for {cfg.module_name} based on {cfg.coverage_replications} replications. \\label{{tab:{cfg.module_name}}} }}\n")
        f.write("\\end{table}\n")

    # print(f"--- {time.time()-start_time} seconds for rank {my_rank}")
    # compare_dist()


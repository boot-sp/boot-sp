import sys
import time
import numpy as np
import json
import mpisppy.utils.config as config
import mpisppy.confidence_intervals.ciutils as ciutils
import bootsp.boot_utils as boot_utils
import bootsp.boot_sp as boot_sp
import bootsp.simulate_boot as simulate_boot
import datetime

# TBD: we are using the mpi-sppy MPI wrapper to help windows users live without MPI.
import mpisppy.MPI as MPI
my_rank = MPI.COMM_WORLD.Get_rank()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("need 2 json files")
        print("usage, e.g.: python simulate_experiments.py farmer.json experiments.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)
    module = boot_utils.module_name_to_module(cfg.module_name)
    
    setup_fname = sys.argv[2]
    with open(setup_fname, "r") as f:
        options = json.load(f)
    problem_sizelist = options["problem_sizelist"]
    nB_list = options["nB_list"]
    method_kfac = options["method_kfac"]
  
    if my_rank == 0:
        append_name = cfg.module_name +"_table.tex"
        with open(append_name,  "w") as f:
            f.write("% "+cfg.module_name+"\n")
            f.write("\\begin{table}\n")
            f.write("\\begin{center}\n")
            f.write("\\begin{tabular}{||l|rrr|rr||}\n")
            f.write("\\hline\\hline\n")
            f.write("method & N & nB & k & avg len & coverage\\\\\n")
            f.write("\\hline\\hline\n")

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
                            mtex = cfg.boot_method.replace("_","\\_")
                            f.write(mtex+" & " )
                            f.write(f"{cfg.sample_size}"+" & ")
                            f.write(f"{cfg.nB}"+" & ")
                            f.write(f"{cfg.subsample_size}"+" & ")
                            f.write(f"{avg_len:.2f}" +" & ")
                            f.write(f"{coverage:.3f}\\\\ \n")
    
    with open(append_name,  "a+") as f:
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{center}\n")
        mntex = cfg.module_name.replace("_","\\_")
        f.write(f"\\caption{{Results for {mntex} with $\\alpha$={cfg.alpha} based on {cfg.coverage_replications} replications. \\label{{tab:{cfg.module_name}}} }}\n")
        f.write("\\end{table}\n")

    # print(f"--- {time.time()-start_time} seconds for rank {my_rank}")
    # compare_dist()


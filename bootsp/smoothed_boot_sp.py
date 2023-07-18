# general-purpose bootstrap code
from os import dup, replace
from typing import Any
from mpisppy import global_toc
import mpisppy.utils.sputils as sputils
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.ciutils as ciutils
import mpisppy.confidence_intervals.mmw_ci as mmw_ci
import pyomo.environ as pyo
import sys
import scipy
import numpy as np
from numpy.random import default_rng
import scipy.stats as ss
import matplotlib.pyplot as plt
# TBD: we are using the mpi-sppy MPI wrapper to try to help windows users live without MPI.
import mpisppy.MPI as MPI
import json
import bootsp.boot_sp as boot_sp
import statdist

n_proc = MPI.COMM_WORLD.Get_size()
my_rank = MPI.COMM_WORLD.Get_rank()
comm = MPI.COMM_WORLD
rankcomm = comm.Split(key=my_rank, color=my_rank)  # single rank comm

def fit_distribution(sample_data, distr_type='univariate-epispline'):
    # fit distribution using∏ sample data
    # input either a list(for one variable) or a dictionary(for multivariate)
    # output same type
    distr_func = statdist.distribution_factory(distr_type)
    if isinstance(sample_data[0], (float,int)) : # 1-dim
        fitted_distr = distr_func.fit(sample_data)
    else:
        fitted_distr = {}
        for key in sample_data[0]:
            data = [data_dict[key] for data_dict in sample_data]
            fitted_distr[key] = distr_func.fit(data) 
    return fitted_distr


# def call_MMW(cfg, mpicomm):
#     refmodel = cfg.module_name
#     xhat = ciutils.read_xhat(cfg.xhat_fname)
#     num_batches = cfg.MMW_num_batches
#     batch_size = cfg.MMW_batch_size

#     cfg.quick_assign("EF_solver_name", domain = str, value = cfg.solver_name)
#     cfg.quick_assign("confidence_level", domain = float, value=1-cfg.alpha*2)
#     cfg.quick_assign("EF_2stage", bool, True)
      
#     mmw = mmw_ci.MMWConfidenceIntervals(refmodel, cfg, xhat, num_batches, batch_size=batch_size, start = cfg.seed_offset,verbose=True, mpicomm=mpicomm) 
#     cl = float(cfg.confidence_level)
#     r = mmw.run(confidence_level=cl)
#     return r

def center_original(cfg, module, xhat, scenario_pool):
    rng = default_rng(cfg.seed_offset) 
    
    # estimation of CI center
    cfg.use_fitted = False
    dag_upper = boot_sp.evaluate_scenarios(cfg, module,scenario_pool, xhat, duplication = False)
    dag_ef = boot_sp.solve_routine(cfg, module,scenario_pool, num_threads=2, duplication= False)
    dag_optimal = pyo.value(dag_ef.EF_Obj)
    dag_gap = dag_upper - dag_optimal # this is gamma(D) in the note    

    if my_rank == 0:
        print("use original N points to find center")
        return dag_gap
    else:
        return None

def center_smoothed(cfg, module, xhat, mpicomm):

    ctr_cfg = cfg()
    ctr_cfg.use_fitted = True   
    
    assert cfg.smoothed_center_sample_size is not None, "need a sample size for smoothed bootstrap center estimation"
    scenario_pool = list(range(cfg.seed_offset,  cfg.seed_offset + cfg.smoothed_center_sample_size))

    # print(f" at rank {my_rank}, i={i}, j={j}, scenario_pool is {scenario_pool}")

    center_upper = boot_sp.evaluate_scenarios(cfg, module,scenario_pool, xhat, duplication = False)
    center_ef = boot_sp.solve_routine(cfg, module,scenario_pool, num_threads=2, duplication= False)
    center_optimal = pyo.value(center_ef.EF_Obj)
    center_gap = center_upper - center_optimal
    
    if my_rank == 0:   
        return center_gap
    else:
        return None


def smoothed_resample_helper(cfg, module, xhat,serial=False):
   
    # same as bootstrap_helper

    if serial:
        local_nB = cfg.nB
    else:
        local_nB = boot_sp.slice_lens(cfg.nB)[my_rank]
        
    local_boot_gaps = np.empty(local_nB, dtype=np.float64) 

    boot_cfg = cfg() # for ephemeral changes to deal with seed_offset
    boot_cfg.use_fitted = True
     #use one large batch to find an approximation manually set MMW_nB to 1 in experiment.json
 
    for iter in range(local_nB):
        # seed_offset makes unique samples
        if serial:
            seed_offset = iter
        else:
            seed_offset = sum(boot_sp.slice_lens(boot_cfg.nB)[:my_rank]) + iter
        boot_cfg.seed_offset = cfg.seed_offset + seed_offset

        # if serial:
        #     r = call_MMW(boot_cfg, mpicomm=comm) 
        # else:
        #     r = call_MMW(boot_cfg, mpicomm=rankcomm) 
        scenario_pool = list(range( boot_cfg.seed_offset,  boot_cfg.seed_offset + cfg.subsample_size))

        # print(f" at rank {my_rank}, i={i}, j={j}, scenario_pool is {scenario_pool}")

        local_boot_upper = boot_sp.evaluate_scenarios(cfg, module,scenario_pool, xhat, duplication = False)
        local_boot_ef = boot_sp.solve_routine(cfg, module,scenario_pool, num_threads=2, duplication= False)
        local_boot_optimal = pyo.value(local_boot_ef.EF_Obj)
        local_boot_gaps[iter] = local_boot_upper - local_boot_optimal

        # local_boot_gap[iter] = r["Gbar"]
        # local_boot_std[iter] = r["std"]

    return local_boot_gaps


def smoothed_bootstrap(cfg, module, xhat, distr_type='univariate-epispline', quantile=False, serial=False):
    """ use the original data to estimate the center, then perform a smoothed estimation of width of confidence intervals
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and valuees are mpi-sppy nonant vectors
                     (i.. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        tuple with confidence interval if on MPI rank 0

    """
    
    rng = default_rng(cfg.seed_offset) 
    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False) 

    cfg.use_fitted = False
    sample_data = [module.data_sampler(scenario,cfg) for scenario in scenario_pool]
    cfg.fitted_distribution = fit_distribution(sample_data, distr_type=distr_type)
    
    # estimation of CI center
    # dag_gap = center_original(cfg, module, xhat, scenario_pool)
    global_toc("before fiding center")
    dag_gap = center_smoothed(cfg, module, xhat, mpicomm=comm)
    # dag_gap = center_original(cfg, module, xhat, scenario_pool)
    global_toc("after finding center")
    comm.Barrier()
    cfg.use_fitted = True
    # conduct an m out of n bootstrap, with B = cfg.nB, and m=cfg_MMW_batch_size


    cfg.subsample_size = cfg.sample_size
    local_boot_gaps = smoothed_resample_helper(cfg, module, xhat, serial)
    comm.Barrier()

    # do analysis only on rank 0
    if my_rank == 0:
        boot_gap = np.empty(cfg.nB, dtype=np.float64)
        boot_std = np.empty(cfg.nB, dtype=np.float64)
    else:
        boot_gap = None
        boot_std = None

    # but everyone needs to send to the gather
    lenlist = boot_sp.slice_lens(cfg.nB)
    comm.Gatherv(sendbuf=local_boot_gaps, recvbuf=(boot_gap, lenlist), root=0)


    if my_rank == 0:
        global_toc("Done bootstrap")
        print("subsample size for smoothed boot set to be the same as sample size")

        if not quantile:
            s_g = np.std(boot_gap, ddof = 1)
            ppf = ss.norm.ppf(1-cfg.alpha)
            error = s_g * ppf 
            ci_gap_two_sided =  [dag_gap - error, dag_gap + error]  
            # ci_gap_two_sided = [r["Gbar"] - error, r["Gbar"]+error]  

            # Ssquare = sum(np.square(r["Glist"]-dag_gap)) / cfg.MMW_num_batches
            # t_g = ss.t.ppf(1-cfg.alpha , cfg.MMW_num_batches)
            # eps = np.sqrt(Ssquare) * t_g * np.sqrt(cfg.MMW_batch_size/cfg.sample_size)
            # ci_gap_two_sided = [dag_gap - eps, dag_gap + eps]
        else:
            alpha = cfg.alpha        
 
            eps = np.quantile(boot_gap - dag_gap,  [alpha, 1-alpha]) 
            ci_gap_two_sided =  [dag_gap - eps[1], dag_gap-eps[0]]
        print(f"{ci_gap_two_sided = }")
        return ci_gap_two_sided
    else:
        return None
################################################


def smoothed_bagging(cfg, module, xhat, distr_type='univariate-kernel', serial=False):
    """ perform a bagging-based estimation of confidence intervals
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers
        xhat (dict): keys are scenario tree node names (e.g. ROOT) and values are mpi-sppy nonant vectors
                     (i.e. the specification of a candidate solution)
        serial (bool): indicates that only one MPI rank should be used
    Returns:
        tuple with confidence interval if on MPI rank 0
    """
    
    rng = default_rng(cfg.seed_offset)
    scenario_pool = rng.choice(cfg.max_count, size=cfg.sample_size, replace=False) 

    cfg.use_fitted = False
    sample_data = [module.data_sampler(scenario,cfg) for scenario in scenario_pool]
    cfg.fitted_distribution = fit_distribution(sample_data, distr_type=distr_type)
    cfg.use_fitted = True
    
    # Should Soon quit using MMW_batch_size

    local_nB = boot_sp.slice_lens(cfg.nB)[my_rank]
    local_gaps = np.empty(local_nB, dtype=np.float64)

    if my_rank == 0:
        bagging_gap = np.empty(cfg.nB, dtype=np.float64)
        all_gaps = []
        avg_gaps = []
    else:
        bagging_gap = None
        all_gaps = None
        avg_gaps = None

    assert cfg.smoothed_B_I is not None,  "B_I required for smoothed bagging"
    
    B_I = cfg.smoothed_B_I
    for i in range(B_I):
        seed_offset_base = cfg.seed_offset + cfg.nB * cfg.subsample_size * i

        for j in range(local_nB):
            seed_offset = seed_offset_base + (sum(boot_sp.slice_lens(cfg.nB)[:my_rank])+j) * cfg.subsample_size
            scenario_pool = list(range(seed_offset, seed_offset + cfg.subsample_size))
            scenario_pool[0] = seed_offset_base

            # print(f" at rank {my_rank}, i={i}, j={j}, scenario_pool is {scenario_pool}")

            local_upper = boot_sp.evaluate_scenarios(cfg, module,scenario_pool, xhat, duplication = False)
            local_ef = boot_sp.solve_routine(cfg, module,scenario_pool, num_threads=2, duplication= False)
            local_optimal = pyo.value(local_ef.EF_Obj)
            local_gaps[j] = local_upper - local_optimal
        comm.Barrier()
        lenlist = boot_sp.slice_lens(cfg.nB)
        comm.Gatherv(sendbuf=local_gaps, recvbuf=(bagging_gap, lenlist), root=0)

        if my_rank == 0:
            all_gaps = all_gaps + bagging_gap.tolist()
            avg_gaps.append(np.mean(bagging_gap))

    if my_rank == 0:
        print(f"{np.array(avg_gaps)=}")
        global_toc("Done Smoothed Bagging MH")
        print("subsample size for smoothed bagging set to be sample size//4")

        dag_gap = np.mean(avg_gaps) 
        print(f"{dag_gap=}")
        
        s1 = np.var(avg_gaps)
        s2 = np.var(all_gaps)
        ppf = ss.norm.ppf(1-cfg.alpha)
        s_g_2 = (cfg.subsample_size**2) * s1 / cfg.sample_size + s2/(B_I * cfg.nB)
        error = np.sqrt(s_g_2) * ppf 
        ci_gap_two_sided =  [dag_gap - error, dag_gap + error]  

        print(f"{ci_gap_two_sided = }")
        return ci_gap_two_sided
    else:
        return None

from os import dup, replace
import farmer
import mpisppy.utils.sputils as sputils
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.confidence_intervals.ciutils as ciutils
import pyomo.environ as pyo
import sys
import numpy as np
from numpy.random import default_rng
import scipy.stats as ss
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import json
import boot_sp
import mpisppy.utils.config as config
import boot_utils
import inspect
import importlib

# Compute the optimal function value with max_count scenarios
#(maybe read this from a file)

# find a candidate solution using the last candidate_sample_size scenarios and compute the corresponding optimality gap


def find_optimal(cfg, module):
    opt_ef = boot_sp.solve_routine(cfg, module, range(cfg.max_count), num_threads=16)
    opt_obj = pyo.value(opt_ef.EF_Obj)
    return opt_obj

def find_candidate(cfg, module):
    scenarios = range(cfg.max_count - cfg.candidate_sample_size,  cfg.max_count)
    if len(scenarios) == 1:
        print(f"only one scenario, {scenarios},  for candidate solution")
    candidate_ef = boot_sp.solve_routine(cfg, module, scenarios, num_threads=2, duplication = False)

    xhat = sputils.nonant_cache_from_ef(candidate_ef)
    return xhat

def find_gap(cfg, module, xhat, opt_obj):
    obj_hat = boot_sp.evaluate_scenarios(cfg, module, range(cfg.max_count), xhat, duplication = False)
    opt_gap = obj_hat - opt_obj
    return opt_gap


if __name__ == '__main__':
    if len(sys.argv) <2:
        print("need json file")
        print("usage (e.g.): python boot_general_prep.py little_schultz.json")
        quit()

    json_fname = sys.argv[1]
    cfg = boot_utils.cfg_from_json(json_fname)

    module = boot_utils.module_name_to_module(cfg.module_name)

    xhat_fname = cfg["xhat_fname"]

    opt_obj = find_optimal(cfg, module)
    xhat =  find_candidate(cfg, module)
    opt_gap = find_gap(cfg, module, xhat, opt_obj)
    
    np.save(cfg.optimal_fname, [opt_obj, opt_gap])
    ciutils.write_xhat(xhat, path=xhat_fname)

    print(f"opt_obj: {opt_obj}")
    print(f"opt_gap: {opt_gap}")


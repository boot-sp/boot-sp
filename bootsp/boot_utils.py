# utilities for the bootstrap code

import json
import enum
import inspect
import importlib
import mpisppy.utils.config as config

class BootMethods(enum.Enum):
    Classical_gaussian = "Classical_gaussian"
    Classical_quantile = "Classical_quantile"
    Extended = "Extended"
    Subsampling = "Subsampling"
    Bagging_with_replacement = "Bagging_with_replacement"
    Bagging_without_replacement = "Bagging_without_replacement"

    @classmethod
    def has_member_key(cls, key):
        return key in cls.__members__
    @classmethod
    def list_of_members(cls):
        return list(cls.__members__.keys())
    @classmethod
    def check_for_it(cls, key):
        if not cls.has_member_key(key):
            raise ValueError(f"Token={key} was not found in list={cls.list_of_members()}")

    
def module_name_to_module(module_name):
    if inspect.ismodule(module_name):
        module = mname
    else:
        module = importlib.import_module(module_name)
    return module


def cfg_for_boot():
    """ Create and return a Config object for boot-sp

    Returns:
        cfg (Config): the Pyomo config object with boot-sp options added
    """
    cfg = config.Config()
    # module name gets special parsing
    cfg.add_to_config(name="module_name",
                      description="file name that had scenario creator, etc.",
                      domain=str,
                      default=None,
                      argparse=False)
    cfg.add_to_config(name="max_count",
                      description="The total sample size=M+N",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="candidate_sample_size",
                      description="Sample size to for xhat=M",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="sample_size",
                      description="Sample size for bootstrap=N",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="subsample_size",
                      description="Bagging subsample_size",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="nB",
                      description="number of boot/bag samples",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="alpha",
                      description="significance level two-tailed (e.g. 0.05)",
                      domain=float,
                      default=None)
    cfg.add_to_config(name="seed_offset",
                      description="For some instances this enables replication.",
                      domain=int,
                      default=None)
    cfg.add_to_config(name="optimal_fname",
                      description="(optional for simulations) the name of a npy file with pre-stored optimal; use 'None' when not present",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="xhat_fname",
                      description="(optional) the name of an npy file with a pre-sored xhat; use 'None' when not present",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="solver_name",
                      description="name of solver (e.g. gurobi_direct)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="boot_method",
                      description="A token naming the boot method (e.g Bagging_with_replacement)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="trace_fname",
                      description="(optional) the name of an output file that will be appended to by simulate_experiments.py; use 'None' when not present",
                      domain=str,
                      default=None,
                      argparse = False)
    cfg.add_to_config(name="coverage_replications",
                      description="number of replications for simulating to get coverage rate.",
                      domain=int,
                      default=None,
                      argparse = False)
    cfg.add_to_config(name="smoothed_B_I",
                    description="number of initial fixed points to use in smoothed bagging.",
                    domain=int,
                    default=None)
    cfg.add_to_config(name="smoothed_center_sample_size",
                description="number of points to sample from the fitted distribution for the gap center.",
                domain=int,
                default=None)
    return cfg


def _process_module(mname):
    # factored code
    module = module_name_to_module(mname)
    cfg = cfg_for_boot()
    assert hasattr(module, "inparser_adder"), f"The module {module_name} must have the inparser_adder function"
    module.inparser_adder(cfg)
    assert len(cfg) > 0, f"cfg is empty after inparser_adder in {module_name}"    
    return cfg


def cfg_from_json(json_fname):
    """ create a Pyomo config object for the bootstrap code from a json file
    Args:
        json_fname (str): json file name, perhaps with path
    Returns:
        cfg (Config object): populated Config object
    Note:
        Used by the simulation code
    """
    try:
        with open(json_fname, "r") as read_file:
            options = json.load(read_file)
    except:
        print(f"Could not read the json file: {json_fname}")
        raise
    assert "module_name" in options, "The json file must include module_name"
    cfg = _process_module(options["module_name"])

    badtrip = False

    def _dobool(idx):
        if idx not in options:
            badtrip = True
            # such an index will raise two complaints...
            print(f"ERROR: {idx} must be in json {json_fname}")
            return
        if options[idx].lower().capitalize() == "True":
            options[idx] = True
        elif options[idx].lower().capitalize() == "False":
            options[idx] = False
        else:
            badtrip = True
            print(f"ERROR: Needed 'True' or 'False', got {options[idx]} for {idx}")


    # get every cfg index from the json
    for idx in cfg:
        if idx not in options:
            if "smoothed" in idx and "smoothed" not in cfg.boot_method:
                continue
            badtrip = True
            print(f"ERROR: {idx} not in the options read from {json_fname}")
            continue
        if options[idx] != "None":
            # TBD: query the cfg to see if it is bool
            if str(options[idx]).lower().capitalize() == "True" or str(options[idx]).lower().capitalize() == "False":
                _dobool(idx)  # do not return options, just modify cfg
            cfg[idx] = options[idx]
        else:
            cfg[idx] = None

    BootMethods.check_for_it(options["boot_method"])    
    if badtrip:
        raise RuntimeError(f"There were missing options in the json file: {json_fname}")
    else:
        return cfg


def cfg_from_parse(module_name, name=None):
    """ create a Pyomo config object for the bootstrap code from a json file
    Args:
        module_name (str): name of module with scenario creator and helpers
        name (str): name for parser on the command line (e.g. user_boot)
    Returns:
        cfg (Config object): Config object populated by parsing the command line
    """

    cfg = _process_module(module_name)

    parser = cfg.create_parser(name)
    # the module name is very special because it has to be plucked from argv
    parser.add_argument(
            "module_name", help="amalgamator compatible module (often read from argv)", type=str,
        )
    cfg.module_name = module_name
    
    args = parser.parse_args()  # from the command line
    args = cfg.import_argparse(args)
    
    return cfg


def compute_xhat(cfg, module):
    """  Deal with signatures specified by mpi-sppy to find an xhat (realy local to main_routine)
    Args:
        cfg (Config): paramaters
        module (Python module): contains the scenario creator function and helpers
    Returns:
        xhat (dict): the optimal nonants in a format specified by mpi-sppy
    Note: Basically, the code to solve for xhat must be provided in the module
    """
    xhat_fct_name = f"xhat_generator_{cfg.module_name}"
    if not hasattr(module, xhat_fct_name):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"{xhat_fct_name} when xhat-fname is not given")
    if not hasattr(module, "kw_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"kw_creator when xhat-fname is not given")
    if not hasattr(module, "scenario_names_creator"):
        raise RuntimeError(f"\nModule {cfg.module_name} must contain a function "
                           f"scenario_names_creator when xhat-fname is not given")
    #Computing xhat_k

    xhat_scenario_names = module.scenario_names_creator(cfg.candidate_sample_size, start=cfg.sample_size)
    
    xgo = module.kw_creator(cfg)
    xgo.pop("solver_name", None)  # it will be given explicitly
    ###xgo.pop("solver_options", None)  # it will be given explicitly
    xgo.pop("num_scens", None)
    xgo.pop("scenario_names", None)  # given explicitly
    xhat_fct = getattr(module, xhat_fct_name)
    xhat_k = xhat_fct(xhat_scenario_names, solver_name=cfg.solver_name, **xgo)
    return xhat_k


def check_BFs(cfg):
    BFs = cfg.get("Branching_factors", [0])
    if len(BFs) > 1:
        raise ValueError("Only two-stage problems are presently supported.\n"
                         f"branching_factors was {BFs}")
    

if __name__ == "__main__":
    print("boot_utils does not have a main program.")

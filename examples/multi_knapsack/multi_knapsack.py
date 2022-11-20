# DLW Nov 2022; Vaagen and Wallace, IJPE, 2007
# (see also the chapter in the King/Wallace book)
import json
import numpy as np
import pyomo.environ as pyo
from mpisppy.utils import config
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import mpisppy.utils.amalgamator as amalgamator

# Use this random stream:
sstream = np.random.RandomState(1)

def scenario_creator(scenario_name, cfg=None, seedoffset=0, num_scens=None):
    """ Create the CVaR examples using Schultz method we always use
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Config): the control parameters
        seedoffset (int): used by confidence interval code
    Returns:
        model (ConcreteModel): the Pyomo model
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex then
    # converted mod 3 into one of the below avg./avg./above avg. scenarios
    scennum   = sputils.extract_num(scenario_name)

    sstream.seed(scennum+seedoffset)  # allows for resampling easily

    # Create the concrete model object
    model = pyo.ConcreteModel("multi-knapsack")

    # deterministic data
    json_fname = cfg.deterministic_data_json
    try:
        with open(json_fname, "r") as read_file:
            detdata = json.load(read_file)
    except:
        print(f"Could not read the json file: {json_fname}")
        raise
    v = detdata["v"]
    c = detdata["c"]
    g = detdata["g"]
    alpha = detdata["alpha"]  # a dict of lists

    # use the same variable names as in chapter 6 of the King/Wallace book
    # item numbers start at 1
    model.I = pyo.RangeSet(detdata["num_prods"])

    model.x = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.y = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.z = pyo.Var(model.I, model.I, within=pyo.NonNegativeReals, initialize=0)
    model.zt = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)
    model.w = pyo.Var(model.I, within=pyo.NonNegativeReals, initialize=0)

    # tbd: update this...xxxx
    d = {i: int(detdata["mean_d"]*sstream.rand()/2) for i in model.I}

    # note: the json indexes are strings
    
    def d_rule(m, i):
        return m.y[i] + sum(m.z[j,i] for j in model.I if j != i) <= d[i]
    model.d_constraint = pyo.Constraint(model.I, rule=d_rule)

    def z_rule(m,i,j):
        # note that alpha is a dict of lists
        if i == j:
            return pyo.Constraint.Skip
        else:
            return m.z[i,j] <= alpha[str(i)][j-1] * (d[j]-m.y[j])
    model.z_constraint = pyo.Constraint(model.I, model.I, rule=z_rule)

    def zt_rule(m, i):
        return m.zt[i] == sum(m.z[i,j] for j in model.I if j != i)
    model.zt_constraint = pyo.Constraint(model.I, rule=zt_rule)

    def w_rule(m, i):
        return m.w[i] == m.x[i] - (m.y[i]+m.zt[i])
    model.w_constraint = pyo.Constraint(model.I, rule=w_rule)

    m = model  # typing aid
    model.Obj1 = pyo.Expression(expr=sum(v[str(i)]*(m.y[i]+m.zt[i])
                                         + g[str(i)]*m.w[i]
                                         - c[str(i)]*m.x[i] for i in m.I))

    model.obj = pyo.Objective(expr=model.Obj1)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    varlist = [model.x]
    sputils.attach_root_node(model, model.Obj1, varlist)
    
    #Add the probability of the scenario
    if num_scens is not None :
        model._mpisppy_probability = 1/num_scens
    return model

#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]


#=========
def inparser_adder(cfg):
    # add options unique to the model
    cfg.add_to_config("deterministic_data_json",
                      description="file name for json file with determinstic data",
                      domain=str,
                      default=None)

#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"cfg" : cfg}
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

#============================
def xhat_generator_multi_knapsack(scenario_names, solver_name=None,cfg=None):
    ''' Given scenario names and
    options, create the scenarios and compute the xhat that is minimizing the
    approximate problem associated with these scenarios.

    Parameters
    ----------
    scenario_names: list of str
        Names of the scenario we use
    cfg (Config): control parameters

    Returns
    -------
    xhat: xhat object (dict containing a 'ROOT' key with a np.array)
        A generated xhat.

    NOTE: this is here for testing during development.

    '''
    num_scens = len(scenario_names)
    
    xhat_cfg = cfg()
    xhat_cfg.quick_assign("num_scens", int, num_scens)
    xhat_cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)
    xhat_cfg.quick_assign("EF_2stage", bool, True)
    xhat_cfg.quick_assign("EF_solver_name", str, cfg.solver_name)

    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("multi_knapsack",
                                  xhat_cfg, use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat
        


if __name__ == "__main__":
    # main program just for developer testing (and therefore might not execute)

    solver_name = "cplex"
    cfg = config.Config()
    inparser_adder(cfg)
    cfg.parse_command_line("farmer_cylinders")
    # now, pretty much ignore the command line...
    num_scens = 1
    cfg["num_scens"] = num_scens
    scenario_names = scenario_names_creator(num_scens)
    scenario_creator_kwargs = kw_creator(cfg)
    
    ef = sputils.create_EF(
        scenario_names,
        scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
    )

    solver = pyo.SolverFactory(solver_name)
    if 'persistent' in solver_name:
        solver.set_instance(ef, symbolic_solver_labels=True)
        solver.solve(tee=True)
    else:
        solver.solve(ef, tee=True, symbolic_solver_labels=True,)

    print(f"EF objective: {pyo.value(ef.EF_Obj)}")
    #sputils.ef_ROOT_nonants_npy_serializer(ef, "lam_cvar_nonants.npy")
    solfile = "foo.out"
    representative_scenario = getattr(ef,ef._ef_scenario_names[0])
    sputils.first_stage_nonant_writer(solfile, 
                                        representative_scenario,
                                        bundling=False)
    print(f"Solution written to {solfile}")

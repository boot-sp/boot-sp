# DLW July 2022; CVaR as in Lam, Qian paper
import pyomo.environ as pyo
from mpisppy.utils import config
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import numpy as np

# Use this random stream:
sstream = np.random.RandomState(1)

def scenario_creator(scenario_name, num_scens=None, seedoffset=0):
    """ Create the CVaR examples using Schultz method we always use
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex then
    # converted mod 3 into one of the below avg./avg./above avg. scenarios
    scennum   = sputils.extract_num(scenario_name)

    sstream.seed(scennum+seedoffset)  # allows for resampling easily
    xi = np.random.normal(0,1)
    #print(f"{scenario_name}: {xi}")
    alpha = 0.1

    # Create the concrete model object
    model = pyo.ConcreteModel("Lam_CVaR")

    model.nu = pyo.Var(within=pyo.NonNegativeReals)  # second stage (xi - x)+ in L&Q
    model.eta = pyo.Var(within=pyo.Reals)  # first stage (x in Lam and Qian)

    model.Obj1 = pyo.Expression(expr=model.eta + (model.nu/alpha))

    model.obj = pyo.Objective(expr=model.Obj1)

    def excess_rule(m):
        return m.nu >= xi - m.eta
    model.excess_constraint = pyo.Constraint(rule=excess_rule)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Obj1,
            nonant_list=[model.eta],
            scen_model=model,
        )
    ]
    
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
    pass

#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : cfg.get('num_scens', None)}
    return kwargs


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

if __name__ == "__main__":
    # main program just for developer testing (and therefore might not execute)

    solver_name = "cplex"
    cfg = config.Config()
    inparser_adder(cfg)
    num_scens = 10000
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

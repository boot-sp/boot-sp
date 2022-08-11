# DLW July 2022
import pyomo.environ as pyo
from mpisppy.utils import config
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
import numpy as np

# Use this random stream:
sstream = np.random.RandomState()

def scenario_creator(scenario_name, num_scens=None, seedoffset=0):
    """ Create the little Schultz example
    
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
    scennum = scennum % 121

    sstream.seed(scennum+seedoffset)  # allows for resampling easily
    # ri1 = np.random.randint(5,15)
    # ri2 = np.random.randint(5,15)
    ri1 = scennum // 11 + 5
    ri2 = scennum % 11 + 5
    xi = [ri1, ri2]
    # print(f"{scenario_name}: {ri1}, {ri2}")

    fsc = (-1.5, -4)
    ssc = (-16, -19, -23, -28)
    T = [[2,3,4,5], [6,1,3,2]]

    # Create the concrete model object
    model = pyo.ConcreteModel(scenario_name)

    xrange = range(2)
    yrange = range(4)
    model.x = pyo.Var(xrange, within=pyo.NonNegativeIntegers, bounds=(0,5))
    model.y = pyo.Var(yrange, within=pyo.Binary)

    model.Obj1 = pyo.Expression(expr=sum(model.x[i] * fsc[i] for i in xrange))
    model.Obj2 = pyo.Expression(expr=sum(model.y[i] * ssc[i] for i in yrange))

    model.obj = pyo.Objective(expr=model.Obj1 + model.Obj2, sense=pyo.minimize)
    
    def upper_rule(m, i):
        return sum(T[i][j] * m.y[j] for j in yrange) <= xi[i] - m.x[i]
    model.constraint = pyo.Constraint(xrange, rule=upper_rule)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.Obj1,
            nonant_list=[model.x],
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
    # add options unique to this model
    pass

#=========
def kw_creator(cfg):
    # linked to the scenario_creator and inparser_adder
    kwargs = {"num_scens" : cfg.get('num_scens', None)}
    return kwargs

def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    """ Create a scenario within a sample tree. Mainly for multi-stage and simple for two-stage.
        (this function supports zhat and confidence interval code)
    Args:
        sname (string): scenario name to be created
        stage (int >=1 ): for stages > 1, fix data based on sname in earlier stages
        sample_branching_factors (list of ints): branching factors for the sample tree
        seed (int): To allow random sampling (for some problems, it might be scenario offset)
        given_scenario (Pyomo concrete model): if not None, use this to get data for ealier stages
        scenario_creator_kwargs (dict): keyword args for the standard scenario creator funcion
    Returns:
        scenario (Pyomo concrete model): A scenario for sname with data in stages < stage determined
                                         by the arguments
    """
    # Since this is a two-stage problem, we don't have to do much.
    sca = scenario_creator_kwargs.copy()
    sca["seedoffset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass

if __name__ == "__main__":
    # This is command line callable just to support ad hoc testing by developers
    m = scenario_creator("scen0")
    opt = pyo.SolverFactory('cplex')
    results = opt.solve(m)
    pyo.assert_optimal_termination(results)
    m.pprint()
    

# farmer_fit for fit_resample
# copied from mpi-sppy 19July 2022; xhat code added Aug 2022;
#   num-scens dropped from command line Aug 2022
# unlimited crops
# ALL INDEXES ARE ZERO-BASED
#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2018 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# special scalable farmer for stress-testing

import pyomo.environ as pyo
import numpy as np
import mpisppy.scenario_tree as scenario_tree
import mpisppy.utils.sputils as sputils
from mpisppy.utils import config
import mpisppy.utils.amalgamator as amalgamator
import statdist
from statdist.sampler import Sampler

# Use this random stream:
farmerstream = np.random.RandomState()

def _get_b(c, cv):
    # c is approximately the lower bound of crop yield, cv is approx coefficient of variation 
    return c*cv/(1/np.sqrt(12) -cv/2)

def _get_distr_dict(cfg):
    
    if not cfg.use_fitted:
        uunif = statdist.distribution_factory('univariate-unif')
        distr_dict ={}
        for i in range(cfg.crops_multiplier):
            distr_dict[f"WHEAT{i}"] = uunif(0, _get_b(2.5, cfg.yield_cv))
            distr_dict[f"CORN{i}"] = uunif (0,_get_b(3, cfg.yield_cv))
            distr_dict[f"SUGAR_BEETS{i}"] = uunif(0,_get_b(20, cfg.yield_cv))
    else:
        distr_dict = cfg.fitted_distribution
    return distr_dict

def scenario_creator(
    scenario_name, cfg, sense=pyo.minimize, seed_offset=None
):
    """ Create a scenario for the (scalable) farmer example.  
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        cfg (Config): 
            control parameters
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        seed_offset (int): used by confidence interval code
    """
    # scenario_name has the form <str><int> e.g. scen12, foobar7
    # The digits are scraped off the right of scenario_name using regex then
    # converted mod 3 into one of the below avg./avg./above avg. scenarios
    scennum   = sputils.extract_num(scenario_name)
    basenames = ['BelowAverageScenario', 'AverageScenario', 'AboveAverageScenario']
    basenum   = scennum  % 3
    groupnum  = scennum // 3
    scenname  = basenames[basenum]+str(groupnum)


    # The RNG is seeded with the scenario number so that it is
    # reproducible when used with multiple threads.
    # NOTE: if you want to do replicates, you will need to pass a seed
    # as a kwarg to scenario_creator then use seed+scennum as the seed argument.
    seed_offset = cfg.get("seed_offset",0) if seed_offset is None else seed_offset

    farmerstream.seed(scennum+seed_offset)

    use_integer= cfg.get('use_integer', False)
    crops_multiplier= cfg.get('crops_multiplier', 1)
    num_scens = cfg.get('num_scens', None)

    # Check for minimization vs. maximization
    if sense not in [pyo.minimize, pyo.maximize]:
        raise ValueError("Model sense Not recognized")

    uunif = statdist.distribution_factory('univariate-unif')

    distr_dict = _get_distr_dict(cfg)

    # Create the concrete model object
    model = pysp_instance_creation_callback(
        scenname,
        use_integer=use_integer,
        sense=sense,
        crops_multiplier=crops_multiplier,
        distr_dict=distr_dict,
        num_scens = num_scens
    )

    
    # data = data_sampler(scennum, cfg)
    # print(f"now printing data for scennum {scennum}")
    # print(data)

    return model



def data_sampler(record_num, cfg):
    # return the fluctuation data around the baseline from a sample
    # Note: we are syncronizing using the seed
    # yield as in "crop yield"

    distr_dict = _get_distr_dict(cfg)
    farmerstream.seed(record_num+cfg.seed_offset)
    groupnum  = record_num // 3

    sampler_dict = {}
    for i in range(cfg.crops_multiplier):
        if groupnum != 0:
            sampler_dict[f"WHEAT{i}"] = Sampler([distr_dict[f"WHEAT{i}"]], farmerstream)
            sampler_dict[f"CORN{i}"] = Sampler([distr_dict[f"CORN{i}"]], farmerstream)
            sampler_dict[f"SUGAR_BEETS{i}"] = Sampler([distr_dict[f"SUGAR_BEETS{i}"]], farmerstream)
 
    data = {}
    for i in range(cfg.crops_multiplier):
        if groupnum != 0:
            data[f"WHEAT{i}"] = sampler_dict[f"WHEAT{i}"].sample_one()[0]
            data[f"CORN{i}"] = sampler_dict[f"CORN{i}"].sample_one()[0]
            data[f"SUGAR_BEETS{i}"] = sampler_dict[f"SUGAR_BEETS{i}"].sample_one()[0]
        else:
            data[f"WHEAT{i}"] = 0
            data[f"CORN{i}"] = 0
            data[f"SUGAR_BEETS{i}"] = 0
    return data



def pysp_instance_creation_callback(
    scenario_name, use_integer=False, sense=pyo.minimize, crops_multiplier=1, distr_dict=None,  num_scens=None
):
    # long function to create the entire model
    # scenario_name is a string (e.g. AboveAverageScenario0)
    #
    # Returns a concrete model for the specified scenario

    # scenarios come in groups of three
    # print(scenario_name)
    scengroupnum = sputils.extract_num(scenario_name)
    scenario_base_name = scenario_name.rstrip("0123456789")
    
    model = pyo.ConcreteModel()

    def crops_init(m):
        retval = []
        for i in range(crops_multiplier):
            retval.append("WHEAT"+str(i))
            retval.append("CORN"+str(i))
            retval.append("SUGAR_BEETS"+str(i))
        return retval

    model.CROPS = pyo.Set(initialize=crops_init)

    #
    # Parameters
    #

    model.TOTAL_ACREAGE = 500.0 * crops_multiplier

    def _scale_up_data(indict):
        outdict = {}
        for i in range(crops_multiplier):
           for crop in ['WHEAT', 'CORN', 'SUGAR_BEETS']:
               outdict[crop+str(i)] = indict[crop]
        return outdict
        
    model.PriceQuota = _scale_up_data(
        {'WHEAT':100000.0,'CORN':100000.0,'SUGAR_BEETS':6000.0})

    model.SubQuotaSellingPrice = _scale_up_data(
        {'WHEAT':170.0,'CORN':150.0,'SUGAR_BEETS':36.0})

    model.SuperQuotaSellingPrice = _scale_up_data(
        {'WHEAT':0.0,'CORN':0.0,'SUGAR_BEETS':10.0})

    model.CattleFeedRequirement = _scale_up_data(
        {'WHEAT':200.0,'CORN':240.0,'SUGAR_BEETS':0.0})

    model.PurchasePrice = _scale_up_data(
        {'WHEAT':238.0,'CORN':210.0,'SUGAR_BEETS':100000.0})

    model.PlantingCostPerAcre = _scale_up_data(
        {'WHEAT':150.0,'CORN':230.0,'SUGAR_BEETS':260.0})

    #
    # Stochastic Data
    #
    Yield = {}
    Yield['BelowAverageScenario'] = \
        {'WHEAT':2.0,'CORN':2.4,'SUGAR_BEETS':16.0}
    Yield['AverageScenario'] = \
        {'WHEAT':2.5,'CORN':3.0,'SUGAR_BEETS':20.0}
    Yield['AboveAverageScenario'] = \
        {'WHEAT':3.0,'CORN':3.6,'SUGAR_BEETS':24.0}


    def Yield_init(m, cropname):
        # yield as in "crop yield"
        sampler = Sampler([distr_dict[cropname]], farmerstream)
        crop_base_name = cropname.rstrip("0123456789")
        if scengroupnum != 0:
            pertubation = sampler.sample_one()[0]
            # print(f"{pertubation =}")
            return Yield[scenario_base_name][crop_base_name]+ pertubation
        else:
            return Yield[scenario_base_name][crop_base_name]

    model.Yield = pyo.Param(model.CROPS,
                            within=pyo.NonNegativeReals,
                            initialize=Yield_init,
                            mutable=True)

    #
    # Variables
    #

    if (use_integer):
        model.DevotedAcreage = pyo.Var(model.CROPS,
                                       within=pyo.NonNegativeIntegers,
                                       bounds=(0.0, model.TOTAL_ACREAGE))
    else:
        model.DevotedAcreage = pyo.Var(model.CROPS, 
                                       bounds=(0.0, model.TOTAL_ACREAGE))

    model.QuantitySubQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantitySuperQuotaSold = pyo.Var(model.CROPS, bounds=(0.0, None))
    model.QuantityPurchased = pyo.Var(model.CROPS, bounds=(0.0, None))

    #
    # Constraints
    #

    def ConstrainTotalAcreage_rule(model):
        return pyo.sum_product(model.DevotedAcreage) <= model.TOTAL_ACREAGE

    model.ConstrainTotalAcreage = pyo.Constraint(rule=ConstrainTotalAcreage_rule)

    def EnforceCattleFeedRequirement_rule(model, i):
        return model.CattleFeedRequirement[i] <= (model.Yield[i] * model.DevotedAcreage[i]) + model.QuantityPurchased[i] - model.QuantitySubQuotaSold[i] - model.QuantitySuperQuotaSold[i]

    model.EnforceCattleFeedRequirement = pyo.Constraint(model.CROPS, rule=EnforceCattleFeedRequirement_rule)

    def LimitAmountSold_rule(model, i):
        return model.QuantitySubQuotaSold[i] + model.QuantitySuperQuotaSold[i] - (model.Yield[i] * model.DevotedAcreage[i]) <= 0.0

    model.LimitAmountSold = pyo.Constraint(model.CROPS, rule=LimitAmountSold_rule)

    def EnforceQuotas_rule(model, i):
        return (0.0, model.QuantitySubQuotaSold[i], model.PriceQuota[i])

    model.EnforceQuotas = pyo.Constraint(model.CROPS, rule=EnforceQuotas_rule)

    # Stage-specific cost computations;

    def ComputeFirstStageCost_rule(model):
        return pyo.sum_product(model.PlantingCostPerAcre, model.DevotedAcreage)
    model.FirstStageCost = pyo.Expression(rule=ComputeFirstStageCost_rule)

    def ComputeSecondStageCost_rule(model):
        expr = pyo.sum_product(model.PurchasePrice, model.QuantityPurchased)
        expr -= pyo.sum_product(model.SubQuotaSellingPrice, model.QuantitySubQuotaSold)
        expr -= pyo.sum_product(model.SuperQuotaSellingPrice, model.QuantitySuperQuotaSold)
        return expr
    model.SecondStageCost = pyo.Expression(rule=ComputeSecondStageCost_rule)

    def total_cost_rule(model):
        if (sense == pyo.minimize):
            return model.FirstStageCost + model.SecondStageCost
        return -model.FirstStageCost - model.SecondStageCost
    model.Total_Cost_Objective = pyo.Objective(rule=total_cost_rule, 
                                               sense=sense)

    # Create the list of nodes associated with the scenario (for two stage,
    # there is only one node associated with the scenario--leaf nodes are
    # ignored).
    model._mpisppy_node_list = [
        scenario_tree.ScenarioNode(
            name="ROOT",
            cond_prob=1.0,
            stage=1,
            cost_expression=model.FirstStageCost,
            nonant_list=[model.DevotedAcreage],
            scen_model=model,
        )
    ]
    
    #Add the probability of the scenario
    if num_scens is not None :
        model._mpisppy_probability = 1/num_scens
    # model._mpisppy_probability = "uniform"

    return model

# begin functions not needed by farmer_cylinders
# (but needed by special codes such as confidence intervals)
#=========
def scenario_names_creator(num_scens,start=None):
    # (only for Amalgamator): return the full list of num_scens scenario names
    # if start!=None, the list starts with the 'start' labeled scenario
    if (start is None) :
        start=0
    return [f"scen{i}" for i in range(start,start+num_scens)]
        


#=========
def inparser_adder(cfg):
    # add options unique to farmer
    #cfg.num_scens_required()  Not on the command line for bootstrap.
    cfg.add_to_config("crops_multiplier",
                      description="number of crops will be three times this (default 1)",
                      domain=int,
                      default=1)
    
    cfg.add_to_config("farmer_with_integers",
                      description="make the version that has integers (default False)",
                      domain=bool,
                      default=False)
    cfg.add_to_config("yield_cv",
                      description="approximate farmer crop yield coefficient of variation",
                      domain=float,
                      default=0.03)



#=========
def kw_creator(cfg):
    # (for Amalgamator): linked to the scenario_creator and inparser_adder
    kwargs = {"cfg": cfg}
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
    sca["seed_offset"] = seed
    sca["num_scens"] = sample_branching_factors[0]  # two-stage problem
    return scenario_creator(sname, **sca)


# end functions not needed by farmer_cylinders


#============================
def scenario_denouement(rank, scenario_name, scenario):
    sname = scenario_name
    s = scenario
    if sname == 'scen0':
        print("Arbitrary sanity checks:")
        print ("SUGAR_BEETS0 for scenario",sname,"is",
               pyo.value(s.DevotedAcreage["SUGAR_BEETS0"]))
        print ("FirstStageCost for scenario",sname,"is", pyo.value(s.FirstStageCost))


#============================
def xhat_generator_farmer(scenario_names, solvername="gurobi", solver_options=None, crops_multiplier=1, use_integer=False):
    ''' Given scenario names and
    options, create the scenarios and compute the xhat that is minimizing the
    approximate problem associated with these scenarios.

    Parameters
    ----------
    scenario_names: int
        Names of the scenario we use
    solvername: str, optional
        Name of the solver used. The default is "gurobi".
    solver_options: dict, optional
        Solving options. The default is None.
    crops_multiplier: int, optional
        A parameter of the farmer model. The default is 1.

    Returns
    -------
    xhat: xhat object (dict containing a 'ROOT' key with a np.array)
        A generated xhat.

    NOTE: this is here for testing during development.

    '''
    num_scens = len(scenario_names)
    
    cfg = config.Config()
    cfg.quick_assign("EF_2stage", bool, True)
    cfg.quick_assign("EF_solver_name", str, solvername)
    cfg.quick_assign("EF_solver_options", dict, solver_options)
    cfg.quick_assign("num_scens", int, num_scens)
    cfg.quick_assign("_mpisppy_probability", float, 1/num_scens)

    #We use from_module to build easily an Amalgamator object
    ama = amalgamator.from_module("mpisppy.tests.examples.farmer",
                                  cfg, use_command_line=False)
    #Correcting the building by putting the right scenarios.
    ama.scenario_names = scenario_names
    ama.run()
    
    # get the xhat
    xhat = sputils.nonant_cache_from_ef(ama.ef)

    return xhat
        

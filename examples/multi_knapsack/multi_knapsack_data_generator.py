import json
import random

if __name__ == '__main__': 
    template_fname = "multi_knapsack_data_temp.json"
    setup_fname = "multi_knapsack_data.json"
    num_prods = 6
    g_rate = 0.08
    exchange_rate = 0.1
    random.seed(0)
    
    with open(template_fname, "r") as f:
        options = json.load(f)
    options["num_prods"] = num_prods
    options["mean_d"] = { "high": 1160, "low": 116}
    options["stdev_d"] = {"high": 74, "low": 96}
    options["v"] = {str(i+1) : random.uniform(0, 20) for i in range(num_prods)}
    options["c"] =  {str(i+1) : min(random.uniform(0, 8), options["v"][str(i+1)]/2 ) for i in range(num_prods)}
    options["g"] = {str(i+1) : options["v"][str(i+1)] * g_rate for i in range(num_prods)}

    options["alpha"] = {}
    for i in range(num_prods):
        options["alpha"][str(i+1)] = [exchange_rate] * num_prods
        options["alpha"][str(i+1)][i] = 0
    
    with open(setup_fname, "w") as f:
        json.dump(options, f, indent=4)
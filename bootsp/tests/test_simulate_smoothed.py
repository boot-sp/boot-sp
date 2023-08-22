# This software is distributed under the 3-clause BSD License.
# Author: David L. Woodruff (circa August 2022)

# TBD: This needs to do a better job dealing with files and modules (dlw aug2022)

__version__ = 0.1

import os
import shutil
import unittest
import tempfile
import bootsp.boot_utils as boot_utils
import bootsp.simulate_boot as simulate_boot

import mpisppy.utils.sputils as sputils
print("Disabling tictoc output so there will be very little terminal output.")
sputils.disable_tictoc_output()


from mpisppy.tests.utils import get_solver,round_pos_sig

solver_available,solver_name, persistent_available, persistentsolver_name= get_solver()

my_dir = os.path.dirname(os.path.abspath(__file__))
main_example_path = os.path.join(my_dir,"..","..","examples")
if not os.path.exists(main_example_path):
    raise RuntimeError(f"Directory not found: {main_example_path}")

methods_empirical = ["Classical_gaussian",
         "Classical_quantile",
         "Bagging_with_replacement",
         "Bagging_without_replacement",
         "Subsampling",
         "Extended"]

methods_smoothed = ["Smoothed_boot_kernel",
                    "Smoothed_boot_kernel_quantile",
                    "Smoothed_boot_epi",
                    "Smoothed_bagging"]

#*****************************************************************************
class Test_simulate(unittest.TestCase):
    """ Test the boot_simulate code.
        Assumes naming conventions for filenames"""

    def setUp(self):
        # we might copy files to and cd to this temp dir
        # self.temp_dir = tempfile.TemporaryDirectory()
        # self.cwd = os.getcwd()
        shutil.copy("../../examples/multi_knapsack/multi_knapsack.py", "multi_knapsack.py")
        shutil.copy("../../examples/farmer/farmer.py", "farmer.py")


        
    def tearDown(self):
        # just in case
        # self.temp_dir.cleanup()
        # os.chdir(self.cwd)
        # pass
        os.remove("multi_knapsack.py")
        os.remove("farmer.py")


    def _mdir_path(self, dirname, module_name):
        # uses the directory structure and naming conventions
        # to get the path for the module from the main examples
        mdir = os.path.join(main_example_path, dirname)
        if not os.path.exists(mdir):
            raise RuntimeError(f"Directory not found: {mdir}")
        return mdir

    
    def _json_path(self, dirname, module_name):
        # uses the directory structure and naming conventions
        # to get the path to the json file for the example
        mdir_path = self._mdir_path(dirname, module_name)
        jpath = os.path.join(mdir_path,f"{module_name}.json")
        if not os.path.exists(jpath):
            raise RuntimeError(f"file not found: {jpath}")
        return jpath


    def _do_smoothed(self, dirname, module_name):
        # do the test, return a dictionary of return values
        ret_dict = dict()
        json_fname = self._json_path(dirname, module_name)
        cfg = boot_utils.cfg_from_json(json_fname)

        cfg.coverage_replications = 10
        cfg.seed_offset = 0
        cfg.solver_name = solver_name
        cfg.quick_assign("trace_fname", str, "_test_simuluate_smoothed.app")
        module = boot_utils.module_name_to_module(cfg.module_name)
        for method in methods_smoothed:
            print(f"Trying {method} for {module_name}")
            print(f"{cfg.seed_offset =}")
            # These are *not* good parameters for real use...
            cfg.boot_method = method
            cfg.sample_size = 40
            cfg.subsample_size = 20
            cfg.nB = 10
            cfg.smoothed_B_I = 3
            cfg.smoothed_center_sample_size = 10
            ret_dict[method] = simulate_boot.smoothed_main_routine(cfg, module)
        return ret_dict

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    # @unittest.skip("temporarily skip farmer")
    def test_smoothed_farmer(self):
        results = self._do_smoothed("farmer","smoothed_farmer")
        print(f"farmer {results =}")
        assert "0.2, 0.6" in str(results["Smoothed_boot_kernel"]), "failure on Smoothed boot kernel"
        assert "0.9, 1.0" in str(results["Smoothed_bagging"]), "failure on Smoothed bagging"

    @unittest.skip("temporarily skip multi-knapsack")
    def test_smoothed_knapsack(self):
        results = self._do_smoothed("multi_knapsack","smoothed_multi_knapsack")
        print(f"multi_knapsack {results =}")

if __name__ == '__main__':
    unittest.main()

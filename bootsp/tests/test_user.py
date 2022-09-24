# This software is distributed under the 3-clause BSD License.
# Author: David L. Woodruff (circa August 2022)

# TBD: This needs to do a better job dealing with files and modules (dlw aug2022)

__version__ = 0.1

import os
import shutil
import unittest
import tempfile
import bootsp.boot_utils as boot_utils
import bootsp.user_boot as user_boot

import mpisppy.utils.sputils as sputils

### sputils.disable_tictoc_output()

from mpisppy.tests.utils import get_solver,round_pos_sig

solver_available,solver_name, persistent_available, persistentsolver_name= get_solver()

my_dir = os.path.dirname(os.path.abspath(__file__))
main_example_path = os.path.join(my_dir,"..","..","examples")
if not os.path.exists(main_example_path):
    raise RuntimeError(f"Directory not found: {main_example_path}")

methods = ["Classical_gaussian",
         "Classical_quantile",
         "Bagging_with_replacement",
         "Bagging_without_replacement",
         "Subsampling",
         "Extended"]

#*****************************************************************************
class Test_user(unittest.TestCase):
    """ Test the boot_user code.
        Assumes naming conventions for filenames"""

    def setUp(self):
        # we might copy files to and cd to this temp dir
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()

        
    def tearDown(self):
        # just in case
        self.temp_dir.cleanup()
        os.chdir(self.cwd)
        pass


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

    
    def _make_arglist(self, solver_name):
        # many of these items will be overwritten (e.g., method)
        cmdlist = ["--max-count", "9999",
                   "--candidate-sample-size", "1",
                   "--sample-size", "40",
                   "--subsample-size", "10",
                   "--nB", "10",
                   "--alpha", "0.05",
                   "--seed-offset", "100",
                   "--xhat-fname", "cvar_xhat.npy",
                   "--solver-name", solver_name,
                   "--boot-method", methods[0]]
        return cmdlist
        
    def _do_one(self, dirname, module_name):
        # do the test, return a dictionary of return values
        ret_dict = dict()
        json_fname = self._json_path(dirname, module_name)

        cfg = boot_utils._process_module(module_name)
        parser = cfg.create_parser(f"test user {module_name}")
        arglist = self._make_arglist(solver_name)
        args = parser.parse_args(arglist)
        args = cfg.import_argparse(args)
        cfg.module_name = module_name

        cfg.solver_name = solver_name
        module = boot_utils.module_name_to_module(module_name)
        for method in methods:
            print(f"Trying {method} for {module_name}")
            # These are *not* good parameters for real use...
            cfg.boot_method = method
            cfg.sample_size = 40
            cfg.subsample_size = 20
            cfg.nB = 10
            ret_dict[method] = user_boot.main_routine(cfg, module)
        return ret_dict
    

    @unittest.skipIf(not solver_available,
                     "no solver is available")
    def test_cvar(self):
        results = self._do_one("cvar", "cvar")

if __name__ == '__main__':
    unittest.main()

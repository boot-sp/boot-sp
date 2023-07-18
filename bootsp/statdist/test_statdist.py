# Provide some test for statdist
# Author: David L. Woodruff 2018

import unittest
import pkg_resources
import tempfile
import sys
import os
import shutil
import json
from statdist.distributions import UnivariateDiscrete
from collections import OrderedDict

__author__ = 'David L. Woodruff <DLWoodruff@UCDavis.edu>'
__date__ = 'September 11, 2018'
__version__ = 1.0

class UnivariteDiscreteDistributionTester(unittest.TestCase):
    def setUp(self):
        bpoints = OrderedDict([(0, 0.1), (1, 0.8), (2, 0.1)]) 
        self.distribution = UnivariateDiscrete(bpoints)

    def test_mean(self):
        self.assertEqual(self.distribution.mean, 1)
                              
    def test_at_point(self):
        self.assertEqual(self.distribution.cdf(0), 0.1)
        self.assertEqual(self.distribution.cdf(1), 0.9)
        self.assertEqual(self.distribution.cdf(1.5), 0.9)

    def test_sampling(self):
        N = 1000
        totsofar = 0.0
        for i in range(N):
            totsofar += self.distribution.sample_one()
        meanfromsample = totsofar / N
        self.assertAlmostEqual(meanfromsample, self.distribution.mean, 1)

if __name__ == '__main__':
    unittest.main()

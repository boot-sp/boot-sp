# pseudo-random numbers from distributions
from random import sample


class Sampler:
    """"
        This class enables generation of pseudo random numbers from distributions
        args:
            distributions (list of BaseDistribution): we sample from inverse of the cdf; len implies sample dimension
            stream (np.random): should be seeded and reseeded by the caller
    """
    def __init__(self, distributions, stream):
        self.distributions = distributions
        self.stream = stream

    def sample_one(self):
        """
            Return a single sample from the distribution as a list
        """
        # independent variables
        retval = []

        for distr in self.distributions:
            unorm = self.stream.uniform(0,1)
            # print(f"{unorm=}")
            retval.append(distr.cdf_inverse(unorm))
        return retval
import numpy as np
from scipy.special import gamma
from scipy.stats import norm, t

from statdist.base_distribution import Parameter, params_as_args, params_as_args2
from statdist.distributions import epsilon
from statdist.copula import CopulaBase
from statdist.distribution_factory import register_distribution

class BivariateCopula(CopulaBase):
    """
    This class will serve as an abstract base class for all types of copula
    on two variables.
    """
    def __init__(self, dimkeys=None, parameters=None):
        CopulaBase.__init__(self, 2, dimkeys, parameters)

    def pdf(self, u, v):
        """

        """
        pass

    def fit(self, data, dimkeys=None):
        """

        """
        pass


class ArchimedeanCopula(BivariateCopula):
    """
    This class will serve as an abstract base class for archimedean copulas.
    Archimedean copulas are those determined by a generator function phi,
        C(u, v) = phi^-1(phi(u) + phi(v))
    """

    @classmethod
    def phi(cls, x, params=None):
        """
        The generator function for the Archimedean copula.
        """
        pass

    @classmethod
    def phi_inverse(cls, x, params=None):
        """
        The inverse of the generator function of the Archimedean copula.
        """
        pass

    @classmethod
    def cdf(cls, u, v, params=None):
        return cls.phi_inverse(
                cls.phi(u, params) + cls.phi(v, params), params)

    @classmethod
    def hfunc(cls, u, v, params=None):
        pass


@register_distribution(name='ClaytonCopula2D')
class ClaytonCopula(ArchimedeanCopula):

    theta = Parameter('theta', bounds=(epsilon, None))
    parameters = [theta]

    def __init__(self, theta, dimkeys=None):
        self.theta.set_value(theta)
        ArchimedeanCopula.__init__(self, dimkeys, self.parameters)

    @classmethod
    @params_as_args(['theta'])
    def phi(cls, x, params=None):
        theta = params['theta']

        return (1/theta) * (x**(-theta) - 1)

    @classmethod
    @params_as_args(['theta'])
    def phi_inverse(cls, x, params=None):
        theta = params['theta']

        return (1 + theta*x)**(-1/theta)

    @classmethod
    @params_as_args2(['theta'])
    def pdf(cls, u, v, params=None):
        theta = params['theta']
        factor1 = (1 + theta)*(u*v)**(-1-theta)
        factor2 = (u**(-theta) + v**(-theta) - 1)**(-1/theta - 2)
        return factor1 * factor2

    @classmethod
    @params_as_args2(['theta'])
    def cdf(cls, u, v, params=None):
        theta = params['theta']
        return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

    @classmethod
    @params_as_args2(['theta'])
    def hfunc(cls, u, v, params=None):
        theta = params['theta']
        factor1 = v**(-theta-1)
        factor2 = (u**(-theta) + v**(-theta) - 1)**(-1 - 1/theta)
        return factor1 * factor2

    @classmethod
    @params_as_args2(['theta'])
    def hinv(cls, u, v, params=None):
        theta = params['theta']
        summand1 = (u * v**(theta+1))**(-(theta/(theta+1)))
        summand2 = 1 - v**(-theta)
        return (summand1 + summand2)**(-1/theta)


@register_distribution(name='FrankCopula2D')
class FrankCopula(ArchimedeanCopula):

    theta = Parameter('theta')
    parameters = [theta]

    def __init__(self, theta, dimkeys=None):
        self.theta.set_value(theta)
        ArchimedeanCopula.__init__(self, dimkeys, self.parameters)

    @classmethod
    @params_as_args(['theta'])
    def phi(cls, x, params=None):
        theta = params['theta']

        numer = np.exp(-theta * x) - 1
        denom = np.exp(-theta) - 1

        return -np.log(numer/denom)

    @classmethod
    @params_as_args(['theta'])
    def phi_inverse(cls, x, params=None):
        theta = params['theta']

        arg = 1 + np.exp(-x)*(np.exp(-theta)-1)
        return (-1/theta)*np.log(arg)

    @classmethod
    @params_as_args2(['theta'])
    def pdf(cls, u, v, params=None):
        theta = params['theta']
        numer = (np.exp(theta) - 1)*(theta)*(np.exp(theta*(u+v+1)))
        denom = (np.exp(theta) - np.exp(theta*(1+u))\
                 + np.exp(theta*(u+v)) - np.exp(theta*(1+v)))**2
        return numer / denom

    @classmethod
    @params_as_args2(['theta'])
    def cdf(cls, u, v, params=None):
        theta = params['theta']
        numer = (np.exp(-theta * u) - 1) * (np.exp(-theta * v) - 1)
        denom = np.exp(-theta) - 1
        return -(1/theta) * np.log(1 + numer/denom)


@register_distribution(name='GumbelCopula2D')
class GumbelCopula(ArchimedeanCopula):

    theta = Parameter('theta', bounds=(epsilon, None))
    parameters = [theta]

    def __init__(self, theta, dimkeys=None):
        self.theta.set_value(theta)
        ArchimedeanCopula.__init__(self, dimkeys, self.parameters)

    @classmethod
    @params_as_args(['theta'])
    def phi(cls, x, params=None):
        theta = params['theta']

        return (-np.log(x))**theta

    @classmethod
    @params_as_args(['theta'])
    def phi_inverse(cls, x, params=None):
        theta = params['theta']

        return np.exp(-x**(1/theta))

    @classmethod
    @params_as_args2(['theta'])
    def pdf(cls, u, v, params=None):
        theta = params['theta']

        factor1 = cls.cdf(u, v, params) * (u*v)**(-1)
        factor2 = ((-np.log(u))**theta + (-np.log(v))**theta)**(-2 + 2/theta)
        factor3 = (np.log(u) * np.log(v))**(theta - 1)
        argument = ((-np.log(u))**theta + (-np.log(u))**theta)**(-1/theta)
        factor4 = 1 + (theta - 1)*argument

        return factor1 * factor2 * factor3 * factor4

    @classmethod
    @params_as_args2(['theta'])
    def cdf(cls, u, v, params=None):
        theta = params['theta']

        return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

    @classmethod
    @params_as_args2(['theta'])
    def hfunc(cls, u, v, params=None):
        theta = params['theta']

        factor1 = cls.cdf(u, v, params) * (1/v) * (-np.log(v))**(theta-1)
        factor2 = ((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta - 1)

        return factor1 * factor2


@register_distribution(name='IndependenceCopula2D')
class IndependenceCopula(ArchimedeanCopula):
    @classmethod
    def phi(cls, x, params=None):
        return (-np.log(x))

    @classmethod
    def phi_inverse(cls, x, params=None):
        return np.exp(-x)

    @classmethod
    @params_as_args2([])
    def pdf(cls, u, v, params=None):
        return 1

    @classmethod
    @params_as_args2([])
    def hfunc(cls, u, v, params=None):
        return u


class JoeCopula(ArchimedeanCopula):
    @classmethod
    @params_as_args(['theta'])
    def phi(cls, x, params=None):
        theta = params['theta']

        return -np.log(1 - (1-x)**theta)

    @classmethod
    @params_as_args(['theta'])
    def phi_inverse(cls, x, params=None):
        theta = params['theta']

        return 1 - (1-np.exp(-x))**(1/theta)


@register_distribution(name='GaussianCopula2D')
class GaussianCopula2D(BivariateCopula):
    """

    """

    rho = Parameter('rho')
    parameters = [rho]

    def __init__(self, rho, dimkeys=None):
        self.rho.set_value(rho)
        self.R = np.array([[1, rho], [rho, 1]])
        BivariateCopula.__init__(self, dimkeys, self.parameters)

    @classmethod
    @params_as_args2(['rho'])
    def pdf(cls, u, v, params=None):
        rho = params['rho']
        factor1 = 1 / np.sqrt(1 - rho**2)
        numer = rho**2 * (u**2 + v**2) - 2*rho*u*v
        denom = 2*(1-rho**2)
        factor2 = np.exp(-numer/denom)
        return factor1 * factor2

    @classmethod
    @params_as_args2(['rho'])
    def hfunc(cls, u, v, params=None):
        rho = params['rho']
        numer = norm.ppf(u) - rho * norm.ppf(v)
        denom = np.sqrt(1 - rho**2)
        return norm.cdf(numer/denom)

    @classmethod
    @params_as_args2(['rho'])
    def hinv(cls, u, v, params=None):
        rho = params['rho']
        first = norm.ppf(u) * np.sqrt(1 - rho**2)
        second = rho * norm.ppf(v)
        return norm.cdf(first + second)


@register_distribution(name='StudentCopula2D')
class StudentCopula2D(BivariateCopula):
    rho = Parameter('rho')
    df = Parameter('df', bounds=(epsilon, None))
    parameters = [rho, df]

    def __init__(self, rho, df, dimkeys=None):
        rho.set_value(rho)
        df.set_value(df)
        BivariateCopula.__init__(self, dimkeys, self.parameters)

    @classmethod
    @params_as_args2(['rho', 'df'])
    def pdf(cls, u, v, params=None):
        rho = params['rho']
        df = params['df']

        numer1 = gamma((df+2)/2) / gamma(df/2)
        denom1 = df * np.pi * t.pdf(u, df) * t.pdf(v, df) * np.sqrt(1 - rho**2)
        factor1 = numer1 / denom1

        numer2 = u**2 + v**2 - 2*rho*u*v
        denom2 = df * (1 - rho**2)
        factor2 = (1 + numer2/denom2)**(-(df+1)/2)

        return factor1*factor2

    @classmethod
    @params_as_args2(['rho', 'df'])
    def hfunc(cls, u, v, params=None):
        rho = params['rho']
        df = params['df']

        numer1 = t.ppf(u, df) - rho*t.ppf(v, df)
        numer2 = (df + t.ppf(v, df)**2) * (1 - rho**2)
        denom2 = df + 1
        denom1 = np.sqrt(numer2/denom2)

        return t.pdf(numer1/denom1, df+1)

    @classmethod
    @params_as_args2(['rho', 'df'])
    def hinv(cls, u, v, params=None):
        rho = params['rho']
        df = params['df']

        numer = (df + t.ppf(v, df)**2) * (1 - rho**2)
        denom = df + 1
        summand1 = t.ppf(u, df+1)*np.sqrt(numer/denom)

        summand2 = rho*t.ppf(v, df)

        return t.pdf(summand1+summand2, df)

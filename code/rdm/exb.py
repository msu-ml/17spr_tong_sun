# expression builders.
import numpy as np
import theano
from theano import tensor as T
FX = theano.config.floatX


class Ods:
    """ expression builder for ordernal sigmoid expression. """

    def __init__(self, lvl=2, shp=1.0):
        self.shp = shp
        self.lvl = T.constant(lvl, 'float32')

        from scipy.stats import norm
        bar = norm.ppf(np.linspace(0.0, 1.0, lvl + 1))[1:lvl]
        self.bar = bar.reshape(lvl - 1, 1, 1).astype('f')

    def __call__(self, x):
        _ = T.nnet.sigmoid(self.shp * (x - self.bar))
        return T.sum(_, 0) / (self.lvl - 1) * (1 - 1e-6) + 5e-7


def CE(y, z):
    """ symbolic expression of Binary Cross Entrophy
    y: predicted binary probability.
    z: true binary probability, either 0 or 1, the default is 0.
    CE = - z * log(y) + (1 - z) * log(1 - y)

    The first dimension denotes the sampling units, and the last
    dimension denotes the value.
    """
    # protection against numerical instability
    _ = -z * T.log(y) - (1 - z) * T.log(1 - y)
    return T.sum(_, range(1, _.ndim))


def CM(y, z):
    """ symbolic expression of Multinomial Cross Entrophy
    y: predicted probabilities, supposely sum up to 1.
    z: true probabilites, supposedly sum up to 1.
    CE = - z * log(y) + (1 - ) * log(1 - y)

    The first dimension go through samples, the last dimension go
    through features. Rest of the dimensions falls in between denote
    categories, or levels, of each feature.
    """
    _ = -z * T.log(y)
    return T.sum(_, range(1, _.ndim))

    
def L2(y, z=None):
    """ build symbolic expression of L2 norm
    y: predicted value
    z: true value, the default is 0.

    L2 = sum((y - z)^2, dim=1:-1)

    The first dimension denote sampling units.
    """
    u = y - z if z else y
    return T.sqrt(T.sum(u**2, -1))


def L1(y, z=None):
    """ build symbolic expression of L1 norm
    y: predicted value
    z: true value, the default is 0.

    The first dimension of y, z denote the batch size.
    """
    u = y - z if z else y
    return T.sum(T.abs_(u), -1)


def L0(y, z=None, thd=1e-06):
    """ build symbolic expression of L0 norm. """
    u = y - z if z else y
    return T.sum((T.abs_(u) > thd), -1, FX)


def BH(y, z):
    """ binary hinge loss. Assume the lables are binary coded by {0, 1}.
    """
    a = 2.0 * z - 1             # 1 -> 1, 0 -> -1
    b = 2.0 * y - 1             # + -> 1, - -> -1
    x = a * b                   # + -> correct, - -> loss
    return 0.5 * (T.abs_(x) - x)


# --------  squashers  -------- #
class Relu:
    """ Parametric Relu Class. """
    def __init__(self):
        pass
    
    def __call__(self, x, alpha):
        """ parametric Relu.
        x : symbolic tensor
        Tensor to compute the activation function for.

        alpha : `scalar or tensor, optional`
        Slope for negative input, usually between 0 and 1. The default value
        of 0 will lead to the standard rectifier, 1 will lead to
        a linear activation function, and any value in between will give a
        leaky rectifier. A shared variable (broadcastable against `x`) will
        result in a parameterized rectifier with learnable slope(s).
        """
        if alpha is not None:
            f1 = 0.5 * (1.0 + alpha)
            f2 = 0.5 * (1.0 - alpha)
            return f1 * x + f2 * T.abs_(x)
        else:
            return 0.5 * (x + T.abs_(x))

    def __str__(self):
        return 'relu'


class Fsig:
    """ fast sigmoid. """
    def __init__(self):
        pass

    def __call__(self, x, alpha):
        """ parametric fast sigmoid.
        x: symbolic tensor to compute the activation
        alpha: optional shape parameter """

        if alpha is not None:
            return (1.0 + (alpha * x) / (1.0 + T.abs_(alpha * x))) / 2.0
        else:
            return (1.0 + x / (1 + T.abs_(x))) / 2.0

    def __str__(self):
        return 'fsig'

    
relu = Relu()
fsig = Fsig()
softmax = T.nnet.softmax
sigmoid = T.nnet.sigmoid
softplus = T.nnet.softplus

# squashers to initials
S2I = {'relu': 'RL', 'softmax': 'SM', 'sigmoid': 'SG', 'softplus': 'SP'}

# initials to squashers
I2S = {'RL': 'relu', 'SM': 'softmax', 'SG': 'sigmoid', 'SP': 'softplus'}

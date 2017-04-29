import numpy as np
import theano
from theano import tensor as T
from theano import shared as S


class Nnt(list):
    """
    Generic layer of neural network
    """

    def __init__(self, seed=None, **kwd):
        """
        -- seed: seed for random number generator.

        ** tag: a short representation of the network.
        """
        # numpy random number generator
        seed = kwd.pop('seed', None)
        nrng = kwd.pop('nrng', np.random.RandomState(seed))

        from theano.tensor.shared_randomstreams import RandomStreams
        trng = kwd.pop('trng', RandomStreams(nrng.randint(0x7FFFFFFF)))

        # private members
        self.__seed__ = seed
        self.__nrng__ = nrng
        self.__trng__ = trng

        # rest of the keywords become public members.
        self.__dict__.update(kwd)

    def __call__(self, x, **kwd):
        """
        makes the network a callable object.
        """
        return self.__expr__(x, **kwd)

    def __expr__(self, x, **kwd):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        return x

    def __pcpy__(self, nnt, **kwd):
        """
        shallowly paste parameter values onto another network of exact
        topology.

        nnt: the target network to print parameter values. if None, a new
        network is created.

        kwd: dictionary of additional keywords.

        return:
        the target neural network with parameter pasted.

        for a recursive deep copy, use hlp.cp instead.
        """
        if not self.__homo__(nnt):
            raise ValueError('cannot cp parameters to different shapes.')

        # parameters of target, they are shared tensors
        par = nnt.__parm__()
        dct = nnt.__dict__
        for k, v in self.__parm__().items():
            # get source parameter values:
            v = v.get_value()

            # update values in the target
            if k in par:
                par[k].set_value(v)
            elif k in dct:      # k is a member but not a shared tensor
                raise ValueError('cannot cp to non-shared-tensor.')
            else:
                dct[k] = S(v)  # creat new member if possible
        # done
        return nnt

    def __parm__(self, **kwd):
        """ Sallowly list parameters to be tuned by a trainer, they must
        be shared tensors, and should be on the symbolic expression.
        By default, all shared tensor members are returned.

        """
        # type of the shared tensor differs between CPU and GPU
        if theano.config.device.startswith('gpu'):
            t = theano.sandbox.cuda.var.CudaNdarraySharedVariable
        else:
            t = T.sharedvar.TensorSharedVariable
        return dict(
            (k, v) for k, v in vars(self).viewitems() if isinstance(v, t))

    def __wreg__(self, **kwd):
        """ sallowly list weight parameters. weights will later be subjected
        to regulator terms for model decay. (e.g. LASSO and Regee regression.)
        By default, the shared tensor member named 'w' will be selected.
        """
        p = self.__parm__()
        return dict(
            (k, v) for k, v in p.viewitems() if k.startswith('w'))

    def __homo__(self, that):
        """ sallowly test homogeneity of topology with another network. """
        p1, p2 = self.__parm__(), that.__parm__()
        if p1.keys() != p2.keys():
            return False
        for k in p1.keys():
            if p1[k].get_value().shape != p2[k].get_value().shape:
                return False
        return True

    def __repr__(self):
        sup = super(Nnt, self).__repr__()
        tag = self.__dict__.get('tag', self.__class__.__name__.upper())
        return '{}{}'.format(tag, sup)


def tst1():
    from pcp import Pcp
    p1 = Pcp([100, 200])
    p2 = Pcp([100, 200])
    # print(np.all(p1.w.get_value() == p2.w.get_value()))

    # p1.__pcpy__(p2)
    # print(np.all(p1.w.get_value() == p2.w.get_value()))

    return p1, p2

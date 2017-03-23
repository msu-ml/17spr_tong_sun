import numpy as np
import theano
import theano.tensor as T
from theano import config as cfg
from copy import deepcopy as dp


def S(v, name=None, dtype=None, strict=False):
    """ create shared variable from v
    dtype: forced data type.
    """
    # return None as it is.
    if v is None:
        return v
    
    if type(v) is T.sharedvar.TensorSharedVariable:
        return v
    # wrap python type to numpy type
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype)
    if dtype and v.dtype != dtype:
        v = v.astype(dtype)

    # use float 32 if necessary
    if dtype is None and v.dtype == 'f8':
        v = v.astype(cfg.floatX)

    # broadcasting pattern
    b = tuple(s == 1 for s in v.shape)

    return theano.shared(v, name, strict=strict, broadcastable=b)


# tensor contant creator
def C(v, name=None, dtype=None, strict=False):
    """ creat constant from v. """
    if type(v) is T.TensorConstant:
        v = v.data
        
    if not isinstance(v, np.ndarray):
        v = np.array(v, dtype)

    if dtype is None and v.dtype == 'f8':
        v = v.astype(cfg.floatX)

    # name
    name = str(v) if name is None else name

    return T.constant(v, name, v.ndim)


# type checkers
def is_tvar(x):
    """ is x a theano symbolic variable with no explict value. """
    return type(x) is T.TensorVariable


def is_tshr(x):
    """ is x a theano shared variable """
    if type(x) is T.sharedvar.TensorSharedVariable:
        return True
    if not cfg.device.startswith('gpu'):
        return False
    if type(x) is theano.sandbox.cuda.var.CudaNdarraySharedVariable:
        return True
    return False


def is_tcns(x):
    """ is x a theano tensor constant """
    return type(x) is T.TensorConstant


def is_tnsr(x):
    """ is x a theano tensor """
    return (is_tvar(x) or is_tshr(x) or is_tcns(x))


# fetch parameters
def parms(y, chk=None):
    """
    find parameters in symbolic expression {y}.

    chk: checker for allagible parameter. By default only shared
    variables could pass.
    """
    chk = is_tshr if chk is None else chk

    from collections import OrderedDict

    d = OrderedDict()
    q = [y]
    while len(q) > 0:
        v = q.pop()
        q.extend(v.get_parents())
        if chk(v):
            d[v] = v

    return list(d.keys())


def wrg(nnt):
    """ deeply search the weight parameters of the network."""
    stk, ret = [nnt], []
    while stk:
        s = stk.pop()
        ret.extend(s.__wreg__())
        stk.extend(s)
    return ret


def par(src):
    """ deeply search the parameters of the source network."""
    stk, ret = [src], []
    while stk:
        # pop up the networks at the top
        s = stk.pop()

        # search for parameters
        ret.extend(s.__parm__().viewvalues())

        # push in child networks
        stk.extend(s)

    # return the collection
    return ret


def paint(src, tgt=None):
    """ paint the parameters values of a source onto the target network.
    If the target is None, a new network is automatically instantiated,
    otherwise the call must be sure the two are homogenuous.

    src, tgt: source and target networks.
    """

    if tgt is None:
        return dp(src)
    
    # the stack to hold recursive networks
    stk = [(src, tgt)]
    while len(stk) > 0:
        # pop up the networks at the top
        s, d = stk.pop()

        # do a shallow painting
        s.__pcpy__(d)

        # push in child networks
        stk.extend(zip(s, d))

    # dummy return
    return tgt


def dhom(src, tgt):
    """ deeply test if the source network is topologically homogeneous
    to the target network.

    src, tgt: source and target networks.
    """

    # the stack to hold recursive networks
    stk = [(src, tgt)]
    while len(stk) > 0:
        # pop up the networks at the top
        s, t = stk.pop()

        # homogeneity test:
        if not s.__homo__(t):
            return False

        # push in child networks
        stk.extend(zip(s, t))

    # source and target are deeply homogenuous.
    return True


def cv_msk(x, k, permu=True):
    """ return masks that randomly partition the data into {k} parts.
    x: the sample size or the sample data
    k: the number of partitions.
    """
    n = x if isinstance(x, int) else len(x)
    idx = np.array(np.arange(np.ceil(n / float(k)) * k), '<i4')
    if permu:
        idx = np.random.permutation(idx)
    idx = idx.reshape(k, -1) % n
    msk = np.zeros((k, n), 'bool')
    for _ in range(k):
        msk[_, idx[_, ]] = True
    return msk


def test():
    from sae import SAE
    dm = [100, 200, 300]
    sa1 = SAE.from_dim(dm)
    sa2 = SAE.from_dim(dm)
    return sa1, sa2

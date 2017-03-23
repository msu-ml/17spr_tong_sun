# deep learning test
import numpy as np
from tnr.bas import Base as Trainer
from sae import SAE
import os
from os import path as pt


def main(fnm, nep=20, out=None, rdp=0, **kwd):
    """ Performance test for sigmoid, relu autoencoders. 
    out: output location.
    rdp: reduce network depth.
    """

    # pick data file
    if(pt.isdir(fnm)):
        fnm = pt.join(fnm, np.random.choice(os.listdir(fnm)))
    dat = np.load(fnm)

    # convert to float 32 for GPU
    gmx = dat['gmx'].astype('f')
    
    # flatten the two copies of genotype: [copy1, copy2]
    gmx = gmx.reshape(gmx.shape[0], -1)
    N, D = gmx.shape[0], gmx.shape[1]

    # separate training and testing data
    _ = np.zeros(N, 'i4')
    _[np.random.randint(0, N, N/5)] = 1
    x = gmx[_ > 0]            # training data
    u = gmx[_ < 1]            # testing data

    # train autoencoders
    # each layer half dimensionality
    d = [D]
    while d[-1] > 1:
        d.append(d[-1]/2)
    d[-1] = 1
    [d.pop() for _ in range(rdp)]

    # train the sigmoid network
    n1 = SAE.from_dim(d, s='sigmoid', **kwd)
    t1 = Trainer(n1, x, u=u, lrt=1e-3, err='CE', **kwd)
    t1.tune(nep)
    h1 = t1.query()
    del t1                      # release

    # train the relu network
    n2 = SAE.from_dim(d, s='relu', **kwd)
    n2[-1].s = 'sigmoid'
    t2 = Trainer(n2, x, u=u, lrt=1e-3, err='CE', **kwd)
    t2.tune(nep)
    h2 = t2.query()
    del t2

    # train the parametric softplus network
    n3 = SAE.from_dim(d, s='softplus', shp=1.0, **kwd)
    n3[-1].s = 'sigmoid'
    t3 = Trainer(n3, x, u=u, lrt=1e-3, err='CE', **kwd)
    t3.tune(nep)
    h3 = t3.query()
    del t3

    # save the training histories.
    if out is None:
        out = '.'
    if pt.isdir(out):
        out = pt.join(out, pt.basename(fnm).split('.')[0])
    np.savez_compressed(out, h1=h1, h2=h2, h3=h3)

    return h1, h2, h3

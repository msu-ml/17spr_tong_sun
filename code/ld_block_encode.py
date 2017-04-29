# encode a genomic LD block
from helpers import loadVCF
from xnnt.sae import SAE
from os import path as pt, listdir as ls
from xutl import spz, lpz
from sklearn.decomposition import PCA


# main('../data/1kg/bk1/17_072406743.vcf.gz', nep=20, rdp=5)
def main(vcf, nep=20, out=None, rdp=None, **kwd):
    """ Performance test for sigmoid, relu autoencoders.
    -- fnm: the genomic file.
    -- nep: number of epoches to go through.
    -- out: output location.
    -- rdp: reduce network depth by this much.

    ** sav: location to save training progress.
    ** mdp: maximum network depth.
    """

    # handle filenames
    stm = pt.join(pt.dirname(vcf), pt.basename(vcf).split('.')[0])
    if not pt.exists(vcf):      # name correction
        if pt.exists(stm + '.vcf'):
            vcf = stm + '.vcf'
        elif pt.exists(stm + '.vcf.gz'):
            vcf = stm + '.vcf.gz'
        else:
            raise Exception('non-existing: ', vcf)

    sav = kwd.get('sav', '.')
    if pt.isdir(sav):
        sav = pt.join(sav, pt.basename(stm))
    if not sav.endswith('.pgz'):
        sav = sav + '.pgz'

    # prepare data
    gmx, sbj = loadVCF(vcf)
    gmx = gmx.reshape(gmx.shape[0], -1).astype('f')

    # progress recover:
    if pt.exists(sav):
        print(sav, ": exists,", )

        # do not continue to training?
        ovr = kwd.pop('ovr', 0)
        if ovr == 0:
            print(" skipped.")
            return kwd
    else:
        ovr = 2

    # resume progress, use network stored in {sav}.
    if ovr is 1:
        # options in {kwd} take precedence over {sav}.
        sdt = lpz(sav)
        sdt.update(kwd)
        kwd = sdt
        print("continue training.")
    else:  # restart the training
        print("restart training.")

    # perform PCA on genome data if necessary
    pcs = kwd.pop('pcs', None)
    if pcs is None:
        try:
            pca = PCA(n_components=gmx.shape[0])
            pcs = pca.fit_transform(gmx)
        except numpy.linalg.linalg.LinAlgError as e:
            pcs = e
    kwd.update(pcs=pcs)

    hlt = kwd.get('hlt', 0)     # halted?
    if hlt > 0:
        print 'NT: Halt.\nNT: Done.'
        return kwd
    mdp = kwd.pop('mdp', None)  # maximum network depth
    lrt = kwd.pop('lrt', 1e-4)  # learing rates
    gdy = kwd.pop('gdy', 0)     # some greedy pre-training?

    # train the network, create it if necessary
    nwk = kwd.pop('nwk', None)
    if nwk is None:
        # train autoencoders, each layer roughfly halves the dimensionality
        dim = [gmx.shape[1]] + [1024//2**_ for _ in range(16) if 2**_ <= 1024]
        dim = dim[:mdp]

        nwk = SAE.from_dim(dim, s='sigmoid', **kwd)
        print('create NT: ', nwk)

    print('NT: begin')
    tnr = SAE.Train(nwk, gmx, gmx, lrt=lrt, gdy=gdy, nep=nep, **kwd)
    lrt = tnr.lrt.get_value()   # updated learning rate
    hof = nwk.ec(gmx).eval()    # high order features
    eot = tnr.terr().item()     # error of training
    hlt = tnr.hlt               # halting status
    if hlt > 0:
        print('NT: Halt.')
    print('NT: Done.')

    # update, save the progress, then return
    kwd.update(nwk=nwk, lrt=lrt, hof=hof, eot=eot, hlt=hlt, sbj=sbj)
    spz(sav, kwd)

    return kwd


def collect(fdr):
    """ collect encoder output in a folder. """
    eot = []
    hof = []
    pcs = []
    # __n = 0
    for fname in sorted(ls(fdr)):
        if not fname.endswith('pgz'):
            continue
        # if not __n < 10:
        #     break
        # __n = __n + 1
        fname = pt.join(fdr, fname)
        print(fname)
        output = lpz(fname)
        eot.append(output['eot'])
        hof.append(output['hof'])
        if not isinstance(output['pcs'], Exception):
            pcs.append(output['pcs'][:, 0:16])

    import numpy as np
    eot = np.array(eot, 'f')
    hof = np.array(hof, 'f')
    pcs = np.array(pcs, 'f')
    hof = np.transpose(hof, [1, 2, 0])
    pcs = np.transpose(pcs, [1, 2, 0])
    ret = {'eot': eot, 'hof': hof, 'pcs': pcs}
    return ret

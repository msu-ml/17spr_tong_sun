# helper functions.
import numpy as np
from os import path as pt, listdir as ls


def loadVCF(vcf):
    """ Read genomic VCF file by name.
    Fix MAF and allele orders.
    return genomic matrix and subject IDs.
    """
    # Python VCF reader: pip install pyvcf
    from vcf import Reader as vcfR

    # the two homogenuous chromosomes
    A, B = [], []

    reader = vcfR(filename=vcf)
    sbj = reader.samples        # subject IDs
    for v in reader:
        # copy #1 and #2
        a = [int(g.gt_alleles[0] > '0') for g in v.samples]
        b = [int(g.gt_alleles[1] > '0') for g in v.samples]
        A.append(a)
        B.append(b)

    # compile genomic matrix
    gmx = np.array([A, B], dtype='uint8')

    # MAF fixing
    i = np.where(gmx.sum((0, 2)) > gmx.shape[2])[0]
    gmx[:, i, :] = 1 - gmx[:, i, :]

    # Allele order fix, make sure copy(a) >= copy(b)
    i = np.where(gmx[0, :, :] < gmx[1, :, :])
    gmx[0, :, :][i] = 1
    gmx[1, :, :][i] = 0

    # dim_0: sample index; dim_1: copy number; dim_2: variant index
    gmx = gmx.transpose(2, 0, 1)
    return gmx, sbj


def cache(vcf='../data/1kg_gpk'):
    """ cache input genomic data in numpy format. """
    if pt.isdir(vcf):
        vcfs = [pt.join(vcf, f) for f in ls(vcf) if f.endswith('vcf.gz')]
    else:
        vcfs = vcf if isinstance(vcf, list) else [vcf]
    for vcf in vcfs:
        npz = vcf.replace('vcf.gz', 'npz')
        if pt.exists(npz):
            print(vcf, npz, 'exists.')
            continue
        print(vcf, npz, 'to cache.')
        gmx, sbj = loadVCF(vcf)
        np.savez_compressed(npz, gmx=gmx, sbj=sbj)

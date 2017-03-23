import sys
try:
    sys.path.extend(['..'] if '..' not in sys.path else [])
    from rdm.ae import AE
    from rdm.cat import Cat
except ValueError as e:
    from ae import AE
    from cat import Cat


class SAE(Cat):
    """
    Stacked Auto Encoder
    """

    def __init__(self, AEs):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms in the AEs are initialized by default rule.
        -------- parameters --------
        AEs: a list of autoencoders
        """
        """ the default view of the stacked autoencoders"""
        sa = AEs
        """ the encoder view of the stacked autoencoders """
        ec = Cat([a.ec for a in sa])
        """ the decoder view of the stacked autoencoders """
        dc = Cat([a.dc for a in reversed(sa)])

        self.sa = sa  # default view
        self.ec = ec  # encoder view
        self.dc = dc  # decoder view

        nts = []
        nts.extend(ec)
        nts.extend(dc)
        super(SAE, self).__init__(nts)

    @staticmethod
    def from_dim(dim, **kwd):
        """ create SAE by specifying encoding dimensions
        dim: a list of encoding dimensions
        """
        AEs = [AE(d, **kwd) for d in zip(dim[:-1], dim[1:])]
        return SAE(AEs)

    def sub(self, depth, start=None, copy=False):
        """ get sub stack from of lower encoding depth
        -------- parameters --------
        depth: depth of the sub-stack, should be less then the full
        encoder
        
        start: starting level of sub-stack extraction. by default
        always extract from the lowest level.

        copy: parameters in the sub stack is deeply copied from the
        full stack
        """
        ret = SAE(self.sa[start:depth])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret

from xnnt.ae import AE
from xnnt.cat import Cat


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
        return SAE([AE(d, **kwd) for d in zip(dim[:-1], dim[1:])])

    def sub(self, start=None, stop=None, copy=False):
        """ get sub stack from of lower encoding stop
        -------- parameters --------
        start: starting level of sub-stack extraction. by default
        always extract from the least encoded level.

        stop: stopping level of the sub-stack, which will be cropped
        to the full depth of the original stack if exceeds it.

        copy: should the sub stack deep copy the parameters?
        """
        ret = SAE(self.sa[start:stop])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret

    @staticmethod
    def Train(w, x, u=None, nep=20, gdy=0, **kwd):
        """ train the stacked autoencoder {w}.
        w: the stacked autoencoders
        x: the training features.
        u: the training labels.

        nep: number of epochs to go through, if not converge.
        gdy: number of greedy pre-training to go through per added layer.

        kwd - additional key words to pass on to the trainer.
        ** lrt: initial learning rate.
        ** hte: halting error
        **   v: the testing features.
        **   y: the testing label.

        returns: the used trainer {class: xnnt.tnr.bas}.
        """
        # the trainer class
        from xnnt.tnr.cmb import Comb as Tnr

        # layer-wise greedy pre-training (incremental).
        if gdy > 0:
            for i in range(1, len(w.sa)):
                sw = w.sub(0, i)
                print('pre-train sub-stack:', sw)
                tr = Tnr(sw, x, u=u)
                tr.tune(gdy)

        # whole stack fine-tuning
        print('train stack:', w)
        tr = Tnr(w, x, u=u, **kwd)
        tr.tune(nep)

        return tr

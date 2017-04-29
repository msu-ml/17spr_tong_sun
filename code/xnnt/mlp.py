from xnnt.pcp import Pcp
from xnnt.cat import Cat


class MLP(Cat):
    """
    Multiple Layered Perceptron
    """
    def __init__(self, pts):
        """
        Initialize the stacked auto encoder by a list of code dimensions.
        The weight and bias terms in the AEs are initialized by default rule.
        -------- parameters --------
        pts: a list of (single layered) perceptrons
        """
        super(MLP, self).__init__(pts)

    @staticmethod
    def from_dim(dim, **kwd):
        """ create SAE by specifying encoding dimensions
        dim: a list of encoding dimensions
        """
        return MLP([Pcp(d, **kwd) for d in zip(dim[:-1], dim[1:])])

    def sub(self, start=None, stop=None, copy=False):
        """ get sub stack from of lower encoding stop
        -------- parameters --------
        start: starting layer of sub MLP, the default is the lowest layer.

        stop: stopping layer of the sub MLP, will be cropped to the full
        depth of the original MLP.

        copy: the sub MLP is to be deeply copied, instead of shairing
        components of the current MLP.
        """
        ret = MLP(self[start:stop])
        if copy:
            import copy
            ret = copy.deepcopy(ret)
        return ret

    @staticmethod
    def Train(w, x, y, nep=20, gdy=0, **kwd):
        """ train the stacked autoencoder {w}.
        w: the MLP
        x: the training features.
        y: the training lables.

        nep: number of epochs to go through, if not converge.
        gdy: number of greedy pre-training to go through per added layer.

        kwd - additional key words to pass on to the trainer.
        ** lrt: initial learning rate.
        ** hte: halting error
        **   u: the testing features.
        **   v: the testing label.

        returns: the used trainer {class: xnnt.tnr.bas}.
        """
        # the trainer class
        from xnnt.tnr.bas import Base as Tnr

        # layer-wise greedy pre-training (incremental).
        if gdy > 0:
            for i in range(1, len(w)):
                sw = w.sub(0, i)
                print('pre-train sub-MLP:', sw)
                tr = Tnr(sw, x, y, **kwd)
                tr.tune(gdy)

        # whole network fine-tuning
        print('train stack:', w)
        tr = Tnr(w, x, y, **kwd)
        tr.tune(nep)

        return tr

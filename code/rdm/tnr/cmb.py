import theano
from bdr import Bold
from hlp import S

FX = theano.config.floatX


class Comb(Bold):
    """
    Standardized Trainter.
    """

    def __init__(self, *arg, **kwd):
        """
        ** nce: non-convergence epoch, at which the neural network will be
        re-shuffled and re-trained if also {terr < nct} is not met.

        ** nct: non-convergence threshold, if the lowest training seen so
        far is aboved this, the training is dimmed not converging if also
        {epoch == nce} is reached.
        """
        # non-convergence epoch
        self.nce = S(kwd.get('nce', 20), 'NCE')

        # non-convergence threshold
        self.nct = S(kwd.get('nct', 9.), 'NCT')
        
        super(Comb, self).__init__(*arg, **kwd)

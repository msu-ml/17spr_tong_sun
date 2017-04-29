try:
    from .cat import Cat
    from .pcp import Pcp
except ValueError as e:
    from cat import Cat
    from pcp import Pcp


class AE(Cat):
    """
    Autoencoder
    """

    def __init__(self, dim, **kw):
        """
        Initialize the denosing auto encoder by specifying the the dimension
        of the input  and output.
        The constructor also receives symbolic variables for the weights and
        bias.

        -------- parameters --------
        dim:
        dim[0] dimension of the input (visible units)
        dim[1] dimension of the code (hidden units)
        """
        """ the encoder view of the autoencoder """
        ec = Pcp(dim, **kw)
        kw.pop('w', None)

        """ the decoder view of the autoencoder """
        dc = Pcp([dim[1], dim[0]], ec.w.T, **kw)
        kw.pop('b', None)
        kw.pop('s', None)

        # the default view is a concatinated network of dimension d0, d1, d0
        super(AE, self).__init__([ec, dc])
        self.ec = ec
        self.dc = dc


if __name__ == '__main__':
    pass

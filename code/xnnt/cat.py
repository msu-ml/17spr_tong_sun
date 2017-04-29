try:
    from .nnt import Nnt
except ValueError as e:
    from nnt import Nnt


class Cat(Nnt):
    """
    Neural networks formed by concatinating sub-networks.
    """
    def __init__(self, nts):
        """
        Initialize the super neural network by a list of sub networks.

        -------- parameters --------
        nts: child networks to be chinned up.
        """
        super(Cat, self).__init__()

        # first dimension
        dim = [nts[0].dim[0]]

        for p, q in zip(nts[:-1], nts[1:]):
            if p.dim[-1] != q.dim[0]:
                raise Exception('dimension unmatch: {} to {}'.format(p, q))
            dim.append(q.dim[0])

        # last dimension
        dim.append(nts[-1].dim[-1])

        self.extend(nts)
        self.dim = dim

    def __expr__(self, x):
        """
        build symbolic expression of output given input. x is supposdly
        a tensor object.
        """
        for net in self:
            x = net.__expr__(x)
        return x

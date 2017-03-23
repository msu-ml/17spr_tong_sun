from matplotlib import pyplot as plt


class Plot(object):
    """
    The component class to enable historical plot for neural network trainers.
    This class must be subclassed along with a Trainer class to ensure methods
    'ep', 'nnt', and 'step' do exist.
    """
    def __init__(self, *arg, **kwd):
        """ Constructor for class Snap. """
        super(Plot).__init__(*arg, **kwd)

    def plot(self, dx=None, dy=None):
        """ plot the training graph.
        dx: dimension on x axis, by default it is time.
        dy: dimensions on y axis, by defaut it is training and validation
        errors (if available).
        """
        if 'query' not in vars(self):
            print('method "query" is not available.')
            return
            
        if dx is None:
            dx = ['time']
        if dy is None:
            dy = ['terr']
            if self.u and self.v:
                dy = dy + ['verr']
        else:
            if '__iter__' not in dir(dy):
                dy = [dy]

        clr = 'bgrcmykw'
        dat = self.query(fc=dx + dy)

        plt.close()
        x = dx[0]
        for i, y in enumerate(dy):
            plt.plot(dat[x], dat[y], c=clr[i])

        return plt

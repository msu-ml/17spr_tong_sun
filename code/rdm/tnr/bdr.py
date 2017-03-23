import theano
from hlp import S
from bas import Base
from snp import Snap

FX = theano.config.floatX


class Bold(Snap, Base):
    """
    Bold Driver implementation for nerual networks.
    """
    def __init__(self, *arg, **kwd):
        """
        acc: acceleration when the trainer see an reduction in error.
        dec: deceleration when the trainer see an increase in error.

        """
        # learning rate changes
        self.acc = S(kwd.get('acc', 1.04), 'ACC')  # acceleration
        self.dec = S(kwd.get('dec', 0.85), 'DEC')  # deceleration
        self.bdr = kwd.get('bdr', True)            # enable bold driver?
        # patient of waiting validation error to fall again.
        self.bdr_patient = kwd.get('bdr_patient', 20)

        # initialize super class.
        super(Bold, self).__init__(*arg, **kwd)
        self.snap('meot', 's')  # minimum training error
        self.snap('meov', 's')  # minimum validation error

    def __onep__(self):
        """ called on new epoch. """
        if not self.bdr:
            # super class call
            return super(Bold, self).__onep__()
            
        # history records
        r = self.__hist__[-1]

        # update the learning rate and suprimum of gradient
        if r['terr'] < self.snap('meot', 'l')['terr']:  # accelerate
            self.lrt.set_value(self.lrt.get_value() * self.acc.get_value())
        else:                   # slow down, and restore saved state
            self.snap('meot', 'r')
            self.lrt.set_value(self.lrt.get_value() * self.dec.get_value())
        self.snap('meot', 's')

        # update minimum validation error, also save the state
        if r['verr'] < self.snap('meov', 'l')['verr'] and self.u is not self.x:
            self.snap('meov', 's')

        # super class call
        super(Bold, self).__onep__()

    def __stop__(self):
        """ called on each epoch, return true if the training
        should be halted. """
        # current state
        c = self.__hist__[-1]

        # minimum validation error
        m = self.snap('meov')
        if m is None:
            return False

        # stop on rising validation error.
        hlt = True
        hlt = hlt and c['verr'] > m['verr']
        hlt = hlt and c['terr'] < m['terr']
        # Halt on runing out of Validation Patients
        hlt = hlt and c['ep'] - m['ep'] > self.hvp
        if hlt:
            # restore the state with lowest validation error.
            self.snap('meov', 'r')
            self.hlt = 2  # increased validation error.
            return True
        return super(Bold, self).__stop__()

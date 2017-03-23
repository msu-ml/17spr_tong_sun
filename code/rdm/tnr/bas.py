import numpy as np
import theano
from theano import function as F, tensor as T
from time import time as tm
import exb
from hlp import C, S, parms
import sys


FX = theano.config.floatX


class Base(object):
    """
    Class for neural network training.
    """
    def __init__(self, nnt, x=None, y=None, u=None, v=None, *arg, **kwd):
        """
        : -------- parameters -------- :
        nnt: an expression builder for the neural network to be trained,
        could be a Nnt object.

        x: the inputs, with the first dimension standing for sample units.
        If unspecified, the trainer will try to evaluate the entry point and
        cache the result as source data.

        y: the labels, with the first dimension standing for sample units.
        if unspecified, a simi-unsupervied training is assumed as the labels
        will be identical to the inputs.

        u: the valication data inputs
        v: the validation data labels

        : -------- kwd: keywords -------- :
        ** bsz: batch size.
        ** lrt: learning rate.
        ** lmb: weight decay factor, the lambda

        ** err: expression builder for the computation of training error
        between the network output {yhat} and the label {y}. the expression
        must evaluate to a scalar.

        ** reg: expression builder for the computation of weight panalize
        the vector of parameters {vhat}, the expression must evaluate to a
        scalar.

        ** mmt: momentom of the trainer

        ** vdr: validation disruption rate
        ** hte: the halting training error.
        """
        # numpy random number generator
        seed = kwd.pop('seed', None)
        nrng = kwd.pop('nrng', np.random.RandomState(seed))

        from theano.tensor.shared_randomstreams import RandomStreams
        trng = kwd.pop('trng', RandomStreams(nrng.randint(0x7FFFFFFF)))

        # private members
        self.__seed__ = seed
        self.__nrng__ = nrng
        self.__trng__ = trng

        # expression of error and regulator terms
        err = getattr(exb, kwd.get('err', 'CE'))
        reg = getattr(exb, kwd.get('reg', 'L1'))

        # the validation disruption
        self.vdr = S(kwd.get('vdr'), 'VDR')

        # the denoising
        self.dns = S(kwd.get('dns'), 'DNS')

        # the halting rules, and halting status
        self.hte = kwd.get('hte', 1e-3)  # 1. low training error
        self.hgd = kwd.get('htg', 1e-7)  # 2. low gradient
        self.hlr = kwd.get('hlr', 1e-7)  # 3. low learning rate
        self.hvp = kwd.get('hvp', 0100)  # 4. out of validation patients
        self.hlt = 0

        # current epoch index, use int64
        self.ep = S(0, 'EP')

        # training batch ppsize, use int64
        bsz = kwd.get('bsz', 20)
        self.bsz = S(bsz, 'BSZ')

        # current batch index, use int64
        self.bt = S(0, 'BT')

        # momentumn, make sure momentum is a sane value
        mmt = kwd.get('mmt', .0)
        self.mmt = S(mmt, 'MMT')

        # learning rate
        lrt = kwd.get('lrt', 0.01)  # learning rate
        acc = kwd.get('acc', 1.04)  # acceleration
        dec = kwd.get('dec', 0.85)  # deceleration
        self.lrt = S(lrt, 'LRT', 'f')
        self.acc = S(acc, 'ACC', 'f')
        self.dec = S(dec, 'DEC', 'f')

        # weight decay, lambda
        lmd = kwd.get('lmd', .0)
        self.lmd = S(lmd, 'LMD', 'f')

        # the neural network
        self.nnt = nnt
        self.dim = (nnt.dim[0], nnt.dim[-1])

        # inputs and labels, for modeling and validation
        x = S(np.zeros((bsz * 2, self.dim[0]), 'f') if x is None else x)
        y = x if y is None else S(y)
        u = x if u is None else S(u)
        v = u if v is None else S(v)
        self.x, self.y, self.u, self.v = x, y, u, v

        # -------- construct trainer function -------- *
        # 1) symbolic expressions
        x = T.tensor(name='x', dtype=x.dtype, broadcastable=x.broadcastable)
        y = T.tensor(name='y', dtype=y.dtype, broadcastable=y.broadcastable)
        yhat = nnt(x)

        u = T.tensor(name='u', dtype=u.dtype, broadcastable=u.broadcastable)
        v = T.tensor(name='v', dtype=v.dtype, broadcastable=v.broadcastable)

        # list of symbolic parameters to be tuned
        pars = parms(yhat)

        # list of  symbolic weights to apply decay
        vwgt = T.concatenate([p.flatten() for p in pars if p.name == 'w'])

        # symbolic batch cost, which is the mean trainning erro over all
        # observations and sub attributes.

        # The observations are indexed by the first dimension of y, the last
        # dimension indices data entries for each observation,
        # e.g. voxels in an MRI region, and SNPs in a gene.
        # The objective function, err, returns a scalar of training loss, it
        # can be the L1, L2 norm and CE.
        erro = err(yhat, y).mean()

        # the sum of weights calculated for weight decay.
        wsum = reg(vwgt)
        cost = erro + wsum * self.lmd

        # symbolic gradient of cost WRT parameters
        grad = T.grad(cost, pars)
        gabs = T.concatenate([T.abs_(g.flatten()) for g in grad])
        gsup = T.max(gabs)

        # trainer control
        nwep = ((self.bt + 1) * self.bsz) // self.x.shape[-2]  # new epoch?

        # 2) define updates after each batch training
        up = []

        # update parameters using gradiant decent, and momentum
        for p, g in zip(pars, grad):
            # initialize accumulated gradient
            # NOTE: p.eval() causes mehem!!
            h = S(np.zeros_like(p.get_value()))

            # accumulate gradient, partially historical (due to the momentum),
            # partially noval
            up.append((h, self.mmt * h + (1 - self.mmt) * g))

            # update parameters by stepping down the accumulated gradient
            up.append((p, p - self.lrt * h))

        # update batch and eqoch index
        up.append((self.bt, (self.bt + 1) * (1 - nwep)))
        up.append((self.ep, self.ep + nwep))

        # 3) the trainer functions
        # expression of batch and whole data feed:
        _ = T.arange((self.bt + 0) * self.bsz, (self.bt + 1) * self.bsz)

        # enable denoise training
        if self.dns:
            msk = self.__trng__.binomial(self.y.shape, 1, self.dns, dtype=FX)
            msk = 1 - msk
            bts = {
                x: self.x.take(_, 0, 'wrap'),
                y: self.y.take(_, 0, 'wrap') * msk.take(_, -2, 'wrap')}
            dts = {x: self.x, y: self.y * msk}
        else:
            bts = {
                x: self.x.take(_, 0, 'wrap'),
                y: self.y.take(_, 0, 'wrap')}
            dts = {x: self.x, y: self.y}

        # each invocation sends one batch of training examples to the network,
        # calculate total cost and tune the parameters by gradient decent.
        self.step = F([], cost, name="step", givens=bts, updates=up)

        # training error, training cost
        self.terr = F([], erro, name="terr", givens=dts)
        self.tcst = F([], cost, name="tcst", givens=dts)

        # weights, and parameters
        self.wsum = F([], wsum, name="wsum")
        self.gsup = F([], gsup, name="gsup", givens=dts)
        # * -------- done with trainer functions -------- *

        # * -------- validation functions -------- *
        # enable validation binary disruption (binary)?
        if self.vdr:
            _ = self.__trng__.binomial(self.v.shape, 1, self.vdr, dtype=FX)
            vts = {x: self.u, y: (self.v + _) % C(2.0, FX)}
        else:
            vts = {x: self.u, y: self.v}
        self.verr = F([], erro, name="verr", givens=vts)

        # * ---------- logging and recording ---------- *
        hd, skip = [], ['step']
        for k, v in vars(self).items():
            if k.startswith('__') or k in skip:
                continue
            # possible theano shared cpu variable
            if isinstance(v, type(self.lmd)) and v.ndim < 1:
                hd.append((k, v.get_value))
            # possible theano shared gpu variable
            if isinstance(v, type(self.ep)) and v.ndim < 1:
                hd.append((k, v.get_value))
            if isinstance(v, type(self.step)):
                hd.append((k, v))
            if isinstance(v, float) or isinstance(v, int):
                hd.append((k, v))

        self.__head__ = hd
        self.__time__ = .0

        # the initial record, and history
        self.__hist__ = [self.__rpt__()]
        self.__mver__ = self.__hist__[-1]  # min verr
        self.__mter__ = self.__hist__[-1]  # min terr

        # printing format
        self.__pfmt__ = (
            '{ep:04d}: {tcst:.1e} = {terr:.1e} + {lmd:.1e}*{wsum:.1f}'
            '|{verr:.1e}, {gsup:.2e}, {lrt:.2e}, {hte:.1e}')

    # -------- helper funtions -------- *
    def nbat(self):
        return self.x.get_value().shape[-2] // self.bsz.get_value()

    def yhat(self, x=None, evl=True):
        """
        Predicted outcome given input {x}. By default, use the training samples
        as input.
        x: network input
        evl: evaludate the expression (default = True)
        """
        yhat = self.nnt(self.x if x is None else x)
        return yhat.eval() if evl else yhat

    def __rpt__(self):
        """ report current status. """
        r = dict((k, f().item()) for k, f in self.__head__ if callable(f))
        r.update((k, v) for k, v in self.__head__ if not callable(v))
        r['time'] = self.__time__
        return r

    def __onep__(self):
        """ called on new epoch. """
        # update semi snap shots
        r = self.__hist__[-1]
        if r['verr'] < self.__mver__['verr']:  # min verr
            self.__mver__ = r
        if r['terr'] < self.__mter__['terr']:  # min terr
            self.__mter__ = r

    def __onbt__(self):
        """ called on new batch """
        pass

    def __stop__(self):
        """ return true to signal training stop. """
        # already halted?
        if self.hlt:
            return True
        
        # no training histoty, do not stop.
        if len(self.__hist__) < 1L:
            return False

        # pull out the latest history
        r = self.__hist__[-1]

        # terr flucturation, do not stop!
        if r['terr'] > self.__mter__['terr']:
            return False

        # check the latest history
        if r['terr'] < self.hte:  # early stop on terr
            self.hlt = 1
            return True
        if r['gsup'] < self.hgd:  # converged
            self.hlt = 2
            return True
        # early stop if we run out of patient on rising verr
        if r['ep'] - self.__mver__['ep'] > self.hvp:
            self.hlt = 3
            return True
        if r['lrt'] < self.hlr:  # fail
            self.hlt = 9
            return True
        return False

    def tune(self, nep=1, nbt=0, rec=0, prt=0):
        """ tune the parameters by running the trainer {nep} epoch.
        nep: number of epoches to go through
        nbt: number of extra batches to go through

        rec: frequency of recording. 0 means record after each epoch,
        which is the default, otherwise, record for each batch, which
        can be time consuming.

        prt: frequency of printing.
        """
        bt = self.bt
        b0 = bt.eval().item()  # starting batch
        ei, bi = 0, 0  # counted epochs and batches

        nep = nep + nbt // self.nbat()
        nbt = nbt % self.nbat()

        while (ei < nep or bi < nbt) and not self.__stop__():
            # send one batch for training
            t0 = tm()
            self.step()
            self.__onbt__()     # on each batch
            self.__time__ += tm() - t0

            # update history
            bi = bi + 1  # batch count increase by 1
            if rec > 0:  # record each batch
                self.__hist__.append(self.__rpt__())
            if prt > 0:  # print each batch
                print(self)

            # see the next epoch relative to batch b0?
            if bt.get_value().item() == b0:
                ei = ei + 1  # epoch count increase by 1
                bi = 0  # reset batch count

            # after an epoch
            if bt.get_value().item() == 0:
                # record history
                if rec == 0:  # record at new epoch
                    self.__hist__.append(self.__rpt__())
                # print
                if prt == 0:  # print
                    print(self)

                self.__onep__()  # on each epoch
            sys.stdout.flush()

    def __str__(self, ix=None):
        """
        Print out a range of training records.
        idx: slice of records to be printed, by default the last
        one is shown.
        """
        hs = self.__hist__
        if ix is None:
            hs = [self.__hist__[-1]]
        elif ix is int:
            hs = [self.__hist__[ix]]
        else:
            hs = hs[ix]

        # latest history
        return '\n'.join(self.__pfmt__.format(**h) for h in hs)

    def query(self, fc=None, rc=None, out=None):
        """ report the trainer'shot history.
        fc: the field checker, can be a function taking a string field name,
        or a list of field names. by default no checking is imposed.
        rc: the record checker, a function taking a dictionary record.
        by default all record will pass.
        e.g.:
        >>> query(rc=lambda r: r['eq']<10)  # return the first 10 epoch

        out: the file to flush the output. if specified, the query result is
        written to a a tab-delimited file. if 0 is specified, the STDOUT will
        be used.
        """
        hs = self.__hist__
        h0 = hs[0]

        # field checker
        if fc is None:

            def fc1(f):
                return True
        elif '__iter__' in dir(fc):

            def fc1(f):
                return f in fc
        else:
            fc1 = fc

        # record checker
        if rc is None:

            def rc1(**r):
                return True
        else:
            rc1 = rc

        # numpy types from python native types
        tp = {float: 'f4', int: 'i4'}
        tp = np.dtype([(k, tp[type(v)]) for k, v in h0.items() if fc1(k)])

        # numpy data
        dt = [tuple(v for k, v in _.items() if fc1(k)) for _ in hs if rc1(**_)]
        dt = np.array(dt, tp)

        if out is not None:
            # output format
            fm = {float: '%f', int: '%d'}
            fm = '\t'.join([fm[type(v)] for k, v in h0.items() if fc1(k)])
            hd = '\t'.join(dt.dtype.names)
            np.savetxt(out, dt, fm, header=hd)

        return dt

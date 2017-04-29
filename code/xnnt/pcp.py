import numpy as np
from xnnt.hlp import S, C
from xnnt.nnt import Nnt
from xnnt import exb
from theano import tensor as T
from theano import scan as sc


class Pcp(Nnt):
    """
    A Perceptron, which is the full linear recombination of the input
    elements and a bias(or intercept), followed by an per-element non-
    linear transformation(usually sigmoid).
    """
    def __init__(self, dim, w=None, b=None, c=None, s=None, **kwd):
        """
        Initialize the perceptron by specifying the the dimension of input
        and output.
        The constructor also receives symbolic variables for the input,
        weights and bias. Such a symbolic variables are useful when, for
        example, the input is the result of some computations, or when
        the weights are shared between the layers

        -------- parameters --------
        dim: a 2-tuple of input/output dimensions
        d_1: specify the input dimension P, or # of visible units
        d_2: specify the output dimension Q, or # of hidden units

        w: (optional) weight of dimension (P, Q), randomly initialized
        if not given.

        b: (optional) bias of output (hidden) units, of dimension Q,
        filled with 0 by default.
        c: (optional) bias of input (visible) units, of dimension P,
        filled with 0 by default.
        this bias vector is needed by energy based interpretation.

        s: (optional) nonlinear tranformation of the weighted sum, by
        default the sigmoid is used.
        To suppress nonlinearity, specify 1 instead.
        """
        super(Pcp, self).__init__(**kwd)

        # I/O dimensions
        self.dim = dim

        # note : W' was written as `W_prime` and b' as `b_prime`
        """
        # W is initialized with `initial_W` which is uniformely sampled
        # from -4*sqrt(6./(n_vis+n_hid)) and
        # 4 * sqrt(6./(n_hid+n_vis))the output of uniform if
        # converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        """
        # the activation/squarshing function, the defaut is sigmoid
        if s is None:
            s = 'sigmoid'
        self.s = str(s).lower()

        # levels of the output, the defaut is 2 (binary data)
        lvl = kwd.get('lvl', 2)

        # category of the output, for softmax squarsh only
        cat = kwd.get('cat', 2)

        if w is None:
            if s in ['softplus', 'relu', 'sftp']:
                print('Initalize w for', s)
                w = self.__nrng__.normal(
                    0, np.sqrt(1.0/dim[0] + 1.0/dim[1]), dim)
            elif s in ['sigmoid', 'elliot', 'logistic']:
                print('Initalize w for', s)
                w = self.__nrng__.uniform(
                    low=-4 * np.sqrt(6. / (dim[0] + dim[1])),
                    high=4 * np.sqrt(6. / (dim[0] + dim[1])),
                    size=dim)
            elif s == 'softmax':
                print('Initialize w for softmax', cat)
                w = self.__nrng__.uniform(
                    low=-4 * np.sqrt(6. / (dim[0] + dim[1])),
                    high=4 * np.sqrt(6. / (dim[0] + dim[1])),
                    size=(cat, dim[0], dim[1]))
            else:
                print('Initialize w for affine')
                w = self.__nrng__.normal(
                    0, np.sqrt(1.0/dim[0] + 1.0/dim[1]), dim)
            w = S(w, 'w')

        if b is None:
            if s == 'softmax':
                b = np.zeros((1, cat, dim[1]))
            else:
                b = np.zeros(dim[1])
            b = S(b, 'b')

        if c is None:
            if s == 'softmax':
                c = np.zeros((1, cat, dim[1]))
            else:
                c = np.zeros(dim[1])
            c = S(c, 'c')

        # shape parameters
        self.alpha = None
        self.beta = None
        self.gamma = None

        # flexible shape of the activation function.
        for shp in ('alpha', 'beta', 'gamma'):
            v = kwd.get(shp, None)
            if v is None:
                continue
            vars(self)[shp] = S(v, shp, 'f4')

        # constant shape parameters
        for shp in ('Alpha', 'Beta', 'Gamma'):
            v = kwd.get(shp, None)
            if v is None:
                continue
            shp = shp.lower()
            vars(self)[shp] = C(v, shp, 'f4')

        self.w = w            # weight matrix
        self.b = b            # offset on output (bias of the hidden)
        self.c = c            # offset on bottom (bias of the visible)
        self.lvl = lvl        # output level
        self.cat = cat        # output categorys

    # a perceptron is represented by the nonlinear funciton and dimensions
    def __repr__(self):
        L = {'softmax': 'M', 'sigmoid': 'S', 'relu': 'R', 'softplus': 'P'}
        return '{}({}x{})'.format(L.get(self.s, ""), self.dim[0], self.dim[1])

    def __expr__(self, x):
        """ build symbolic expression of {y} given {x}.
        For a perceptron, it is the full linear recombination of the
        input elements and an bias(or intercept), followed by an
        element-wise non-linear transformation(usually sigmoid)
        """
        # affine transformation of inputs
        # dim(:) = d_1, dim(m) = d_2
        # 1) _[n, c, m] = x[n, :   ]' * w[c, :, m]
        # 2) _[n, c, m] = _[n, c, m]  + b[*, c, m]
        _ = T.dot(x, self.w) + self.b

        # the linear part of the reference category, which is always 0.
        # if callable(self.s) and self.s is T.nnet.softmax:
        # z = T.zeros((x.shape[0], 1, self.dim[1]))
        # _ = T.concatenate((z, _), 1)

        # activation / squashing
        if self.s == 'sigmoid':
            # cross bars to dertermine which level the output is located
            if self.lvl > 2:
                from scipy.stats import norm
                bar = norm.ppf(np.linspace(0, 1, self.lvl+1))[1:self.lvl]
                bar = bar.reshape(self.lvl - 1, 1, 1).astype('f')
                _ = _ - C(bar, 'Bar')
                self.bar = bar

            # apply shape, larger means steeper
            if self.alpha is not None:
                _ = _ * self.alpha

            # activation
            _ = exb.sigmoid(_)

            # sum over levels, standardize to [0, 1]
            if self.lvl > 2:
                _ = T.sum(_, 0) / C(self.lvl - 1, 'f', 'L-1')

        if self.s == 'elliot':
            _ = exb.elliot(_, self.alpha)

        if self.s == 'logistic':
            _ = exb.logistic(_, self.alpha, self.beta)

        # softplus activation
        if self.s in ['softplus', 'sftp']:
            _ = exb.sftp(_, self.alpha, self.beta)

        # softmax
        if self.s == 'softmax':
            v = T.transpose(_, (0, 2, 1))
            _, u = sc(exb.softmax, sequences=[v])
            _ = T.transpose(_, (0, 2, 1))

        # relu activation
        if self.s == 'relu':
            _ = exb.relu(_, self.alpha)
            _.name = 'relu(Xw+b)'
            
        return _

    def __free_energy__(self, x, family='b'):
        ''' symbolic expression of free energy given fixed assignment on
        visible units (x).
        For now it only works for binary units.

        x: the P-vector of visible units at the bottom, linked to the input.
        h: the Q-vector of hidden units at the top, linked to the label.

        W: the Q * P weight matrix, whose (i,j) element gives the negative
        engery contributed by the ith visible and and the jth hidden units
        together.

        b: the Q-vector of bias for hidden units, whose j th. element gives
        the negative energy contributed by the j th. hidden unit.

        c: the P-vector of bias for visible units, whose i th. element gives
        the negative engery contributed by the i th. visible unit.

        # total system engery:
        - E(x, h) = c'x + b'h + h'Wx

        # some intermediate terms:
        # energy of the visibles, which is fixed because we fix x:
        # E(x) = c'x

        # integration of energy contributed by the j th. hidden unit over its
        # support. for binary units, h_j only takes value from {0, 1}.
        # Int:E(h_j) = sum(h_j, e^[h_j(c_j + w_j x)])
        
        # integration or sum of engery contributed by all hidden units, is
        # the direct sum of individual units, because in a RBM all hidden
        # units are not connected.
        # sum(he_) = sum(j, sum(hej))

        FE(x) = -ve - /sum_i{/log{/sum_{h_i}{e^{h_i(c_i + W_i %*% x)}}}}.

        For binary units, it is simplified since h_i only takes {0, 1}:
        FE(x) = -ve - /sum_i{/log{(1 + e^(c_i + W_i %*% x))}
        '''
        # engery contribution of active hidden units (h == [1])
        he1 = T.dot(x, self.w) + self.b  # energy when h_j == 1
        # he0 = 0                        # energy when h_j == 0

        # sum over all instances and all dimensions of h
        # he_ = T.sum(T.log(T.exp(he0) + T.exp(he1)), axis=-1)
        he_ = T.sum(T.log(1 + T.exp(he1)), axis=-1)

        # engery contributed by visible x
        ve_ = T.dot(x, self.c)

        # total negative energy
        return -ve_ - he_

    def propup(self, x0):
        ''' This function propagates the visible units activation upwards
        to the hidden units

        x0: N * P instances of visible units, one row per observation.

        returns:
        [0] ay: the activation of hidden units
        [1] py: the mean of hidden units, for a binary case, it is also the
        probability of activation, that is, Prb(y_j==1) for j = 1 .. Q
        
        Also returned is the pre-sigmoid activation of the layer. As it will
        turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write down a
        more stable computational graph (see details in the reconstruction
        cost function)
        '''
        ay = T.dot(x0, self.w) + self.b
        py = T.nnet.sigmoid(ay)
        return [ay, py]

    def yox(self, x0):
        ''' Hidden units of Visible units, or, y of x. the function infers state
        of hidden units (y) given visible units (x).
        x: instance of visible units, one row per observation.

        returns:
        [0] ay: the raw activation of hidden units
        [1] py: the mean of hidden units, for a binary case, it is also the
        probability of activation, that is, P(y_j == 1) for j = 1 .. Q
        [2] sy: the sampled hidden units.
        '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        ay, py = self.propup(x0)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        sy = self.__trng__.binomial(
            size=py.shape, n=1, p=py, dtype='float32')
        return [ay, py, sy]

    def propdown(self, y0):
        ''' This function propagates the hidden units activation downwards to
        the visible units

        y0: N * Q instances of hidden units, one row per observation.
        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        ax = T.dot(y0, self.w.T) + self.c
        px = T.nnet.sigmoid(ax)
        return [ax, px]

    def xoy(self, y0):
        ''' visible units of hidden units, x of y. this function infers state
        of visible units, given hidden units y0.
        y0: instance of hidden units, one row per observation.

        returns:
        [0] ax: the raw activation of visible units
        [1] px: the mean of visible units, for binary cases, it is also the
        probability of activation, that is, P(x_i == 1) for i = 1 .. P
        [2] x1: the sampled visible units at.
        '''
        # compute the activation of the visible given the hidden sample
        ax, px = self.propdown(y0)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        x1 = self.__trng__.binomial(
            size=px.shape, n=1, p=px, dtype='float32')
        return [ax, px, x1]

    def yxy(self, y0):
        ''' build symbolic expression of one step of Gibbs sampling,
        from the hidden state y0, to the intermediate visible state
        x?, then to the next hidden state y1.
        y0: hidden units at t0
        returns:
        [0] ax: raw activation of visible units due to y0
        [1] px: mean activation of visible units
        [2] x1: sampled visible units
        [3] ay: raw activation of hidden units at t1
        [4] py: mean activation of hidden units at t1
        [5] y1: sampled hidden units at t1
        '''
        ax, px, sx = self.xoy(y0)
        ay, py, y1 = self.yox(sx)
        return [ax, px, sx, ay, py, y1]

    def xyx(self, x0):
        ''' build symbolic expression of one step of Gibbs sampling,
        from the visible state x0, to the intermediate hidden state
        y?, then to the next visible state x1.
        x0: hidden units at t0
        returns:
        [0] ay: raw activation of hidden units due to x0
        [1] py: mean activation of hidden units
        [2] sy: sampled visible units
        [3] ax: raw activation of visible units at t1
        [4] px: mean activation of visible units at t1
        [5] x1: sampled visible states at t1
        '''
        ay, py, sy = self.xoy(x0)
        ax, px, x1 = self.yox(sy)
        return [ay, py, sy, ax, px, x1]

    def get_cost_updates(self, x, lr=0.1, persistent=None, k=1):
        """This functions implements one step of CD-k or PCD-k
        lr: learning rate used to train the RBM
        persistent: None for CD. For PCD, shared variable containing old state
        of Gibbs chain. This must be a shared variable of size (batch size,
        number of hidden units).
        x: input to be wired to the visible units
        k: number of Gibbs steps to do in CD-k/PCD-k

        Returns a proxy for the cost and the updates dictionary. The dictionary
        contains the update rules for weights and biases but also an update of
        the shared variable used to store the persistent chain, if one is used.
        """

        # compute positive phase? seriousely?

        # decide how to initialize persistent chain:
        # for CD, we use the newly generate hidden sample
        # for PCD, we initialize from the old state of the chain
        if persistent:
            y_0 = persistent
        else:
            ay0, py0, y_0 = self.yox(x)

        # perform actual negative phase
        # in order to implement CD-k/PCD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        import theano
        ([ax, px, sx, ay, py, sy], updates) = theano.scan(
            self.yxy,
            # the None are place holders, saying that
            # MCy0 is the initial state corresponding to the
            # 6th output
            outputs_info=[None, None, None, None, None, y_0],
            n_steps=k)

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        # sample x from p(x) = F(x)/Z, where Z=sum(F(_x_))
        x_k = sx[-1]

        # positive phase and negative phase
        pp = self.__free_energy__(x)
        np = self.__free_energy__(x_k)

        # psudo cost = -log[p(x)]
        cost = T.mean(pp) - T.mean(np)
        # compute the gradient of cost w.r.t. the parameters, [w, b, c], but
        # skip the parameters' involvement in the gibbs chain, because the
        # sampling only serves to approximate the normalizing constant Z.
        parm = [self.w, self.b, self.c]
        grad = T.grad(cost, parm, consider_constant=[x_k])

        # constructs the update dictionary
        for g, p in zip(grad, parm):
            # make sure that the learning rate is of the right dtype
            updates[p] = p - g * T.cast(lr, 'float32')
        if persistent:
            # Note that this works only if persistent is a shared variable
            updates[persistent] = sy[-1]
            # pseudo-likelihood is a better proxy for PCD
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            # reconstruction cross-entropy is a better proxy for CD
            monitoring_cost = self.get_reconstruction_cost(updates, ax[-1])

        return monitoring_cost, updates
        # end-snippet-4

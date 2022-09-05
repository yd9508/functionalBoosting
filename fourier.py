import numpy as np
from basis import *
from basis import basis
from locfdr import *


def pendiagfn(basisobj, nderiv):
    nbasis = basisobj.nbasis
    period = basisobj.params
    rangev = basisobj.rangeval
    omega = 2 * np.pi / period
    halfper = period / 2
    twonde = 2 * nderiv
    pendiag = np.zeros(nbasis)
    if nderiv == 0:
        pendiag[0] = period / 2.0
    else:
        pendiag[0] = 0
    j = [x for x in range(1, nbasis - 2, 2)]
    fac = halfper * (j * omega / 2) ^ period
    pendiag[range(1, nbasis - 2, 2)] = fac
    pendiag[range(2, nbasis - 1, 2)] = fac
    pendiag = 2 * pendiag / period
    return pendiag


def fourierpen(basisobj, Lfdobj=int2Lfd(0)):
    if not isinstance(basisobj, basis):
        raise ValueError("First argument is not a basis object.")

    nbasis = basisobj.nbasis
    if (nbasis % 2 == 0):
        basisobj.nbasis = nbasis + 1

    type = basisobj.type
    if (type != 'fourier'):
        raise ValueError("Wrong basis type.")

    Lfdobj = int2Lfd(Lfdobj)
    width = basisobj.rangeval[1] - basisobj.rangeval[0]
    period = basisobj.params[0]
    ratio = np.round(width / period)
    nderiv = Lfdobj.nderiv
    if width / period == ratio:
        penaltymatrix = np.diag(pendiagfn(basisobj, nderiv))
    else:
        penaltymatrix = inprod(basisobj, basisobj, Lfdobj, Lfdobj)

    return penaltymatrix


def fourier(x, nbasis, period, nderiv=0):
    # check x and set up range
    n = len(x)
    onen = np.ones(n)
    xrange = [np.min(x), np.max(x)]
    span = xrange[1] - xrange[0]

    # check period and set up omega

    if (period <= 0):
        raise ValueError("Period not positive.")

    omega = 2 * np.pi / period
    omegax = np.inner(omega, x)

    # check nbasis
    if (nbasis <= 0):
        raise ValueError("NBASIS not positive.")

    # check nderiv
    if (nderiv < 0):
        raise ValueError("nderiv is negative.")
    if (nderiv > 0):
        raise ValueError("positive nderiv is not supported currently.")

    # if nbasis is even, add one
    if (nbasis % 2 == 0):
        nbasis = nbasis + 1

    # compute basis matrix
    basismat = np.zeros((n, nbasis))
    if (nderiv == 0):
        basismat[:, 0] = 1 / np.sqrt(2)
        if (nbasis > 1):
            j = range(1, nbasis - 2, 2)
            k = [int(x / 2) for x in j]
            args = np.outer(omegax, k)
            basismat[:, j] = np.sin(args)
            basismat[:, j + 1] = np.cos(args)
    basismat = basismat / np.sqrt(period / 2)
    return basismat

def create_fourier_basis(rangeval=[0, 1], nbasis=3, period=None, dropind=None, quadvals=None, values=None,
                         basisvalues=None):
    type = 'fourier'
    if period is None:
        period = np.diff(rangeval)
    # 1. check RANGEVAL
    if (len(rangeval) < 1):
        if (rangeval <= 0):
            raise ValueError("'len(rangeval) = 0', not allowed.")
        rangeval = [0, rangeval]

    if not rangechk(rangeval):
        raise ValueError("Argument RNAGEVAL is not correct.")

    # Set up the basis object
    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=period, dropind=dropind, quadvals=quadvals,
                     values=values, basisvalues=basisvalues)

    return basisobj

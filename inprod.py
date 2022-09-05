import numpy as np

from basis import create_constant_basis, basis, create_bspline_basis
from fRegress import fdchk, knotmultchk, ppBspline, ppderiv, polyprod
from fd import Lfd, fd, eval_fd


def knotmultchk(basisobj, knotmult):
    type = basisobj.type
    if type == 'bspline':
        params = basisobj.params
        nparams = len(params)
        norder = basisobj.nbasis - nparams
        if norder == 1:
            knotmult = np.concatenate((knotmult, params), axis=None)
        else:
            if nparams > 1:
                for i in range(1, nparams):
                    if params[i] == params[i - 1]:
                        knotmult = np.concatenate((knotmult, params[i]), axis=None)

    return knotmult


def inprod_bspline(fdobj1, fdobj2=None, nderiv1=0, nderiv2=0):
    if fdobj2 is None:
        fdobj2 = fdobj1

    if not isinstance(fdobj1, fd):
        raise ValueError("FD1 is not a functional data object.")
    if not isinstance(fdobj2, fd):
        raise ValueError("FD2 is not a functional data object.")

    basis1 = fdobj1.basisobj
    type1 = basis1.type
    if type1 != 'bspline':
        raise ValueError("FDOBJ1 does not have a B-spline basis.")

    range1 = np.array(basis1.rangeval)
    breaks1 = np.concatenate((np.concatenate(([range1[0]], basis1.params)), [range1[1]]))
    nbasis1 = basis1.nbasis
    norder1 = nbasis1 - len(breaks1) + 2

    basis2 = fdobj2.basisobj
    type2 = basis2.type
    if type2 != 'bspline':
        raise ValueError("FDOBJ2 does not have a B-spline basis.")

    range2 = np.array(basis2.rangeval)
    breaks2 = np.concatenate((np.concatenate(([range2[0]], basis2.params)), [range2[1]]))
    nbasis2 = basis2.nbasis
    norder2 = nbasis2 - len(breaks2) + 2

    if np.any(range1 - range2 != 0):
        raise ValueError("The argument ranges from FDOBJ1 and FDOBJ2 are not identical.")

    # check that break values are equal and set up common array
    if len(breaks1) != len(breaks2):
        raise ValueError("The number of knots for FDOBJ1 and FDOBJ2 are not identical.")

    if np.any(breaks1 - breaks2 != 0):
        raise ValueError("The knots of  FDOBJ1 and FDOBJ2 are not identical.")
    else:
        breaks = breaks1

    if len(breaks) < 2:
        raise ValueError("The length of argument BREAKS is less than 2.")

    breakdiff = np.diff(breaks)
    if np.min(breakdiff) <= 0:
        raise ValueError("Argument BREAKS is not strictly increasing.")

    # set up the two coefficient matrixs

    coef1 = np.matrix(fdobj1.coef)
    coef2 = np.matrix(fdobj2.coef)
    if len(np.shape(coef1)) > 2:
        raise ValueError("FDOBJ1 is not univariate.")
    if len(np.shape(coef2)) > 2:
        raise ValueError("FDOBJ2 is not univariate.")

    nbreaks = len(breaks)
    ninterval = nbreaks - 1
    nbasis1 = ninterval + norder1 - 1
    nbasis2 = ninterval + norder2 - 1
    if np.shape(coef1)[0] != nbasis1 or np.shape(coef2)[0] != nbasis2:
        raise ValueError(
            "Error: coef1 should have length no. breaks1 + norder1 - 2, and coef2 no. breaks2 + norder2 - 2.")

    breaks1 = breaks[0]
    breaksn = breaks[nbreaks - 1]

    # the knot sequences are built so that there are no continuity conditions
    # at the first and last breaks. There are k-1 continuity conditions at
    # the other breaks.

    temp = breaks[1: nbreaks - 1]
    knots1 = np.concatenate((np.concatenate((breaks1 * np.ones(norder1), temp)), breaksn * np.ones(norder1)))
    knots2 = np.concatenate((np.concatenate((breaks1 * np.ones(norder2), temp)), breaksn * np.ones(norder2)))

    # construct the piecewise polynomial representations of
    # f^DERIV1 and g^DERIV2

    nrep1 = np.shape(coef1)[1]
    polycoef1 = np.zeros([ninterval, norder1 - nderiv1[0], nrep1])
    for i in range(nbasis1):
        # compute polynomial representation of B(i, norder1, knots1)(x)
        ppBlist = ppBspline(knots1[range(i, i + norder1 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        # convert the index of the breaks in knots1 to the index in the variable 'breaks'
        index = index + i - norder1
        CoeffD = ppderiv(Coeff, nderiv1)
        if nrep1 == 1:
            polycoef1[index, :, 1] = coef1[i] * CoeffD + polycoef1[index, :, 1]
        else:
            for j in range(len(index)):
                temp = np.outer(CoeffD[j, :], coef1[i, :])
                polycoef1[index[j], :, :] = temp + polycoef1[index[j], :, :]

    nrep2 = np.shape(coef2)[1]
    polycoef2 = np.zeros([ninterval, norder2 - nderiv2[0], nrep2])
    for i in range(nbasis2):
        # compute polynomial representation of B(i, norder2, knots2)(x)
        ppBlist = ppBspline(knots2[range(i, i + norder2 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        # convert the index of the breaks in knots1 to the index in the variable 'breaks'
        index = index + i - norder2
        CoeffD = ppderiv(Coeff, nderiv2)
        if nrep2 == 1:
            polycoef2[index, :, 1] = coef2[i] * CoeffD + polycoef2[index, :, 1]
        else:
            for j in range(len(index)):
                temp = np.outer(CoeffD[j, :], coef2[i, :])
                polycoef2[index[j], :, :] = temp + polycoef2[index[j], :, :]

    # Compute the scalar product between f and g

    prodmat = np.zeros([nrep1, nrep2])
    for j in range(ninterval):
        # multiply f(i1) and g(i2) piecewise and integrate
        c1 = np.matrix(polycoef1[j, :, :])
        c2 = np.matrix(polycoef2[j, :, :])
        polyprodmat = polyprod(c1, c2)
        # compute the coefficients of the anti-derivative
        N = np.shape(polyprodmat)[2]
        delta = breaks[j + 1] - breaks[j]
        power = delta
        prodmati = np.zeros([nrep1, nrep2])
        for i in range(N):
            prodmati = prodmati + power * polyprodmat[:, :, N - i - 1] / (i + 1)
            power = power * delta
        # add the integral to s
        prodmat = prodmat + prodmati

    return prodmat


def int2Lfd(m=[0]):
    if isinstance(m, Lfd):
        Lfdobj = m
        return Lfdobj

    if isinstance(m, int):
        m = [m]
        # raise ValueError("Argument not numeric and not a linear differential operator.")

    if len(m) != 1:
        raise ValueError("Argument is not a scalar.")

    if round(m[0]) != m[0]:
        raise ValueError("Argument is not an integer.")

    if m[0] < 0:
        raise ValueError("Argument is negative.")

    if m[0] == 0:
        bwtlist = None
    else:
        basisobj = create_constant_basis([0, 1])
        bwtlist = []
        for j in range(m[0]):
            bwtlist.append(fd(np.zeros(0), basisobj))

    Lfdobj = Lfd(m, bwtlist)
    return Lfdobj


def fdchk(fdobj):
    if isinstance(fdobj, fd):
        coef = fdobj.coef
    else:
        if isinstance(fdobj, basis):
            coef = np.zeros([fdobj.nbasis - len(fdobj.dropind), fdobj.nbasis - len(fdobj.dropind)])
            np.fill_diagonal(coef, 1)
            fdobj = fd(coef, fdobj)
        else:
            raise ValueError("FDOBJ is not an FD object.")

    # extract the number of replications and basis object

    coefd = np.shape(np.matrix(coef))
    if len(coefd) > 2:
        raise ValueError("Functional data object must be univariate.")
    nrep = coefd[1]
    basisobj = fdobj.basisobj
    return [nrep, fdobj]


def inprod(fdobj1, fdobj2=None, Lfdobj1=int2Lfd(0), Lfdobj2=int2Lfd(0), rng=None, wtfd=0):
    # check FDOBJ1 and get no. replications and basis object
    result1 = fdchk(fdobj1)
    nrep1 = result1[0]
    fdobj1 = result1[1]
    coef1 = fdobj1.coef
    basisobj1 = fdobj1.basisobj
    type1 = basisobj1.type
    range1 = basisobj1.rangeval
    if rng is None:
        rng = range1
    # Default FDOBJ2 to a constant function, using a basis that matches
    # that of FDOBJ1 if possible

    if fdobj2 is None:
        tempfd = fdobj1
        tempbasis = fdobj1.basisobj
        temptype = basisobj1.type
        temprange = basisobj1.rangeval
        if temptype == 'bspline':
            basis2 = create_bspline_basis(temprange, 1, 1)
        else:
            basis2 = create_constant_basis(temprange)
        fdobj2 = fd(1, basis2)

    # check FDOBJ2 and get no. replications and basis object
    result2 = fdchk(fdobj2)
    nrep2 = result2[0]
    fdobj2 = result2[1]
    coef2 = fdobj2.coef
    basisobj2 = fdobj2.basisobj
    type2 = basisobj2.type
    range2 = basisobj2.rangeval

    # check ranges

    if rng[0] < range1[0] or rng[1] > range1[1]:
        raise ValueError("Limits of integration are inadmissible.")

    if isinstance(fdobj1, fd) and isinstance(fdobj2,
                                             fd) and type1 == 'bspline' and type2 == 'bspline' and basisobj1 == basisobj2 and len(
        basisobj1.dropind) == 0 and len(
        basisobj2.dropind) == 0 and wtfd == 0 and np.all(rng == range1):
        inprodmat = inprod_bspline(fdobj1, fdobj2, Lfdobj1.nderiv, Lfdobj2.nderiv)
        return inprodmat

    # check LFDOBJ1 and LFDOBJ2
    Lfdobj1 = int2Lfd(Lfdobj1)
    Lfdobj2 = int2Lfd(Lfdobj2)

    # else proceed with the use of the Romberg integration.

    # set iter
    iter = 0

    # The default case, no multiplicities
    rngvec = rng

    # check for any knot multiplicaties in either argument
    knotmult = []
    if type1 == 'bspline':
        knotmult = knotmultchk(basisobj1, knotmult)
    if type2 == 'bspline':
        knotmult = knotmultchk(basisobj2, knotmult)

    # Modify RNGVEC defining subinverals if there are any
    # knot multiplicities

    if len(knotmult) > 0:
        knotmult = np.unique(knotmult).sort()
        index = np.logical_and(knotmult > rng[0], knotmult < rng[1])
        knotmult = knotmult[index]
        rngvec = np.concatenate((np.concatenate((rng[0], knotmult)), rng[1]))

    # check for either coefficient aray being zero
    if np.all(coef1 == 0) or np.all(coef2 == 0):
        return np.zeros([nrep1, nrep2])

    ##################################
    # loop through sub-intervals
    ##################################

    JMAX = 15
    JMIN = 5
    EPS = 1e-4

    inprodmat = np.zeros([nrep1, nrep2])
    nrng = len(rngvec)
    for irng in range(1, nrng):
        rngi = [rngvec[irng - 1], rngvec[irng]]
        # change range so as to avoid being exactly one
        # multiple knot values
        if irng > 2:
            rngi[0] = rngi[0] + 1e-10
        if irng < nrng:
            rngi[1] = rngi[1] - 1e-10

        # set up first iteration

        iter = 1
        width = rngi[1] - rngi[0]
        JMAXP = JMAX + 1
        h = np.ones(JMAXP)
        h[1] = 0.25
        s = np.zeros([JMAXP, nrep1, nrep2])
        sdim = len(np.shape(s))
        # the first iterations uses just the endpoints
        fx1 = eval_fd(rngi, fdobj1, Lfdobj1)
        fx2 = eval_fd(rngi, fdobj2, Lfdobj2)
        # multiply by values of weight function if necessary
        # if isinstance(wtfd, fd):
        #     wtd = eval_fd(rngi, wtfd, 0)
        #     fx2 = np.multiply(wtd.reshape((np.shape(wtd)[0], np.shape(fx2)[1])), fx2)
        s[0, :, :] = width * np.matmul(np.transpose(fx1), fx2).reshape((nrep1, nrep2)) / 2
        tnm = 0.5

        # now iterate to convergence
        for iter in range(2, JMAX):
            tnm = tnm * 2
            if iter == 2:
                x = np.mean(rngi)
            else:
                dele = width / tnm
                x = np.linspace(rngi[0] + dele / 2, rngi[1] - dele / 2, int((rngi[1] - rngi[0]) / dele))
            fx1 = eval_fd(x, fdobj1, Lfdobj1)
            fx2 = eval_fd(x, fdobj2, Lfdobj2)
            if not isinstance(wtfd, int):
                wtd = eval_fd(wtfd, x, 0)
                fx2 = np.multiply(np.repeat(wtd, np.shape(fx2)[1]).reshape((np.shape(wtd)[0], np.shape(fx2)[1])), fx2)
            chs = width * np.matmul(np.transpose(fx1), fx2).reshape((nrep1, nrep2)) / tnm
            s[iter - 1, :, :] = (s[iter - 2, :, :] + chs) / 2
            if iter >= 5:
                ind = range(iter - 5, iter)
                ya = s[ind, :, :]
                ya = ya.reshape((5, nrep1, nrep2))
                xa = h[ind]
                absxa = np.abs(xa)
                absxamin = np.min(absxa)
                ns = int(np.min(np.linspace(0, len(absxa) - 1, len(absxa))[absxa == absxamin]))
                cs = ya
                ds = ya
                y = ya[ns, :, :]
                ns = ns - 1
                for m in range(4):
                    for i in range(4 - m):
                        ho = xa[i]
                        hp = xa[i + m]
                        w = (cs[i + 1, :, :] - ds[i, :, :]) / ((ho - hp) if (ho - hp) > 1e-6 else 1e-6)
                        ds[i, :, :] = hp * w
                        cs[i, :, :] = ho * w
                    if 2 * ns < 4 - m:
                        dy = cs[ns, :, :]
                    else:
                        dy = ds[ns - 1, :, :]
                        ns = ns - 1
                    y = y + dy
                ss = y
                errval = np.max(np.abs(dy))
                ssqval = np.max(np.abs(ss))
                if np.all(ssqval > 0):
                    crit = errval / ssqval
                else:
                    crit = errval
                if crit < EPS and iter >= JMIN:
                    break
            s[iter, :, :] = s[iter - 1, :, :]
            h[iter] = 0.25 * h[iter - 1]
        inprodmat = inprodmat + ss

    if len(np.shape(inprodmat)) == 2:
        return np.matrix(inprodmat)
    else:
        return inprodmat

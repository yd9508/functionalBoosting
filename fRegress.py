import numpy as np
from fd import *
from basis import *
from smooth_basis import *
import copy
from fd import *

def ppBspline(t):
    norder = len(t) - 1
    ncoef = 2 * (norder - 1)
    if norder > 1:
        adds = np.ones(norder - 1)
        tt = np.concatenate((np.concatenate((adds * t[0], t)), adds * t[norder]))
        gapin = np.linspace(1, len(tt) - 1, len(tt) - 1)[np.diff(tt) > 0]
        ngap = len(gapin)
        iseq = [i for i in range(2 - norder, norder)]
        ind = np.outer(np.ones(ngap), iseq) + np.outer(gapin, np.ones(ncoef))
        ind = ind.astype(int)
        tx = np.matrix([tt[i - 1] for i in ind.flatten()]).reshape((ngap, ncoef))
        gapin = gapin.astype(int)
        ty = tx - np.outer([tt[i - 1] for i in gapin], np.ones(ncoef))
        b = np.outer(np.ones(ngap), [i for i in range(1 - norder, 1)]) + np.outer(gapin, np.ones(norder))
        a = np.concatenate((np.concatenate((adds * 0, [1])), adds * 0))
        b = b.astype(int)
        d = np.matrix([a[i - 1] for i in b.flatten()]).reshape((ngap, norder))
        for j in range(1, norder):
            for i in range(1, norder - j + 1):
                ind1 = i + norder - 1
                ind2 = i + j - 1
                d[:, i - 1] = (np.multiply(ty[:, ind1 - 1], d[:, i - 1]) - np.multiply(ty[:, ind2 - 1], d[:, i])) / (
                        ty[:, ind1 - 1] - ty[:, ind2 - 1])
        Coeff = d
        for j in range(2, norder + 1):
            factor = (norder - j + 1) / (j - 1)
            ind = [i for i in range(norder, j - 1, -1)]
            for i in ind:
                Coeff[:, i - 1] = factor * (Coeff[:, i - 1] - Coeff[:, i - 2]) / ty[:, i + norder - j - 1]
        ind = [i for i in range(norder, 0, -1)]
        if ngap > 1:
            Coeff = np.flip(Coeff, 1)
        else:
            Coeff = np.matrix(np.flip(Coeff, 1)).reshape((1, norder))
        index = gapin - norder + 1
    else:
        Coeff = np.matrix(1)
        index = np.matrix(1)
    return [Coeff, index]


def ppderiv(Coeff, Deriv=[0]):
    m = np.shape(Coeff)[0]
    k = np.shape(Coeff)[1]

    if Deriv[0] < 1:
        CoeffD = np.matrix(Coeff)
        return CoeffD

    if k - Deriv[0] < 1:
        CoeffD = np.zeros([m, 1])
        return CoeffD
    else:
        CoeffD = Coeff[:, range(1, k - Deriv[0] + 1)]
        if not isinstance(CoeffD, np.ndarray):
            CoeffD = np.transpose(np.matrix(CoeffD))
            for j in range(1, k - 1):
                bound1 = np.max(1, j - Deriv[0] + 1)
                bound2 = np.min(j, k - Deriv[0])
                CoeffD[:, range(bound1 - 1, bound2)] = (k - j) * CoeffD[:, range(bound1 - 1, bound2)]
        return CoeffD


def polyprod(Coeff1, Coeff2):
    polyorder1 = np.shape(Coeff1)[0]
    norder1 = np.shape(Coeff1)[1]
    polyorder2 = np.shape(Coeff2)[0]
    norder2 = np.shape(Coeff2)[1]
    ndegree1 = polyorder1 - 1
    ndegree2 = polyorder2 - 1

    # if the degrees are not equal, pad out the smaller matrix with 0s
    if ndegree1 != ndegree2:
        if ndegree1 > ndegree2:
            Coeff2 = np.concatenate((Coeff2, np.zeros(ndegree1 - ndegree2, norder2)))
        else:
            Coeff1 = np.concatenate((Coeff1, np.zeros(ndegree2 - ndegree1, norder1)))

    # find order of the product
    D = np.max([ndegree1, ndegree2])
    N = 2 * D + 1

    # compute the coefficients for the products
    convmat = np.zeros([norder1, norder2, N])
    for i in range(D):
        ind = np.array([x + 1 for x in range(i + 1)])
        if len(ind) == 1:
            convmat[:, :, i] = np.outer(Coeff1[ind - 1, :], Coeff2[i - ind + 1, :])
            convmat[:, :, N - i - 1] = np.outer(Coeff1[D - ind + 1, :], Coeff2[D - i + ind - 1, :])
        else:
            convmat[:, :, i] = np.matmul(np.transpose(Coeff1[ind - 1, :]), Coeff2[i - ind + 1, :])
            convmat[:, :, N - i - 1] = np.matmul(np.transpose(Coeff1[D - ind + 1, :]), Coeff2[D - i + ind - 1, :])
    ind = np.array([x + 1 for x in range(D + 1)])
    convmat[:, :, D] = np.matmul(np.transpose(Coeff1[ind - 1, :]), Coeff2[D - ind + 1, :])
    if ndegree1 != ndegree2:
        convmat = convmat[:, :, range(ndegree1 + ndegree2 + 1)]
        convmat = convmat.reshape((norder1, norder2, ndegree1 + ndegree2 + 1))
    return convmat


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


def eigchk(Cmat):
    if np.max(np.abs(Cmat - np.transpose(Cmat))) / np.max(np.abs(Cmat)) > 1e-10:
        raise ValueError("Cmat is not symmetric.")
    else:
        Cmat = (Cmat + np.transpose(Cmat)) / 2

    # check Cmat  for singularity

    eigval = np.linalg.eigvals(Cmat)
    ncoef = len(eigval)
    if np.min(eigval) <= 0:
        raise ValueError("Non-positive eigenvalue of coefficient matrix.")
    logcondition = np.log10(np.max(eigval)) - np.log10(np.min(eigval))
    if logcondition > 12:
        print("Near singluarity in coefficient matrix.")


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


class fRegressArg:
    def __init__(self, yfd, xfdlist, betalist, wt):
        self.yfd = yfd
        self.xfdlist = xfdlist
        self.betalist = betalist
        self.wt = wt


class fRegressList:
    def __init__(self, yfdobj, xfdlist, betalist, betaestlist, yhatfdobj, Cmat, Dmat, Cmatinv, wt, df, y2cMap, SigmaE,
                 betastderrlist, bvar, c2bMap):
        self.yfdobj = yfdobj
        self.xfdlist = xfdlist
        self.betalist = betalist
        self.betaestlist = betaestlist
        self.yhatfdobj = yhatfdobj
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.Cmatinv = Cmatinv
        self.wt = wt
        self.df = df
        self.y2cMap = y2cMap
        self.SigmaE = SigmaE
        self.betastderrlist = betastderrlist
        self.bvar = bvar
        self.c2bMap = c2bMap


def fRegressArgCheck(yfd, xfdlist, betalist, wt=None):
    if not (isinstance(yfd, fdPar) or isinstance(yfd, fd)):
        raise ValueError("First argument is not of class 'fdPar' and 'fd'.")

    if isinstance(yfd, fdPar):
        yfd = yfd.fd

    if isinstance(yfd, fd):
        ycoef = yfd.coef
        N = np.shape(ycoef)[1]
    else:
        N = len(yfd)

    # check that xfdlist is a list object and compute number of covaraites p

    # check XFDLIST

    if isinstance(xfdlist, fd):
        xfdlist = [xfdlist]

    # get number of independent variables p

    p = len(xfdlist)

    # check BETALIST

    if isinstance(betalist, fd):
        betalist = [betalist]

    if len(betalist) != p:
        raise ValueError("Number of regression coefficients does not match number of independent variables.")

    # extract the range if YFD is functional

    if isinstance(yfd, fd):
        rangeval = yfd.basisobj.rangeval
    else:
        rangeval = [0, 1]

    # check contents of XFDLIST

    onebasis = create_constant_basis(rangeval)
    onesfd = fd(1, onebasis)

    xerror = False

    for j in range(p):
        xfdj = xfdlist[j]
        if isinstance(xfdj, fd):
            xcoef = xfdj.coef
            if len(np.shape(xcoef)) > 2:
                raise ValueError("Covariate ", j, " is not univariate.")

            Nj = np.shape(xcoef)[1]
            if Nj != N:
                print("Incorrect number of replications in XFDLIST for covariate ", j)
                xerror = True
        elif isinstance(xfdj, np.ndarray):
            xfdj = np.matrix(xfdj)
            Zdimj = np.shape(xfdj)
            if Zdimj[0] != N:
                print("vector in XFDLIST has wrong length.")
                xerror = True
            if Zdimj[1] != 1:
                print("vector in XFDLIST has wrong length.")
                xerror = True
            xfdlist[j] = fd(np.matrix(xfdj).reshape((1, N)), onebasis)

    # check contents of BETALIST

    berror = False
    for j in range(p):
        betafdParj = betalist[j]
        if isinstance(betafdParj, fd) or isinstance(betafdParj, basis):
            betafdParj = fdPar(betafdParj)
            betalist[j] = betafdParj
        if not isinstance(betafdParj, fdPar):
            print("BETALIST  is not  a fdPar object.")
            berror = True

    if xerror or berror:
        raise ValueError("An error has been found in either XFDLIST or BETALIST.")

    if wt is None:
        wt = np.ones(N)

    return fRegressArg(yfd, xfdlist, betalist, wt)


def fRegress(y, xfdlist, betalist, wt=None, y2cMap=None, SigmaE=None, returnMatrix=False, method='fRegress', sep='.'):
    arglist = fRegressArgCheck(y, xfdlist, betalist, wt)
    yfdobj = arglist.yfd
    xfdlist = arglist.xfdlist
    betalist = arglist.betalist
    wt = arglist.wt
    p = len(xfdlist)
    wtconst = np.var(wt) == 0
    ycoef = yfdobj.coef
    ycoefdim = np.shape(ycoef)
    N = ycoefdim[1]
    ybasisobj = yfdobj.basisobj
    rangeval = ybasisobj.rangeval
    ynbasis = ybasisobj.nbasis
    onesbasis = create_constant_basis(rangeval)
    onesfd = fd(1, onesbasis)
    if len(ycoefdim) > 2:
        raise ValueError("YFDOBJ from YFD is not univariate.")
    ncoef = 0
    for j in range(p):
        betafdParj = betalist[j]
        if betafdParj.estimate:
            ncoefj = betafdParj.fd.basisobj.nbasis
            ncoef = ncoef + ncoefj
    Cmat = np.zeros([ncoef, ncoef])
    Dmat = np.zeros(ncoef)
    mj2 = 0
    # j = 1
    # xfdj = xfdlist[j]
    # xyfdj = xfdj * yfdobj
    # wtfdj = xyfdj.sum()
    # betafdj = betalist[j].fd
    # betabasisj = betafdj.basis
    # ncoefj = betabasisj.nbasis
    # k =3
    # betafdk = betalist[k].fd
    # betabasisk = betafdk.basis
    # ncoefk = betabasisk.nbasis
    for j in range(p):
        betafdParj = betalist[j]
        if betafdParj.estimate:
            betafdj = betafdParj.fd
            betabasisj = betafdj.basisobj
            ncoefj = betabasisj.nbasis
            mj1 = mj2
            mj2 = mj1 + ncoefj
            xfdj = xfdlist[j]
            if wtconst:
                xyfdj = xfdj * yfdobj
            else:
                xyfdj = xfdj * wt * yfdobj
            wtfdj = xyfdj.sum()
            Dmatj = inprod(betabasisj, onesfd, 0, 0, rangeval, wtfdj)
            Dmat[mj1: mj2] = Dmatj.reshape(mj2 - mj1)
            mk2 = 0
            for k in range(j + 1):
                betafdPark = betalist[k]
                if betafdPark.estimate:
                    betafdk = betafdPark.fd
                    betabasisk = betafdk.basisobj
                    ncoefk = betabasisk.nbasis
                    mk1 = mk2
                    mk2 = mk1 + ncoefk
                    xfdk = xfdlist[k]
                    if wtconst:
                        xxfdjk = xfdj * xfdk
                    else:
                        xxfdjk = xfdj * wt * xfdk
                    wtfdjk = xxfdjk.sum()
                    Cmatjk = inprod(betabasisj, betabasisk, 0, 0, rangeval, wtfdjk)
                    Cmat[mj1: mj2, mk1:mk2] = Cmatjk
                    Cmat[mk1:mk2, mj1: mj2] = np.transpose(Cmatjk)
            lamdbaj = betafdParj.lamdba
            if lamdbaj > 0:
                Rmatj = betafdParj.penmat
                if Rmatj is None:
                    Lfdj = betafdParj.Lfd
                    Rmatj = eval_penalty(betabasisj, Lfdj)
                Cmat[mj1: mj2, mj1: mj2] = Cmat[mj1: mj2, mj1: mj2] + lamdbaj * Rmatj
    Cmat = (Cmat + np.transpose(Cmat)) / 2

    eigchk(Cmat)
    Cmatinv = np.linalg.inv(Cmat)
    # Lmat = np.linalg.cholesky(Cmat)
    # Lmatinv = np.linalg.inv(Lmat)
    # Cmatinv = np.matmul(Lmatinv, np.transpose(Lmatinv))
    betacoef = np.dot(Cmatinv, Dmat)
    betaestlist = copy.deepcopy(betalist)
    mj2 = 0
    for j in range(p):
        betafdParj = copy.deepcopy(betalist[j])
        if betafdParj.estimate:
            betafdj = betafdParj.fd
            ncoefj = betafdj.basisobj.nbasis
            mj1 = mj2
            mj2 = mj2 + ncoefj
            coefj = betacoef[mj1: mj2]
            betafdj.coef = np.transpose(np.matrix(coefj))
            betafdParj.fd = betafdj
        betaestlist[j] = betafdParj
    nfine = max(501, 10 * ynbasis + 1)
    tfine = np.linspace(rangeval[0], rangeval[1], nfine)
    yhatmat = np.zeros([nfine, N])
    for j in range(p):
        xfdj = xfdlist[j]
        xmatj = eval_fd(tfine, xfdj, 0, returnMatrix)
        betafdParj = betaestlist[j]
        betafdj = betafdParj.fd
        betavecj = eval_fd(tfine, betafdj, 0, returnMatrix)
        for k in range(np.shape(xmatj)[1]):
            xmatj[:, k] = np.multiply(xmatj[:, k], np.matrix(betavecj))
        yhatmat = yhatmat + xmatj
    yhatfdobj = smooth_basis(tfine, yhatmat, ybasisobj).fd
    df = None
    if not (y2cMap is None or SigmaE is None):
        y2cdim = np.shape(y2cMap)
        if y2cdim[0] != ynbasis or y2cdim[1] != np.shape(SigmaE)[0]:
            raise ValueError("Dimensions of Y2CMAP not correct.")
        ybasismat = eval_basis(tfine, ncoef, ynbasis * N)
        deltat = tfine[1] - tfine[0]
        basisprodmat = np.zeros([ncoef, ynbasis * N])
        mj2 = 0
        for j in range(p):
            betafdParj = betalist[j]
            betabasisj = betafdParj.fd.basisobj
            ncoefj = betabasisj.nbasis
            bbasismatj = eval_basis(tfine, betabasisj, 0, returnMatrix)
            xfdj = xfdlist[j]
            tempj = eval_fd(tfine, xfdj, 0, returnMatrix)
            mj1 = mj2 + 1
            mj2 = mj2 + ncoefj
            mk2 = 0
            for k in range(ynbasis):
                mk1 = mk2 + 1
                mk2 = mk2 + N
                tempk = bbasismatj * ybasismat[:, k]
                basisprodmat[mj1 - 1: mj2, mk1 - 1: mk2] = deltat * np.matmul(np.transpose(tempk), tempj)
        c2bMap = np.linalg.solve(Cmat, basisprodmat)
        VarCoef = np.dot(np.dot(y2cMap, SigmaE), np.transpose(y2cMap))
        CVariance = np.kron(VarCoef, np.identity(N))
        bvar = np.dot(np.dot(c2bMap, CVariance), np.transpose(c2bMap))
        betastderrlist = []
        mj2 = 0
        for j in range(p):
            betafdParj = betalist[j]
            betabasisj = betafdParj.fd.basisobj
            ncoefj = betabasisj.nbasis
            mj1 = mj2 + 1
            mj2 = mj2 + ncoefj
            bbasismat = eval_basis(tfine, betabasisj, 0, returnMatrix)
            bvarj = bvar[mj1 - 1: mj2, mj1 - 1: mj2]
            bstderrj = np.sqrt(np.dot(np.dot(bbasismat, bvarj), np.transpose(bbasismat)).diagonal())
            bstderrfdj = smooth_basis(tfine, bstderrj, betabasisj).fd
            betastderrlist.append(bstderrfdj)
    else:
        betastderrlist = None
        bvar = None
        c2bMap = None

    fr = fRegressList(yfdobj, xfdlist, betalist, betaestlist, yhatfdobj, Cmat, Dmat, Cmatinv, wt, df, y2cMap, SigmaE,
                      betastderrlist, bvar, c2bMap)
    return fr







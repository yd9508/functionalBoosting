import sys
import warnings
import numpy as np
from basis import *
from basis import basis
from locfdr import *

class fd:
    def __init__(self, coef=None, basisobj=None, fdnames=None):

        # # check coef and get its dimension
        # if not isinstance(coef, np.ndarray) and not isinstance(coef , int):
        #     raise ValueError("'coef' is not an instance of numpy ndarray, please check.")
        #
        # if coef is None:
        #     coef = np.zeros([basisobj.nbasis])
        #
        # coef = np.matrix(coef)
        # coefd = coef.shape
        # ndim = len(coefd)
        #
        # if ndim > 3:
        #     raise ValueError("'coef' not of dimension 1, 2 or 3")
        #
        # # check basisobj
        #
        # nbasis = basisobj.nbasis
        # dropind = basisobj.dropind
        # ndropind = len(dropind)
        #
        # if coefd[0] != nbasis - ndropind:
        #     raise ValueError("First dim. of 'coef' not equal to 'nbasis - ndropind'.")
        #
        # # setup number of replicates and number of variables
        # if ndim > 1:
        #     nrep = coefd[1]
        # else:
        #     nrep = 1
        #
        # if ndim > 2:
        #     nvar = coefd[2]
        # else:
        #     nvar = 1

        self.coef = coef
        self.basisobj = basisobj
        self.fdnames = fdnames

    def __add__(self, other):
        if isinstance(other, fd):
            return fd(self.coef + other.coef, self.basisobj, self.fdnames)
        else:
            pass

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, fd):
            coef = self.coef - other.coef
            return fd(coef, self.basisobj, self.fdnames)

    def __mul__(self, other):
        if isinstance(other, fd):
            coef1 = self.coef
            coef2 = other.coef
            coefd1 = np.shape(coef1)
            coefd2 = np.shape(coef2)
            ndim1 = len(coefd1)
            ndim2 = len(coefd2)

            if len(coefd1) != len(coefd2):
                raise ValueError("Number of dimensions of coefficient arrays does not match.")
            # allow for one function having a single replicate
            # and if so, copy it as many times as there are replicates
            # in the other function

            # self is single, other has replications

            if coefd1[1] == 1 and coefd2[1] > 1:
                if ndim1 == 2:
                    coef1 = np.matrix(coef1).reshape(coefd1[0], coefd2[1])
                elif ndim1 == 3:
                    temp = np.zeros(coefd2)
                    for j in range(coefd1[2]):
                        temp[:, :, j] = np.outer(coef1[:, 0, j], np.ones(coefd2[1]))
                    coef1 = temp
                else:
                    raise ValueError("Dimensions of coefficient matrices not compatable.")
                coefd1 = np.shape(coef1)
                self.coef = coef1

            # self has replications, other is single

            if coefd1[1] > 1 and coefd2[1] == 1:
                if ndim2 == 2:
                    coef2 = np.matrix(coef2).reshape(coefd2[0], coefd1[1])
                elif ndim1 == 3:
                    temp = np.zeros(coefd1)
                    for j in range(coefd2[2]):
                        temp[:, :, j] = np.outer(coef2[:, 0, j], np.ones(coefd1[1]))
                    coef2 = temp
                else:
                    raise ValueError("Dimensions of coefficient matrices not compatible.")
                coefd2 = np.shape(coef2)
                other.coef = coef2

            # check that numbers of replications are equal
            if coefd1[1] != coefd2[1]:
                raise ValueError("Number of replications are not equal.")

            # check for matching in the muktivariate case

            if ndim1 > 2 and ndim2 > 2 and ndim1 != ndim2:
                raise ValueError("Both arguments multivariate, but involve different numbers of functions.")

            # extract the two bases

            basisobj = self.basisobj * other.basisobj
            nbasis1 = self.basisobj.nbasis
            nbasis2 = other.basisobj.nbasis
            rangeval = self.basisobj.rangeval

            # set up a fine mesh for evaluating the product
            neval = max(10 * max(nbasis1, nbasis2) + 1, 201)
            evalarg = np.linspace(rangeval[0], rangeval[1], neval)

            # set up arrays for function values
            fdarray1 = eval_fd(evalarg, self)
            fdarray2 = eval_fd(evalarg, other)

            # compute product arrays

            if (ndim1 <= 2 and ndim2 <= 2) or (ndim1 > 2 and ndim2 > 2):
                fdarray = np.multiply(fdarray1, fdarray2)
            else:
                # product array where the number of dimensions don't match
                if ndim1 == 2 and ndim2 > 2:
                    fdarray = np.zeros(coefd2)
                    for ivar in range(coefd2[2]):
                        fdarray[:, :, ivar] = fdarray1 * fdarray2[:, :, ivar]
                if ndim1 > 2 and ndim2 == 2:
                    fdarray = np.zeros(coefd1)
                    for ivar in range(coefd1[2]):
                        fdarray[:, :, ivar] = fdarray1[:, :, ivar] * fdarray2

            # set up the coefficient by projecting on to the
            # product basis

            coefprod = project_basis(fdarray, evalarg, basisobj)
        else:
            fac = other
            fac = np.matrix(fac)
            coef = self.coef
            coefd = np.shape(coef)
            coefprod = np.multiply(fac, coef)
            basisobj = self.basisobj

        fdprod = fd(coefprod, basisobj)

        return fdprod

    __rmul__ = __mul__

    def mean(self):
        coef = self.coef
        coefd = np.shape(coef)
        ndim = len(coefd)
        basisobj = self.basisobj
        nbasis = basisobj.nbasis
        dropind = basisobj.dropind
        ndropind = len(dropind)
        if ndim == 2:
            coefmean = np.reshape(coef.mean(1), (nbasis - ndropind, 1))
        else:
            nvar = coefd[2]
            coefmean = np.zeros((coefd[0], 1, nvar))
            for j in range(nvar):
                coefmean[:, 0, j] = coef[:, :, j].mean(1)
        meanfd = fd(coefmean, basisobj)
        return meanfd

    def sum(self):
        coef = self.coef
        coefd = np.shape(coef)
        ndim = len(coefd)
        basisobj = self.basisobj
        nbasis = basisobj.nbasis
        dropind = basisobj.dropind
        ndropind = len(dropind)

        if ndim == 2:
            if nbasis - ndropind == 0:
                coefsum = np.matrix(np.sum(coef, axis=1))
            else:
                coefsum = np.matrix(np.sum(coef, axis=1)).reshape((nbasis - ndropind, 1))
        else:
            nvar = coefd[2]
            coefsum = np.zeros([coefd[0], 1, nvar])
            for j in range(nvar):
                coefsum[:, 1, j] = np.sum(coef[:, :, j], axis=1)

        sumfd = fd(coefsum, basisobj)
        return sumfd


def eval_basis(evalarg, basisobj, Lftobj=0, returnMatrix=False):
    # 1. check
    if (isinstance(basisobj, int) or isinstance(basisobj, list)) and isinstance(evalarg, basis):
        temp = basisobj
        basisobj = evalarg
        evalarg = temp

    # check basisobj
    if not isinstance(basisobj, basis):
        raise ValueError("Second argument is not a basis object.")

    # check LFDOBJ

    Lftobj = int2Lfd(Lftobj)

    # 2. Set up
    # determine the highest order of derivative NDERIV required

    nderiv = Lftobj.nderiv

    # get weight coefficient functions

    bwtlist = Lftobj.bwlist

    # 3. Do
    # get highest order of basis matrix

    basismat = getbasismatrix(evalarg, basisobj, nderiv, returnMatrix)

    # Compute the weightes combination of derivations is
    # evaluated here if the operator is not defined by an integer
    # and the order of derivative is positive

    if nderiv > [0]:
        nbasis = np.shape(basismat)[1]
        oneb = np.ones([1, nbasis])
        nonintwrd = False
        for j in range(nderiv[0]):
            bfd = bwtlist[j]
            bbasis = bfd.basisobj
            if bbasis.type != 'const' or bfd.coef != 0:
                nonintwrd = True
        if nonintwrd:
            for j in range(nderiv[0]):
                bfd = bwtlist[j]
                temp = bfd.coef == 0.0
                if not temp.all():
                    wjarry = eval_fd(evalarg, bfd, 0, returnMatrix)
                    Dbasismat = getbasismatrix(evalarg, basisobj, [j - 1], returnMatrix)
                    basismat = basismat + np.multiply(np.dot(wjarry, oneb), Dbasismat)

    # if (not returnMatrix) and len(np.shape(basismat)) == 2:
    #     return basismat.tolist()
    # else:
    #     return basismat
    return basismat


def eval_fd(evalarg, fdobj, Lfdobj=0, returnMatrix=False):
    Lfdobj = int2Lfd(Lfdobj)
    if ((isinstance(fdobj, int) or isinstance(fdobj, np.ndarray)) or isinstance(fdobj, float)) and isinstance(evalarg,
                                                                                                              fd):
        temp = fdobj
        fdobj = evalarg
        evalarg = temp

    # check EVALARG
    Evalarg = evalarg
    # if not isinstance(evalarg, int):
    #     evalarg = int(Evalarg)
    #     nNa = sum(np.isnan(evalarg))
    #     if nNa > 0:
    #         raise ValueError("as.numeric(evalarg) contains", nNa, "NA")

    evaldim = np.shape(evalarg)
    if len(evaldim) >= 3:
        raise ValueError("Argument 'evalarg' is not a vector or a matrix.")

    # check FDOBJ
    if not isinstance(fdobj, fd):
        raise ValueError("Argument FD is not a functional data object.")

    basisobj = fdobj.basisobj
    nbasis = basisobj.nbasis
    rangeval = basisobj.rangeval
    onerow = [1] * nbasis

    # temp = [i for i in evalarg if not np.isnan(i)]
    # EPS = 5 * sys.float_info.epsilon
    # if min(temp) < rangeval[0] - EPS or max(temp) > rangeval[1] + EPS:
    #     warnings.warn("Values in argument 'evalarg' are outside of permitted range and will be ignored.")

    # get maximum number of evaluation values
    if isinstance(evalarg, np.ndarray):
        n = len(evalarg)
    elif isinstance(evalarg, float):
        evalarg = [evalarg]
        n = len(evalarg)
    else:
        n = evaldim[0]

    # set up coefficient array for FD
    coef = fdobj.coef
    coef = np.array(coef)
    coefd = np.shape(coef)
    ndim = len(coefd)
    nrep = 1 if ndim <= 1 else coefd[1]
    nvar = 1 if ndim <= 2 else coefd[2]

    # check coef is conformable with evalarg
    if len(evaldim) > 1:
        if evaldim[1] != 1 and evaldim[1] != coefd[1]:
            raise ValueError("'evalarg' has ", evaldim[1], "columns; does not match ", ndim[1],
                             " = number of columns of fdobj.coefs.")

    # Set up array for function values
    if ndim <= 2:
        evalarray = np.zeros([n, nrep])
    else:
        evalarray = np.zeros((n, nrep, nvar))

    # case where EVALARG is a vector of values to be used for all curves
    evalarg = np.array(evalarg)
    if len(evaldim) <= 1:
        # evalarg[evalarg < rangeval[0]  - 1e-10] = np.nan
        # evalarg[evalarg > rangeval[1] + 1e-10] = np.nan
        basismat = eval_basis(evalarg, basisobj, Lfdobj, returnMatrix)
        # print(evalarg)
        # evaluate the functions at arguments in EVALARG

        if ndim <= 2:
            evalarray = np.dot(basismat, coef)
        else:
            evalarray = np.zeros([n, nrep, nvar])
            for i in range(nvar):
                evalarray[:, :, i] = np.dot(basismat, coef[:, :, i])
    else:
        # case of evaluation values varying from curve to curve
        for i in range(nrep):
            evalargi = evalarg[:, i]
            basismat = eval_basis(evalargi, basisobj, Lfdobj, returnMatrix)
            if ndim == 2:
                evalarray[:, i] = np.dot(basismat, coef[:, i])
            elif ndim == 3:
                for j in range(nvar):
                    evalarray[:, i, j] = np.dot(basismat, coef[:, i, j])

    return np.matrix(evalarray)


class fdname:
    def __init__(self, time, reps, values):
        self.args = time
        self.reps = reps
        self.funs = values


def getbasismatrix(evalarg, basisobj, nderiv=[0], returnMatrix=False):
    if (isinstance(basisobj, int) or isinstance(basisobj, list)) and isinstance(evalarg, basis):
        temp = basisobj
        basisobj = evalarg
        evalarg = temp

    # check EVALARG
    if evalarg is None:
        raise ValueError("evalarg required; is None.")
    Evalarg = evalarg

    # check BASISOBJ
    if not isinstance(basisobj, basis):
        raise ValueError("Second argument is not a basis object.")

    # Search for stored basis matrix and return it if found
    if not (basisobj.basisvalues is None or len(basisobj.basisvalues) == 0):
        # one or more stored basis matrix found,
        # check that requested derivative is available
        basisvalues = basisobj.basisvalues
        if not isinstance(basisvalues, list):
            raise ValueError("BASISVALUES is not a list.")
        nvalues = len(basisvalues)
        N = len(evalarg)
        OK = False
        for ivalues in range(nvalues):
            basisvaluesi = basisvalues[ivalues]
            if len(basisvaluesi) >= nderiv[0] + 2:
                if N == len(basisvaluesi):
                    if all(basisvaluesi == evalarg):
                        basismat = basisvaluesi[nderiv[0] + 2]
                        OK = True

        if OK:
            return basismat
        # if len(np.shape(basismat)) == 2:
        #     return basismat

    # Compute the basis matrix and return it
    # Extract information about the basis
    type = basisobj.type
    nbasis = basisobj.nbasis
    params = basisobj.params
    rangeval = basisobj.rangeval
    dropind = basisobj.dropind

    # select basis and evaluate it at EVALARG values

    #   B-Spline Basis
    if type == 'bspline':
        if len(params) == 0:
            breaks = [rangeval[0], rangeval[1]]
        else:
            breaks = [rangeval[0]] + params.tolist() + [rangeval[1]]
        norder = nbasis - len(breaks) + 2
        basismat = bsplineS(evalarg, breaks, norder, nderiv)
    elif type == 'const':
        basismat = np.ones([len(evalarg), 1])
    else:
        raise ValueError("Only constant and bspline basis are supported now. Basis type not recognized.")
    basismat = np.array(basismat)

    return basismat


def bsplineS(x, breaks, norder=4, nderiv=0, returnMatrix=False):
    x = np.array(x)
    n = len(x)
    tol = 1e-14
    nbreaks = len(breaks)
    if nbreaks < 2:
        raise ValueError("Number of knots less than 2.")
    if min(np.diff(np.array(breaks))) < 0:
        raise ValueError("Knots are not increasing.")

    if max(x) > (max(breaks) + tol) or min(x) < (min(breaks) - tol):
        raise ValueError("Knots do not span the values of 'X'.")

    if x[n - 1] > breaks[nbreaks - 1]:
        breaks[nbreaks - 1] = x[n - 1]
    if x[0] < breaks[0]:
        breaks[0] = x[0]

    if norder > 20:
        raise ValueError("NORDER exceeds 20.")
    if norder < 1:
        raise ValueError("NORDER less than 1.")
    if nderiv[0] > 19:
        raise ValueError("NDERIV exceeds 19.")
    if norder < 0:
        raise ValueError("NDERIV less than 0.")
    if nderiv[0] >= norder:
        raise ValueError("NDERIV cannot be as large as order of B-Spline.")

    knots = [breaks[0]] * (norder - 1) + breaks + [breaks[nbreaks - 1]] * (norder - 1)
    deriv = [nderiv[0]] * n
    nbasis = nbreaks + norder - 2

    if nbasis >= norder:
        basismat = splineDesign(knots, x, norder, deriv[0])
        return basismat
    else:
        raise ValueError("NBASIS is less than NORDER.")


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


def fd2list(fdobj):
    # get the coefficient matrix and the basis
    coef = fdobj.coef
    coefsize = np.shape(coef)
    nrep = coefsize[1]

    # check whether FDOBJ is univariate

    if len(coefsize) > 2:
        raise ValueError("FDOBJ is not univariate.")

    fdlist = []
    for i in range(nrep):
        fdlist.append(fd(coef[:, i], fdobj.basisobj, fdobj.fdnames))

    return fdlist


class fdPar:
    def __init__(self, fdobj=None, Lfdobj=None, lamdba=0, estimate=True, penmat=None):
        # default fdPar object
        if not isinstance(fdobj, fd):
            if fdobj is None:
                fdobj = fd()
            else:
                if isinstance(fdobj, basis):
                    nbasis = fdobj.nbasis
                    dropind = fdobj.dropind
                    nbasis = nbasis - dropind if len(dropind) != 0 else nbasis
                    coefs = np.zeros([nbasis, nbasis])
                    # fdnames = ['time', 'reps 1', 'values']
                    # if fdobj.name is not None:
                    #     basisnames = [i for i in fdobj.names if fdobj.names.index(i) not in dropind] if len(
                    #         dropind) > 0 else fdobj.names
                    #     fdnames[1] = basisnames
                    fdobj = fd(coefs, fdobj, fdnames=None)
                elif isinstance(fdobj, int) or isinstance(fdobj, np.ndarray) or isinstance(fdobj, list):
                    fdobj = fd(basisobj=fdobj)
                else:
                    raise ValueError("First argument is neither a functional data object nor a basis object.")
        else:
            nbasis = fdobj.basisobj.nbasis

        # check parameters
        # check Lfobj
        if Lfdobj is None:
            norder = norder_bspline(fdobj.basisobj) if fdobj.basisobj.type == 'bspline' else 2
            Lfdobj = int2Lfd(max(0, norder - 2))
        else:
            Lfdobj = int2Lfd(Lfdobj)

        if not isinstance(Lfdobj, Lfd):
            raise ValueError("'Lfobj' is not a linear differential operator object.")

        # check lambda

        if not isinstance(lamdba, int):
            raise ValueError("Class of LAMBDA is not numeric.")
        if lamdba < 0:
            raise ValueError("LAMBDA is negative.")

        # check estimate

        if not isinstance(estimate, bool):
            raise ValueError("Class of ESIMATE is not logical.")

        # check penmat

        if penmat is not None:
            penmatsize = np.shape(penmat)
            if any(i != nbasis for i in penmatsize):
                raise ValueError("Dimensions of PENMAT are not correct.")

        # setup attributes

        self.fd = fdobj
        self.Lfd = Lfdobj
        self.lamdba = lamdba
        self.estimate = estimate
        self.penmat = penmat


def fdParcheck(fdParobj, ncurve=None):
    if isinstance(fdParobj, basis) and ncurve is None:
        raise ValueError("First argument is basis object and second argument is missing.")
    if not isinstance(fdParobj, fdPar):
        if isinstance(fdParobj, fd):
            fdParobj = fdPar(fdParobj)
        if isinstance(fdParobj, basis):
            nbasis = fdParobj.nbasis
            fdParobj = fdPar(fd(np.zeros([nbasis, ncurve]), fdParobj))
        else:
            raise ValueError(
                "'fdParobj' is not a functional parameter object, not a functional data object and not a basis object.")

    return fdParobj


class Lfd:
    def __init__(self, nderiv=[0], bwtlist=[]):
        if round(nderiv[0]) != nderiv[0]:
            raise ValueError("Order of operator is not an integer.")
        if nderiv[0] < 0:
            raise ValueError("Order of operator is negative.")

        # check that bwtlist is either a list or a fd object

        if not isinstance(bwtlist, list) and not isinstance(bwtlist, fd) and bwtlist is not None:
            raise ValueError("BWTLIST is neither a LIST or a FD object.")

        # if bwtlist is missing or NULL, convert it to a constant basis FD object

        if bwtlist is None:
            bwtlist = []
            if nderiv[0] > 0:
                conbasis = create_constant_basis()
                for j in range(nderiv[0]):
                    bwtlist.append(fd(coef=0, basisobj=conbasis))

        # if BWTLIST is a fd object, convert to a list object

        if isinstance(bwtlist, fd):
            bwtlist = fd2list(bwtlist)

        # check size of bwtlist

        nbwt = len(bwtlist)

        if nbwt != nderiv[0] and nbwt != nderiv[0] + 1:
            raise ValueError("The size of bwtlist inconsistent with NDERIV.")

        # check individual list entries for class
        # and find a default range

        if nderiv[0] > 0:
            rangevec = [0, 1]
            for j in range(nbwt):
                bfdj = bwtlist[j]

                if isinstance(bfdj, fdPar):
                    bfdj = bfdj.fd
                    bwtlist[j] = bfdj

                if not isinstance(bfdj, fd) and not isinstance(bfdj, int):
                    raise ValueError("AN element of BWTLIST contains something other than an fd object or an integer.")

                if isinstance(bfdj, fd):
                    bbasis = bfdj.basisobj
                    rangevec = bbasis.rangeval
                else:
                    if len(bfdj) == 1:
                        bwtfd = fd(bfdj, conbasis)
                        bwtlist[j] = bwtfd
                    else:
                        raise ValueError("An element of BWTLIST conatins a more than one integer.")

            # check that the ranges are compatible

            for j in range(nbwt):
                bfdj = bwtlist[j]
                if isinstance(bfdj, fdPar):
                    bfdj = bfdj.fd
                bbasis = bfdj.basisobj
                btype = bbasis.type
                if btype != 'const':
                    brange = bbasis.rangeval
                    if rangevec != brange:
                        raise ValueError("Ranges are not compatible.")

        self.nderiv = nderiv
        self.bwlist = bwtlist


def norder_bspline(basisobj):
    return basisobj.nbasis - len(basisobj.params)


def project_basis(y, argvals, basisobj, penalize=False):
    if not isinstance(basisobj, basis):
        raise InputError("Third argument BASISOBJ is not a basis object.")
    basismat = getbasismatrix(argvals, basisobj, [0])
    Bmat = np.matmul(np.transpose(basismat), basismat)

    if penalize:
        penmat = eval_penalty(basisobj)
        penmat = penmat + 1e-10 * max(penmat) * np.identity(np.shape(penmat)[0])
        lamdba = (0.0001 * sum(np.diagonal(Bmat))) / sum(np.diagonal(Bmat))
        Cmat = Bmat + lamdba * penmat
    else:
        Cmat = Bmat

    if len(np.shape(y)) <= 2:
        Dmat = transpose(basismat) * y
        coef = np.linalg.solve(Cmat, Dmat)
    else:
        nvar = np.shape(y)[2]
        coef = np.zeros([basisobj.nbasism, np.shape(y)[1], nvar])
        for ivar in range(nvar):
            Dmat = transpose(basismat) * y[:, :, ivar]
            coef[:, :, ivar] = np.linalg.solve(Cmat, Dmat)

    return coef


def eval_penalty(basisobj, Lfdobj=int2Lfd(0), rng=[0, 1]):
    if isinstance(basisobj, fd):
        basisobj = basisobj.basisobj

    if isinstance(basisobj, fdPar):
        fdobj = basisobj.fd
        basisobj = fdobj.basisobj

    if not isinstance(basisobj, basis):
        raise ValueError("Argument BASISOBJ is not a functional basis object.")

    # set up default values
    rangeval = basisobj.rangeval

    # deal with the case where LFDOBJ is an integer
    LFdobj = int2Lfd(Lfdobj)

    # determine basis type
    type = basisobj.type

    # choose appropriate penalty matrix function
    if type == 'bspline':
        penaltymat = bsplinepen(basisobj, Lfdobj, rangeval)
    elif type == 'const':
        rangeval = getbasisrange(basisobj)
        penaltymat = rangeval[1] - rangeval[0]
    else:
        raise ValueError("Basis type not reconizable, cannot find penalty matrix.")

    dropind = basisobj.dropind
    nbasis = basisobj.nbasis

    if len(dropind) > 0:
        index = range(nbasis)
        index = [i for i in index if index.index(i) not in dropind]
        penaltymat = penaltymat[index, index]

    return penaltymat


def getbasisrange(basisobj):
    if not isinstance(basisobj, basis):
        raise ValueError("'basisobj' is not a functional basis object.")

    rangeval = basisobj.rangeval
    return rangeval


def bsplinepen(basisobj, Lfdobj=2, returnMatrix=False):
    pass
    # if not isinstance(basisobj, basis):
    #     raise ValueError("First argument is not a basis object.")
    # rng = basisobj.rangeval
    # # check basis type
    # type = basisobj.type
    #
    # # check LFDOBJ
    # Lfdobj = int2Lfd(Lfdobj)
    #
    # # get basis information
    # nbasis = basisobj.nbasis
    # params = basisobj.params
    #
    # # if there are no internal knots, use the monomial penalty
    # if len(params) == 0:
    #     basisobj = create_monomial_basis(basisobj.rangeval, nbasis, np.array(range(nbasis - 1)))
    #     penaltymatrix = monomialpen(basisobj, Lfdobj, rng)
    #     return penaltymatrix
    #
    # # normal case
    # # check
    # nNA = sum(np.isnan(rng))
    # if nNA > 0:
    #     raise ValueError("rng constains NA.")
    # nNAp = sum(np.isnan(params))
    # if nNAp > 0:
    #     raise ValueError("params constains NA")
    #
    # breaks = [rng[0]] + params + [rng[1]]
    # nbreaks = len(breaks)
    # ninterval = nbreaks - 1
    #
    # # check break values
    # if len(breaks) > 2:
    #     raise ValueError("The length of argument breaks is less than 2.")
    #
    # # find the highest order derivative on LFD
    # nderiv = Lfdobj.nderiv
    # norder = nbasis - len(params)
    #
    # # check for order of derivative being equal or greater than order of spline
    #
    # if nderiv >= norder:
    #     raise ValueError("Derivative of order cannot be taken for B-spline of order. \n",
    #                      "Probable cause is a value of the argument.\n",
    #                      "in function create.basis.fd that is too small.")
    #
    # # check for order of derivative being equal to order of spline
    # # minus one, in which case following code won't work
    #
    # if nderiv > 0 and nderiv == norder - 1:
    #     raise ValueError("Penalty matrix cannot be evaluated for derivative of order", nderiv,
    #                      " for B-spline of order ", norder)
    #
    # # special case where LFD is D^NDERIV and NDERIV = NORDER -1
    # bwtlist = Lfdobj.bwlist
    # isintLfd = True
    #
    # if nderiv > 0:
    #     for ideriv in range(nderiv):
    #         fdj = bwtlist[ideriv]
    #         if fdj is not None:
    #             if any(fdj.coefs != 0):
    #                 isintLfd = False
    #                 breaks
    #
    # if isintLfd and nderiv == norder - 1:
    #     halfseq = (breaks[1:(nbreaks - 1)] + breaks[0:(nbreaks - 2)]) / 2
    #     halfmat = bsplineS(halfseq, breaks, norder, nderiv, returnMatrix)
    #     brwidth = np.diff(breaks)
    #     penaltymat = np.matmul(np.matmul(transpose(halfmat), np.diag(breaks)),halfmat)
    #     return penaltymat
    #
    # # look for knot multiplicities within the range
    #
    # intbreaks = [rng[0]]  + params +[rng[1]]
    # index =  (intbreaks )


class bifd:
    def __init__(self, coef=np.zeros([2, 1]), sbasisobj=create_bspline_basis(), tbasisobj=create_bspline_basis(),
                 fdname=None):

        if not isinstance(coef, np.ndarray):
            raise ValueError("coef must be numerical vector or matrix.")
        elif len(np.shape(coef)) == 1:
            raise ValueError("Argument coef is not at least 2 dimensional.")
        else:
            coefd = np.shape(coef)
            ndim = len(coefd)

        if ndim > 4:
            raise ValueError("Argument COEF is not correct.")

        # check SBASISOBJ

        if not isinstance(sbasisobj, basis):
            raise ValueError("Argument SBASISOBJ is not of basis class.")

        if coefd[0] != sbasisobj.nbasis:
            raise ValueError("Number of coefficients does not match number of basis functions from SBASISOBJ.")

        # check TBASISOBJ

        if not isinstance(tbasisobj, basis):
            raise ValueError("Argument "
                             "TBASISOBJ is not of basis class.")

        if coefd[1] != tbasisobj.nbasis:
            raise ValueError("Number of coefficients does not match number of basis functions from TBASISOBJ.")

        # set up number of replicates and number of variables

        if ndim > 2:
            nrep = coefd[2]
        else:
            nrep = 1

        if ndim > 3:
            nvar = coefd[3]
        else:
            nvar = 1

        self.coef = coef
        self.sbasis = sbasisobj
        self.tbasis = tbasisobj
        self.bifdname = fdname


class bifdPar:
    def __init__(self, bifdobj, Lfdobjs=int2Lfd(2), Lfdobjt=int2Lfd(2), lamdbas=0, lamdbat=0, estimate=True):

        if not isinstance(bifdobj, bifd):
            raise ValueError("BIFDOBJ is not a bivariate functional data object.")

        # check the linear differential operators

        LFdobjs = int2Lfd(Lfdobjs)
        LFdobjt = int2Lfd(Lfdobjt)

        if not isinstance(LFdobjs, Lfd):
            raise ValueError("LFDOBJS is not a linear differential operator object.")

        if not isinstance(LFdobjt, Lfd):
            raise ValueError("LFDOBJT is not a linear differential operator object.")

        # check the roughness penalty parameters

        if not isinstance(lamdbas, int):
            raise ValueError("LAMDBAS is not numeric.")

        if lamdbas < 0:
            lamdbas = 0

        if not isinstance(lamdbat, int):
            raise ValueError("LAMDBAT is not numeric.")

        if lamdbat < 0:
            lamdbat = 0

        if not isinstance(estimate, bool):
            raise ValueError('ESTIMATE is not logical.')

        # set up the bifdPar object

        self.bifd = bifdobj
        self.estimate = estimate
        self.lamdbas = lamdbas
        self.lamdbat = lamdbat
        self.Lfds = Lfdobjs
        self.Lfdt = Lfdobjt

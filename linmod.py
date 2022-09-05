from fd import *
from basis import *
from smooth_basis import *


class linmodList:
    def __init__(self, beta0estfd, beta1estbifd, yhatfdobj):
        self.beta0estfd = beta0estfd
        self.beta1estbifd = beta1estbifd
        self.yhatfdobj = yhatfdobj


def linmod(xfdobj, yfdobj, betalist, wtvec=None):
    if not isinstance(xfdobj, fd):
        raise ValueError("XFD is not a functional data object,")

    if not isinstance(yfdobj, fd):
        raise ValueError("YFD is not a functional data object,")

    ybasis = yfdobj.basisobj
    ynbasis = ybasis.nbasis
    ranget = ybasis.rangeval

    xbasis = xfdobj.basisobj
    ranges = xbasis.rangeval

    nfine = np.max((201, 10 * ynbasis + 1))
    tfine = np.linspace(ranget[0], ranget[1], nfine)

    # get dimensions of data

    coefy = yfdobj.coef
    coefx = xfdobj.coef
    coefdx = np.shape(coefx)
    coefdy = np.shape(coefy)
    ncurves = coefdx[1]
    if coefdy[1] != ncurves:
        raise ValueError("Number of obervations in first two arguments do not match.")

    # set up or check weight vector

    if wtvec is not None:
        wtvec = wtcheck(ncurves, wtvec)

    # get basis parameter objects

    if not isinstance(betalist, list):
        raise ValueError("betalist is not a list object.")

    if len(betalist) != 2:
        raise ValueError("betalist not of length 2.")

    alphafdPar = betalist[0]
    betabifdPar = betalist[1]

    if not isinstance(alphafdPar, fdPar):
        raise ValueError("BETACELL[0] is not a fdPar object.")

    if not isinstance(betabifdPar, bifdPar):
        raise ValueError("BETACELL[1] is not a bifdPar object.")

    # get Lfd objects

    alphaLfd = alphafdPar.Lfd
    betasLfd = betabifdPar.Lfds
    betatLfd = betabifdPar.Lfdt

    # get smoothing parameters

    alphalamdba = alphafdPar.lamdba
    betaslamdba = betabifdPar.lamdbas
    betatlamdba = betabifdPar.lamdbat

    # get basis object
    alphafd = alphafdPar.fd
    alphabasis = alphafd.basisobj
    alpharange = alphabasis.rangeval

    if (alpharange[0] != ranget[0]) or (alpharange[1] != ranget[1]):
        raise ValueError("Range of ALPHAFD coefficient and YFD not compatible.")

    betabifd = betabifdPar.bifd

    betasbasis = betabifd.sbasis
    betasrange = betasbasis.rangeval
    if (betasrange[0] != ranges[0]) or (betasrange[1] != ranges[1]):
        raise ValueError("Range of BETASFD coefficient and XFD not compatible.")

    betatbasis = betabifd.tbasis
    betatrange = betatbasis.rangeval
    if (betatrange[0] != ranget[0]) or (betatrange[1] != ranget[1]):
        raise ValueError("Range of BETATFD coefficient and YFD not compatible.")

    # get numbers of basis functions

    alphanbasis = alphabasis.nbasis
    betasnbasis = betasbasis.nbasis
    betatnbasis = betatbasis.nbasis

    # get inner product of basis functions and data functions

    Finprod = inprod(ybasis, alphabasis)
    Ginprod = inprod(ybasis, betatbasis)
    Hinprod = inprod(xbasis, betasbasis)

    ycoef = yfdobj.coef
    xcoef = xfdobj.coef
    Fmat = np.matmul(np.transpose(ycoef), Finprod)
    Gmat = np.matmul(np.transpose(ycoef), Ginprod)
    Hmat = np.matmul(np.transpose(xcoef), Hinprod)

    if wtvec is None:
        HHCP = np.matmul(np.transpose(Hmat), Hmat)
        HGCP = np.matmul(np.transpose(Hmat), Gmat)
        H1CP = np.matrix(np.sum(Hmat, axis=0))
        F1CP = np.matrix(np.sum(Fmat, axis=0))
    else:
        HHCP = np.matmul(np.transpose(Hmat), np.multiply(np.outer(wtvec, betasnbasis), Hmat))
        HGCP = np.matmul(np.transpose(Hmat), np.multiply(np.outer(wtvec, betatnbasis), Gmat))
        H1CP = np.matmul(np.transpose(Hmat), wtvec)
        F1CP = np.matmul(np.transpose(Fmat), wtvec)

    # get inner product of basis functions

    alphattmat = inprod(alphabasis, alphabasis)
    betalttmat = inprod(betatbasis, alphabasis)
    betassmat = inprod(betasbasis, betasbasis)
    betattmat = inprod(betatbasis, betatbasis)

    # get penalty matrix
    if alphalamdba > 0:
        alphapenmat = eval_penalty(alphabasis, alphaLfd)
    else:
        alphapenmat = 0

    if betaslamdba > 0:
        betaspenmat = eval_penalty(betasbasis, betasLfd)
    else:
        betaspenmat = 0

    if betatlamdba > 0:
        betatpenmat = eval_penalty(betatbasis, betatLfd)
    else:
        betatpenmat = 0

    # set up coefficient matrix and right side fro stationary equations

    betan = betasnbasis * betatnbasis
    ncoef = alphanbasis + betan
    Cmat = np.zeros([ncoef, ncoef])
    Dmat = np.zeros([ncoef, 1])

    # row for alpha

    ind11 = 0
    ind12 = alphanbasis
    ind21 = ind11
    ind22 = ind12
    Cmat[ind11:ind12, ind21:ind22] = ncurves * alphattmat
    if alphalamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + alphalamdba * alphapenmat
    ind21 = alphanbasis
    ind22 = alphanbasis + betan
    Cmat[ind11:ind12, ind21:ind22] = np.kron(H1CP, betalttmat)

    Dmat[ind11:ind12] = np.transpose(F1CP)

    # row for data
    ind11 = alphanbasis
    ind12 = alphanbasis + betan
    ind21 = 0
    ind22 = alphanbasis
    Cmat[ind11:ind12, ind21:ind22] = np.transpose(Cmat[ind21:ind22, ind11:ind12])
    ind21 = ind11
    ind22 = ind12
    Cmat[ind11:ind12, ind21:ind22] = np.kron(HHCP, betattmat)
    if betaslamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + betaslamdba * np.kron(betaspenmat, betattmat)

    if betatlamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + betatlamdba * np.kron(betassmat, betatpenmat)

    Dmat[ind11:ind12] = HGCP.reshape((betan, 1))

    # solve the equations
    coefvec = np.linalg.solve(Cmat, Dmat)

    # set up the coefficient function estimates
    # functional structure for the alpha function

    ind11 = 0
    ind12 = alphanbasis
    alphacoef = coefvec[ind11:ind12]
    alphafd = fd(alphacoef, alphabasis)

    # bi-functional structure for the beta function
    ind11 = alphanbasis
    ind12 = betan + alphanbasis
    betacoef = np.transpose(np.matrix(coefvec[ind11:ind12]).reshape((betatnbasis, betasnbasis)))
    betafd = bifd(np.transpose(betacoef), betasbasis, betatbasis)

    # functional data structure for the yhat functions
    xbetacoef = np.dot(betacoef, np.transpose(Hmat))
    xbetafd = fd(xbetacoef, betatbasis)
    yhatmat = np.matmul(eval_fd(tfine, alphafd), np.ones([1, ncurves])) + eval_fd(tfine, xbetafd)
    yhatfd = smooth_basis(tfine, yhatmat, ybasis).fd

    linmodres = linmodList(beta0estfd=alphafd, beta1estbifd=betafd, yhatfdobj=yhatfd)
    return linmodres


def predit_linmod(linmodres, newdata=None):
    if newdata is None:
        return linmodres.yhatfdobj
    else:
        xbasis = newdata.basisobj
        xnbasis = xbasis.nbasis
        ranget = xbasis.rangeval
        coefx = newdata.coef
        coefdx = np.shape(coefx)
        ncurves = coefdx[1]

        nfine = np.max((201, 10 * xnbasis + 1))
        tfine = np.linspace(ranget[0], ranget[1], nfine)

        alphafd = linmodres.beta0estfd
        betasbasis = linmodres.beta1estbifd.sbasis
        Hinprod = inprod(xbasis, betasbasis)
        xcoef = coefx
        Hmat = np.dot(np.transpose(xcoef), Hinprod)
        betacoef = np.transpose(linmodres.beta1estbifd.coef)
        xbetacoef = np.dot(betacoef, np.transpose(Hmat))
        xbetafd = fd(xbetacoef, linmodres.beta1estbifd.tbasis)
        yhatmat = np.dot(eval_fd(tfine, alphafd), np.ones([1, ncurves])) + eval_fd(tfine, xbetafd)
        return smooth_basis(tfine, yhatmat, xbasis).fd


def functionalBoosting(x_function1, x_function2, yfdobj, betalist, boost_control, step_len):
    result = []
    init = yfdobj.mean()
    coefdim = np.shape(yfdobj.coef)
    init.coef = np.repeat(init.coef, repeats=coefdim[1], axis=1)
    residual = yfdobj - init

    result.append([-1, init])

    i = 2
    while i <= boost_control:
        model_function1 = linmod(x_function1, residual, betalist)
        model_function2 = linmod(x_function2, residual, betalist)

        sse1 = inprod(((model_function1.yhatfdobj - residual) * (model_function1.yhatfdobj - residual))).sum()
        sse2 = inprod(((model_function2.yhatfdobj - residual) * (model_function2.yhatfdobj - residual))).sum()

        resid = [sse1, sse2]
        # print(sse1, sse2)
        best = np.argmin(resid)
        if best == 0:
            bestmodel = model_function1
        else:
            bestmodel = model_function2
        result.append([best, bestmodel])
        # print("Iteration ", i, "Best model is ", best)
        residual = yfdobj - pred_gradboost1(result, step_len)
        i = i + 1

    return result


def pred_gradboost1(res, step_length):
    length = len(res)
    y_pred = res[0][1]
    for i in range(1, length):
        y_pred = step_length * res[i][1].yhatfdobj + y_pred
    return y_pred


def pred_gradboost2(res, x_function1, x_function2, step_len):
    length = len(res)
    coefdim = np.shape(x_function1.coef)
    res[0][1].coef = np.repeat(res[0][1].coef, repeats=coefdim[1], axis=1)

    y_pred = res[0][1]
    for i in range(1, length):
        if res[i][0] == 0:
            y_pred = step_len * predit_linmod(res[i][1], x_function1) + y_pred
        else:
            y_pred = step_len * predit_linmod(res[i][1], x_function2) + y_pred

    return y_pred

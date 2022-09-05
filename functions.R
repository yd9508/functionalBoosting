############################################################################
############################################################################
meanfd <- function(x, ...)
{
  if(!inherits(x, 'fd'))
    stop("'x' is not of class 'fd'")
  #
  coef      <- x$coefs
  coefd     <- dim(coef)
  ndim      <- length(coefd)
  basisobj  <- x$basis
  nbasis    <- basisobj$nbasis
  dropind   <- basisobj$dropind
  ndropind  <- length(dropind)
  if (ndim == 2) {
    coefmean  <- matrix(apply(coef,1,mean),nbasis-ndropind,1)
    coefnames <- list(dimnames(coef)[[1]],"Mean")
  } else {
    nvar <- coefd[3]
    coefmean  <- array(0,c(coefd[1],1,nvar))
    for (j in 1:nvar) coefmean[,1,j] <- apply(coef[,,j],1,mean)
    coefnames <- list(dimnames(coef)[[1]], "Mean", dimnames(coef)[[3]])
  }
  fdnames <- x$fdnames
  fdnames[[2]] <- "mean"
  fdnames[[3]] <- paste("mean",fdnames[[3]])
  meanfd <- fd(coefmean, basisobj, fdnames)
  #
  meanfd
}
############################################################################
############################################################################
predict_linmod =function(linmodres, newdata = NULL){
  if(is.null(newdata)){
    return (linmodres$yhatfdobj)
  }
  xbasis = newdata$basis
  xnbasis = xbasis$nbasis
  ranget = xbasis$rangeval
  coefx = newdata$coefs
  coefdx = dim(coefx)
  ncurves = coefdx[2]
  
  nfine = max(201, 10 * xnbasis  + 1)
  tfine = seq(ranget[1], ranget[2], len = nfine)
  
  alphafd = linmodres$beta0estfd
  betasbasis = linmodres$beta1estbifd$sbasis
  Hinprod = inprod(xbasis, betasbasis)
  xcoef = coefx
  Hmat = t(xcoef) %*% Hinprod
  betacoef = t(linmodres$beta1estbifd$coef)
  xbetacoef = betacoef %*% t(Hmat)
  xbetafd = fd(xbetacoef, linmodres$beta1estbifd$tbasis)
  yhatmat = eval.fd(tfine, alphafd) %*% matrix(1, 1, ncurves) + 
    eval.fd(tfine, xbetafd)
  res = smooth.basis(tfine, yhatmat, xbasis)$fd
  
  return(res)
}
############################################################################
############################################################################
gradboost = function(x1, x2, y, betaList, boost_control, step_length, id){
  res = list()
  init = meanfd(y)
  if(id == 1){
    coefdim = dim(y$coefs)
    init$coefs = matrix(rep(x1$coefs[1, ],each=coefdim[1]),nrow = coefdim[1])
    residual = y - init
    res = append(res, list(init))
  }else{
    coefdim = dim(y$coefs)
    init$coefs = matrix(rep(x2$coefs[1, ],each=coefdim[1]),nrow = coefdim[1])
    residual = y - init
    res = append(res, list(init))
  }
  
  
  i = 2
  while(i <= boost_control){
    model1 = linmod_new(x1, residual, betaList)
    model2 = linmod_new(x2, residual, betaList)
    sse1 = sum(inprod((model1$yhatfdobj - residual)^2))
    sse2 = sum(inprod((model2$yhatfdobj - residual)^2))
    best = which.min(c(sse1, sse2))
    print(best)
    if(best == 1){
      model1$id = 1
      res = append(res, list(model1))
    }else if(best == 2){
      model2$id = 2
      res = append(res, list(model2))
    }
    residual = y - pred_gradboost1(res, step_length)
    i = i + 1  
  }
  return(res)
}
############################################################################
############################################################################
pred_gradboost1 = function(res, step_length){
  len = length(res)
  y_pred = res[[1]]
  
  for(i in 2:len){
    y_pred = step_length * res[[i]]$yhatfdobj + y_pred
  }
  return(y_pred)
}
############################################################################
############################################################################
pred_gradboost2 = function(res, x1, x2, step_length){
  len = length(res)
  
  coefdim = dim(x1$coefs)
  if(res[[2]]$id == 1){
    res[[1]]$coefs = matrix(rep(x1$coefs[1, ],each=coefdim[1]),nrow = coefdim[1])
  }else{
    res[[1]]$coefs = matrix(rep(x2$coefs[1, ],each=coefdim[1]),nrow = coefdim[1])
  }

  y_pred = res[[1]]
  
  for(i in 2:len){
    if(res[[i]]$id == 1){
      y_pred = step_length * predict_linmod(res[[i]], x1) + y_pred
    }else if(res[[i]]$id == 2){
      y_pred = step_length * predict_linmod(res[[i]], x2) + y_pred
    }else{
      pred = predict.fRegress(res[[i]])
      pred$coefs = pred$coefs[, 1:coefdim[2]]
      y_pred = step_length * pred + y_pred
    }
  }
  return(y_pred)
}
############################################################################
############################################################################
linmod_new = function (xfdobj, yfdobj, betaList, wtvec = NULL) 
{
  if (!is.fd(xfdobj)) {
    stop("XFD is not a functional data object.")
  }
  if (!is.fd(yfdobj)) {
    stop("YFD is not a functional data object.")
  }
  ybasis = yfdobj$basis
  ynbasis = ybasis$nbasis
  ranget = ybasis$rangeval
  xbasis = xfdobj$basis
  ranges = xbasis$rangeval
  nfine = max(c(201, 10 * ynbasis + 1))
  tfine = seq(ranget[1], ranget[2], len = nfine)
  coefy = yfdobj$coef
  coefx = xfdobj$coef
  coefdx = dim(coefx)
  coefdy = dim(coefy)
  ncurves = coefdx[2]
  if (coefdy[2] != ncurves) {
    stop("Numbers of observations in first two arguments do not match.")
  }
  if (!is.null(wtvec)) 
    wtvec = wtcheck(ncurves, wtvec)
  if (!inherits(betaList, "list")) 
    stop("betaList is not a list object.")
  if (length(betaList) != 2) 
    stop("betaList not of length 2.")
  alphafdPar = betaList[[1]]
  betabifdPar = betaList[[2]]
  if (!inherits(alphafdPar, "fdPar")) {
    stop("BETACELL[[1]] is not a fdPar object.")
  }
  if (!inherits(betabifdPar, "bifdPar")) {
    stop("BETACELL[[2]] is not a bifdPar object.")
  }
  alphaLfd = alphafdPar$Lfd
  betasLfd = betabifdPar$Lfds
  betatLfd = betabifdPar$Lfdt
  alphalambda = alphafdPar$lambda
  betaslambda = betabifdPar$lambdas
  betatlambda = betabifdPar$lambdat
  alphafd = alphafdPar$fd
  alphabasis = alphafd$basis
  alpharange = alphabasis$rangeval
  if (alpharange[1] != ranget[1] || alpharange[2] != ranget[2]) {
    stop("Range of ALPHAFD coefficient and YFD not compatible.")
  }
  betabifd = betabifdPar$bifd
  betasbasis = betabifd$sbasis
  betasrange = betasbasis$rangeval
  if (betasrange[1] != ranges[1] || betasrange[2] != ranges[2]) {
    stop("Range of BETASFD coefficient and XFD not compatible.")
  }
  betatbasis = betabifd$tbasis
  betatrange = betatbasis$rangeval
  if (betatrange[1] != ranget[1] || betatrange[2] != ranget[2]) {
    stop("Range of BETATFD coefficient and YFD not compatible.")
  }
  alphanbasis = alphabasis$nbasis
  betasnbasis = betasbasis$nbasis
  betatnbasis = betatbasis$nbasis
  Finprod = inprod(ybasis, alphabasis)
  Ginprod = inprod(ybasis, betatbasis)
  Hinprod = inprod(xbasis, betasbasis)
  ycoef = yfdobj$coef
  xcoef = xfdobj$coef
  Fmat = t(ycoef) %*% Finprod
  Gmat = t(ycoef) %*% Ginprod
  Hmat = t(xcoef) %*% Hinprod
  if (is.null(wtvec)) {
    HHCP = t(Hmat) %*% Hmat
    HGCP = t(Hmat) %*% Gmat
    H1CP = as.matrix(apply(Hmat, 2, sum))
    F1CP = as.matrix(apply(Fmat, 2, sum))
  }
  else {
    HHCP = t(Hmat) %*% (outer(wtvec, rep(betasnbasis)) * 
                          Hmat)
    HGCP = t(Hmat) %*% (outer(wtvec, rep(betatnbasis)) * 
                          Gmat)
    H1CP = t(Hmat) %*% wtvec
    F1CP = t(Fmat) %*% wtvec
  }
  alphattmat = inprod(alphabasis, alphabasis)
  betalttmat = inprod(betatbasis, alphabasis)
  betassmat = inprod(betasbasis, betasbasis)
  betattmat = inprod(betatbasis, betatbasis)
  if (alphalambda > 0) {
    alphapenmat = eval.penalty(alphabasis, alphaLfd)
  }
  else {
    alphapenmat = NULL
  }
  if (betaslambda > 0) {
    betaspenmat = eval.penalty(betasbasis, betasLfd)
  }
  else {
    betaspenmat = NULL
  }
  if (betatlambda > 0) {
    betatpenmat = eval.penalty(betatbasis, betatLfd)
  }
  else {
    betatpenmat = NULL
  }
  betan = betasnbasis * betatnbasis
  ncoef = alphanbasis + betan
  Cmat = matrix(0, ncoef, ncoef)
  Dmat = matrix(0, ncoef, 1)
  ind1 = 1:alphanbasis
  ind2 = ind1
  Cmat[ind1, ind2] = ncurves * alphattmat
  if (alphalambda > 0) {
    Cmat[ind1, ind2] = Cmat[ind1, ind2] + alphalambda * 
      alphapenmat
  }
  ind2 = alphanbasis + (1:betan)
  Cmat[ind1, ind2] = t(kronecker(H1CP, betalttmat))
  Dmat[ind1] = F1CP
  ind1 = alphanbasis + (1:betan)
  ind2 = 1:alphanbasis
  Cmat[ind1, ind2] = t(Cmat[ind2, ind1])
  ind2 = ind1
  Cmat[ind1, ind2] = kronecker(HHCP, betattmat)
  if (betaslambda > 0) {
    Cmat[ind1, ind2] = Cmat[ind1, ind2] + betaslambda * 
      kronecker(betaspenmat, betattmat)
  }
  if (betatlambda > 0) {
    Cmat[ind1, ind2] = Cmat[ind1, ind2] + betatlambda * 
      kronecker(betassmat, betatpenmat)
  }
  Dmat[ind1] = matrix(t(HGCP), betan, 1)
  coefvec = ginv(Cmat) %*% Dmat 
  ind1 = 1:alphanbasis
  alphacoef = coefvec[ind1]
  alphafdnames = yfdobj$fdnames
  alphafdnames[[3]] = "Intercept"
  alphafd = fd(alphacoef, alphabasis, alphafdnames)
  ind1 = alphanbasis + (1:betan)
  betacoef = matrix(coefvec[ind1], betatnbasis, betasnbasis)
  betafdnames = xfdobj$fdnames
  betafdnames[[3]] = "Reg. Coefficient"
  betafd = bifd(t(betacoef), betasbasis, betatbasis, betafdnames)
  xbetacoef = betacoef %*% t(Hmat)
  xbetafd = fd(xbetacoef, betatbasis)
  yhatmat = eval.fd(tfine, alphafd) %*% matrix(1, 1, ncurves) + 
    eval.fd(tfine, xbetafd)
  yhatfd = smooth.basis(tfine, yhatmat, ybasis)$fd
  linmodList = list(beta0estfd = alphafd, beta1estbifd = betafd, 
                    yhatfdobj = yhatfd)
  return(linmodList)
}










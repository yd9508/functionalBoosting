setwd("D:\\OneDrive - Binghamton University\\FunctionalBoosting")
y_pred_expx = read.csv("y_pred_expx_coef.csv")
y_pred_expy = read.csv("y_pred_expy_coef.csv")
library(fda)
library(ggplot2)
basis = create.bspline.basis(c(1, 200), 20)
expx_test = as.matrix(read.csv("expx_test.csv", header = FALSE))
simx_test = as.matrix(read.csv("simx_test.csv", header = FALSE))
expy_test = as.matrix(read.csv("expy_test.csv", header = FALSE))
simy_test = as.matrix(read.csv("simy_test.csv", header = FALSE))
expx_train = as.matrix(read.csv("expx_train.csv", header = FALSE))
simx_train = as.matrix(read.csv("simx_train.csv", header = FALSE))
expy_train = as.matrix(read.csv("expy_train.csv", header = FALSE))
simy_train = as.matrix(read.csv("simy_train.csv", header = FALSE))
expx = cbind(expx_test, expx_train)
expy = cbind(expy_test, expy_train)
simx = cbind(simx_test, simx_train)
simy = cbind(simy_test, simy_train)
len = dim(expx)[1]
n = dim(expx)[2]
expx_fd = smooth.basis(seq(1, 200), expx, basis)$fd
expy_fd = smooth.basis(seq(1, 200), expy, basis)$fd
simx_fd = smooth.basis(seq(1, 200), simx, basis)$fd
simy_fd = smooth.basis(seq(1, 200), simy, basis)$fd

strings = as.character(seq(1, n, 1))
for(i in 1:n){
  expx_fd = smooth.basis(seq(1, 200), expx[, i], basis)$fd
  expy_fd = smooth.basis(seq(1, 200), expy[, i], basis)$fd
  simx_fd = smooth.basis(seq(1, 200), simx[, i], basis)$fd
  simy_fd = smooth.basis(seq(1, 200), simy[, i], basis)$fd
  sse1 = sum(inprod((simx_fd - expx_fd)^2))
  sse2 = sum(inprod((simy_fd - expy_fd)^2))
  cat(i, sse1, sse2,'\n')
  x = array(0, 400)
  x[1:200] = expx[, i]
  x[201:400] = simx[, i]
  y = array(0, 400)
  y[1:200] = expy[, i]
  y[201:400] = simy[, i]
  group = array(0, 400)
  group[1:200] = rep(1, 200)
  group[201:400] = rep(2, 200)
  group = as.factor(group)
  df = data.frame( x= x, y = y, group = group)
  myplot = ggplot(data = df, aes(x = x, y = y, group= group, color = group)) + geom_line()  +  scale_colour_discrete(name  ="Lines",breaks=c( "1","2"),labels=c( "Experiment", "Simulated"))
  png(paste0(strings[i],"plot",".png", seq = ""))
  print(myplot)
  dev.off()
}
exclude = c(6, 7, 15, 20, 22, 23, 24, 26, 33, 35, 37, 41, 48, 53, 55, 60, 62, 67, 68, 75, 77, 90, 96, 98, 102, 104, 110)
simx = simx[,-exclude]
simy = simy[,-exclude]
expx = expx[,-exclude]
expy = expy[,-exclude]

simx_test = simx_train =  vector('numeric', 0)
simy_test = simy_train =  vector('numeric', 0)
expx_test = expx_train =  vector('numeric', 0)
expy_test = expy_train =  vector('numeric', 0)
for(i in 1:dim(simx)[2]){
  rand = runif(1)
  if(rand < 0.8){
    simx_train = cbind(simx_train, simx[, i])
    simy_train = cbind(simy_train, simy[, i])
    expx_train = cbind(expx_train, expx[, i])
    expy_train = cbind(expy_train, expy[, i])
  }else{
    simx_test = cbind(simx_test, simx[, i])
    simy_test = cbind(simy_test, simy[, i])
    expx_test = cbind(expx_test, expx[, i])
    expy_test = cbind(expy_test, expy[, i])
  }
}


# expx_test_fd = smooth.basis(seq(1, 200), expx_test, basis)$fd
# expy_test_fd = smooth.basis(seq(1, 200), expy_test, basis)$fd

# for(i in 1: dim(expx_test)[2]){
#   for(j in 1: dim(y_pred_expx)[1]){
#     expx_test_fd$coefs[j, i] = y_pred_expx[j, i + 1]
#     expy_test_fd$coefs[j, i] = y_pred_expy[j, i + 1]
#   }
# }
# i = 4
# x_pred = eval.fd(seq(1, 200), expx_test_fd)[,i]
# y_pred = eval.fd(seq(1, 200), expy_test_fd)[,i]

x = array(0, 400)
# x[1:200] = x_pred
x[1:200] = expx_test[, i]
x[201:400] = simx_test[, i]
y = array(0, 400)
# y[1:200] = y_pred
y[1:200] = expy_test[, i]
y[201:400] = simy_test[, i]
group = array(0, 400)
# group[1:200] = rep(1, 200)
group[1:200] = rep(2, 200)
group[201:400] = rep(3, 200)
group = as.factor(group)
df = data.frame( x= x, y = y, group = group)
myplot = ggplot(data = df, aes(x = x, y = y, group= group, color = group)) + geom_line()  +  scale_colour_discrete(name  ="Lines",breaks=c( "2","3"),labels=c( "Experiment", "Simulated"))
myplot

# for(i in 1: dim(y_pred_expx)[2]){
#   x_pred = eval.fd(seq(1, 200), expx_test_fd)[,i]
#   y_pred = eval.fd(seq(1, 200), expy_test_fd)[,i]
# 
#   x = array(0, 600)
#   x[1:200] = x_pred
#   x[201:400] = expx_test[, i]
#   x[401:600] = simx_test[, i]
#   y = array(0, 600)
#   y[1:200] = y_pred
#   y[201:400] = expy_test[, i]
#   y[401:600] = simy_test[, i]
#   group = array(0, 600)
#   group[1:200] = rep(1, 200)
#   group[201:400] = rep(2, 200)
#   group[401:600] = rep(3, 200)
#   group = as.factor(group)
#   df = data.frame( x= x, y = y, group = group)
#   myplot = ggplot(data = df, aes(x = x, y = y, group= group, color = group)) + geom_line()  +  scale_colour_discrete(name  ="Lines",breaks=c("1", "2","3"),labels=c("Predicted", "Experiment", "Simulated"))
#   strings = as.character(seq(1, dim(y_pred_expx)[2], 1))
#   png(paste0(strings[i],"test_result",".png", seq = ""))
#   print(myplot)
#   dev.off()
# }






























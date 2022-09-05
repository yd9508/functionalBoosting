library(fda)
library(ggplot2)

setwd("/Users/yuding/Library/CloudStorage/OneDrive-BinghamtonUniversity/FunctionalBoosting")
source("functions.R")
simx_test = as.matrix(read.csv("simx_test.csv", header = F))
simy_test = as.matrix(read.csv("simy_test.csv", header = F))
expx_test = as.matrix(read.csv("expx_test.csv", header = F))
expy_test = as.matrix(read.csv("expy_test.csv", header = F))

simx_train = as.matrix(read.csv("simx_train.csv", header = F))
simy_train = as.matrix(read.csv("simy_train.csv", header = F))
expx_train = as.matrix(read.csv("expx_train.csv", header = F))
expy_train = as.matrix(read.csv("expy_train.csv", header = F))

expx = as.matrix(cbind(expx_test, expx_train))
expy = as.matrix(cbind(expy_test, expy_train))
simx = as.matrix(cbind(simx_test, simx_train))
simy = as.matrix(cbind(simy_test, simy_train))


# for(i in 1: dim(expx)[2]){
#   x = array(0, 400)
#   x[1:200] = expx[, i]
#   x[201:400] =  simx[, i]
#   y = array(0, 400)
#   y[1:200] = expy[, i]
#   y[201:400] =  simy[, i]
#   group = array(0, 400)
#   group[1:200] = rep(1, 200)
#   group[201:400] = rep(2, 200)
#   group = as.factor(group)
#   df = data.frame( x= x, y = y, group = group)
#   myplot = ggplot(data = df, aes(x = x, y = y, group= group, color = group)) + geom_line()  +  scale_colour_discrete(name  ="Lines",breaks=c("1", "2"),labels=c( "Experiment", "Simulated"))
#   strings = as.character(seq(1, dim(expx)[2], 1))
#   png(paste0(strings[i],"test_result",".png", seq = ""))
#   print(myplot)
#   dev.off()
# }
x = array(dim = dim(expx)[2])
y = array(dim = dim(expx)[2])
angle1 = array(dim = dim(expx)[2])
angle2 = array(dim = dim(expx)[2])
color = array(dim = dim(expx)[2])
for(i in 1:dim(expx)[2]){
  x[i] = expx[1, i]
  y[i] = expy[1, i]

  angle1[i] = findAngle((expx[100, i] - expx[1, i]), (expy[100, i] - expy[1, i]))
  angle2[i] = findAngle((simx[100, i] - simx[1, i]), (simy[100, i] - simy[1, i]))
  if(angle1[i] >= angle2[i] && angle2[i] >= 1){
    color[i] = 1
  }else if(angle1[i] >= angle2[i] && angle2[i] < 1){
    color[i] = 2
  }else if(angle1[i] < angle2[i] && angle2[i] >= 1){
    color[i] = 3
  }else{
    color[i] = 4
  }
  
  #cat(expx[1, i], expy[1, i], "\n")
}
plot(x, y ,col = color)
text(x + 5, y + 5, labels=angle2)




findAngle = function(x, y){
  if(x >= 0 && y >= 0){
    return(round(atan(y / x), 1))
  }else if(x < 0 && y < 0){
    return(1 + round(atan(y / x), 1))
  }else if(x < 0 && y >= 0){
    return(1.5 + round(atan(y / x), 1))
  }else{
    return(2 + round(atan(y / x), 1))
  }
}

classifier = function(x, y){
  return(1100 - 4 * x[1] /11 - y[1] >= 0)
}




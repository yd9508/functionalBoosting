import numpy as np
from fda import *

def main():
    daybasis = create_bspline_basis([0, 365], nbasis=20)
    day = np.genfromtxt('day.csv', delimiter=',')
    temperature = np.genfromtxt('Temperature.csv', delimiter=',')
    precipitation = np.genfromtxt('Precipitation.csv', delimiter=',')
    scalar = np.genfromtxt('scalar.csv', delimiter=',')
    code = np.genfromtxt('regioncode.csv', delimiter=',')
    argvals = day
    y = temperature
    fdParobj = daybasis
    tempfd = smooth_basis(day, temperature, fdParobj).fd
    precfd = smooth_basis(day, precipitation, fdParobj).fd
    betabasis = create_bspline_basis([0, 365], 20)
    betaPar = fdPar(betabasis, 0, 0)
    beta2_2 = bifd(np.linspace(1, pow(20, 2), pow(20, 2)).reshape((20, 20)), create_bspline_basis([0, 365], 20),
                   create_bspline_basis([0, 365], 20))
    bifdbasis = bifdPar(beta2_2, 0, 0, 0, 0)
    betaList = [betaPar, bifdbasis]
    x_function = precfd
    yfdobj = tempfd
    boost_control = 20
    step_len = 0.1
    duplicates_sample = 20
    duplicates_learner = 20
    fb = functionalBoosting(x_function, x_function, yfdobj, betaList, boost_control, step_len, duplicates_sample, duplicates_learner)
    y_pred = pred_gradboost1(fb, 0.1)
    return y_pred


if __name__ == "__main__":
    main()


